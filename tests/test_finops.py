"""Smoke tests for the FinOps Arbitrage Engine. Run with: python -m unittest tests.test_finops -v"""

import os
import unittest

from finops.cost_calculator import analyze_api_costs, resolve_pricing
from finops.data_loader import DataLoadError, load_records
from finops.infrastructure_audit import analyze_infrastructure
from finops.report_generator import generate_report


class TestPricingResolution(unittest.TestCase):
    def test_exact_and_prefix_matches(self):
        self.assertEqual(resolve_pricing("gpt-4o")[0], (2.50, 10.00))
        self.assertEqual(resolve_pricing("gpt-4o-2024-08-06")[0], (2.50, 10.00))
        self.assertEqual(resolve_pricing("anthropic/claude-3-5-sonnet")[1], "frontier")
        self.assertEqual(resolve_pricing("llama-3.1-8b-instruct")[1], "budget")

    def test_unknown_model_assumes_frontier(self):
        price, tier = resolve_pricing("totally-unknown-model")
        self.assertEqual(tier, "unknown(frontier-assumed)")
        self.assertGreater(price[1], 0)


class TestArbitrage(unittest.TestCase):
    def _rows(self):
        return [
            {"timestamp": "2025-11-01T09:00:00", "model_used": "gpt-4o",
             "input_tokens": 1_000_000, "output_tokens": 1_000_000,
             "task_type": "classification"},
            {"timestamp": "2025-11-08T09:00:00", "model_used": "gpt-4o",
             "input_tokens": 1_000_000, "output_tokens": 1_000_000,
             "task_type": "complex_reasoning"},
        ]

    def test_routing_saves_money_and_keeps_frontier_for_hard_tasks(self):
        result = analyze_api_costs(self._rows())
        # classification -> budget model (much cheaper); reasoning stays on gpt-4o.
        self.assertEqual(result.records_rerouted, 1)
        self.assertLess(result.optimized_monthly_spend, result.current_monthly_spend)
        routed = [i for i in result.line_items if i.rerouted_to]
        self.assertEqual(routed[0].task_type, "classification")
        self.assertEqual(routed[0].model_used, "gpt-4o")

    def test_savings_scale_with_window(self):
        one_week = analyze_api_costs(self._rows(), sample_days=7)
        four_weeks = analyze_api_costs(self._rows(), sample_days=28)
        # Both are already monthly extrapolations; a 7-day window is 4x denser
        # than a 28-day window for the same record count.
        self.assertAlmostEqual(one_week.current_monthly_spend,
                               four_weeks.current_monthly_spend * 4, places=2)


class TestInfrastructure(unittest.TestCase):
    def _rows(self):
        return [
            {"instance_id": "i-idle-gpu", "instance_type": "g5.12xlarge",
             "avg_cpu_utilization_pct": 8, "avg_gpu_utilization_pct": 6,
             "running_hours_per_day": 24},
            {"instance_id": "i-busy", "instance_type": "g5.2xlarge",
             "avg_cpu_utilization_pct": 55, "avg_gpu_utilization_pct": 80,
             "running_hours_per_day": 24},
        ]

    def test_flags_idle_gpu_but_not_busy_instance(self):
        result = analyze_infrastructure(self._rows())
        self.assertEqual(result.findings[0].instance_id, "i-idle-gpu")
        self.assertIn("i-busy", result.healthy_instances)
        self.assertGreater(result.total_monthly_waste, 0)
        self.assertLessEqual(result.findings[0].monthly_waste,
                             result.findings[0].monthly_cost * 0.85 + 0.01)

    def test_hourly_rate_override(self):
        rows = self._rows()
        rows[0]["hourly_rate"] = 10.0  # e.g. reserved/committed rate
        result = analyze_infrastructure(rows)
        self.assertEqual(result.findings[0].hourly_rate, 10.0)


class TestLoaderAndReport(unittest.TestCase):
    def test_missing_file_raises(self):
        with self.assertRaises(DataLoadError):
            load_records("does_not_exist.csv", ("a",))

    def test_report_contains_dollar_figures(self):
        arb = analyze_api_costs(TestArbitrage()._rows())
        infra = analyze_infrastructure(TestInfrastructure()._rows())
        report = generate_report(arbitrage=arb, infra=infra)
        self.assertIn("FinOps Arbitrage Audit", report)
        self.assertIn("$", report)
        self.assertIn("Module A", report)
        self.assertIn("Module B", report)


class TestPricingConfig(unittest.TestCase):
    def test_load_pricing_merges_overrides(self):
        from finops.pricing_config import Pricing, load_pricing
        pricing = load_pricing()  # bundled config/ dir, mirrors built-ins
        self.assertEqual(pricing.frontier["gpt-4o"], (2.50, 10.00))
        self.assertEqual(pricing.ec2_hourly["g5.12xlarge"], 5.672)
        self.assertIn("config", pricing.source)

        empty = Pricing()  # pure built-in defaults
        self.assertEqual(empty.frontier["gpt-4o"], (2.50, 10.00))


class TestAttribution(unittest.TestCase):
    def _rows(self):
        return [
            {"timestamp": "2025-11-01T09:00:00", "model_used": "gpt-4o",
             "input_tokens": 1_000_000, "output_tokens": 100_000,
             "task_type": "classification", "team": "growth",
             "feature": "lead-scoring", "customer_id": "cust-acme"},
            {"timestamp": "2025-11-08T09:00:00", "model_used": "gpt-4o",
             "input_tokens": 1_000_000, "output_tokens": 100_000,
             "task_type": "classification"},  # untagged
        ]

    def test_tagged_rollups_exclude_untagged(self):
        result = analyze_api_costs(self._rows(), sample_days=30)
        self.assertEqual(result.tagged_records, 1)
        self.assertEqual(set(result.by_team), {"growth"})
        # Feature and customer rollups share the by_feature dimension,
        # with customer keys namespaced under a 'customer:' prefix.
        self.assertEqual(set(result.by_feature), {"lead-scoring", "customer:cust-acme"})
        self.assertGreater(result.by_team["growth"]["current"], 0)
        self.assertLess(result.by_team["growth"]["optimized"],
                        result.by_team["growth"]["current"])

    def test_report_shows_unit_economics_section(self):
        result = analyze_api_costs(self._rows(), sample_days=30)
        report = generate_report(arbitrage=result)
        self.assertIn("Unit economics", report)
        self.assertIn("lead-scoring", report)


class TestLiteLLMAdapter(unittest.TestCase):
    def test_loads_and_maps_fields(self):
        from finops.adapters.litellm import load_litellm_records
        rows = load_litellm_records(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "sample_data", "litellm_logs.json"))
        self.assertEqual(len(rows), 10)
        first = rows[0]
        self.assertEqual(first["model_used"], "openai/gpt-4o")
        self.assertEqual(first["input_tokens"], 1240)
        self.assertEqual(first["task_type"], "classification")
        self.assertEqual(first["team"], "growth")
        self.assertEqual(first["customer_id"], "cust-acme")
        # Tag-based fallback (r-010 uses tags list, not metadata keys)
        tagged = [r for r in rows if r["feature"] == "dedup"]
        self.assertEqual(len(tagged), 1)
        self.assertEqual(tagged[0]["team"], "ml-infra")

    def test_adapter_rows_feed_module_a(self):
        from finops.adapters.litellm import load_litellm_records
        rows = load_litellm_records(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "sample_data", "litellm_logs.json"))
        result = analyze_api_costs(rows, sample_days=2)
        self.assertGreater(result.current_monthly_spend, 0)
        self.assertGreater(result.records_rerouted, 0)

    def test_missing_file_raises(self):
        from finops.adapters.litellm import load_litellm_records
        with self.assertRaises(FileNotFoundError):
            load_litellm_records("no_such_file.json")


if __name__ == "__main__":
    unittest.main()