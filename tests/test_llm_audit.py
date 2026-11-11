import dataclasses
import unittest

from finops.llm_audit import analyze_llm_spend, infer_window_days
from finops.models import Confidence, TaskSource
from tests.helpers import BOOK, POLICY, TODAY, rec


def run(records, **kw):
    kw.setdefault("today", TODAY)
    kw.setdefault("sample_days", 30.4)   # scale factor 1 unless a test says otherwise
    return analyze_llm_spend(records, BOOK, POLICY, **kw)


class TestRouting(unittest.TestCase):
    def test_routes_routable_task_to_cheaper_same_provider_model(self):
        r = run([rec("gpt-5.5", 1_000_000, 10_000, "classification")])
        (line,) = r.lever_lines
        self.assertEqual((line.lever, line.target), ("routing", "gpt-6-luna"))
        self.assertEqual(line.confidence, Confidence.ESTIMATED)   # no proof yet
        expected = BOOK.get("gpt-5.5").cost(1e6, 1e4) - BOOK.get("gpt-6-luna").cost(1e6, 1e4)
        self.assertAlmostEqual(r.monthly_savings, expected, places=6)

    def test_never_routes_to_a_more_expensive_model(self):
        # v1 bug: gemini-1.5-flash summarization was "routed" to a pricier model,
        # producing negative savings.
        r = run([rec("gemini-1.5-flash", 1_000_000, 100_000, "summarization")])
        self.assertEqual(r.lever_lines, [])
        self.assertAlmostEqual(r.monthly_savings, 0.0)

    def test_complex_tasks_only_get_same_tier_refresh(self):
        r = run([rec("claude-sonnet-4-5", 1_000_000, 100_000, "complex_reasoning")])
        (line,) = r.lever_lines
        self.assertEqual((line.lever, line.target), ("model_refresh", "claude-sonnet-5-5"))
        self.assertEqual(line.model, "claude-sonnet-4-5")       # shown as the customer names it
        self.assertEqual(line.confidence, Confidence.MEASURED)

    def test_proof_pass_verifies_and_fail_blocks(self):
        records = [rec("gpt-5.5", 1_000_000, 10_000, "classification"),
                   rec("claude-sonnet-5-5", 1_000_000, 100_000, "extraction")]
        proof = {("classification", "gpt-6-luna"): "pass",
                 ("extraction", "claude-haiku-5-5"): "fail"}
        r = run(records, proof=proof)
        routing = [l for l in r.lever_lines if l.lever == "routing"]
        self.assertEqual(len(routing), 1)
        self.assertEqual(routing[0].confidence, Confidence.VERIFIED)
        self.assertEqual(list(r.blocked_by_proof), ["extraction: claude-sonnet-5-5 → claude-haiku-5-5"])

    def test_inferred_task_is_never_verified(self):
        r = run([rec("gpt-5.5", 1_000_000, 10_000, "classification",
                     task_source=TaskSource.INFERRED)],
                proof={("classification", "gpt-6-luna"): "pass"})
        self.assertEqual(r.lever_lines[0].confidence, Confidence.ESTIMATED)

    def test_cross_provider_policy_picks_cheapest(self):
        policy = dataclasses.replace(POLICY, cross_provider=True)
        r = analyze_llm_spend([rec("claude-sonnet-5-5", 1_000_000, 10_000, "classification")],
                              BOOK, policy, sample_days=30.4, today=TODAY)
        targets = {t for t in POLICY.targets_for("classification").values()}
        cheapest = min(targets, key=lambda n: BOOK.get(n).cost(1e6, 1e4))
        self.assertEqual(r.lever_lines[0].target, cheapest)


class TestOtherLevers(unittest.TestCase):
    def test_batch_for_nightly_features(self):
        r = run([rec("gpt-5.4-mini", 1_000_000, 100_000, "tagging", feature="nightly-sync")])
        levers = {l.lever for l in r.lever_lines}
        self.assertIn("batch", levers)

    def test_already_batched_traffic_not_counted_again(self):
        r = run([rec("gpt-5.4", 1_000_000, 100_000, "bulk_enrichment", batch=True)])
        self.assertNotIn("batch", {l.lever for l in r.lever_lines})

    def test_caching_measured_from_shared_prompt_prefix(self):
        system = "You are a careful assistant. " * 400          # ~12k chars shared prefix
        records = [rec("claude-sonnet-5-5", 4_000, 50, "complex_reasoning",
                       prompt=system + f"question {i}", cache_reported=True) for i in range(5)]
        r = run(records)
        (line,) = [l for l in r.lever_lines if l.lever == "caching"]
        self.assertEqual(line.confidence, Confidence.MEASURED)
        self.assertGreater(line.monthly_savings, 0)

    def test_openai_without_cache_data_is_not_assumed_uncached(self):
        records = [rec("gpt-6.1-sol", 5_000_000, 1_000, "code_gen", requests=2_000)]
        self.assertNotIn("caching", {l.lever for l in run(records).lever_lines})
        records[0].cache_reported = True     # data says 0 cached tokens -> real opportunity
        self.assertIn("caching", {l.lever for l in run(records).lever_lines})

    def test_scheduled_price_increase(self):
        r = run([rec("gemini-4-argon", 1_000_000, 100_000, "code_gen")])
        (change,) = r.upcoming_price_changes
        self.assertEqual(change["effective"], "2026-12-01")
        self.assertAlmostEqual(change["monthly_delta"], BOOK.get("gemini-4-argon").cost(1e6, 1e5))


class TestAccounting(unittest.TestCase):
    def test_lever_totals_sum_to_total_savings(self):
        records = [rec("gpt-5.5", 2_000_000, 10_000, "classification", feature="nightly-x"),
                   rec("claude-opus-4-6", 3_000_000, 200_000, "complex_reasoning"),
                   rec("gpt-4o", 1_000_000, 100_000, "summarization")]
        r = run(records)
        self.assertAlmostEqual(sum(r.lever_totals.values()), r.monthly_savings, places=6)
        self.assertAlmostEqual(sum(r.savings_by_confidence.values()), r.monthly_savings, places=6)

    def test_customers_are_their_own_slice(self):
        # v1 bug: customer rollups were written into the feature table.
        r = run([rec("gpt-5.5", 1_000, 10, "classification", feature="triage", customer_id="acme")])
        self.assertIn("acme", r.by_customer)
        self.assertEqual(list(r.by_feature), ["triage"])

    def test_unpriced_models_are_excluded_and_reported(self):
        r = run([rec("gpt-5.5", 1_000, 10, "classification"), rec("mystery-model", 10**6, 10**6)])
        self.assertIn("mystery-model", r.unpriced)
        self.assertAlmostEqual(r.monthly_current, BOOK.get("gpt-5.5").cost(1_000, 10))
        self.assertTrue(any("no price" in w for w in r.warnings))

    def test_window_and_sample_rate_scaling(self):
        records = [rec("gpt-5.5", 1_000, 0, "x", ts="2026-11-01"),
                   rec("gpt-5.5", 1_000, 0, "x", ts="2026-11-07")]
        self.assertAlmostEqual(infer_window_days(records), 7.0)   # date-only = whole day
        r = analyze_llm_spend(records, BOOK, POLICY, sample_rate=0.1, today=TODAY)
        self.assertAlmostEqual(r.monthly_current, 2 * 1_000 * 5 / 1e6 * 30.4 / 7 / 0.1)

    def test_reconciliation_flags_negotiated_rates(self):
        r = run([rec("gpt-5.5", 1_000_000, 0, "x", requests=10, actual_spend=2.5)])  # list $5.00
        self.assertAlmostEqual(r.reconciliation["variance_pct"], 100.0)
        self.assertTrue(any("negotiated" in w for w in r.warnings))

    def test_short_window_warning(self):
        records = [rec("gpt-5.5", 1_000, 0, "x", ts="2026-11-01T00:00:00"),
                   rec("gpt-5.5", 1_000, 0, "x", ts="2026-11-01T02:00:00")]
        r = analyze_llm_spend(records, BOOK, POLICY, today=TODAY)
        self.assertTrue(any("hours" in w for w in r.warnings))


if __name__ == "__main__":
    unittest.main()
