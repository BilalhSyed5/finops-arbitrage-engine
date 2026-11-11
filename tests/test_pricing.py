import json
import os
import tempfile
import unittest
from datetime import date

from finops.pricing import PricingError, load_pricebook, normalize_model_name
from tests.helpers import BOOK


class TestModelResolution(unittest.TestCase):
    def name(self, raw):
        price = BOOK.resolve(raw)
        return price.name if price else None

    def test_mini_is_not_priced_as_its_bigger_sibling(self):
        # v1 bug: "gpt-4o-mini" matched "gpt-4o" by prefix (16x overstatement).
        self.assertEqual(self.name("gpt-4o-mini"), "gpt-4o-mini")
        self.assertEqual(self.name("openai/gpt-4o-mini-2024-07-18"), "gpt-4o-mini")
        self.assertEqual(BOOK.resolve("gpt-4o-mini").input, 0.15)

    def test_shorter_name_does_not_capture_longer_model(self):
        # v1 bug: "gpt-4" was priced as "gpt-4o".
        self.assertIsNone(self.name("gpt-4"))

    def test_dated_and_prefixed_names(self):
        self.assertEqual(self.name("gpt-4o-2024-08-06"), "gpt-4o")
        self.assertEqual(self.name("anthropic/claude-3-5-sonnet-20241022"), "claude-sonnet-4-6")
        self.assertEqual(self.name("models/gemini-2.5-pro"), "gemini-2.5-pro")

    def test_bedrock_and_vertex_names(self):
        self.assertEqual(self.name("us.anthropic.claude-sonnet-4-5-20250929-v1:0"), "claude-sonnet-4-6")
        self.assertEqual(self.name("claude-sonnet-4-5@20250929"), "claude-sonnet-4-6")

    def test_dotted_claude_versions(self):
        self.assertEqual(self.name("claude-opus-5.5"), "claude-opus-5-5")

    def test_unknown_newer_version_is_unpriced_not_guessed(self):
        self.assertIsNone(self.name("claude-opus-5-6"))
        self.assertIsNone(self.name("gpt-5.7"))
        self.assertIsNone(self.name("totally-unknown-model"))

    def test_variant_suffix_resolves_to_base(self):
        self.assertEqual(self.name("gpt-6-luna-experimental"), "gpt-6-luna")

    def test_normalize_candidates(self):
        self.assertEqual(normalize_model_name("openai/gpt-4o-mini-2024-07-18"),
                         ["gpt-4o-mini-2024-07-18", "gpt-4o-mini"])


class TestCostMath(unittest.TestCase):
    def test_plain_cost(self):
        p = BOOK.get("gpt-5.5")
        self.assertAlmostEqual(p.cost(1_000, 100), (1_000 * 5 + 100 * 30) / 1e6)

    def test_cached_tokens_billed_at_cached_rate(self):
        p = BOOK.get("gpt-5.5")
        self.assertAlmostEqual(p.cost(1_000, 0, cached_input_tokens=400),
                               (600 * 5 + 400 * 0.5) / 1e6)

    def test_cache_writes(self):
        p = BOOK.get("claude-sonnet-5-5")
        self.assertAlmostEqual(p.cost(1_000, 0, cache_write_tokens=1_000), 1_000 * 2.5 / 1e6)

    def test_batch_discount(self):
        p = BOOK.get("gpt-5.5")
        self.assertAlmostEqual(p.cost(1_000, 100, batch=True), p.cost(1_000, 100) / 2)

    def test_long_context_uses_per_request_average(self):
        haiku = BOOK.get("claude-haiku-5-5")
        # 2 requests of 150k tokens each -> over the 100k threshold -> 5x rates.
        self.assertAlmostEqual(haiku.cost(300_000, 0, requests=2), 300_000 * 0.5 / 1e6)
        # 10 requests of 30k each -> standard rate.
        self.assertAlmostEqual(haiku.cost(300_000, 0, requests=10), 300_000 * 0.1 / 1e6)
        # Unknown request count (0) -> no surcharge unless flagged explicitly.
        self.assertAlmostEqual(haiku.cost(300_000, 0, requests=0), 300_000 * 0.1 / 1e6)
        self.assertAlmostEqual(haiku.cost(300_000, 0, requests=0, long_context=True),
                               300_000 * 0.5 / 1e6)

    def test_unpublished_cached_rate_falls_back_to_full_price(self):
        p = BOOK.get("gpt-5.5-pro")
        self.assertEqual(p.cached_rate, p.input)
        self.assertFalse(p.supports_caching)


class TestPriceBookLoading(unittest.TestCase):
    def test_bundled_book_is_consistent(self):
        self.assertTrue(BOOK.as_of)
        for price in BOOK.models.values():
            self.assertGreater(price.output, 0, price.name)
            if price.successor:
                self.assertIn(price.successor, BOOK.models)
        self.assertEqual(BOOK.instance("p5.48xlarge").hourly, 55.04)  # post-2025 price cut
        self.assertEqual(BOOK.instance("g5.12xlarge").gpus, 4)

    def test_override_merges_and_accepts_v1_format(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "model_pricing.json"), "w") as f:
                json.dump({"frontier": {"gpt-5.5": [4.0, 24.0]},
                           "models": {"my-ft-model": {"input": 1, "output": 2, "tier": "mid"}}}, f)
            with open(os.path.join(d, "ec2_pricing.json"), "w") as f:
                json.dump({"instances": {"g5.xlarge": 0.7}}, f)
            book = load_pricebook(d)
        self.assertEqual(book.get("gpt-5.5").input, 4.0)
        self.assertEqual(book.get("my-ft-model").output, 2.0)
        self.assertEqual(book.instance("g5.xlarge").hourly, 0.7)
        self.assertIsNotNone(book.get("claude-opus-5-5"))  # defaults still present

    def test_malformed_override_is_an_error_not_a_silent_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "model_pricing.json"), "w") as f:
                f.write("{not json")
            with self.assertRaises(PricingError):
                load_pricebook(d)
            with open(os.path.join(d, "model_pricing.json"), "w") as f:
                json.dump({"models": {"x": {"input": 1}}}, f)
            with self.assertRaises(PricingError):
                load_pricebook(d)

    def test_staleness(self):
        self.assertFalse(BOOK.is_stale(date(2026, 11, 20)))
        self.assertTrue(BOOK.is_stale(date(2027, 3, 1)))


class TestShippedExamples(unittest.TestCase):
    def test_example_configs_load(self):
        from finops.policy import load_policy
        from tests.helpers import ROOT
        book = load_pricebook(os.path.join(ROOT, "config", "example_pricing_override"))
        self.assertEqual(book.get("gpt-5.5").input, 4.25)
        self.assertEqual(book.instance("g5.12xlarge").hourly, 3.9704)
        policy = load_policy(os.path.join(ROOT, "config", "example_policy.json"), book)
        self.assertTrue(policy.cross_provider)
        self.assertNotIn("summarization", policy.routable_tasks)
        self.assertIn("sentiment", policy.routable_tasks)
        self.assertEqual(policy.proof_pass_threshold, 0.95)


if __name__ == "__main__":
    unittest.main()
