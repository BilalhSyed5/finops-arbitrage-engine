import contextlib
import io
import json
import os
import tempfile
import unittest

from finops.cli import main
from tests.helpers import SAMPLE, write_tmp

try:
    import yaml  # optional: only used to validate generated YAML
except ImportError:  # pragma: no cover
    yaml = None


def read(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def run_cli(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(list(argv))
    return code, out.getvalue(), err.getvalue()


class TestCli(unittest.TestCase):
    def test_demo_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            code, out, err = run_cli("audit", "--demo", "--today", "2026-11-11", "-q",
                                     "--json", os.path.join(d, "a.json"),
                                     "--actions-dir", os.path.join(d, "actions"))
            self.assertEqual(code, 0, err)
            self.assertIn("## Executive summary", out)
            self.assertIn("✅ verified", out)
            data = json.loads(read(os.path.join(d, "a.json")))
            self.assertGreater(data["summary"]["monthly_savings"], 0)
            self.assertAlmostEqual(sum(data["summary"]["savings_by_confidence"].values()),
                                   data["summary"]["monthly_savings"], places=2)
            config = read(os.path.join(d, "actions", "litellm_router.yaml"))
            self.assertIn("finops-classification", config)
            self.assertIn("# UNVERIFIED", config.replace("(UNVERIFIED", "# UNVERIFIED"))
            if yaml:
                parsed = yaml.safe_load(config)
                names = [m["model_name"] for m in parsed["model_list"]]
                self.assertIn("finops-classification", names)
                self.assertNotIn("finops-extraction", names)   # failed proof
            self.assertTrue(os.access(os.path.join(d, "actions", "ec2_schedules.sh"), os.X_OK))

    def test_v1_flags_still_work(self):
        path = write_tmp("timestamp,model_used,input_tokens,output_tokens,task_type\n"
                         "2026-11-01T00:00:00,gpt-4o,1000,10,classification\n"
                         "2026-11-08T00:00:00,gpt-4o,1000,10,classification\n", ".csv")
        code, out, _ = run_cli("--log", path, "-q")
        self.assertEqual(code, 0)
        self.assertIn("LLM API run-rate", out)

    def test_error_codes(self):
        self.assertEqual(run_cli("audit", "-q")[0], 2)
        self.assertEqual(run_cli("audit", "--usage", "/nope.csv", "-q")[0], 1)
        self.assertEqual(run_cli("audit", "--usage", os.path.join(SAMPLE, "usage_daily.csv"),
                                 "--pricing-config", "/no/such/dir", "-q")[0], 1)

    def test_pricing_command(self):
        code, out, _ = run_cli("pricing", "gpt-4o-mini", "gpt-9")
        self.assertEqual(code, 0)
        self.assertIn("gpt-4o-mini -> gpt-4o-mini", out)
        self.assertIn("NOT PRICED", out)

    def test_proof_command_writes_results(self):
        with tempfile.TemporaryDirectory() as d:
            out_path = os.path.join(d, "proof.json")
            code, out, err = run_cli("proof", "--usage", os.path.join(SAMPLE, "request_samples.jsonl"),
                                     "--candidate-outputs", os.path.join(SAMPLE, "candidate_outputs.jsonl"),
                                     "--per-task", "100", "-o", out_path, "-q")
            self.assertEqual(code, 0, err)
            self.assertIn("**PASS**", out)
            self.assertIn("**FAIL**", out)
            self.assertEqual(len(json.loads(read(out_path))["results"]), 2)

    def test_provider_exports_with_attribution(self):
        code, out, err = run_cli("audit", "--usage", os.path.join(SAMPLE, "openai_usage.json"),
                                 "--usage", os.path.join(SAMPLE, "anthropic_usage.json"),
                                 "--attribution", os.path.join(SAMPLE, "attribution.json"), "-q")
        self.assertEqual(code, 0, err)
        self.assertIn("support-triage", out)
        self.assertIn("Batch API", out)


if __name__ == "__main__":
    unittest.main()
