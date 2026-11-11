import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from finops.ingest import infer_tasks, load_usage_file
from finops.proof import (OpenAICompatibleClient, load_candidate_outputs, run_proof, score_json,
                          score_labels, token_f1, wilson_interval)
from tests.helpers import BOOK, POLICY, SAMPLE, rec


class TestScorers(unittest.TestCase):
    def test_labels(self):
        self.assertTrue(score_labels("billing", "Billing.")[0])
        self.assertTrue(score_labels("billing", "billing\nbecause the card...")[0])
        self.assertTrue(score_labels("bug", "Category: bug")[0])
        self.assertFalse(score_labels("bug", "feature_request")[0])

    def test_json(self):
        ref = '{"total": 10.5, "vendor": "Acme"}'
        self.assertTrue(score_json(ref, '```json\n{"vendor": "acme", "total": 10.50}\n```')[0])
        self.assertFalse(score_json(ref, '{"vendor": "Acme", "total": 11}')[0])
        self.assertFalse(score_json(ref, "not json")[0])

    def test_f1_and_wilson(self):
        self.assertTrue(token_f1("the cat sat on the mat", "the cat sat on a mat")[0])
        low, high = wilson_interval(59, 60)
        self.assertGreater(low, 0.9)
        low, high = wilson_interval(49, 50)
        self.assertLess(low, 0.9)


class TestRunProof(unittest.TestCase):
    def setUp(self):
        self.records, _ = load_usage_file(os.path.join(SAMPLE, "request_samples.jsonl"))
        infer_tasks(self.records)

    def test_demo_verdicts_from_candidate_outputs(self):
        outputs = load_candidate_outputs(os.path.join(SAMPLE, "candidate_outputs.jsonl"))
        rep = run_proof(self.records, POLICY, BOOK, per_task=100, candidate_outputs=outputs)
        verdicts = {(g.task_type, g.candidate): g.verdict for g in rep.groups}
        self.assertEqual(verdicts[("classification", "gpt-6-luna")], "pass")
        self.assertEqual(verdicts[("extraction", "claude-haiku-5-5")], "fail")
        json.dumps(rep.to_json())  # serializable

    def test_requires_exactly_one_source(self):
        with self.assertRaises(ValueError):
            run_proof(self.records, POLICY, BOOK)

    def test_no_replayable_records(self):
        rep = run_proof([rec("gpt-5.5", 10, 1, "classification")], POLICY, BOOK, candidate_outputs={})
        self.assertEqual(rep.groups, [])
        self.assertTrue(rep.notes)


class _FakeLLM(BaseHTTPRequestHandler):
    calls = 0

    def do_POST(self):  # noqa: N802
        type(self).calls += 1
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if type(self).calls == 1:          # first call: transient 503 -> client retries
            self.send_response(503)
            self.end_headers()
            return
        assert self.headers["Authorization"] == "Bearer test-key"
        text = body["messages"][-1]["content"]
        answer = "billing" if "charged" in text or "receipt" in text else "other"
        payload = json.dumps({"choices": [{"message": {"content": answer}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


class TestHttpClient(unittest.TestCase):
    def test_openai_compatible_client_with_retry(self):
        server = HTTPServer(("127.0.0.1", 0), _FakeLLM)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = OpenAICompatibleClient(f"http://127.0.0.1:{server.server_port}/v1",
                                            "test-key", timeout=5, max_retries=2)
            out = client.chat("gpt-6-luna", [{"role": "user", "content": "I was charged twice"}])
            self.assertEqual(out, "billing")
            records = [rec("gpt-5.5", 1800, 4, "classification", prompt=f"Ticket {i}: I was charged",
                           response="billing", request_id=f"r{i}") for i in range(35)]
            rep = run_proof(records, POLICY, BOOK, client=client, concurrency=1)
            (g,) = rep.groups
            self.assertEqual((g.n, g.passed, g.verdict), (35, 35, "pass"))
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
