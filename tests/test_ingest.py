import json
import os
import unittest

from finops.ingest import (DataLoadError, apply_attribution, detect_format, infer_tasks,
                           load_attribution_map, load_usage_file)
from finops.models import TaskSource
from tests.helpers import SAMPLE, rec, write_tmp


class TestGenericCsv(unittest.TestCase):
    def test_bad_rows_are_skipped_and_counted(self):
        path = write_tmp(
            "timestamp,model_used,input_tokens,output_tokens,task_type\n"
            "2026-11-01T00:00:00,gpt-5.5,100,5,classification\n"
            "2026-11-01T00:01:00,gpt-5.5,abc,5,classification\n"
            "2026-11-01T00:02:00,,100,5,classification\n"
            "2026-11-01T00:03:00,gpt-5.5,100,-4,classification\n", ".csv")
        records, report = load_usage_file(path)
        self.assertEqual(len(records), 1)
        self.assertEqual(report.skipped, 3)
        self.assertIn("non-numeric input_tokens", report.skip_reasons)

    def test_all_bad_rows_raise(self):
        path = write_tmp("model,input_tokens,output_tokens\ngpt-5.5,x,y\n", ".csv")
        with self.assertRaises(DataLoadError):
            load_usage_file(path)

    def test_flexible_column_names_and_cache_flag(self):
        path = write_tmp("Model,Prompt_Tokens,Completion_Tokens,cached_tokens,requests\n"
                         "gpt-5.5,1000,10,400,3\n", ".csv")
        (r,), _ = load_usage_file(path)
        self.assertEqual((r.input_tokens, r.output_tokens, r.cached_input_tokens, r.requests),
                         (1000, 10, 400, 3))
        self.assertTrue(r.cache_reported)

    def test_missing_and_empty_files(self):
        with self.assertRaises(DataLoadError):
            load_usage_file("/nonexistent/file.csv")
        with self.assertRaises(DataLoadError):
            load_usage_file(write_tmp("", ".csv"))
        with self.assertRaises(DataLoadError):
            load_usage_file(write_tmp("[1, 2", ".json"))

    def test_jsonl_with_messages(self):
        line = {"model": "gpt-5.5", "input_tokens": 10, "output_tokens": 1,
                "messages": [{"role": "system", "content": "Classify"},
                             {"role": "user", "content": [{"type": "text", "text": "hi"}]}],
                "response": {"choices": [{"message": {"content": "billing"}}]}}
        path = write_tmp(json.dumps(line) + "\n" + json.dumps(line) + "\n", ".jsonl")
        records, _ = load_usage_file(path)
        self.assertEqual(records[0].prompt, "system: Classify\nuser: hi")
        self.assertEqual(records[0].response, "billing")
        self.assertEqual(len(records[0].extra["messages"]), 2)


class TestProviderFormats(unittest.TestCase):
    def test_detection(self):
        self.assertEqual(detect_format(os.path.join(SAMPLE, "openai_usage.json")), "openai-usage")
        self.assertEqual(detect_format(os.path.join(SAMPLE, "anthropic_usage.json")), "anthropic-usage")
        self.assertEqual(detect_format(os.path.join(SAMPLE, "request_samples.jsonl")), "litellm")
        self.assertEqual(detect_format(os.path.join(SAMPLE, "usage_daily.csv")), "generic")

    def test_openai_usage_buckets(self):
        records, rep = load_usage_file(os.path.join(SAMPLE, "openai_usage.json"))
        self.assertEqual(len(records), 6)
        r = records[0]
        self.assertEqual(r.requests, 39_000)
        self.assertEqual(r.project, "proj_support_triage")
        self.assertTrue(r.period_end > r.timestamp)
        self.assertTrue(r.cache_reported)

    def test_anthropic_usage_counts_cache_reads_and_writes_as_input(self):
        page = {"data": [{"starting_at": "2026-11-01T00:00:00Z", "ending_at": "2026-11-02T00:00:00Z",
                          "results": [{"model": "claude-sonnet-5-5", "uncached_input_tokens": 100,
                                       "cache_read_input_tokens": 50,
                                       "cache_creation": {"ephemeral_5m_input_tokens": 10,
                                                          "ephemeral_1h_input_tokens": 5},
                                       "output_tokens": 7, "service_tier": "batch",
                                       "context_window": "200k-1M"}]}]}
        (r,), _ = load_usage_file(write_tmp(json.dumps(page), ".json"))
        self.assertEqual((r.input_tokens, r.cached_input_tokens, r.cache_write_tokens), (165, 50, 15))
        self.assertTrue(r.batch)
        self.assertTrue(r.long_context)
        self.assertEqual(r.requests, 0)

    def test_litellm_tags_and_metadata(self):
        records, _ = load_usage_file(os.path.join(SAMPLE, "request_samples.jsonl"))
        r = records[0]
        self.assertEqual(r.task_type, "classification")
        self.assertEqual(r.feature, "support-triage")
        self.assertTrue(r.customer_id.startswith("cust-"))
        self.assertTrue(r.prompt and r.response)
        entry = {"model": "gpt-5.5", "prompt_tokens": 5, "completion_tokens": 1, "spend": 0.1,
                 "request_tags": ["team:growth", "feature:lead-scoring", "task:tagging"]}
        (r,), _ = load_usage_file(write_tmp(json.dumps([entry]), ".json"))
        self.assertEqual((r.team, r.feature, r.task_type), ("growth", "lead-scoring", "tagging"))
        self.assertEqual(r.actual_spend, 0.1)


class TestEnrichment(unittest.TestCase):
    def test_attribution_map(self):
        records, _ = load_usage_file(os.path.join(SAMPLE, "openai_usage.json"))
        amap = load_attribution_map(os.path.join(SAMPLE, "attribution.json"))
        apply_attribution(records, amap)
        triage = [r for r in records if r.project == "proj_support_triage"]
        self.assertTrue(all(r.task_type == "classification" and r.team == "support" for r in triage))
        self.assertTrue(all(r.task_source == TaskSource.MAPPED for r in triage))

    def test_attribution_does_not_override_provided_values(self):
        r = rec("gpt-5.5", 10, 1, "summarization", project="proj_support_x")
        amap = load_attribution_map(os.path.join(SAMPLE, "attribution.json"))
        apply_attribution([r], amap)
        self.assertEqual(r.task_type, "summarization")
        self.assertEqual(r.team, "support")

    def test_invalid_attribution_map(self):
        path = write_tmp(json.dumps({"rules": [{"match": {"colour": "x"}, "set": {"team": "a"}}]}),
                         ".json")
        with self.assertRaises(DataLoadError):
            load_attribution_map(path)

    def test_task_inference(self):
        by_prompt = rec("gpt-5.5", 500, 200, prompt="Please summarize the following call notes")
        by_shape = rec("gpt-5.5", 2_000, 6, requests=2)
        unknown = rec("gpt-5.5", 2_000, 900)
        stats = infer_tasks([by_prompt, by_shape, unknown])
        self.assertEqual(by_prompt.task_type, "summarization")
        self.assertEqual(by_shape.task_type, "classification")
        self.assertEqual(by_shape.task_source, TaskSource.INFERRED)
        self.assertEqual(unknown.task_type, "unknown")
        self.assertEqual(stats, {"prompt": 1, "shape": 1, "unknown": 1})


if __name__ == "__main__":
    unittest.main()
