"""Exercise scripts/export_cloudwatch.py against a fake boto3 (no AWS needed)."""

import csv
import importlib.util
import os
import sys
import tempfile
import types
import unittest
from datetime import datetime, timedelta, timezone

from finops.infra_audit import analyze_infra, load_inventory, load_utilization
from tests.helpers import BOOK, ROOT

T0 = datetime(2026, 11, 1, tzinfo=timezone.utc)


class _Paginator:
    def __init__(self, pages):
        self.pages = pages

    def paginate(self, **_):
        return iter(self.pages)


class _EC2:
    def get_paginator(self, name):
        assert name == "describe_instances"
        return _Paginator([{"Reservations": [{"Instances": [
            {"InstanceId": "i-gpu", "InstanceType": "g5.12xlarge",
             "Tags": [{"Key": "Name", "Value": "infer"}, {"Key": "team", "Value": "ml"}]},
            {"InstanceId": "i-cpu", "InstanceType": "m5.xlarge"}]}]}])


class _CloudWatch:
    def get_paginator(self, name):
        assert name == "list_metrics"
        dims = lambda idx: [{"Name": "InstanceId", "Value": "i-gpu"}, {"Name": "index", "Value": idx}]
        return _Paginator([{"Metrics": [{"Dimensions": dims("0")}, {"Dimensions": dims("1")}]}])

    def get_metric_data(self, MetricDataQueries, StartTime, EndTime, NextToken=None):  # noqa: N803
        stamps = [T0 + timedelta(hours=h) for h in range(48)]
        results = []
        for q in MetricDataQueries:
            gpu = q["Id"].startswith("gpu")
            value = 20.0 if q["Id"] in ("gpu0",) else (40.0 if gpu else 3.0)
            results.append({"Id": q["Id"], "Timestamps": stamps, "Values": [value] * len(stamps)})
        return {"MetricDataResults": results}


class TestExportCloudWatch(unittest.TestCase):
    def test_export_with_fake_boto3(self):
        fake = types.ModuleType("boto3")
        fake.Session = lambda **_: types.SimpleNamespace(
            client=lambda name: _EC2() if name == "ec2" else _CloudWatch())
        spec = importlib.util.spec_from_file_location(
            "export_cloudwatch", os.path.join(ROOT, "scripts", "export_cloudwatch.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules["boto3"] = fake
        try:
            with tempfile.TemporaryDirectory() as d:
                self.assertEqual(module.main(["--days", "2", "--out-dir", d]), 0)
                with open(os.path.join(d, "utilization.csv"), encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                gpu_rows = [r for r in rows if r["instance_id"] == "i-gpu"]
                self.assertEqual(len(gpu_rows), 48)
                self.assertEqual(gpu_rows[0]["gpu_pct"], "30.00")   # mean of 2 GPUs (20, 40)
                self.assertEqual([r for r in rows if r["instance_id"] == "i-cpu"][0]["gpu_pct"], "")
                # The export loads straight into the audit.
                inv, _ = load_inventory(os.path.join(d, "inventory.csv"))
                ser, typ, _ = load_utilization(os.path.join(d, "utilization.csv"))
                result = analyze_infra(inv, BOOK, ser, typ)
                self.assertEqual(result.instances_audited, 2)
                self.assertEqual(inv[0]["team"], "ml")
        finally:
            sys.modules.pop("boto3", None)


if __name__ == "__main__":
    unittest.main()
