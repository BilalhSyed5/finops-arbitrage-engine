#!/usr/bin/env python3
"""Export EC2 inventory + hourly CPU/GPU utilization from CloudWatch.

Runs inside the customer's AWS account with read-only permissions
(ec2:DescribeInstances, cloudwatch:GetMetricData, cloudwatch:ListMetrics),
so no data leaves their environment until they choose to share the CSVs.

    pip install boto3
    python scripts/export_cloudwatch.py --days 14 --region us-east-1 --out-dir audit_data/

Writes:
    inventory.csv    instance_id, instance_type, name, team
    utilization.csv  instance_id, instance_type, timestamp, cpu_pct, gpu_pct

GPU utilization is not a default EC2 metric. It's read from the CloudWatch
agent's NVIDIA plugin (namespace CWAgent, metric nvidia_smi_utilization_gpu),
averaged across the GPUs of each instance. Without the agent, gpu_pct is
blank and the audit judges GPU instances on CPU only (and says so).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Tuple

GPU_NAMESPACE = "CWAgent"
GPU_METRIC = "nvidia_smi_utilization_gpu"
PERIOD = 3600
MAX_QUERIES = 500  # GetMetricData limit per call


def _chunks(items: List, size: int) -> Iterable[List]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


def list_instances(ec2, tag_filter: Optional[str]) -> List[Dict[str, str]]:
    filters = [{"Name": "instance-state-name", "Values": ["running", "stopped"]}]
    if tag_filter:
        key, _, value = tag_filter.partition("=")
        filters.append({"Name": f"tag:{key}", "Values": [value or "*"]})
    out = []
    for page in ec2.get_paginator("describe_instances").paginate(Filters=filters):
        for res in page["Reservations"]:
            for inst in res["Instances"]:
                tags = {t["Key"]: t["Value"] for t in inst.get("Tags", [])}
                out.append({
                    "instance_id": inst["InstanceId"],
                    "instance_type": inst["InstanceType"],
                    "name": tags.get("Name", ""),
                    "team": tags.get("team") or tags.get("Team") or tags.get("owner", ""),
                })
    return out


def gpu_dimension_sets(cw, instance_ids: List[str]) -> Dict[str, List[List[Dict[str, str]]]]:
    """The agent publishes one series per GPU with extra dimensions (index,
    name, ...); find them all so we can average per instance."""
    wanted = set(instance_ids)
    found: Dict[str, List[List[Dict[str, str]]]] = defaultdict(list)
    for page in cw.get_paginator("list_metrics").paginate(Namespace=GPU_NAMESPACE,
                                                          MetricName=GPU_METRIC):
        for metric in page["Metrics"]:
            dims = metric["Dimensions"]
            iid = next((d["Value"] for d in dims if d["Name"] == "InstanceId"), None)
            if iid in wanted:
                found[iid].append(dims)
    return found


def fetch(cw, queries: List[Dict], start: datetime, end: datetime) -> Dict[str, Dict[datetime, float]]:
    """Run GetMetricData in chunks; returns {query_id: {timestamp: value}}."""
    out: Dict[str, Dict[datetime, float]] = defaultdict(dict)
    for chunk in _chunks(queries, MAX_QUERIES):
        kwargs = {"MetricDataQueries": chunk, "StartTime": start, "EndTime": end}
        while True:
            resp = cw.get_metric_data(**kwargs)
            for series in resp["MetricDataResults"]:
                for ts, val in zip(series["Timestamps"], series["Values"]):
                    out[series["Id"]][ts.astimezone(timezone.utc).replace(tzinfo=None)] = val
            token = resp.get("NextToken")
            if not token:
                break
            kwargs["NextToken"] = token
    return out


def build_rows(instances: List[Dict[str, str]], cpu: Dict[str, Dict[datetime, float]],
               gpu: Dict[str, List[Dict[datetime, float]]]) -> List[Tuple]:
    rows = []
    for idx, inst in enumerate(instances):
        cpu_series = cpu.get(f"cpu{idx}", {})
        gpu_series = gpu.get(inst["instance_id"], [])
        for ts in sorted(cpu_series):
            vals = [s[ts] for s in gpu_series if ts in s]
            gpu_pct = f"{sum(vals) / len(vals):.2f}" if vals else ""
            rows.append((inst["instance_id"], inst["instance_type"], ts.isoformat(),
                         f"{cpu_series[ts]:.2f}", gpu_pct))
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=int, default=14, help="history to export (≥7 recommended)")
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", "us-east-1"))
    parser.add_argument("--profile", help="AWS named profile")
    parser.add_argument("--tag", help="only instances with this tag, e.g. team=ml or env")
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args(argv)

    try:
        import boto3  # noqa: PLC0415 - optional dependency
    except ImportError:
        print("error: this script needs boto3 (pip install boto3)", file=sys.stderr)
        return 2

    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    ec2, cw = session.client("ec2"), session.client("cloudwatch")
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=args.days)

    instances = list_instances(ec2, args.tag)
    if not instances:
        print("No instances found.", file=sys.stderr)
        return 1
    print(f"Found {len(instances)} instances; fetching {args.days} days of hourly metrics...",
          file=sys.stderr)

    cpu_queries = [{
        "Id": f"cpu{i}",
        "MetricStat": {"Metric": {"Namespace": "AWS/EC2", "MetricName": "CPUUtilization",
                                  "Dimensions": [{"Name": "InstanceId",
                                                  "Value": inst["instance_id"]}]},
                       "Period": PERIOD, "Stat": "Average"},
        "ReturnData": True,
    } for i, inst in enumerate(instances)]
    cpu = fetch(cw, cpu_queries, start, end)

    gpu_dims = gpu_dimension_sets(cw, [i["instance_id"] for i in instances])
    gpu_queries, owner = [], {}
    for iid, dim_sets in gpu_dims.items():
        for dims in dim_sets:
            qid = f"gpu{len(gpu_queries)}"
            owner[qid] = iid
            gpu_queries.append({"Id": qid, "MetricStat": {
                "Metric": {"Namespace": GPU_NAMESPACE, "MetricName": GPU_METRIC,
                           "Dimensions": dims},
                "Period": PERIOD, "Stat": "Average"}, "ReturnData": True})
    gpu_raw = fetch(cw, gpu_queries, start, end) if gpu_queries else {}
    gpu: Dict[str, List[Dict[datetime, float]]] = defaultdict(list)
    for qid, series in gpu_raw.items():
        gpu[owner[qid]].append(series)

    os.makedirs(args.out_dir, exist_ok=True)
    inv_path = os.path.join(args.out_dir, "inventory.csv")
    util_path = os.path.join(args.out_dir, "utilization.csv")
    with open(inv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["instance_id", "instance_type", "name", "team"])
        writer.writeheader()
        writer.writerows(instances)
    rows = build_rows(instances, cpu, gpu)
    with open(util_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["instance_id", "instance_type", "timestamp", "cpu_pct", "gpu_pct"])
        writer.writerows(rows)
    print(f"Wrote {inv_path} and {util_path} ({len(rows):,} hourly samples).", file=sys.stderr)
    if not gpu_queries and any(i["instance_type"].startswith(("g", "p")) for i in instances):
        print("note: no GPU metrics found; install the CloudWatch agent NVIDIA plugin for "
              "GPU-aware results.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
