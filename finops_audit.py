#!/usr/bin/env python3
"""FinOps Arbitrage Engine — CLI entrypoint.

A developer-first diagnostic that ingests raw LLM API cost logs and cloud
resource inventories, then emits an executive-ready Markdown report showing:

  Module A  Model arbitrage savings (Frontier vs. Budget model routing)
  Module B  Idle GPU / EC2 right-sizing waste

Usage:
    python finops_audit.py --log api_logs.csv --inventory inventory.json
    python finops_audit.py --log api_logs.csv --inventory inventory.json -o report.md
    python finops_audit.py --demo          # run against bundled sample data

Exit codes: 0 success, 1 data error, 2 usage error.
"""

from __future__ import annotations

import argparse
import os
import sys

from finops import cost_calculator, infrastructure_audit, report_generator
from finops.adapters.litellm import load_litellm_records
from finops.cost_calculator import analyze_api_costs
from finops.data_loader import DataLoadError, load_records
from finops.infrastructure_audit import analyze_infrastructure
from finops.pricing_config import load_pricing
from finops.report_generator import generate_report

API_LOG_FIELDS = ("timestamp", "model_used", "input_tokens", "output_tokens", "task_type")
API_LOG_NUMERIC = ("input_tokens", "output_tokens")
INVENTORY_FIELDS = ("instance_id", "instance_type", "avg_cpu_utilization_pct",
                    "running_hours_per_day")
INVENTORY_NUMERIC = ("avg_cpu_utilization_pct", "running_hours_per_day")

DEMO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="finops_audit",
        description="Quantify model-routing savings and idle cloud/GPU waste.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--log", metavar="CSV|JSON",
                        help="API usage log (timestamp, model_used, input_tokens, "
                             "output_tokens, task_type).")
    parser.add_argument("--inventory", metavar="CSV|JSON",
                        help="Cloud inventory export (instance_id, instance_type, "
                             "avg_cpu_utilization_pct, avg_gpu_utilization_pct, "
                             "running_hours_per_day).")
    parser.add_argument("-o", "--output", metavar="FILE",
                        help="Write the Markdown report to FILE (default: stdout).")
    parser.add_argument("--sample-days", type=float, default=None,
                        help="Override the inferred API-log sampling window in days.")
    parser.add_argument("--litellm", metavar="JSON|JSONL",
                        help="LiteLLM proxy spend-log export instead of --log "
                             "(supports metadata tags for attribution).")
    parser.add_argument("--pricing-config", metavar="DIR",
                        help="Directory containing model_pricing.json / "
                             "ec2_pricing.json overrides (default: bundled config/).")
    parser.add_argument("--demo", action="store_true",
                        help="Run against the bundled sample data instead of file inputs.")
    return parser


def run_audit(log_path: str | None, inventory_path: str | None,
              sample_days: float | None, litellm_path: str | None = None) -> str:
    """Execute the requested audit modules and return the Markdown report."""
    sources: dict[str, str] = {}
    arbitrage = None
    infra = None

    if litellm_path:
        rows = load_litellm_records(litellm_path)
        sources["LiteLLM spend logs"] = litellm_path
        arbitrage = analyze_api_costs(rows, sample_days=sample_days)

    if log_path:
        rows = load_records(log_path, API_LOG_FIELDS, API_LOG_NUMERIC)
        sources["API usage log"] = log_path
        arbitrage = analyze_api_costs(rows, sample_days=sample_days)

    if inventory_path:
        rows = load_records(inventory_path, INVENTORY_FIELDS, INVENTORY_NUMERIC)
        sources["Cloud inventory"] = inventory_path
        infra = analyze_infrastructure(rows)

    if arbitrage is None and infra is None:
        raise DataLoadError("Nothing to audit: provide --log and/or --inventory.")

    return generate_report(arbitrage=arbitrage, infra=infra, source_files=sources or None)


def _force_utf8_stdio() -> None:
    """Ensure the report (which contains emoji/unicode) survives Windows
    cp1252 consoles instead of crashing with UnicodeEncodeError."""
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdio()
    args = build_arg_parser().parse_args(argv)

    # Pricing snapshot must be loaded before any analysis runs.
    pricing = load_pricing(args.pricing_config)
    cost_calculator.use_pricing(pricing)
    infrastructure_audit.use_pricing(pricing)
    print(f"[info] Pricing source: {pricing.source}", file=sys.stderr)

    if args.demo:
        args.log = os.path.join(DEMO_DIR, "api_logs.csv")
        args.inventory = os.path.join(DEMO_DIR, "inventory.json")

    if not args.log and not args.inventory and not args.litellm:
        build_arg_parser().print_help()
        print("\nerror: provide --log, --litellm and/or --inventory (or use --demo).",
              file=sys.stderr)
        return 2

    try:
        report = run_audit(args.log, args.inventory, args.sample_days,
                           litellm_path=args.litellm)
    except DataLoadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as handle:
                handle.write(report)
            print(f"Report written to {args.output}")
        except OSError as exc:
            print(f"error: cannot write '{args.output}': {exc}", file=sys.stderr)
            return 1
    else:
        print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())