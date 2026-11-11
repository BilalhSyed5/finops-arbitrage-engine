"""Command-line interface.

    finops audit   ...   cost audit -> Markdown/JSON report + ready-to-apply configs
    finops proof   ...   quality eval of proposed model routes on your own traffic
    finops pricing ...   inspect the price book / resolve model names

Running without a subcommand (``python finops_audit.py --log ...``) is
treated as ``audit`` for backward compatibility with v1 flags.

Exit codes: 0 success, 1 data/pricing error, 2 usage error.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from typing import List, Optional, Tuple

from finops import __version__
from finops.actions import build_actions, write_artifacts
from finops.infra_audit import analyze_infra, load_inventory, load_utilization
from finops.ingest import (FORMATS, DataLoadError, apply_attribution, infer_tasks,
                           load_attribution_map, load_usage_file)
from finops.llm_audit import analyze_llm_spend
from finops.margin import compute_margins, load_revenue
from finops.policy import load_policy
from finops.pricing import PricingError, load_pricebook
from finops.report import AuditBundle, render_json, render_markdown, usd

DEMO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_data")


def _info(msg: str, quiet: bool = False) -> None:
    if not quiet:
        print(f"[finops] {msg}", file=sys.stderr)


def _force_utf8_stdio() -> None:
    """Reports contain unicode; keep Windows cp1252 consoles from crashing."""
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--pricing-config", metavar="DIR",
                   help="directory with model_pricing.json / ec2_pricing.json overrides "
                        "(e.g. your negotiated rates)")
    p.add_argument("--policy", metavar="JSON", help="routing/lever policy override")
    p.add_argument("--attribution", metavar="JSON",
                   help="rules mapping project/key/model -> team, feature, task_type")
    p.add_argument("--format", choices=FORMATS, default="auto",
                   help="usage file format (default: auto-detect)")
    p.add_argument("-q", "--quiet", action="store_true", help="no progress output")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="finops", description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command")

    a = sub.add_parser("audit", help="cost audit and savings plan",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    a.add_argument("--usage", metavar="FILE", action="append", default=[],
                   help="LLM usage export (repeatable): CSV/JSON/JSONL, LiteLLM spend logs, "
                        "OpenAI Usage API or Anthropic usage report pages")
    a.add_argument("--log", metavar="FILE", action="append", default=[], help=argparse.SUPPRESS)
    a.add_argument("--litellm", metavar="FILE", action="append", default=[], help=argparse.SUPPRESS)
    a.add_argument("--inventory", metavar="FILE", help="EC2 inventory (CSV/JSON)")
    a.add_argument("--utilization", metavar="FILE",
                   help="hourly CPU/GPU samples (scripts/export_cloudwatch.py output)")
    a.add_argument("--revenue", metavar="FILE", help="customer_id, monthly_revenue (enables margin view)")
    a.add_argument("--margin-alert-pct", type=float, default=30.0,
                   help="flag customers whose AI cost exceeds this %% of revenue")
    a.add_argument("--proof", metavar="JSON", help="results from `finops proof`")
    a.add_argument("--sample-days", type=float, help="length of the data window in days "
                                                     "(default: inferred from timestamps)")
    a.add_argument("--sample-rate", type=float, default=1.0,
                   help="fraction of traffic present in a sampled request log")
    a.add_argument("-o", "--output", metavar="FILE", help="write Markdown report here (default: stdout)")
    a.add_argument("--json", metavar="FILE", dest="json_out", help="also write the full JSON result")
    a.add_argument("--actions-dir", metavar="DIR",
                   help="write litellm_router.yaml and ec2_schedules.sh here")
    a.add_argument("--demo", action="store_true", help="run on the bundled sample company")
    a.add_argument("--today", help=argparse.SUPPRESS)   # pin 'today' for reproducible output
    _add_common(a)

    pr = sub.add_parser("proof", help="quality-test proposed routes on your logged traffic",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    pr.add_argument("--usage", metavar="FILE", action="append", required=True,
                    help="request-level log with prompts and responses")
    src = pr.add_mutually_exclusive_group(required=True)
    src.add_argument("--candidate-outputs", metavar="JSONL",
                     help="pre-computed candidate answers: {request_id|prompt_sha1, output}")
    src.add_argument("--base-url", help="OpenAI-compatible API base URL "
                                        "(e.g. https://api.openai.com/v1 or your LiteLLM proxy)")
    pr.add_argument("--api-key-env", default="FINOPS_API_KEY",
                    help="environment variable holding the API key for --base-url")
    pr.add_argument("--model-id", action="append", default=[], metavar="NAME=ID",
                    help="map a price-book model to the id your endpoint expects (repeatable)")
    pr.add_argument("--candidate", help="test this model for every routable task instead of the policy's")
    pr.add_argument("--judge-model", help="model id used to grade free-text tasks")
    pr.add_argument("--per-task", type=int, default=50, help="samples per task/candidate")
    pr.add_argument("--threshold", type=float, help="required pass rate (default: policy)")
    pr.add_argument("--concurrency", type=int, default=4)
    pr.add_argument("--max-cost", type=float, default=5.0,
                    help="abort if the estimated replay cost (USD) exceeds this")
    pr.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    pr.add_argument("-o", "--output", default="proof_results.json")
    _add_common(pr)

    pp = sub.add_parser("pricing", help="inspect the price book")
    pp.add_argument("models", nargs="*", help="model names to resolve")
    pp.add_argument("--list", nargs="?", const="all", metavar="PROVIDER",
                    help="list prices (optionally one provider)")
    pp.add_argument("--pricing-config", metavar="DIR")
    return parser


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------

def cmd_audit(args: argparse.Namespace) -> int:
    quiet = args.quiet
    usage: List[Tuple[str, str]] = [(p, args.format) for p in args.usage + args.log]
    usage += [(p, "litellm") for p in args.litellm]
    if args.demo:
        usage = [(os.path.join(DEMO_DIR, "usage_daily.csv"), "auto")]
        args.inventory = args.inventory or os.path.join(DEMO_DIR, "inventory.csv")
        args.utilization = args.utilization or os.path.join(DEMO_DIR, "utilization.csv")
        args.revenue = args.revenue or os.path.join(DEMO_DIR, "revenue.csv")
        args.proof = args.proof or os.path.join(DEMO_DIR, "proof_results.json")
    if not usage and not args.inventory and not args.utilization:
        print("error: nothing to audit. Provide --usage and/or --inventory/--utilization, "
              "or try --demo.", file=sys.stderr)
        return 2

    today = date.fromisoformat(args.today) if args.today else None
    book = load_pricebook(args.pricing_config)
    policy = load_policy(args.policy, book)
    _info(f"Prices as of {book.as_of} ({len(book.models)} models, {len(book.instances)} instance types)",
          quiet)
    bundle = AuditBundle(pricebook=book, policy=policy)
    if today:
        bundle.generated_at = today.isoformat()

    if usage:
        records = []
        for path, fmt in usage:
            recs, rep = load_usage_file(path, fmt)
            records.extend(recs)
            bundle.ingest.append(rep)
            _info(rep.summary(), quiet)
        if args.attribution:
            amap = load_attribution_map(args.attribution)
            apply_attribution(records, amap)
            unused = [i + 1 for i, r in enumerate(amap.rules) if r.hits == 0]
            if unused:
                _info(f"attribution rules that matched nothing: {unused}", quiet)
        bundle.task_inference = infer_tasks(records)
        proof_index = None
        if args.proof:
            from finops.proof import load_proof_index
            proof_index = load_proof_index(args.proof)
            bundle.proof_path = args.proof
            for verdict in proof_index.values():
                bundle.proof_counts[verdict] = bundle.proof_counts.get(verdict, 0) + 1
        bundle.llm = analyze_llm_spend(records, book, policy, sample_days=args.sample_days,
                                       sample_rate=args.sample_rate, proof=proof_index, today=today)
        if args.revenue:
            revenue, rep = load_revenue(args.revenue)
            bundle.ingest.append(rep)
            bundle.margins = compute_margins(bundle.llm, revenue, args.margin_alert_pct)

    if args.inventory or args.utilization:
        inventory = []
        if args.inventory:
            inventory, rep = load_inventory(args.inventory)
            bundle.ingest.append(rep)
        series, types = {}, {}
        if args.utilization:
            series, types, rep = load_utilization(args.utilization)
            bundle.ingest.append(rep)
        bundle.infra = analyze_infra(inventory, book, series, types)

    bundle.actions = build_actions(bundle.llm, bundle.infra)
    if args.actions_dir:
        bundle.artifacts = write_artifacts(args.actions_dir, bundle.llm, bundle.infra, book)

    report = render_markdown(bundle)
    if args.output:
        _write(args.output, report)
        _info(f"Report written to {args.output}", quiet)
    else:
        print(report)
    if args.json_out:
        _write(args.json_out, render_json(bundle))
        _info(f"JSON written to {args.json_out}", quiet)
    _info(f"Identified savings: {usd(bundle.monthly_savings)}/mo "
          f"({usd(bundle.monthly_savings * 12)}/yr)", quiet)
    return 0


def _write(path: str, text: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


# ---------------------------------------------------------------------------
# proof
# ---------------------------------------------------------------------------

def cmd_proof(args: argparse.Namespace) -> int:
    from finops.proof import (OpenAICompatibleClient, estimate_replay_cost,
                              load_candidate_outputs, make_judge, run_proof, select_samples)
    quiet = args.quiet
    book = load_pricebook(args.pricing_config)
    policy = load_policy(args.policy, book)
    records = []
    for path in args.usage:
        recs, rep = load_usage_file(path, args.format)
        records.extend(recs)
        _info(rep.summary(), quiet)
    if args.attribution:
        apply_attribution(records, load_attribution_map(args.attribution))
    infer_tasks(records)

    model_ids = {}
    for item in args.model_id:
        if "=" not in item:
            print(f"error: --model-id expects NAME=ID, got '{item}'", file=sys.stderr)
            return 2
        k, v = item.split("=", 1)
        model_ids[k.strip().lower()] = v.strip()

    client = outputs = judge = None
    if args.candidate_outputs:
        outputs = load_candidate_outputs(args.candidate_outputs)
    else:
        key = os.environ.get(args.api_key_env)
        if not key:
            print(f"error: set {args.api_key_env} to the API key for {args.base_url}", file=sys.stderr)
            return 2
        client = OpenAICompatibleClient(args.base_url, key)
        samples = select_samples(records, policy, book, args.per_task,
                                 candidate_override=args.candidate)
        est = estimate_replay_cost(samples, book)
        n = sum(len(v) for v in samples.values())
        _info(f"{n} samples across {len(samples)} route(s); estimated replay cost {usd(est)}"
              + (" (+ judge calls)" if args.judge_model else ""), quiet)
        if est > args.max_cost and not args.yes:
            print(f"error: estimated cost {usd(est)} exceeds --max-cost {usd(args.max_cost)}; "
                  "lower --per-task or pass --yes.", file=sys.stderr)
            return 1
    if args.judge_model:
        if client is None:
            print("error: --judge-model needs --base-url (the judge is called through it)",
                  file=sys.stderr)
            return 2
        judge = make_judge(client, args.judge_model)

    report = run_proof(records, policy, book, per_task=args.per_task, threshold=args.threshold,
                       client=client, candidate_outputs=outputs, model_ids=model_ids,
                       judge=judge, candidate_override=args.candidate,
                       concurrency=args.concurrency,
                       progress=None if quiet else (lambda m: _info(m)))
    _write(args.output, json.dumps(report.to_json(), indent=2, ensure_ascii=False))

    lines = ["# Quality proof", "",
             f"Threshold: {report.threshold:.0%} pass rate (Wilson 95% lower bound).", "",
             "| Task | Candidate | Scorer | Samples | Pass rate | 95% CI | Verdict |",
             "| --- | --- | --- | ---: | ---: | --- | --- |"]
    for g in report.groups:
        lines.append(f"| {g.task_type} | {g.candidate} | {g.scorer} | {g.n} | {g.pass_rate:.0%} | "
                     f"{g.ci_low:.0%}–{g.ci_high:.0%} | **{g.verdict.upper()}** |")
    for note in report.notes:
        lines.append(f"\n- {note}")
    for g in report.groups:
        for f in g.failures[:2]:
            prod = " ".join(f["production"].split())[:70]
            cand = " ".join(f["candidate"].split())[:70]
            lines.append(f"\n> {g.task_type} miss ({f['reason']}): production `{prod}` "
                         f"vs candidate `{cand}`")
    print("\n".join(lines))
    _info(f"Results written to {args.output}; use: finops audit ... --proof {args.output}", quiet)
    return 0


# ---------------------------------------------------------------------------
# pricing
# ---------------------------------------------------------------------------

def cmd_pricing(args: argparse.Namespace) -> int:
    book = load_pricebook(args.pricing_config)
    print(f"Price book as of {book.as_of} ({book.age_days()} days old)"
          + ("  ⚠ STALE" if book.is_stale() else ""))
    for name in args.models:
        price = book.resolve(name)
        if price:
            print(f"{name} -> {price.name} [{price.provider}/{price.tier}] in ${price.input}/M "
                  f"cached ${price.cached_rate}/M out ${price.output}/M"
                  + (f" batch -{price.batch_discount:.0%}" if price.batch_discount else "")
                  + (f" successor {price.successor}" if price.successor else ""))
        else:
            print(f"{name} -> NOT PRICED (add it to a --pricing-config override)")
    if args.list:
        print(f"\n{'model':28} {'provider':12} {'tier':9} {'input':>8} {'cached':>8} {'output':>8}")
        for p in sorted(book.models.values(), key=lambda m: (m.provider, m.tier, m.input)):
            if args.list != "all" and p.provider != args.list:
                continue
            print(f"{p.name:28} {p.provider:12} {p.tier:9} {p.input:8.3f} {p.cached_rate:8.3f} "
                  f"{p.output:8.3f}" + ("  legacy" if p.status == "legacy" else ""))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    _force_utf8_stdio()
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0].startswith("-") and argv[0] not in ("-h", "--help", "--version"):
        argv = ["audit"] + argv          # v1 compatibility: flags without a subcommand
    args = build_parser().parse_args(argv)
    handlers = {"audit": cmd_audit, "proof": cmd_proof, "pricing": cmd_pricing}
    try:
        return handlers[args.command](args)
    except (DataLoadError, PricingError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
