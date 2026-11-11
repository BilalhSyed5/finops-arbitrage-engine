#!/usr/bin/env python3
"""Generate the bundled demo dataset (a fictional AI SaaS company).

Deterministic (fixed seed), so the committed sample_data/ and the demo
report are reproducible:

    python scripts/generate_demo_data.py
    python finops_audit.py proof --usage sample_data/request_samples.jsonl \
        --candidate-outputs sample_data/candidate_outputs.jsonl -o sample_data/proof_results.json
    python finops_audit.py audit --demo

Every company, customer and instance name here is invented.
"""

from __future__ import annotations

import csv
import json
import os
import random
import sys
from datetime import datetime, timedelta, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from finops.pricing import load_pricebook  # noqa: E402

OUT = os.path.join(ROOT, "sample_data")
START = datetime(2026, 10, 28, tzinfo=timezone.utc)
DAYS = 14
rng = random.Random(42)

CUSTOMERS = {  # share of customer-facing traffic, target AI cost as % of revenue
    "cust-northwind": (0.30, 0.08), "cust-bluepeak": (0.20, 0.42),
    "cust-redfern": (0.13, 0.12), "cust-oakline": (0.11, 0.18),
    "cust-silverbay": (0.10, 0.21), "cust-ironleaf": (0.08, 0.34),
    "cust-mapleton": (0.05, 0.09), "cust-trial-042": (0.03, None),
}

# feature: (team, model, task, requests/month, avg input, avg cached, avg output, per-customer?)
FEATURES = {
    "support-triage":     ("support", "gpt-5.5", "classification", 1_200_000, 1800, 0, 5, True),
    "invoice-extraction": ("finance-ops", "claude-sonnet-4-5-20250929", "extraction", 600_000, 3000, 0, 400, True),
    "doc-summaries":      ("product", "gpt-4o-2024-08-06", "summarization", 150_000, 6000, 0, 500, True),
    "research-agent":     ("research", "claude-opus-4-6", "complex_reasoning", 40_000, 20000, 12000, 2000, False),
    "code-copilot":       ("platform", "gpt-6.1-sol", "code_gen", 300_000, 4000, 2500, 800, True),
    "nightly-enrichment": ("data", "gpt-5.4-mini", "bulk_enrichment", 1_000_000, 800, 0, 150, False),
    "":                   ("", "gpt-5.5", "", 60_000, 900, 0, 4, False),          # untagged
    "fine-tune-pilot":    ("ml", "ft:gpt-4o-mini:acme-demo::f9c2", "classification", 20_000, 700, 0, 3, False),
}


def daily_usage() -> list:
    rows = []
    for d in range(DAYS):
        day = START + timedelta(days=d)
        weekday_factor = 1.0 if day.weekday() < 5 else 0.55
        for feature, (team, model, task, per_month, tin, tcached, tout, per_cust) in FEATURES.items():
            daily = per_month / 30.4 * (weekday_factor if feature != "nightly-enrichment" else 1.0)
            splits = ({c: s for c, (s, _) in CUSTOMERS.items()} if per_cust else {"": 1.0})
            for cust, share in splits.items():
                reqs = max(1, int(daily * share * rng.uniform(0.9, 1.1)))
                jitter = rng.uniform(0.95, 1.05)
                rows.append({
                    "date": day.date().isoformat(), "model": model, "team": team,
                    "feature": feature, "task_type": task, "customer_id": cust,
                    "requests": reqs,
                    "input_tokens": int(reqs * tin * jitter),
                    "cached_input_tokens": int(reqs * tcached * jitter),
                    "output_tokens": int(reqs * tout * rng.uniform(0.95, 1.05)),
                })
    return rows


def revenue(rows: list) -> list:
    """Set each customer's revenue so their AI cost lands near its target %."""
    book = load_pricebook()
    cost = {c: 0.0 for c in CUSTOMERS}
    for r in rows:
        if r["customer_id"]:
            price = book.resolve(r["model"])
            cost[r["customer_id"]] += price.cost(r["input_tokens"], r["output_tokens"],
                                                 cached_input_tokens=r["cached_input_tokens"],
                                                 requests=r["requests"])
    out = []
    for cust, (_, target) in CUSTOMERS.items():
        if target is None:
            continue   # free trial: cost with no revenue
        monthly_cost = cost[cust] * 30.4 / DAYS
        out.append({"customer_id": cust, "monthly_revenue": round(monthly_cost / target, -2)})
    return out


# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------

INSTANCES = [  # id, name, type, team, profile
    ("i-0a1b2c3d4e5f60011", "llm-inference-prod", "g5.12xlarge", "ml-platform", "business_hours"),
    ("i-0a1b2c3d4e5f60012", "training-sandbox", "p4d.24xlarge", "research", "zombie"),
    ("i-0a1b2c3d4e5f60013", "embedding-batch", "g5.4xlarge", "data", "nightly_job"),
    ("i-0a1b2c3d4e5f60014", "rag-api", "g6.12xlarge", "ml-platform", "oversized"),
    ("i-0a1b2c3d4e5f60015", "ci-runner", "c5.4xlarge", "platform", "ci"),
    ("i-0a1b2c3d4e5f60016", "api-backend", "m5.2xlarge", "platform", "steady"),
    ("i-0a1b2c3d4e5f60017", "analytics-db", "r5.2xlarge", "data", "steady"),
]
AVERAGE_ONLY = [  # in the inventory with averages but no hourly data
    {"instance_id": "i-0a1b2c3d4e5f60018", "instance_type": "g5.2xlarge", "name": "jupyter-gpu",
     "team": "research", "avg_cpu_utilization_pct": 4.0, "avg_gpu_utilization_pct": 2.5,
     "running_hours_per_day": 24, "hourly_rate": ""},
]


def sample(profile: str, ts: datetime) -> tuple:
    h, wd = ts.hour, ts.weekday()
    noise = rng.uniform(-3, 3)
    if profile == "business_hours":
        busy = wd < 5 and 8 <= h < 20
        return (max(0, (35 if busy else 2) + noise), max(0, (65 if busy else 1) + noise))
    if profile == "zombie":
        spike = ts.date() == (START + timedelta(days=2)).date() and 14 <= h < 17
        return (max(0, (60 if spike else 1.5) + noise / 3), max(0, (80 if spike else 0.5) + noise / 3))
    if profile == "nightly_job":
        busy = 2 <= h < 4
        return (max(0, (55 if busy else 2) + noise), max(0, (85 if busy else 1) + noise))
    if profile == "oversized":     # busy 24/7, but one GPU's worth of load on four
        return (max(0, 9 + noise), max(0, 12 + abs(noise)))
    if profile == "ci":
        busy = wd < 5 and 13 <= h < 23
        return (max(0, (45 if busy else 2) + noise), None)
    return (max(0, 45 + 3 * noise), None)   # steady


def infrastructure() -> tuple:
    inventory = [{"instance_id": i, "instance_type": t, "name": n, "team": team,
                  "avg_cpu_utilization_pct": "", "avg_gpu_utilization_pct": "",
                  "running_hours_per_day": "", "hourly_rate": ""}
                 for i, n, t, team, _ in INSTANCES] + AVERAGE_ONLY
    util = []
    for hour in range(DAYS * 24):
        ts = START + timedelta(hours=hour)
        for iid, _, itype, _, profile in INSTANCES:
            cpu, gpu = sample(profile, ts)
            util.append({"instance_id": iid, "instance_type": itype,
                         "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                         "cpu_pct": f"{cpu:.1f}", "gpu_pct": "" if gpu is None else f"{gpu:.1f}"})
    return inventory, util


# ---------------------------------------------------------------------------
# Request-level samples for proof mode
# ---------------------------------------------------------------------------

TRIAGE_SYSTEM = (
    "You are the support triage assistant for a B2B invoicing platform. Classify the customer "
    "ticket into exactly one category and respond with only the category name. Categories: "
    "billing, bug, feature_request, account_access, other. " + "Policy notes: " +
    " ".join(f"Rule {i}: route edge cases conservatively and never invent categories." for i in range(1, 40))
)
TICKETS = {
    "billing": ["I was charged twice for the October invoice run.", "Can you send a VAT receipt for our plan?",
                "Our card was declined but the plan still shows active.", "Why did our seat price change this month?"],
    "bug": ["The PDF export crashes when an invoice has more than 200 lines.", "Dashboard totals don't match the CSV export.",
            "Saving a draft invoice throws a 500 error.", "Line-item tax is rounding incorrectly in EUR."],
    "feature_request": ["Could you add bulk editing for due dates?", "We'd love a Slack notification when invoices are paid.",
                        "Please support multi-currency on a single invoice.", "Can the API return paginated customer lists?"],
    "account_access": ["I'm locked out after enabling 2FA on a new phone.", "Our admin left; how do we transfer ownership?",
                       "Password reset emails never arrive.", "SSO login loops back to the sign-in page."],
    "other": ["Do you have a partner program?", "Where are your data centers located?",
              "Can we schedule a call with your sales team?", "Is there a status page we can subscribe to?"],
}
VENDORS = ["Lumen Supply Co", "Harbor Freight Partners", "Cobalt Office Ltd", "Meadow Logistics", "Quarry Labs"]


def proof_samples() -> tuple:
    samples, candidates = [], []
    t0 = START
    for i in range(60):   # classification on gpt-5.5 -> gpt-6-luna; candidate right 59/60
        label = rng.choice(list(TICKETS))
        ticket = rng.choice(TICKETS[label])
        rid = f"req-triage-{i:03d}"
        samples.append({
            "request_id": rid, "model": "openai/gpt-5.5", "prompt_tokens": 1790, "completion_tokens": 4,
            "startTime": (t0 + timedelta(minutes=37 * i)).isoformat().replace("+00:00", "Z"),
            "end_user": rng.choice(list(CUSTOMERS)),
            "metadata": {"task_type": "classification", "team": "support", "feature": "support-triage"},
            "messages": [{"role": "system", "content": TRIAGE_SYSTEM},
                         {"role": "user", "content": f"Ticket #{4100 + i}: {ticket}"}],
            "response": label})
        wrong = i == 17
        candidates.append({"request_id": rid,
                           "output": "other" if wrong and label != "other" else
                           ("billing" if wrong else f"{label.upper()}." if i % 9 == 0 else label)})
    for i in range(40):   # extraction on sonnet -> haiku; candidate right 30/40 (fails)
        vendor = rng.choice(VENDORS)
        total = round(rng.uniform(120, 9800), 2)
        due = (START + timedelta(days=rng.randint(10, 60))).date().isoformat()
        ref = {"vendor": vendor, "invoice_number": f"INV-{20500 + i}", "total": total,
               "currency": "USD", "due_date": due}
        rid = f"req-extract-{i:03d}"
        samples.append({
            "request_id": rid, "model": "anthropic/claude-sonnet-4-5-20250929",
            "prompt_tokens": 2950, "completion_tokens": 410,
            "startTime": (t0 + timedelta(minutes=53 * i)).isoformat().replace("+00:00", "Z"),
            "end_user": rng.choice(list(CUSTOMERS)),
            "metadata": {"task_type": "extraction", "team": "finance-ops", "feature": "invoice-extraction"},
            "messages": [{"role": "user", "content":
                          "Extract vendor, invoice_number, total, currency and due_date from this invoice "
                          f"and return only JSON.\n\nINVOICE\nFrom: {vendor}\nInvoice no. INV-{20500 + i}\n"
                          f"Amount due: ${total:,.2f} USD\nPayment due by {due}\n"}],
            "response": json.dumps(ref)})
        cand = dict(ref)
        if i % 4 == 1:          # the cheaper model drops cents / misreads dates on 10 of 40
            cand["total"] = round(total)
            if i % 8 == 1:
                cand["due_date"] = "unknown"
                cand["currency"] = "usd$"
        candidates.append({"request_id": rid, "output": "```json\n" + json.dumps(cand) + "\n```"})
    return samples, candidates


# ---------------------------------------------------------------------------
# Small provider-format examples (used by tests and docs)
# ---------------------------------------------------------------------------

def provider_examples() -> tuple:
    day = int(START.timestamp())
    openai_page = {"object": "page", "has_more": False, "next_page": None, "data": [
        {"object": "bucket", "start_time": day + 86400 * d, "end_time": day + 86400 * (d + 1),
         "results": [
             {"object": "organization.usage.completions.result", "model": "gpt-5.5-2026-04-23",
              "project_id": "proj_support_triage", "api_key_id": None, "user_id": None, "batch": False,
              "input_tokens": 70_000_000, "input_cached_tokens": 0, "output_tokens": 200_000,
              "num_model_requests": 39_000, "input_audio_tokens": 0, "output_audio_tokens": 0},
             {"object": "organization.usage.completions.result", "model": "gpt-5.4-mini",
              "project_id": "proj_nightly", "api_key_id": None, "user_id": None, "batch": False,
              "input_tokens": 26_000_000, "input_cached_tokens": 0, "output_tokens": 4_900_000,
              "num_model_requests": 33_000, "input_audio_tokens": 0, "output_audio_tokens": 0}]}
        for d in range(3)]}
    anthropic_page = {"has_more": False, "next_page": None, "data": [
        {"starting_at": (START + timedelta(days=d)).isoformat().replace("+00:00", "Z"),
         "ending_at": (START + timedelta(days=d + 1)).isoformat().replace("+00:00", "Z"),
         "results": [{"model": "claude-sonnet-4-5-20250929", "workspace_id": "wrkspc_finance",
                      "api_key_id": None, "service_tier": "standard", "context_window": "0-200k",
                      "uncached_input_tokens": 59_000_000, "cache_read_input_tokens": 0,
                      "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
                      "output_tokens": 7_900_000, "server_tool_use": {"web_search_requests": 0}}]}
        for d in range(3)]}
    attribution = {"rules": [
        {"match": {"project": "proj_support*"},
         "set": {"team": "support", "feature": "support-triage", "task_type": "classification"}},
        {"match": {"project": "proj_nightly"},
         "set": {"team": "data", "feature": "nightly-enrichment", "task_type": "bulk_enrichment"}},
        {"match": {"project": "wrkspc_finance"},
         "set": {"team": "finance-ops", "feature": "invoice-extraction", "task_type": "extraction"}}]}
    return openai_page, anthropic_page, attribution


def _write_csv(path: str, rows: list) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: str, rows: list) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    usage = daily_usage()
    _write_csv(os.path.join(OUT, "usage_daily.csv"), usage)
    _write_csv(os.path.join(OUT, "revenue.csv"), revenue(usage))
    inventory, util = infrastructure()
    _write_csv(os.path.join(OUT, "inventory.csv"), inventory)
    _write_csv(os.path.join(OUT, "utilization.csv"), util)
    samples, candidates = proof_samples()
    _write_jsonl(os.path.join(OUT, "request_samples.jsonl"), samples)
    _write_jsonl(os.path.join(OUT, "candidate_outputs.jsonl"), candidates)
    openai_page, anthropic_page, attribution = provider_examples()
    for name, obj in (("openai_usage.json", openai_page), ("anthropic_usage.json", anthropic_page),
                      ("attribution.json", attribution)):
        with open(os.path.join(OUT, name), "w", encoding="utf-8") as handle:
            json.dump(obj, handle, indent=2)
    print(f"Wrote demo data to {OUT}: {len(usage)} usage rows, {len(util)} utilization samples, "
          f"{len(samples)} proof samples.")


if __name__ == "__main__":
    main()
