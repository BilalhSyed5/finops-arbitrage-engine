# FinOps Arbitrage Engine

A lightweight, developer-first diagnostic CLI that ingests **raw LLM API cost logs**
and **cloud resource inventories** and outputs an executive-ready Markdown P&L
savings report. It quantifies:

1. **Model arbitrage (Module A)** — how much you'd save by routing
   classification / extraction / summarization traffic from frontier models
   (GPT-4o, Claude 3.5 Sonnet) to budget open-weight models
   (Llama 3.x via Groq/Together).
2. **Infrastructure right-sizing (Module B)** — idle and under-utilized
   GPU/EC2 instances, dollarized per instance ("Instance X runs 24/7 but sits
   idle 18h → schedule auto-shutdown / serverless inference to save $X/mo").

## Quick start

```bash
# Run against the bundled sample data
python finops_audit.py --demo

# Audit your own data (Markdown report to stdout)
python finops_audit.py --log path/to/api_logs.csv --inventory path/to/inventory.json

# Save the report to a file
python finops_audit.py --log api_logs.csv -o savings_report.md

# Audit a LiteLLM proxy spend-log export (with team/feature attribution)
python finops_audit.py --litellm litellm_spend_logs.json

# Use a custom pricing config directory
python finops_audit.py --demo --pricing-config path/to/my_config

# Run the smoke tests
python -m unittest discover -s tests -v
```

## Input formats

**LiteLLM proxy logs** (`--litellm`, JSON array or JSONL): the adapter maps `model`/`prompt_tokens`/`completion_tokens`/`startTime` and pulls attribution from `metadata.task_type`, `metadata.team`, `metadata.feature`, `end_user`, or `tags: ["team:x", "feature:y"]`. See [sample_data/litellm_logs.json](sample_data/litellm_logs.json).

**API usage log** (`--log`, CSV or JSON list):

| timestamp | model_used | input_tokens | output_tokens | task_type |
|---|---|---|---|---|
| 2025-11-03T09:14:00 | gpt-4o | 1240 | 86 | classification |
| 2025-11-03T09:15:03 | claude-3-5-sonnet | 3200 | 1450 | complex_reasoning |

Supported `task_type` values (case-insensitive): `classification`, `extraction`,
`summarization`, `simple_qa`, `tagging`, `routing`, `deduplication`,
`formatting` (routed to budget models), plus `complex_reasoning`, `code_gen`,
and anything else (kept on frontier). Unknown models are priced conservatively
at the most expensive frontier rate so waste is never under-reported.

**Cloud inventory** (`--inventory`, CSV or JSON list):

```json
{
  "instance_id": "i-0a1b2c3d4e5f60011",
  "instance_type": "g5.12xlarge",
  "avg_cpu_utilization_pct": 8.2,
  "avg_gpu_utilization_pct": 6.1,
  "running_hours_per_day": 24
}
```

An optional `hourly_rate` field overrides the pricing table (useful for
reserved/spot rates or other clouds).

## Pricing configuration

Prices are **config-driven**, not hardcoded. `config/model_pricing.json`
(frontier + budget token prices, USD per 1M tokens as `[input, output]`) and
`config/ec2_pricing.json` (EC2 hourly rates) merge over built-in defaults.
Override a few SKUs or replace whole tables — commit a snapshot alongside
each report for reproducibility:

```bash
python finops_audit.py --log api_logs.csv --pricing-config my_config/
```

## Attribution & unit economics

Add optional `team`, `feature`, and `customer_id` columns (or LiteLLM
metadata tags) and the report gains a **Unit economics** section: monthly
cost and savings per team, feature, and customer. Join those lines against
revenue to get true AI gross margin. Untagged records are excluded and the
report shows your attribution coverage so you know what's invisible.

## Architecture

```
finops_audit.py            CLI entrypoint (argparse, error handling, exit codes)
finops/
  data_loader.py           Tolerant CSV/JSON ingestion + strict row validation
  pricing_config.py        Config-driven pricing (model + EC2), built-in fallbacks
  cost_calculator.py       Module A: routing policy, run-rate math, attribution rollups
  infrastructure_audit.py  Module B: idle/under-utilization detection, $ waste
  report_generator.py      Executive Markdown report rendering
  adapters/
    litellm.py             LiteLLM proxy spend-log adapter (with tag attribution)
config/                    model_pricing.json, ec2_pricing.json (override defaults)
tests/test_finops.py       Smoke tests (unittest, stdlib only)
sample_data/               Demo api_logs.csv, inventory.json, litellm_logs.json
```

Each module is importable standalone, so the analysis layer can be wrapped in
a FastAPI endpoint or web UI later without touching the CLI:

```python
from finops.data_loader import load_records
from finops.cost_calculator import analyze_api_costs

rows = load_records("api_logs.csv",
                    ("timestamp", "model_used", "input_tokens",
                     "output_tokens", "task_type"),
                    ("input_tokens", "output_tokens"))
result = analyze_api_costs(rows)
print(result.annual_savings)
```

## Maintenance notes

- **Pricing tables** live in `config/*.json` (with built-in defaults in
  `finops/pricing_config.py`). They are reasonable published list prices but
  should be refreshed from provider pricing pages before board-level
  decisions.
- **Routing policy** is the `BUDGET_ELIGIBLE_TASKS` set in `cost_calculator.py`.
  Tune it per product: add a task type there (and a budget model in
  `DEFAULT_BUDGET_MODEL_BY_TASK`) once an A/B eval proves quality parity.
- **Utilization thresholds** (`IDLE_UTILIZATION_PCT`, `SEVERE_UNDER_UTILIZATION_PCT`)
  are in `infrastructure_audit.py`.
- Malformed rows are skipped with a warning to stderr; if *every* row is
  malformed, the tool exits with code 1 and a clear message.
