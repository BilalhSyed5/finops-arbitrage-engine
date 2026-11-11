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

# Run the smoke tests
python -m unittest discover -s tests -v
```

## Input formats

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

An optional `hourly_rate` field overrides the built-in AWS on-demand pricing
table (useful for reserved/spot rates or other clouds).

## Architecture

```
finops_audit.py            CLI entrypoint (argparse, error handling, exit codes)
finops/
  data_loader.py           Tolerant CSV/JSON ingestion + strict row validation
  cost_calculator.py       Module A: pricing tables, routing policy, run-rate math
  infrastructure_audit.py  Module B: idle/under-utilization detection, $ waste
  report_generator.py      Executive Markdown report rendering
tests/test_finops.py       Smoke tests (unittest, stdlib only)
sample_data/               Demo api_logs.csv and inventory.json
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

- **Pricing tables** live at the top of `cost_calculator.py` (token prices)
  and `infrastructure_audit.py` (EC2 hourly rates). They are reasonable
  published list prices but should be refreshed from provider pricing pages
  before board-level decisions.
- **Routing policy** is the `BUDGET_ELIGIBLE_TASKS` set in `cost_calculator.py`.
  Tune it per product: add a task type there (and a budget model in
  `DEFAULT_BUDGET_MODEL_BY_TASK`) once an A/B eval proves quality parity.
- **Utilization thresholds** (`IDLE_UTILIZATION_PCT`, `SEVERE_UNDER_UTILIZATION_PCT`)
  are in `infrastructure_audit.py`.
- Malformed rows are skipped with a warning to stderr; if *every* row is
  malformed, the tool exits with code 1 and a clear message.
