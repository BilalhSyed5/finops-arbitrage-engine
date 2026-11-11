# FinOps Arbitrage Engine

**Find what your AI and GPU spend could cost, prove the cheaper option holds up on your own traffic, and hand engineers the config to ship it.**

A developer-first CLI that reads LLM usage exports (OpenAI, Anthropic, LiteLLM, or any CSV/JSON) and EC2/GPU utilization, then produces:

1. **A dollarized savings plan.** Every number is labelled ✅ *verified*, 📏 *measured* or 🔶 *estimated*, so nobody has to take it on faith.
2. **A quality proof.** It replays a sample of your real prompts on the cheaper model and scores the answers against production. A route that fails is blocked, not recommended.
3. **Ready-to-apply changes.** A LiteLLM router config and an EventBridge Scheduler script for idle instances.
4. **Unit economics.** Cost per team, feature and customer, and AI cost as a share of each customer's revenue.

The core is pure Python standard library: no dependencies, runs inside a customer's VPC, and their data never has to leave it.

```bash
python finops_audit.py audit --demo          # full sample company: report + configs
```

See the output in [`examples/demo_report.md`](examples/demo_report.md) and [`examples/actions/`](examples/actions/).

---

## What it finds

| Lever | What it does | Typical confidence |
|---|---|---|
| **Model refresh** | Same-tier successor that is cheaper (e.g. Claude Sonnet 4.5 → 5.5, GPT-4o → GPT-6.1 Sol). Works for every task type. | 📏 measured |
| **Task routing** | Classification, extraction, tagging, summarization… → a cheaper model, *only if it's actually cheaper for that traffic*. Defaults to the same provider (a config change, not a vendor migration). | 🔶 → ✅ after `finops proof` |
| **Prompt caching** | Measured from shared prompt prefixes when prompts are logged; otherwise a stated, conservative assumption. | 📏 / 🔶 |
| **Batch API** | Non-interactive traffic (nightly jobs, backfills) at 50% off. | 📏 measured |
| **Idle instances** | Hour-by-hour idle windows from CloudWatch data → a concrete stop/start schedule, with the busy hours it would interrupt. | 📏 measured |
| **Zombies** | Instances idle ~all the time. | 📏 measured |
| **Right-sizing** | Smallest same-family size that fits p95 active load (CPU and GPU count). | 🔶 estimated (GPU memory isn't visible) |

The levers stack without double counting: each is applied to the cost left by the previous one, so they always sum to the total.

It also flags **announced price increases** on models you use, **unpriced models** (fine-tunes, new releases: excluded, never guessed), and **reconciliation** gaps between list-price math and what you were billed (a sign you have negotiated rates to load).

## Quick start

```bash
# Python 3.10+. No install needed; or `pip install .` to get a `finops` command.
python finops_audit.py audit --demo -o report.md --actions-dir out/

# Your data
python finops_audit.py audit \
  --usage litellm_spend_logs.json \
  --inventory inventory.csv --utilization utilization.csv \
  --revenue revenue.csv \
  -o report.md --json report.json --actions-dir out/

# Prove the routes before switching (replays prompts through any OpenAI-compatible API)
export FINOPS_API_KEY=sk-...
python finops_audit.py proof --usage request_log.jsonl --base-url http://localhost:4000 \
  --judge-model gpt-6.1-sol -o proof.json
python finops_audit.py audit --usage litellm_spend_logs.json --proof proof.json ...

# Inspect prices
python finops_audit.py pricing claude-sonnet-4-5 gpt-4o-mini-2024-07-18
python finops_audit.py pricing --list anthropic

# Tests
python -m unittest discover -s tests -t .
```

v1 flags (`--log`, `--litellm`, no subcommand) still work.

## Getting the data

| Source | How to export | Flag |
|---|---|---|
| **LiteLLM proxy** (best: per-request, with team/feature/customer tags) | `GET /spend/logs` or export the `LiteLLM_SpendLogs` table. Turn on `store_prompts_in_spend_logs` to enable proof mode and measured caching. | `--usage` |
| **OpenAI** | Usage API (admin key): `GET /v1/organization/usage/completions?start_time=…&bucket_width=1d&group_by=model&group_by=project_id`. Save the raw page(s). | `--usage` + `--attribution` |
| **Anthropic** | Admin API: `GET /v1/organizations/usage_report/messages?starting_at=…&bucket_width=1d&group_by[]=model&group_by[]=workspace_id`. Save the raw page(s). | `--usage` + `--attribution` |
| **Anything else** | CSV/JSON/JSONL with `model`, `input_tokens`, `output_tokens` (aliases like `prompt_tokens` work) and optionally `timestamp`, `requests`, `cached_input_tokens`, `task_type`, `team`, `feature`, `customer_id`, `actual_spend`, `prompt`/`messages`, `response`. | `--usage` |
| **EC2 + CloudWatch** | `pip install boto3 && python scripts/export_cloudwatch.py --days 14` (read-only IAM). GPU % needs the CloudWatch agent's NVIDIA plugin. | `--inventory`, `--utilization` |
| **Revenue** | CSV `customer_id, monthly_revenue`. | `--revenue` |

Formats are auto-detected; force one with `--format`. Bad rows are skipped and counted (shown under *Inputs* in the report), never silently dropped.

**Provider exports don't say what a call was for.** Most teams use one project or API key per feature, so a short [`--attribution`](sample_data/attribution.json) map (glob rules: project/key/model → team, feature, task type) usually covers most of the spend. Whatever is left is inferred from prompt text or token shape. Inferred task types are labelled, and savings that rely on them are never marked verified.

## The confidence model

| Label | Meaning |
|---|---|
| ✅ **verified** | The route passed a quality proof on your traffic: Wilson 95% lower bound of the pass rate ≥ threshold (default 90%, ≥30 samples). |
| 📏 **measured** | Computed directly from your data at list (or your override) prices. |
| 🔶 **estimated** | Depends on an assumption the report states: an unproven route, an assumed cacheable prefix, average-only utilization, right-sizing. |

**Proof scoring** depends on the task: exact label match (classification, tagging, routing), JSON field-level match (extraction, formatting), or an LLM judge for free text (`--judge-model`). Without a judge, free text is scored by lexical F1, and that alone can never produce a pass. Point `--base-url` at OpenAI, at Anthropic's or Google's OpenAI-compatible endpoints, or at your LiteLLM proxy. Use `--candidate-outputs` to score answers you generated in your own pipeline. A `--max-cost` guard (default $5) stops an expensive replay before it starts.

## Configuration

Defaults (prices as of 2026-11-11, routing policy, thresholds) live in `finops/data/`. Override them by merging, not by editing: `--pricing-config DIR` for negotiated rates and fine-tunes, `--policy FILE` for routing targets, cross-provider moves and proof thresholds. See [`config/README.md`](config/README.md).

Prices go stale fast. Every report shows the price-book date and warns after 45 days.

## Architecture

```
finops_audit.py          CLI shim  →  finops/cli.py (audit | proof | pricing)
finops/
  models.py              UsageRecord (normalized usage) + Confidence labels
  pricing.py             PriceBook: model-name resolution, cache/batch/long-context cost math
  policy.py              Lever policy (routing targets, caching/batch/proof settings)
  ingest/                adapters (generic, LiteLLM, OpenAI, Anthropic), attribution map, task inference
  llm_audit.py           Run-rate, attribution, the four stacked LLM levers
  infra_audit.py         Time-series idle schedules, zombies, right-sizing; average-mode fallback
  proof.py               Sampling, OpenAI-compatible client, scorers, Wilson verdicts
  margin.py              AI cost vs. customer revenue
  actions.py             Ranked action plan, LiteLLM config, EC2 scheduler script
  report.py              Markdown + JSON rendering
  data/                  Default price book and policy
scripts/
  export_cloudwatch.py   Customer-side EC2 + CloudWatch export
  generate_demo_data.py  Rebuilds sample_data/ (fictional company)
tests/                   71 unittest cases (stdlib only)
```

Every module works without the CLI, so a FastAPI service or web UI can wrap it directly:

```python
from finops.ingest import load_usage
from finops.pricing import load_pricebook
from finops.policy import load_policy
from finops.llm_audit import analyze_llm_spend

book = load_pricebook()
records, reports, _ = load_usage(["spend_logs.json"])
result = analyze_llm_spend(records, book, load_policy(None, book))
print(result.monthly_savings, result.savings_by_confidence)
```

## Known limits

- **Untested against live exports.** The OpenAI and Anthropic usage-export adapters follow the documented formats and are tested on fixtures, not yet on real customer files. Validate on the first real export.
- **Routing assumes similar token counts on the new model.** Reasoning models can spend more output tokens; the proof step is where that shows up.
- **Prices are list prices with an `as_of` date.** OpenAI and Gemini have promotional and introductory rates that change on announced dates. Refresh `finops/data/model_pricing.json` regularly.
- **Infrastructure covers EC2 only for now.** No Kubernetes, SageMaker, or GCP/Azure yet. Instances with only average utilization get estimates, not schedules.
