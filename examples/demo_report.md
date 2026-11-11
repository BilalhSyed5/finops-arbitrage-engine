# AI & Cloud Cost Audit

_Generated 2026-11-11 · finops-arbitrage-engine v2.0.0 · prices as of 2026-11-11_

> **Data notes**
> - 1 model(s) have no price (ft:gpt-4o-mini:acme-demo::f9c2; 12,516,894 tokens/mo). They are excluded from all dollar figures — add them to a pricing override.

## Executive summary

|  | Monthly | Annualized |
| --- | ---: | ---: |
| LLM API run-rate | $28,014 | $336,170 |
| Compute (8 instances audited) | $26,729 | $320,749 |
| **Identified savings** | **$39,165** | **$469,984** |

Savings by confidence: ✅ verified $9,363/mo (24%) · 📏 measured $24,032/mo (61%) · 🔶 estimated $5,771/mo (15%)

> ✅ **verified**: a quality test on your own traffic passed. 📏 **measured**: computed directly from your data. 🔶 **estimated**: depends on a stated assumption (see Methodology).

### Top actions

| # | Action | Saves / mo | Confidence | Effort |
| --- | --- | ---: | --- | --- |
| 1 | Stop idle instance training-sandbox (p4d.24xlarge) | $16,020 | 📏 measured | infra change (snapshot + stop) |
| 2 | Route `classification` calls from gpt-5.5 to gpt-6-luna | $9,363 | ✅ verified | config change at the gateway (alias swap) |
| 3 | Upgrade claude-sonnet-4-5 → claude-sonnet-5-5 (newer, cheaper, same tier) | $2,664 | 📏 measured | config change + regression test |
| 4 | Schedule llm-inference-prod (g5.12xlarge): stop 20:00→08:00 UTC on weeknights; off Sat–Sun | $2,660 | 📏 measured | infra change (EventBridge Scheduler) |
| 5 | Right-size rag-api: g6.12xlarge → g6.4xlarge | $2,392 | 🔶 estimated | infra change (stop → resize → start) |
| 6 | Route `summarization` calls from gpt-4o to gpt-6.1-terra | $1,498 | 🔶 estimated | config change at the gateway (alias swap) |
| 7 | Schedule embedding-batch (g5.4xlarge): stop 04:00→02:00 UTC every night | $1,086 | 📏 measured | infra change (EventBridge Scheduler) |
| 8 | Upgrade claude-opus-4-6 → claude-opus-5-5 (newer, cheaper, same tier) | $761 | 📏 measured | config change + regression test |
| 9 | Schedule jupyter-gpu (g5.2xlarge): auto-stop during idle hours | $752 | 🔶 estimated | infra change (EventBridge Scheduler) |
| 10 | Enable prompt caching for `invoice-extraction` on claude-sonnet-5-5 | $679 | 🔶 estimated | prompt restructuring / cache markers |

## 1. LLM API spend

- **Data:** 504 records (490 priced) over a 14.0-day window (inferred from timestamps); figures are scaled to a 30.4-day month.
- **Run-rate:** $28,014/mo → **$11,945/mo** after all levers (**−$16,069/mo, 57%**, $192,828/yr).

### Where the money goes

| Model | Provider | Tier | Requests/mo | $/mo | Share | Savable/mo |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `gpt-5.5` | openai | frontier | 1,092,574 | $9,790 | 35% | $9,635 |
| `claude-sonnet-4-5` | anthropic | mid | 532,337 | $7,993 | 29% | $3,344 |
| `claude-opus-4-6` | anthropic | frontier | 35,053 | $3,381 | 12% | $761 |
| `gpt-6.1-sol` | openai | frontier | 262,107 | $2,937 | 10% | $0.00 |
| `gpt-4o` | openai | frontier (legacy) | 130,514 | $2,605 | 9% | $1,676 |
| `gpt-5.4-mini` | openai | mid | 1,027,344 | $1,308 | 5% | $654 |

### Savings by lever

| Lever | Saves / mo | % of LLM spend |
| --- | ---: | ---: |
| Model refresh | $3,425 | 12% |
| Task routing | $11,092 | 40% |
| Prompt caching | $898 | 3% |
| Batch API | $654 | 2% |

### Lever detail

| Lever | Model | Task | Feature | Affected $/mo | Saves / mo | Confidence | Basis |
| --- | --- | --- | --- | ---: | ---: | --- | --- |
| Task routing | gpt-5.5 → gpt-6-luna | classification | support-triage | $9,553 | $9,363 | ✅ verified | quality proof passed |
| Model refresh | claude-sonnet-4-5 → claude-sonnet-5-5 | extraction | invoice-extraction | $7,993 | $2,664 | 📏 measured | same-tier successor; regression-test before switching |
| Task routing | gpt-4o → gpt-6.1-terra | summarization | doc-summaries | $2,605 | $1,498 | 🔶 estimated | pending quality proof (run `finops proof`) |
| Model refresh | claude-opus-4-6 → claude-opus-5-5 | complex_reasoning | research-agent | $3,381 | $761 | 📏 measured | same-tier successor; regression-test before switching |
| Prompt caching | claude-sonnet-5-5 | extraction | invoice-extraction | $5,329 | $679 | 🔶 estimated | assumes 30% of each long prompt is a stable prefix; log prompts to measure it |
| Batch API | gpt-5.4-mini | bulk_enrichment | nightly-enrichment | $1,308 | $654 | 📏 measured | non-interactive traffic; move to the provider's Batch API |
| Task routing | gpt-5.5 → gpt-6-luna | classification | — | $236 | $232 | 🔶 estimated | quality proof passed; task type inferred |
| Prompt caching | gpt-6.1-terra | summarization | doc-summaries | $1,107 | $178 | 🔶 estimated | assumes 30% of each long prompt is a stable prefix; log prompts to measure it |
| Prompt caching | gpt-6-luna | classification | support-triage | $191 | $40.60 | 🔶 estimated | assumes 30% of each long prompt is a stable prefix; log prompts to measure it |

### Routes blocked by the quality proof

- extraction: claude-sonnet-4-5 → claude-haiku-5-5: would have saved $7,727/mo, but failed the quality threshold; kept on the current model.

### Unpriced models (excluded from all figures)

| Model | Requests/mo | Tokens/mo |
| --- | ---: | ---: |
| `ft:gpt-4o-mini:acme-demo::f9c2` | 17,602 | 12,516,894 |

## 2. Unit economics

Attribution coverage (share of spend tagged): team 99% · feature 99% · customer 82%. Untagged spend is invisible in the tables below.
Task type known from: provided 99%, inferred 1%.

### By team

| Name | Requests/mo | Current $/mo | Optimized $/mo | Savings |
| --- | ---: | ---: | ---: | ---: |
| support | 1,042,112 | $9,553 | $150 | $9,403 |
| finance-ops | 532,337 | $7,993 | $4,650 | $3,344 |
| research | 35,053 | $3,381 | $2,620 | $761 |
| platform | 262,107 | $2,937 | $2,937 | $0.00 |
| product | 130,514 | $2,605 | $929 | $1,676 |
| data | 1,027,344 | $1,308 | $654 | $654 |

### By feature

| Name | Requests/mo | Current $/mo | Optimized $/mo | Savings |
| --- | ---: | ---: | ---: | ---: |
| support-triage | 1,042,112 | $9,553 | $150 | $9,403 |
| invoice-extraction | 532,337 | $7,993 | $4,650 | $3,344 |
| research-agent | 35,053 | $3,381 | $2,620 | $761 |
| code-copilot | 262,107 | $2,937 | $2,937 | $0.00 |
| doc-summaries | 130,514 | $2,605 | $929 | $1,676 |
| nightly-enrichment | 1,027,344 | $1,308 | $654 | $654 |

### By customer (top 10)

| Name | Requests/mo | Current $/mo | Optimized $/mo | Savings |
| --- | ---: | ---: | ---: | ---: |
| cust-northwind | 595,262 | $7,007 | $2,622 | $4,384 |
| cust-bluepeak | 388,299 | $4,568 | $1,733 | $2,835 |
| cust-redfern | 258,780 | $3,023 | $1,112 | $1,911 |
| cust-oakline | 215,718 | $2,523 | $945 | $1,578 |
| cust-silverbay | 195,400 | $2,295 | $866 | $1,429 |
| cust-ironleaf | 156,178 | $1,838 | $698 | $1,140 |
| cust-mapleton | 98,446 | $1,147 | $433 | $713 |
| cust-trial-042 | 58,985 | $688 | $256 | $432 |

### AI cost vs revenue

7 customers matched to revenue: AI cost is 13% of their revenue today, 5% after optimization. **2 customer(s) above 30%.**

| Customer | Revenue/mo | AI cost/mo | AI % of rev | After optimization |  |
| --- | ---: | ---: | ---: | ---: | --- |
| cust-bluepeak | $10,900 | $4,568 | 42% | 16% | ⚠️ |
| cust-ironleaf | $5,400 | $1,838 | 34% | 13% | ⚠️ |
| cust-silverbay | $10,900 | $2,295 | 21% | 8% |  |
| cust-oakline | $14,000 | $2,523 | 18% | 7% |  |
| cust-redfern | $25,200 | $3,023 | 12% | 4% |  |
| cust-mapleton | $12,700 | $1,147 | 9% | 3% |  |
| cust-northwind | $87,600 | $7,007 | 8% | 3% |  |

_1 customer id(s) with AI cost but no revenue row ($688/mo): free tier, trials, or an id mismatch._

## 3. Infrastructure (EC2 / GPU)

- **Audited:** 8 instances, $26,729/mo (list or inventory rates).
- **Recoverable:** **$23,096/mo** ($277,156/yr) across 6 instance(s); 2 look healthy.

| Instance | Type | Data | Idle | Cost/mo | Schedule | Right-size | Saves/mo | Confidence | Severity |
| --- | --- | --- | ---: | ---: | --- | --- | ---: | --- | --- |
| `i-0a1b2c3d4e5f60012` training-sandbox | p4d.24xlarge | timeseries (14d) | 99% | $16,020 | stop (idle) | — | **$16,020** | 📏 measured | CRITICAL |
| `i-0a1b2c3d4e5f60011` llm-inference-prod | g5.12xlarge | timeseries (14d) | 64% | $4,138 | stop 20:00→08:00 UTC on weeknights; off Sat–Sun | — | **$2,660** | 📏 measured | CRITICAL |
| `i-0a1b2c3d4e5f60014` rag-api | g6.12xlarge | timeseries (14d) | 0% | $3,357 | — | g6.4xlarge | **$2,392** | 🔶 estimated | CRITICAL |
| `i-0a1b2c3d4e5f60013` embedding-batch | g5.4xlarge | timeseries (14d) | 92% | $1,185 | stop 04:00→02:00 UTC every night | — | **$1,086** | 📏 measured | CRITICAL |
| `i-0a1b2c3d4e5f60018` jupyter-gpu | g5.2xlarge | average | 93% | $884 | auto-stop | — | **$752** | 🔶 estimated | HIGH |
| `i-0a1b2c3d4e5f60015` ci-runner | c5.4xlarge | timeseries (14d) | 70% | $496 | stop 23:00→08:00 UTC every night | — | **$186** | 📏 measured | MEDIUM |

**Details**

- `i-0a1b2c3d4e5f60012`: Idle 99% of the 14 days observed: stop it (snapshot first) or confirm an owner.
- `i-0a1b2c3d4e5f60011`: Schedule: stop 20:00→08:00 UTC on weeknights; off Sat–Sun (saves 108 h/week).
- `i-0a1b2c3d4e5f60014`: Right-size g6.12xlarge → g6.4xlarge (p95 active load: CPU 12%, GPU 15%).
  - _caveat: check GPU memory: the model must still fit on the smaller instance_
- `i-0a1b2c3d4e5f60013`: Schedule: stop 04:00→02:00 UTC every night (saves 154 h/week).
- `i-0a1b2c3d4e5f60018`: ~93% of running time looks idle: add auto-stop or move to serverless/on-demand inference.
  - _caveat: averages only: idle time is inferred, not measured; upload hourly utilization for a measured schedule_
- `i-0a1b2c3d4e5f60015`: Schedule: stop 23:00→08:00 UTC every night (saves 63 h/week).

## Ready-to-apply changes

- `examples/actions/litellm_router.yaml`
- `examples/actions/ec2_schedules.sh`

## Methodology & assumptions

- **Prices:** list prices as of 2026-11-11 (0 days old); files: model_pricing.json, ec2_pricing.json. Negotiated rates belong in a `--pricing-config` override.
- **Routing** moves only task types the policy marks routable, only to a model that is cheaper for that traffic, and assumes similar token counts on the new model. Targets stay with the current provider (no vendor migration needed).
- **Caching** is *measured* when prompts are logged (shared prefix ≥ 1024 tokens); otherwise *estimated* by assuming 30% of long prompts is a stable prefix with a 80% hit rate, only for groups above 1000 requests/mo. OpenAI traffic without cache data is skipped (OpenAI caches automatically).
- **Batch** applies to tasks/features the policy marks non-interactive: deduplication, backfill, offline_eval, bulk_enrichment, *batch*, *nightly*, *backfill*, *offline*, *bulk*.
- **Quality proof:** a route is verified when the Wilson 95% lower bound of its pass rate is ≥ 90% over ≥ 30 samples. Results loaded from `sample_data/proof_results.json` (pass: 1, fail: 1).
- **Infrastructure:** time-series instances are judged hour by hour (idle = GPU < 5% and CPU < 10% on GPU hosts, CPU < 5% otherwise); a weekly hour is stoppable when idle in ≥ 90% of observed weeks. Average-only instances are estimates. Right-sizing targets p95 active load at 70% and is always *estimated* (GPU memory isn't visible).
- **Task inference:** 0 records labelled from prompt text, 14 from token shape, 0 left unknown.

**Inputs**

- usage_daily.csv [generic]: 504 records from 504 rows
- revenue.csv [revenue]: 7 records from 7 rows
- inventory.csv [inventory]: 8 records from 8 rows
- utilization.csv [utilization]: 2,352 records from 2,352 rows
