# 💰 FinOps Arbitrage Audit — P&L Savings Report

_Generated 2026-11-10 19:31 · AI & Cloud FinOps Arbitrage Engine v1.0_

- Source (LiteLLM spend logs): `sample_data/litellm_logs.json`

## Executive Summary

- **API burn (current):** $6.50/mo → **$0.86/mo** recoverable via model routing
- **Total annualized savings opportunity: $10.28**

## 1. Model ROI & Arbitrage (Module A)

- **Records analyzed:** 10 over a 1.0-day window (extrapolated to a 30.4-day month)
- **Current monthly run-rate:** **$6.50**
- **Optimized run-rate (smart routing):** **$5.64**
- **Net monthly savings:** **$0.86** (13.2% reduction)
- **Net annual savings delta:** **$10.28**

### Spend attribution (top cost centers)

| Model | Task | Calls/mo | Current $/mo | Optimized $/mo | Savings $/mo | Routed To |
|---|---|---:|---:|---:|---:|---|
| `openai/gpt-4o` | code_gen | 58 | $3.22 | $3.22 | **$0.00** | — (frontier retained) |
| `anthropic/claude-3-5-sonnet` | complex_reasoning | 58 | $2.31 | $2.31 | **$0.00** | — (frontier retained) |
| `openai/gpt-4o` | summarization | 58 | $0.55 | $0.11 | **$0.44** | llama-3.1-70b-instruct |
| `openai/gpt-4o` | classification | 58 | $0.22 | $0.00 | **$0.22** | llama-3.1-8b-instruct |
| `openai/gpt-4o` | extraction | 29 | $0.14 | $0.00 | **$0.14** | llama-3.1-8b-instruct |
| `openai/gpt-4o` | tagging | 29 | $0.06 | $0.00 | **$0.06** | llama-3.1-8b-instruct |

### By task type

| Task | Current $/mo | Optimized $/mo | Savings |
|---|---:|---:|---:|
| code_gen | $3.22 | $3.22 | **$0.00** |
| complex_reasoning | $2.31 | $2.31 | **$0.00** |
| summarization | $0.55 | $0.11 | **$0.44** |
| classification | $0.22 | $0.00 | **$0.22** |
| extraction | $0.14 | $0.00 | **$0.14** |
| tagging | $0.06 | $0.00 | **$0.06** |

> **Action:** Route `classification`, `extraction`, and `summarization` traffic to budget open-weight models (Llama 3.x via Groq/Together). Run an A/B quality eval on the top rerouted task before full cutover.

### Unit economics (attribution coverage: 100.0% of records)

**Team**

| Key | Calls/mo | Current $/mo | Optimized $/mo | Savings |
|---|---:|---:|---:|---:|
| platform | 58 | $3.22 | $3.22 | **$0.00** |
| core-product | 117 | $2.86 | $2.42 | **$0.44** |
| growth | 88 | $0.36 | $0.01 | **$0.36** |
| ml-infra | 29 | $0.06 | $0.00 | **$0.06** |

**Feature / customer**

| Key | Calls/mo | Current $/mo | Optimized $/mo | Savings |
|---|---:|---:|---:|---:|
| copilot | 58 | $3.22 | $3.22 | **$0.00** |
| agent-planner | 58 | $2.31 | $2.31 | **$0.00** |
| inbox-digest | 58 | $0.55 | $0.11 | **$0.44** |
| customer:cust-initech | 29 | $0.39 | $0.08 | **$0.31** |
| lead-scoring | 88 | $0.36 | $0.01 | **$0.36** |
| customer:cust-globex | 58 | $0.30 | $0.04 | **$0.27** |
| customer:cust-acme | 58 | $0.22 | $0.00 | **$0.22** |
| dedup | 29 | $0.06 | $0.00 | **$0.06** |

> **Action:** Join these cost lines against revenue for the matching feature/customer keys to get true AI gross margin. Untagged traffic is invisible to unit economics — instrument it via LiteLLM metadata tags.

---
_Assumptions: published on-demand list pricing; utilization averages from the supplied export; smart-routing assumes budget open-weight models meet quality bars for low-complexity tasks. Validate with an A/B eval before cutover._