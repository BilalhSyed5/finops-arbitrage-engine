# 💰 FinOps Arbitrage Audit — P&L Savings Report

_Generated 2026-11-10 19:00 · AI & Cloud FinOps Arbitrage Engine v1.0_

- Source (API usage log): `C:\Users\19726\.lmstudio\apps\bionic\projects\d49037d8-47f4-5808-9028-c707de117f8f\workspace\sample_data\api_logs.csv`
- Source (Cloud inventory): `C:\Users\19726\.lmstudio\apps\bionic\projects\d49037d8-47f4-5808-9028-c707de117f8f\workspace\sample_data\inventory.json`

## Executive Summary

- **API burn (current):** $5.23/mo → **$0.81/mo** recoverable via model routing
- **Cloud waste (flagged):** $19,805.47/mo across 4 of 6 instances
- **Total annualized savings opportunity: $237,675.37**

## 1. Model ROI & Arbitrage (Module A)

- **Records analyzed:** 52 over a 7.1-day window (extrapolated to a 30.4-day month)
- **Current monthly run-rate:** **$5.23**
- **Optimized run-rate (smart routing):** **$4.42**
- **Net monthly savings:** **$0.81** (15.5% reduction)
- **Net annual savings delta:** **$9.73**

### Spend attribution (top cost centers)

| Model | Task | Calls/mo | Current $/mo | Optimized $/mo | Savings $/mo | Routed To |
|---|---|---:|---:|---:|---:|---|
| `gpt-4o` | code_gen | 34 | $2.32 | $2.32 | **$0.00** | — (frontier retained) |
| `claude-3-5-sonnet` | complex_reasoning | 39 | $1.71 | $1.71 | **$0.00** | — (frontier retained) |
| `gpt-4o` | summarization | 34 | $0.52 | $0.10 | **$0.42** | llama-3.1-70b-instruct |
| `gpt-4o` | complex_reasoning | 4 | $0.28 | $0.28 | **$0.00** | — (frontier retained) |
| `gpt-4o` | classification | 56 | $0.18 | $0.00 | **$0.18** | llama-3.1-8b-instruct |
| `gpt-4o` | extraction | 39 | $0.18 | $0.00 | **$0.18** | llama-3.1-8b-instruct |
| `gpt-4o` | tagging | 17 | $0.04 | $0.00 | **$0.04** | llama-3.1-8b-instruct |

### By task type

| Task | Current $/mo | Optimized $/mo | Savings |
|---|---:|---:|---:|
| code_gen | $2.32 | $2.32 | **$0.00** |
| complex_reasoning | $2.00 | $2.00 | **$0.00** |
| summarization | $0.52 | $0.10 | **$0.42** |
| classification | $0.18 | $0.00 | **$0.18** |
| extraction | $0.18 | $0.00 | **$0.18** |
| tagging | $0.04 | $0.00 | **$0.04** |

> **Action:** Route `classification`, `extraction`, and `summarization` traffic to budget open-weight models (Llama 3.x via Groq/Together). Run an A/B quality eval on the top rerouted task before full cutover.

## 2. Infrastructure & GPU Right-Sizing Audit (Module B)

- **Instances audited:** 6
- **Total monthly compute spend:** **$31,206.45**
- **Flagged dead capital:** **$19,805.47/month** (**$237,665.64/year**)

### Flagged instances (sorted by monthly waste)

| Instance | Type | GPU% | CPU% | Hrs/Day | Idle Hrs/Day | Cost $/mo | Waste $/mo | Severity |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `i-0a1b2c3d4e5f60012` | `p4d.24xlarge` | 11% | 22% | 24 | 15 | $23,908.99 | **$14,943.12** | CRITICAL |
| `i-0a1b2c3d4e5f60011` | `g5.12xlarge` | 6% | 8% | 24 | 21 | $4,138.29 | **$3,517.55** | CRITICAL |
| `i-0a1b2c3d4e5f60013` | `g5.4xlarge` | 10% | 12% | 24 | 19 | $1,481.09 | **$1,184.87** | CRITICAL |
| `i-0a1b2c3d4e5f60016` | `m5.2xlarge` | 0% | 19% | 20 | 14 | $233.47 | **$159.93** | HIGH |

### Recommended actions

- **i-0a1b2c3d4e5f60012** (`p4d.24xlarge`, $14,943.12/mo at stake): Running 24h/day but sits idle ~15h. Shift to serverless inference (e.g., Bedrock / RunPod / Lambda GPU) or schedule auto-stop outside business hours.
- **i-0a1b2c3d4e5f60011** (`g5.12xlarge`, $3,517.55/mo at stake): Running 24h/day but sits idle ~21h. Shift to serverless inference (e.g., Bedrock / RunPod / Lambda GPU) or schedule auto-stop outside business hours.
- **i-0a1b2c3d4e5f60013** (`g5.4xlarge`, $1,184.87/mo at stake): Running 24h/day but sits idle ~19h. Shift to serverless inference (e.g., Bedrock / RunPod / Lambda GPU) or schedule auto-stop outside business hours.
- **i-0a1b2c3d4e5f60016** (`m5.2xlarge`, $159.93/mo at stake): Running 20h/day but sits idle ~14h. Schedule auto-stop nights/weekends or move the workload to a smaller instance.

_Healthy / right-sized (no action): `i-0a1b2c3d4e5f60014`, `i-0a1b2c3d4e5f60015`_

---
_Assumptions: published on-demand list pricing; utilization averages from the supplied export; smart-routing assumes budget open-weight models meet quality bars for low-complexity tasks. Validate with an A/B eval before cutover._