"""AI gross margin per customer.

Joins per-customer LLM cost (from the audit) with a revenue file
(``customer_id, monthly_revenue``) to answer the question boards actually
ask: *which customers are we losing money on because of AI costs, and what
does the margin look like after optimization?*
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from finops.ingest.tabular import DataLoadError, IngestReport, read_rows
from finops.llm_audit import LLMAuditResult

__all__ = ["CustomerMargin", "MarginResult", "load_revenue", "compute_margins"]

DEFAULT_ALERT_PCT = 30.0   # AI cost above this share of revenue is flagged


@dataclass
class CustomerMargin:
    customer_id: str
    monthly_revenue: float
    ai_cost: float
    ai_cost_optimized: float

    @property
    def ai_cost_pct(self) -> float:
        return 100.0 * self.ai_cost / self.monthly_revenue if self.monthly_revenue else float("inf")

    @property
    def ai_cost_pct_optimized(self) -> float:
        return (100.0 * self.ai_cost_optimized / self.monthly_revenue
                if self.monthly_revenue else float("inf"))

    @property
    def margin_after_ai(self) -> float:
        return self.monthly_revenue - self.ai_cost


@dataclass
class MarginResult:
    customers: List[CustomerMargin] = field(default_factory=list)
    alert_pct: float = DEFAULT_ALERT_PCT
    cost_without_revenue: Dict[str, float] = field(default_factory=dict)
    revenue_without_usage: int = 0

    @property
    def flagged(self) -> List[CustomerMargin]:
        return [c for c in self.customers if c.ai_cost_pct > self.alert_pct]

    @property
    def total_revenue(self) -> float:
        return sum(c.monthly_revenue for c in self.customers)

    @property
    def total_ai_cost(self) -> float:
        return sum(c.ai_cost for c in self.customers)

    @property
    def total_ai_cost_optimized(self) -> float:
        return sum(c.ai_cost_optimized for c in self.customers)


def load_revenue(path: str) -> Tuple[Dict[str, float], IngestReport]:
    report = IngestReport(path=path, fmt="revenue")
    out: Dict[str, float] = {}
    for raw in read_rows(path):
        report.rows_read += 1
        if not isinstance(raw, dict):
            report.skip("not an object")
            continue
        row = {str(k).strip().lower(): v for k, v in raw.items()}
        cid = str(row.get("customer_id") or row.get("customer") or "").strip()
        value = row.get("monthly_revenue", row.get("mrr", row.get("revenue")))
        try:
            amount = float(str(value).replace(",", "").replace("$", "").strip())
        except (TypeError, ValueError):
            report.skip("missing or non-numeric monthly_revenue")
            continue
        if not cid:
            report.skip("missing customer_id")
            continue
        out[cid] = out.get(cid, 0.0) + amount
        report.records += 1
    if not out:
        raise DataLoadError(f"No usable rows in revenue file '{path}' "
                            "(need customer_id, monthly_revenue).")
    return out, report


def compute_margins(llm: LLMAuditResult, revenue: Dict[str, float],
                    alert_pct: float = DEFAULT_ALERT_PCT) -> MarginResult:
    result = MarginResult(alert_pct=alert_pct)
    for cid, agg in llm.by_customer.items():
        if cid in revenue:
            result.customers.append(CustomerMargin(cid, revenue[cid], agg.current, agg.optimized))
        else:
            result.cost_without_revenue[cid] = agg.current
    result.revenue_without_usage = sum(1 for cid in revenue if cid not in llm.by_customer)
    result.customers.sort(key=lambda c: c.ai_cost_pct, reverse=True)
    return result
