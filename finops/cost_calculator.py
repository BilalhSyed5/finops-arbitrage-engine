"""Module A: Model ROI & Arbitrage Calculator.

Ingests raw API usage logs (timestamp, model_used, input_tokens,
output_tokens, task_type) and quantifies:

* Current monthly run-rate spend per model and per task type.
* Optimized run-rate under a "smart routing" policy that sends
  low-complexity tasks to budget open-weight models.
* Net annualized savings delta.

Pricing is expressed in USD per 1M tokens and loaded from config-driven
tables (see finops/pricing_config.py). Defaults are reasonable published
list prices; refresh via config/model_pricing.json before relying on the
dollar figures for board-level decisions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from finops.pricing_config import Pricing, load_pricing

__all__ = [
    "PRICING",
    "MODEL_PRICING",
    "BUDGET_MODEL_PRICING",
    "use_pricing",
    "HOURS_PER_MONTH",
    "CostRecord",
    "ModelLineItem",
    "ArbitrageResult",
    "analyze_api_costs",
]

# ---------------------------------------------------------------------------
# Pricing (USD per 1M tokens: (input, output))
#
# Tables are config-driven (finops/pricing_config.py, overridable via
# config/model_pricing.json). PRICING is loaded once at import; call
# use_pricing() to swap in a different snapshot (e.g. a committed-rate
# card passed via --pricing-config).
# ---------------------------------------------------------------------------

PRICING: Pricing = load_pricing()

# Compatibility aliases (kept for external callers; resolve_pricing and the
# routing policy read PRICING directly so use_pricing() stays effective).
MODEL_PRICING: Dict[str, Tuple[float, float]] = PRICING.frontier
BUDGET_MODEL_PRICING: Dict[str, Tuple[float, float]] = PRICING.budget


def use_pricing(pricing: Pricing) -> None:
    """Swap the active pricing snapshot (and refresh the legacy aliases)."""
    global PRICING, MODEL_PRICING, BUDGET_MODEL_PRICING
    PRICING = pricing
    MODEL_PRICING = pricing.frontier
    BUDGET_MODEL_PRICING = pricing.budget

# ---------------------------------------------------------------------------
# Task-complexity routing policy
# ---------------------------------------------------------------------------

# Tasks that a budget open-weight model can typically handle with negligible
# quality loss. Everything else (complex reasoning, agentic planning, code
# generation at the frontier) stays on the frontier model.
BUDGET_ELIGIBLE_TASKS = {
    "classification",
    "extraction",
    "summarization",
    "simple_qa",
    "tagging",
    "routing",
    "deduplication",
    "formatting",
}

# Default budget model per task class. Classification/extraction are short and
# latency-sensitive -> small model; summarization needs more context handling
# -> 70B-class model.
DEFAULT_BUDGET_MODEL_BY_TASK = {
    "classification": "llama-3.1-8b-instruct",
    "extraction": "llama-3.1-8b-instruct",
    "tagging": "llama-3.1-8b-instruct",
    "routing": "llama-3.1-8b-instruct",
    "deduplication": "llama-3.1-8b-instruct",
    "formatting": "llama-3.1-8b-instruct",
    "simple_qa": "llama-3.1-8b-instruct",
    "summarization": "llama-3.1-70b-instruct",
}

HOURS_PER_MONTH = 730.0
DAYS_PER_MONTH = 30.4


def resolve_pricing(model_used: str) -> Tuple[Tuple[float, float], str]:
    """Return ((input, output) price per 1M tokens, pricing tier) for a model.

    Matches case-insensitively and tolerates provider prefixes such as
    ``openai/gpt-4o`` or version suffixes such as ``gpt-4o-2024-08-06``.
    Falls back to the most expensive frontier model so unknown models are
    treated conservatively rather than silently priced at zero.
    """
    key = model_used.strip().lower()
    # Strip provider prefix ("anthropic/claude-3-5-sonnet").
    if "/" in key:
        key = key.split("/")[-1]
    # Strip dated snapshot suffixes ("gpt-4o-2024-08-06", "...-preview-0827").
    key = re.sub(r"-(?:\d{4}-\d{2}-\d{2}|\d{6,}|latest|preview)$", "", key)

    for table, tier in ((PRICING.frontier, "frontier"),
                        (PRICING.budget, "budget")):
        for name, price in table.items():
            if key == name or key.startswith(name + "-") or name.startswith(key):
                return price, tier

    # Unknown model: assume worst case so waste is not under-reported.
    return PRICING.frontier["gpt-4o"], "unknown(frontier-assumed)"


@dataclass(frozen=True)
class CostRecord:
    """One normalized API usage record.

    The attribution fields (team / feature / customer_id) are optional
    metadata used for unit-economics rollups; records without them are
    simply excluded from those views.
    """

    timestamp: str
    model_used: str
    input_tokens: int
    output_tokens: int
    task_type: str
    team: str = ""
    feature: str = ""
    customer_id: str = ""


@dataclass
class ModelLineItem:
    """Spend attribution for one (model, task_type) pair."""

    model_used: str
    task_type: str
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    current_cost: float = 0.0
    optimized_cost: float = 0.0
    rerouted_to: str | None = None  # budget model if this bucket gets routed

    @property
    def savings(self) -> float:
        return self.current_cost - self.optimized_cost


@dataclass
class ArbitrageResult:
    """Aggregated output of the arbitrage analysis."""

    current_monthly_spend: float = 0.0
    optimized_monthly_spend: float = 0.0
    records_analyzed: int = 0
    records_rerouted: int = 0
    line_items: List[ModelLineItem] = field(default_factory=list)
    by_model: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_task: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_team: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_feature: Dict[str, Dict[str, float]] = field(default_factory=dict)
    tagged_records: int = 0
    sample_window_days: float = 0.0

    @property
    def monthly_savings(self) -> float:
        return self.current_monthly_spend - self.optimized_monthly_spend

    @property
    def annual_savings(self) -> float:
        return self.monthly_savings * 12.0

    @property
    def savings_pct(self) -> float:
        if self.current_monthly_spend <= 0:
            return 0.0
        return 100.0 * self.monthly_savings / self.current_monthly_spend


def parse_cost_records(rows: Sequence[Dict]) -> List[CostRecord]:
    """Convert raw loader rows into typed :class:`CostRecord` objects."""
    records: List[CostRecord] = []
    for row in rows:
        records.append(CostRecord(
            timestamp=str(row.get("timestamp", "")),
            model_used=str(row["model_used"]),
            input_tokens=int(float(row["input_tokens"])),
            output_tokens=int(float(row["output_tokens"])),
            task_type=str(row["task_type"]).strip().lower(),
            team=str(row.get("team") or "").strip(),
            feature=str(row.get("feature") or "").strip(),
            customer_id=str(row.get("customer_id") or "").strip(),
        ))
    return records


def _cost_usd(price: Tuple[float, float], input_tokens: int, output_tokens: int) -> float:
    """Cost of one request in USD given per-1M-token pricing."""
    return (input_tokens * price[0] + output_tokens * price[1]) / 1_000_000.0


def analyze_api_costs(rows: Sequence[Dict], sample_days: float | None = None) -> ArbitrageResult:
    """Run the Module A arbitrage analysis over raw log rows.

    Args:
        rows: Records from :mod:`finops.data_loader` with fields
            ``timestamp, model_used, input_tokens, output_tokens, task_type``.
        sample_days: Optional explicit window length in days. When omitted it
            is inferred from the min/max ``timestamp`` values (ISO dates such
            as ``2025-01-15`` or full ISO datetimes). Falls back to 30 days
            when timestamps are unusable.

    Returns:
        An :class:`ArbitrageResult` with current vs. optimized run-rates and
        per-model / per-task attribution.
    """
    records = parse_cost_records(rows)
    result = ArbitrageResult(records_analyzed=len(records))

    if not records:
        return result

    # ---- Infer the sampling window ----------------------------------------
    if sample_days and sample_days > 0:
        result.sample_window_days = sample_days
    else:
        stamps = sorted(r.timestamp for r in records if r.timestamp)
        result.sample_window_days = 30.0
        if len(stamps) >= 2:
            try:
                from datetime import datetime
                fmt_candidates = ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d")
                start = end = None
                for fmt in fmt_candidates:
                    try:
                        start = datetime.fromisoformat(stamps[0].replace("Z", "+00:00")) \
                            if "T" in stamps[0] or " " in stamps[0] else datetime.fromisoformat(stamps[0])
                        end = datetime.fromisoformat(stamps[-1].replace("Z", "+00:00")) \
                            if "T" in stamps[-1] or " " in stamps[-1] else datetime.fromisoformat(stamps[-1])
                        break
                    except ValueError:
                        continue
                if start and end and end > start:
                    result.sample_window_days = max(1.0, (end - start).total_seconds() / 86400.0)
            except Exception:  # noqa: BLE001 - window inference is best-effort
                pass

    scale = DAYS_PER_MONTH / result.sample_window_days  # sample -> monthly factor

    # ---- Bucket by (model, task) + attribution rollups ---------------------
    buckets: Dict[Tuple[str, str], ModelLineItem] = {}
    tag_dims = (result.by_team, result.by_feature)
    for rec in records:
        price, _tier = resolve_pricing(rec.model_used)
        rec_current = _cost_usd(price, rec.input_tokens, rec.output_tokens)

        # ---- Smart-routing policy ------------------------------------------
        budget_model: str | None = None
        if rec.task_type in BUDGET_ELIGIBLE_TASKS:
            candidate = DEFAULT_BUDGET_MODEL_BY_TASK.get(
                rec.task_type, "llama-3.1-70b-instruct")
            budget_price = PRICING.budget.get(candidate)
            if budget_price is not None:
                # Budget models are priced identically in/out for simplicity.
                rec_optimized = _cost_usd(
                    budget_price, rec.input_tokens, rec.output_tokens)
                budget_model = candidate
            else:
                # Budget model not in the pricing snapshot: keep frontier
                # pricing so we never claim savings we can't price.
                rec_optimized = rec_current
        else:
            rec_optimized = rec_current

        key = (rec.model_used, rec.task_type)
        item = buckets.setdefault(key, ModelLineItem(
            model_used=rec.model_used, task_type=rec.task_type))
        item.calls += 1
        item.input_tokens += rec.input_tokens
        item.output_tokens += rec.output_tokens
        item.current_cost += rec_current
        item.optimized_cost += rec_optimized
        if budget_model:
            item.rerouted_to = budget_model
            result.records_rerouted += 1

        # ---- Attribution rollups (unscaled; scaled in the aggregate pass) --
        if rec.team or rec.feature:
            result.tagged_records += 1
        for dim, label in ((tag_dims[0], rec.team),
                           (tag_dims[1], rec.feature),
                           (tag_dims[1], rec.customer_id and f"customer:{rec.customer_id}")):
            if label:
                agg = dim.setdefault(
                    label, {"current": 0.0, "optimized": 0.0, "calls": 0})
                agg["current"] += rec_current
                agg["optimized"] += rec_optimized
                agg["calls"] += 1

    # ---- Aggregate ----------------------------------------------------------
    result.line_items = sorted(buckets.values(),
                               key=lambda i: i.current_cost, reverse=True)
    for item in result.line_items:
        item.current_cost *= scale
        item.optimized_cost *= scale
        result.current_monthly_spend += item.current_cost
        result.optimized_monthly_spend += item.optimized_cost

        for dim, label in ((result.by_model, item.model_used),
                           (result.by_task, item.task_type)):
            agg = dim.setdefault(label, {
                "current": 0.0, "optimized": 0.0, "calls": 0})
            agg["current"] += item.current_cost
            agg["optimized"] += item.optimized_cost
            agg["calls"] += item.calls

    # Scale the attribution rollups by the same sample -> monthly factor.
    for dim in (result.by_team, result.by_feature):
        for agg in dim.values():
            agg["current"] *= scale
            agg["optimized"] *= scale

    return result