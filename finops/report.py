"""Report rendering: executive Markdown and machine-readable JSON.

Rendering is pure formatting over :class:`AuditBundle`; no analysis happens
here. Every dollar figure in the Markdown carries its confidence label.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from finops import __version__
from finops.actions import Action
from finops.infra_audit import InfraAuditResult
from finops.ingest.tabular import IngestReport
from finops.llm_audit import LLMAuditResult
from finops.margin import MarginResult
from finops.models import Confidence
from finops.policy import Policy
from finops.pricing import PriceBook

__all__ = ["AuditBundle", "render_markdown", "render_json", "usd"]

LEVER_LABEL = {"model_refresh": "Model refresh", "routing": "Task routing",
               "caching": "Prompt caching", "batch": "Batch API"}
CONF_BADGE = {"verified": "✅ verified", "measured": "📏 measured", "estimated": "🔶 estimated"}


@dataclass
class AuditBundle:
    pricebook: PriceBook
    policy: Policy
    llm: Optional[LLMAuditResult] = None
    infra: Optional[InfraAuditResult] = None
    margins: Optional[MarginResult] = None
    actions: List[Action] = field(default_factory=list)
    ingest: List[IngestReport] = field(default_factory=list)
    task_inference: Dict[str, int] = field(default_factory=dict)
    proof_path: Optional[str] = None
    proof_counts: Dict[str, int] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc)
                              .strftime("%Y-%m-%d %H:%M UTC"))

    @property
    def monthly_spend(self) -> float:
        return ((self.llm.monthly_current if self.llm else 0.0)
                + (self.infra.monthly_compute_spend if self.infra else 0.0))

    @property
    def monthly_savings(self) -> float:
        return ((self.llm.monthly_savings if self.llm else 0.0)
                + (self.infra.monthly_savings if self.infra else 0.0))

    def savings_by_confidence(self) -> Dict[str, float]:
        out = {c.value: 0.0 for c in Confidence}
        if self.llm:
            for k, v in self.llm.savings_by_confidence.items():
                out[k] += v
        if self.infra:
            for k, v in self.infra.savings_by_confidence().items():
                out[k] += v
        return out

    def warnings(self) -> List[str]:
        out: List[str] = []
        if self.pricebook.is_stale():
            out.append(f"Price book is {self.pricebook.age_days()} days old (as of "
                       f"{self.pricebook.as_of}). Refresh prices before sharing these numbers.")
        if self.llm:
            out.extend(self.llm.warnings)
        if self.infra:
            out.extend(self.infra.warnings)
        return out


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def usd(amount: float) -> str:
    if abs(amount) >= 100:
        return f"${amount:,.0f}"
    return f"${amount:,.2f}"


def _short_path(path: str) -> str:
    import os
    try:
        rel = os.path.relpath(path)
    except ValueError:
        return path
    return rel if not rel.startswith("..") else path


def _pct(part: float, whole: float) -> str:
    return f"{100.0 * part / whole:.0f}%" if whole else "—"


def _table(headers: List[str], rows: List[List[str]], align: Optional[str] = None) -> List[str]:
    align = align or "l" * len(headers)
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(sep) + " |"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out + [""]


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def _summary(b: AuditBundle) -> List[str]:
    out = ["## Executive summary", ""]
    rows = []
    if b.llm:
        rows.append(["LLM API run-rate", usd(b.llm.monthly_current),
                     usd(b.llm.monthly_current * 12)])
    if b.infra:
        rows.append([f"Compute ({b.infra.instances_audited} instances audited)",
                     usd(b.infra.monthly_compute_spend), usd(b.infra.monthly_compute_spend * 12)])
    rows.append(["**Identified savings**", f"**{usd(b.monthly_savings)}**",
                 f"**{usd(b.monthly_savings * 12)}**"])
    out += _table(["", "Monthly", "Annualized"], rows, "lrr")
    conf = b.savings_by_confidence()
    total = sum(conf.values()) or 1.0
    out.append("Savings by confidence: " + " · ".join(
        f"{CONF_BADGE[k]} {usd(v)}/mo ({v / total:.0%})" for k, v in conf.items() if v > 0) + "")
    out.append("")
    out.append("> ✅ **verified**: a quality test on your own traffic passed. "
               "📏 **measured**: computed directly from your data. "
               "🔶 **estimated**: depends on a stated assumption (see Methodology).")
    out.append("")

    if b.actions:
        out += ["### Top actions", ""]
        rows = [[str(i), a.title, usd(a.monthly_savings), CONF_BADGE[a.confidence], a.effort]
                for i, a in enumerate(b.actions[:10], start=1)]
        out += _table(["#", "Action", "Saves / mo", "Confidence", "Effort"], rows, "llrll")
    if b.llm and b.llm.upcoming_price_changes:
        rises = [c for c in b.llm.upcoming_price_changes if float(c["monthly_delta"]) > 1]
        if rises:
            out.append("**Heads-up — announced price increases on models you use:** " + "; ".join(
                f"{c['model']} from {c['effective']} (+{usd(float(c['monthly_delta']))}/mo, "
                f"{c['reason']})" for c in rises) + ".")
            out.append("")
    return out


def _llm_section(b: AuditBundle) -> List[str]:
    r = b.llm
    assert r is not None
    out = ["## 1. LLM API spend", ""]
    window = (f"{r.window_days:.1f}-day window " + ("(inferred from timestamps)"
                                                     if r.window_inferred else "(set by --sample-days)"))
    out.append(f"- **Data:** {r.records_analyzed:,} records ({r.records_priced:,} priced) over a "
               f"{window}" + (f", sampled at {r.sample_rate:.1%} of traffic" if r.sample_rate < 1 else "")
               + "; figures are scaled to a 30.4-day month.")
    out.append(f"- **Run-rate:** {usd(r.monthly_current)}/mo → **{usd(r.monthly_optimized)}/mo** "
               f"after all levers (**−{usd(r.monthly_savings)}/mo, {r.savings_pct:.0f}%**, "
               f"{usd(r.annual_savings)}/yr).")
    out.append("")

    out += ["### Where the money goes", ""]
    rows = []
    for name, agg in sorted(r.by_model.items(), key=lambda kv: -kv[1].current)[:12]:
        price = b.pricebook.resolve(name)
        rows.append([f"`{name}`", price.provider if price else "?",
                     (price.tier + (" (legacy)" if price.status == "legacy" else "")) if price else "?",
                     f"{agg.requests:,.0f}", usd(agg.current), _pct(agg.current, r.monthly_current),
                     usd(agg.savings)])
    out += _table(["Model", "Provider", "Tier", "Requests/mo", "$/mo", "Share", "Savable/mo"],
                  rows, "lllrrrr")

    out += ["### Savings by lever", ""]
    rows = [[LEVER_LABEL[k], usd(v), _pct(v, r.monthly_current)]
            for k, v in r.lever_totals.items() if v > 0.005]
    out += _table(["Lever", "Saves / mo", "% of LLM spend"], rows or [["—", "$0", "—"]], "lrr")

    if r.lever_lines:
        out += ["### Lever detail", ""]
        rows = []
        for l in r.lever_lines[:20]:
            move = f"{l.model} → {l.target}" if l.target else l.model
            rows.append([LEVER_LABEL[l.lever], move, l.task_type or "—", l.feature or "—",
                         usd(l.affected_monthly_cost), usd(l.monthly_savings),
                         CONF_BADGE[l.confidence.value], l.method])
        out += _table(["Lever", "Model", "Task", "Feature", "Affected $/mo", "Saves / mo",
                       "Confidence", "Basis"], rows, "llllrrll")
        if len(r.lever_lines) > 20:
            out.append(f"_{len(r.lever_lines) - 20} smaller items in the JSON output._\n")

    if r.blocked_by_proof:
        out += ["### Routes blocked by the quality proof", ""]
        out += [f"- {k}: would have saved {usd(v)}/mo, but failed the quality threshold; "
                "kept on the current model." for k, v in r.blocked_by_proof.items()]
        out.append("")
    if r.unpriced:
        out += ["### Unpriced models (excluded from all figures)", ""]
        rows = [[f"`{m}`", f"{v['requests']:,.0f}", f"{v['input_tokens'] + v['output_tokens']:,.0f}"]
                for m, v in sorted(r.unpriced.items(), key=lambda kv: -kv[1]["input_tokens"])]
        out += _table(["Model", "Requests/mo", "Tokens/mo"], rows, "lrr")
    if r.reconciliation:
        rc = r.reconciliation
        out.append(f"**Reconciliation:** on {rc['records']:,.0f} records with billed spend, list-price "
                   f"math gives {usd(rc['computed_list_price'])} vs {usd(rc['actual'])} billed "
                   f"({rc['variance_pct']:+.1f}%).")
        out.append("")
    return out


def _attribution_section(b: AuditBundle) -> List[str]:
    r = b.llm
    assert r is not None
    out = ["## 2. Unit economics", ""]
    cov = r.coverage
    out.append(f"Attribution coverage (share of spend tagged): team {cov.get('team', 0):.0f}% · "
               f"feature {cov.get('feature', 0):.0f}% · customer {cov.get('customer', 0):.0f}%. "
               "Untagged spend is invisible in the tables below.")
    src = r.spend_by_task_source
    if src:
        out.append("Task type known from: " + ", ".join(
            f"{k} {_pct(v, r.monthly_current)}" for k, v in sorted(src.items(), key=lambda kv: -kv[1])) + ".")
    out.append("")
    for title, data in (("By team", r.by_team), ("By feature", r.by_feature),
                        ("By customer (top 10)", r.by_customer)):
        if not data:
            continue
        rows = [[k, f"{v.requests:,.0f}", usd(v.current), usd(v.optimized), usd(v.savings)]
                for k, v in sorted(data.items(), key=lambda kv: -kv[1].current)[:10]]
        out += [f"### {title}", ""]
        out += _table(["Name", "Requests/mo", "Current $/mo", "Optimized $/mo", "Savings"],
                      rows, "lrrrr")
    m = b.margins
    if m:
        out += ["### AI cost vs revenue", ""]
        out.append(f"{len(m.customers)} customers matched to revenue: AI cost is "
                   f"{_pct(m.total_ai_cost, m.total_revenue)} of their revenue today, "
                   f"{_pct(m.total_ai_cost_optimized, m.total_revenue)} after optimization. "
                   f"**{len(m.flagged)} customer(s) above {m.alert_pct:.0f}%.**")
        out.append("")
        def share(pct: float) -> str:
            return f"{pct:.0f}%" if pct != float("inf") else "— (no revenue)"
        rows = [[c.customer_id, usd(c.monthly_revenue), usd(c.ai_cost),
                 share(c.ai_cost_pct), share(c.ai_cost_pct_optimized),
                 "⚠️" if c.ai_cost_pct > m.alert_pct else ""] for c in m.customers[:15]]
        out += _table(["Customer", "Revenue/mo", "AI cost/mo", "AI % of rev", "After optimization", ""],
                      rows, "lrrrrl")
        if m.cost_without_revenue:
            out.append(f"_{len(m.cost_without_revenue)} customer id(s) with AI cost but no revenue row "
                       f"({usd(sum(m.cost_without_revenue.values()))}/mo): free tier, trials, or an id mismatch._")
            out.append("")
    return out


def _infra_section(b: AuditBundle) -> List[str]:
    r = b.infra
    assert r is not None
    out = ["## 3. Infrastructure (EC2 / GPU)", ""]
    out.append(f"- **Audited:** {r.instances_audited} instances, {usd(r.monthly_compute_spend)}/mo "
               "(list or inventory rates).")
    out.append(f"- **Recoverable:** **{usd(r.monthly_savings)}/mo** ({usd(r.annual_savings)}/yr) "
               f"across {len(r.flagged)} instance(s); {len(r.healthy)} look healthy.")
    out.append("")
    if r.flagged:
        rows = []
        for f in r.flagged:
            rows.append([f"`{f.instance_id}`" + (f" {f.name}" if f.name else ""), f.instance_type,
                         f.method + (f" ({f.days_observed:.0f}d)" if f.method == "timeseries" else ""),
                         f"{f.idle_pct:.0f}%", usd(f.monthly_cost),
                         ("stop (idle)" if f.zombie else (f.schedule.describe() if f.schedule
                                                          else ("auto-stop" if f.schedule_savings else "—"))),
                         f.rightsize_to or "—", f"**{usd(f.monthly_savings)}**",
                         CONF_BADGE[f.overall_confidence.value], f.severity.upper()])
        out += _table(["Instance", "Type", "Data", "Idle", "Cost/mo", "Schedule", "Right-size",
                       "Saves/mo", "Confidence", "Severity"], rows, "lllrrllrll")
        out += ["**Details**", ""]
        for f in r.flagged:
            out.append(f"- `{f.instance_id}`: {f.recommendation}")
            for c in f.caveats:
                out.append(f"  - _caveat: {c}_")
        out.append("")
    return out


def _methodology(b: AuditBundle) -> List[str]:
    pb, pol = b.pricebook, b.policy
    out = ["## Methodology & assumptions", ""]
    out.append(f"- **Prices:** list prices as of {pb.as_of or 'unknown'}"
               + (f" ({pb.age_days()} days old)" if pb.age_days() is not None else "")
               + f"; files: {', '.join(p.split('/')[-1] for p in pb.loaded_from)}. Negotiated "
               "rates belong in a `--pricing-config` override.")
    out.append("- **Routing** moves only task types the policy marks routable, only to a model that "
               "is cheaper for that traffic, and assumes similar token counts on the new model. "
               + ("Cross-provider moves are allowed." if pol.cross_provider else
                  "Targets stay with the current provider (no vendor migration needed)."))
    c = pol.caching
    out.append(f"- **Caching** is *measured* when prompts are logged (shared prefix ≥ "
               f"{c.get('min_shared_prefix_tokens', 1024):.0f} tokens); otherwise *estimated* by assuming "
               f"{float(c.get('assumed_cacheable_fraction', 0.3)):.0%} of long prompts is a stable prefix "
               f"with a {float(c.get('assumed_hit_rate', 0.8)):.0%} hit rate, only for groups above "
               f"{c.get('min_requests_for_assumption', 1000):.0f} requests/mo. OpenAI traffic without "
               "cache data is skipped (OpenAI caches automatically).")
    out.append("- **Batch** applies to tasks/features the policy marks non-interactive: "
               + ", ".join(pol.batch_tasks + pol.batch_feature_patterns) + ".")
    out.append(f"- **Quality proof:** a route is verified when the Wilson 95% lower bound of its pass "
               f"rate is ≥ {pol.proof_pass_threshold:.0%} over ≥ {pol.proof_min_samples} samples."
               + (f" Results loaded from `{_short_path(b.proof_path)}` "
                  f"({', '.join(f'{k}: {v}' for k, v in b.proof_counts.items())})." if b.proof_path
                  else " No proof results loaded: every routing saving is *estimated*."))
    out.append("- **Infrastructure:** time-series instances are judged hour by hour (idle = GPU < 5% and "
               "CPU < 10% on GPU hosts, CPU < 5% otherwise); a weekly hour is stoppable when idle in "
               "≥ 90% of observed weeks. Average-only instances are estimates. Right-sizing targets "
               "p95 active load at 70% and is always *estimated* (GPU memory isn't visible).")
    if b.task_inference:
        ti = b.task_inference
        out.append(f"- **Task inference:** {ti.get('prompt', 0):,} records labelled from prompt text, "
                   f"{ti.get('shape', 0):,} from token shape, {ti.get('unknown', 0):,} left unknown.")
    out.append("")
    if b.ingest:
        out += ["**Inputs**", ""]
        out += [f"- {rep.summary()}" for rep in b.ingest]
        out.append("")
    return out


def render_markdown(b: AuditBundle) -> str:
    out = ["# AI & Cloud Cost Audit", "",
           f"_Generated {b.generated_at} · finops-arbitrage-engine v{__version__} · "
           f"prices as of {b.pricebook.as_of or 'unknown'}_", ""]
    warnings = b.warnings()
    if warnings:
        out += ["> **Data notes**"]
        out += [f"> - {w}" for w in warnings]
        out.append("")
    out += _summary(b)
    if b.llm:
        out += _llm_section(b)
        out += _attribution_section(b)
    if b.infra:
        out += _infra_section(b)
    if b.artifacts:
        out += ["## Ready-to-apply changes", ""]
        out += [f"- `{p}`" for p in b.artifacts]
        out.append("")
    out += _methodology(b)
    return "\n".join(out).rstrip() + "\n"


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def _jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        data = {f.name: _jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)
                if not f.name.startswith("_")}
        for prop in ("monthly_savings", "annual_savings", "savings", "savings_pct",
                     "overall_confidence", "ai_cost_pct", "ai_cost_pct_optimized"):
            if hasattr(type(obj), prop) and isinstance(getattr(type(obj), prop), property):
                data[prop] = _jsonable(getattr(obj, prop))
        return data
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        items = [_jsonable(v) for v in obj]
        return sorted(items) if isinstance(obj, set) else items
    if isinstance(obj, float):
        return round(obj, 4) if obj == obj and abs(obj) != float("inf") else None
    return obj


def render_json(b: AuditBundle) -> str:
    payload = {
        "schema": "finops-audit/2",
        "engine_version": __version__,
        "generated_at": b.generated_at,
        "pricing": {"as_of": b.pricebook.as_of, "files": b.pricebook.loaded_from,
                    "stale": b.pricebook.is_stale()},
        "summary": {"monthly_spend": b.monthly_spend, "monthly_savings": b.monthly_savings,
                    "annual_savings": b.monthly_savings * 12,
                    "savings_by_confidence": b.savings_by_confidence()},
        "warnings": b.warnings(),
        "actions": _jsonable(b.actions),
        "llm": _jsonable(b.llm),
        "infra": _jsonable(b.infra),
        "margins": _jsonable(b.margins),
        "inputs": [{"path": r.path, "format": r.fmt, "rows": r.rows_read, "records": r.records,
                    "skipped": r.skipped, "skip_reasons": dict(r.skip_reasons)} for r in b.ingest],
        "artifacts": b.artifacts,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)
