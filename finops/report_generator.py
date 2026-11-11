"""Executive-ready Markdown report generation for both audit modules.

Everything here is pure formatting: it takes :class:`ArbitrageResult` and/or
:class:`InfraAuditResult` objects and renders them as a Markdown document
suitable for pasting into Slack, Notion, or a terminal.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from finops.cost_calculator import ArbitrageResult
from finops.infrastructure_audit import InfraAuditResult

__all__ = ["generate_report", "format_usd"]


def format_usd(amount: float) -> str:
    """Format a dollar amount for an executive summary ($1,234.56)."""
    return f"${amount:,.2f}"


def format_pct(pct: float) -> str:
    """Format a percentage (34.2%)."""
    return f"{pct:.1f}%"


def _arbitrage_section(result: ArbitrageResult) -> list[str]:
    lines: list[str] = []
    lines.append("## 1. Model ROI & Arbitrage (Module A)\n")
    lines.append(
        f"- **Records analyzed:** {result.records_analyzed:,} over a "
        f"{result.sample_window_days:.1f}-day window "
        f"(extrapolated to a 30.4-day month)"
    )
    lines.append(f"- **Current monthly run-rate:** "
                 f"**{format_usd(result.current_monthly_spend)}**")
    lines.append(f"- **Optimized run-rate (smart routing):** "
                 f"**{format_usd(result.optimized_monthly_spend)}**")
    lines.append(f"- **Net monthly savings:** **{format_usd(result.monthly_savings)}** "
                 f"({format_pct(result.savings_pct)} reduction)")
    lines.append(f"- **Net annual savings delta:** "
                 f"**{format_usd(result.annual_savings)}**\n")

    if not result.line_items:
        return lines

    lines.append("### Spend attribution (top cost centers)\n")
    lines.append("| Model | Task | Calls/mo | Current $/mo | Optimized $/mo | Savings $/mo | Routed To |")
    lines.append("|---|---|---:|---:|---:|---:|---|")
    for item in result.line_items[:15]:  # cap the table for readability
        calls_mo = item.calls * (30.4 / result.sample_window_days)
        routed = item.rerouted_to or "— (frontier retained)"
        lines.append(
            f"| `{item.model_used}` | {item.task_type} | {calls_mo:,.0f} | "
            f"{format_usd(item.current_cost)} | {format_usd(item.optimized_cost)} | "
            f"**{format_usd(item.savings)}** | {routed} |"
        )
    lines.append("")

    lines.append("### By task type\n")
    lines.append("| Task | Current $/mo | Optimized $/mo | Savings |")
    lines.append("|---|---:|---:|---:|")
    for task, agg in sorted(result.by_task.items(),
                            key=lambda kv: kv[1]["current"], reverse=True):
        saved = agg["current"] - agg["optimized"]
        lines.append(f"| {task} | {format_usd(agg['current'])} | "
                     f"{format_usd(agg['optimized'])} | **{format_usd(saved)}** |")
    lines.append("")

    lines.append("> **Action:** Route `classification`, `extraction`, and "
                 "`summarization` traffic to budget open-weight models "
                 "(Llama 3.x via Groq/Together). Run an A/B quality eval on "
                 "the top rerouted task before full cutover.\n")

    lines.extend(_attribution_section(result))
    return lines


def _attribution_section(result: ArbitrageResult) -> list[str]:
    """Unit-economics rollups by team / feature / customer (when tagged)."""
    lines: list[str] = []
    if not result.tagged_records:
        return lines

    coverage = 100.0 * result.tagged_records / max(1, result.records_analyzed)
    lines.append(f"### Unit economics (attribution coverage: "
                 f"{format_pct(coverage)} of records)\n")

    for label, dim in (("Team", result.by_team),
                       ("Feature / customer", result.by_feature)):
        if not dim:
            continue
        lines.append(f"**{label}**\n")
        lines.append("| Key | Calls/mo | Current $/mo | Optimized $/mo | Savings |")
        lines.append("|---|---:|---:|---:|---:|")
        for key, agg in sorted(dim.items(),
                               key=lambda kv: kv[1]["current"], reverse=True):
            calls_mo = agg["calls"] * (30.4 / result.sample_window_days)
            saved = agg["current"] - agg["optimized"]
            lines.append(f"| {key} | {calls_mo:,.0f} | "
                         f"{format_usd(agg['current'])} | "
                         f"{format_usd(agg['optimized'])} | "
                         f"**{format_usd(saved)}** |")
        lines.append("")

    lines.append("> **Action:** Join these cost lines against revenue for the "
                 "matching feature/customer keys to get true AI gross margin. "
                 "Untagged traffic is invisible to unit economics — instrument "
                 "it via LiteLLM metadata tags.\n")
    return lines


def _infra_section(result: InfraAuditResult) -> list[str]:
    lines: list[str] = []
    lines.append("## 2. Infrastructure & GPU Right-Sizing Audit (Module B)\n")
    lines.append(f"- **Instances audited:** {result.instances_audited}")
    lines.append(f"- **Total monthly compute spend:** "
                 f"**{format_usd(result.total_monthly_compute_spend)}**")
    lines.append(f"- **Flagged dead capital:** "
                 f"**{format_usd(result.total_monthly_waste)}/month** "
                 f"(**{format_usd(result.annual_waste)}/year**)\n")

    if not result.findings:
        lines.append("No idle or severely under-utilized instances detected. ✅\n")
        return lines

    lines.append("### Flagged instances (sorted by monthly waste)\n")
    lines.append("| Instance | Type | GPU% | CPU% | Hrs/Day | Idle Hrs/Day | Cost $/mo | Waste $/mo | Severity |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|")
    for f in result.findings:
        lines.append(
            f"| `{f.instance_id}` | `{f.instance_type}` | "
            f"{f.avg_gpu_utilization_pct:.0f}% | {f.avg_cpu_utilization_pct:.0f}% | "
            f"{f.running_hours_per_day:.0f} | {f.idle_hours_per_day:.0f} | "
            f"{format_usd(f.monthly_cost)} | **{format_usd(f.monthly_waste)}** | "
            f"{f.severity.upper()} |"
        )
    lines.append("")

    lines.append("### Recommended actions\n")
    for f in result.findings:
        lines.append(
            f"- **{f.instance_id}** (`{f.instance_type}`, "
            f"{format_usd(f.monthly_waste)}/mo at stake): {f.recommendation}"
        )
    lines.append("")

    if result.healthy_instances:
        lines.append(f"_Healthy / right-sized (no action): "
                     f"{', '.join('`' + i + '`' for i in result.healthy_instances)}_\n")
    return lines


def generate_report(arbitrage: Optional[ArbitrageResult] = None,
                    infra: Optional[InfraAuditResult] = None,
                    source_files: Optional[dict] = None) -> str:
    """Render the full executive Markdown report.

    Args:
        arbitrage: Optional Module A result.
        infra: Optional Module B result.
        source_files: Optional mapping of input label -> file path, echoed in
            the report header for provenance.

    Returns:
        The complete Markdown report as a string.
    """
    total_annual = (arbitrage.annual_savings if arbitrage else 0.0) + \
                   (infra.annual_waste if infra else 0.0)

    lines: list[str] = []
    lines.append("# 💰 FinOps Arbitrage Audit — P&L Savings Report\n")
    lines.append(f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} · "
                 "AI & Cloud FinOps Arbitrage Engine v1.0_\n")
    if source_files:
        for label, path in source_files.items():
            lines.append(f"- Source ({label}): `{path}`")
        lines.append("")

    lines.append("## Executive Summary\n")
    if arbitrage:
        lines.append(f"- **API burn (current):** {format_usd(arbitrage.current_monthly_spend)}/mo "
                     f"→ **{format_usd(arbitrage.monthly_savings)}/mo** recoverable via model routing")
    if infra:
        lines.append(f"- **Cloud waste (flagged):** {format_usd(infra.total_monthly_waste)}/mo "
                     f"across {len(infra.findings)} of {infra.instances_audited} instances")
    lines.append(f"- **Total annualized savings opportunity: "
                 f"{format_usd(total_annual)}**\n")

    if arbitrage:
        lines.extend(_arbitrage_section(arbitrage))
    if infra:
        lines.extend(_infra_section(infra))

    lines.append("---")
    lines.append("_Assumptions: published on-demand list pricing; utilization "
                 "averages from the supplied export; smart-routing assumes "
                 "budget open-weight models meet quality bars for "
                 "low-complexity tasks. Validate with an A/B eval before "
                 "cutover._")
    return "\n".join(lines)