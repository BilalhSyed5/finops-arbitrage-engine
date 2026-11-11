"""Module B: Infrastructure & GPU Right-Sizing Audit.

Ingests a cloud resource inventory / utilization export (instance_id,
instance_type, avg_cpu_utilization_pct, avg_gpu_utilization_pct,
running_hours_per_day) and flags dead capital:

* 24/7 instances that sit idle for most of the day.
* Severely under-utilized GPU instances that should be right-sized,
  scheduled, or moved to serverless inference.
* Dollarized monthly savings for each recommendation.

Hourly rates are config-driven (finops/pricing_config.py, overridable via
config/ec2_pricing.json) with built-in us-east-1 list prices as defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from finops.pricing_config import Pricing, load_pricing

__all__ = [
    "PRICING",
    "EC2_ON_DEMAND_HOURLY_USD",
    "use_pricing",
    "InstanceFinding",
    "InfraAuditResult",
    "analyze_infrastructure",
]

# ---------------------------------------------------------------------------
# On-demand hourly pricing (us-east-1 list prices, USD/hr).
# Config-driven: see finops/pricing_config.py; override via
# config/ec2_pricing.json. Refresh from
# https://aws.amazon.com/ec2/pricing/on-demand/ as needed.
# ---------------------------------------------------------------------------

PRICING: Pricing = load_pricing()

# Compatibility alias (kept for external callers; analyze_infrastructure
# reads PRICING directly so use_pricing() stays effective).
EC2_ON_DEMAND_HOURLY_USD: Dict[str, float] = PRICING.ec2_hourly


def use_pricing(pricing: Pricing) -> None:
    """Swap the active pricing snapshot (and refresh the legacy alias)."""
    global PRICING, EC2_ON_DEMAND_HOURLY_USD
    PRICING = pricing
    EC2_ON_DEMAND_HOURLY_USD = pricing.ec2_hourly

HOURS_PER_MONTH = 730.0
DAYS_PER_MONTH = 30.4

# Utilization thresholds (percent) used by the audit heuristics.
IDLE_UTILIZATION_PCT = 15.0        # below this, an instance is "idle"
SEVERE_UNDER_UTILIZATION_PCT = 25.0  # below this, right-sizing is warranted


@dataclass
class InstanceFinding:
    """One flagged instance with a dollarized recommendation."""

    instance_id: str
    instance_type: str
    hourly_rate: float
    avg_cpu_utilization_pct: float
    avg_gpu_utilization_pct: float
    running_hours_per_day: float
    idle_hours_per_day: float          # estimated hours/day at near-zero load
    monthly_cost: float                # what it costs today
    monthly_waste: float               # recoverable spend
    severity: str                      # "critical" | "high" | "medium"
    recommendation: str                # concrete action text

    @property
    def annual_waste(self) -> float:
        return self.monthly_waste * 12.0


@dataclass
class InfraAuditResult:
    """Aggregated output of the infrastructure audit."""

    total_monthly_compute_spend: float = 0.0
    total_monthly_waste: float = 0.0
    instances_audited: int = 0
    findings: List[InstanceFinding] = field(default_factory=list)
    healthy_instances: List[str] = field(default_factory=list)

    @property
    def annual_waste(self) -> float:
        return self.total_monthly_waste * 12.0


def _estimate_idle_hours(avg_cpu: float, avg_gpu: float,
                         running_hours: float) -> float:
    """Estimate idle hours/day from average utilization.

    Average utilization is a time-weighted mean over the running window. We
    invert that mean into an implied "idle fraction": if an instance averages
    ``u``% load while it is doing real work at a typical working load of
    ``WORKING_LOAD``%, the implied idle fraction is ``1 - u / WORKING_LOAD``.
    This is a conservative heuristic — it deliberately under-counts idle time
    rather than over-counting it.
    """
    utilization = max(avg_gpu, avg_cpu)  # GPU workloads can idle the CPU and vice versa
    working_load = 60.0  # assumed % load while genuinely working
    idle_fraction = max(0.0, 1.0 - (utilization / working_load))
    idle_fraction = min(idle_fraction, 1.0)
    return round(idle_fraction * running_hours, 1)


def _recommendation(finding: InstanceFinding, gpu_instance: bool) -> str:
    """Human-readable, executive-ready remediation for a finding."""
    idle = finding.idle_hours_per_day
    if idle >= 12:
        action = (
            f"Running {finding.running_hours_per_day:.0f}h/day but sits idle "
            f"~{idle:.0f}h. "
        )
        if gpu_instance:
            action += ("Shift to serverless inference (e.g., Bedrock / "
                       "RunPod / Lambda GPU) or schedule auto-stop outside "
                       "business hours.")
        else:
            action += ("Schedule auto-stop nights/weekends or move the "
                       "workload to a smaller instance.")
    else:
        action = (f"Right-size: sustained low utilization "
                  f"({max(finding.avg_gpu_utilization_pct, finding.avg_cpu_utilization_pct):.0f}% peak) "
                  "supports a 1-2 size step-down or spot/reserved pricing.")
    return action


def analyze_infrastructure(rows: Sequence[Dict]) -> InfraAuditResult:
    """Run the Module B right-sizing audit over raw inventory rows.

    Args:
        rows: Records from :mod:`finops.data_loader` with fields
            ``instance_id, instance_type, avg_cpu_utilization_pct,
            avg_gpu_utilization_pct, running_hours_per_day``. An optional
            ``hourly_rate`` field overrides the built-in pricing table
            (useful for reserved pricing or non-AWS clouds).

    Returns:
        An :class:`InfraAuditResult` with total spend, total flagged waste,
        and per-instance findings sorted by monthly waste (largest first).
    """
    result = InfraAuditResult(instances_audited=len(rows))

    for row in rows:
        instance_id = str(row["instance_id"])
        instance_type = str(row["instance_type"]).strip().lower()
        avg_cpu = float(row["avg_cpu_utilization_pct"])
        avg_gpu = float(row.get("avg_gpu_utilization_pct") or 0.0)
        running_hours = min(24.0, max(0.0, float(row["running_hours_per_day"])))

        # Hourly rate: explicit field wins; else pricing table; else estimate
        # from GPU count heuristics so unknown SKUs still get a dollar figure.
        if row.get("hourly_rate") is not None:
            hourly_rate = float(row["hourly_rate"])
        else:
            hourly_rate = PRICING.ec2_hourly.get(instance_type)
            if hourly_rate is None:
                # Rough fallback: $0.10/core-hour heuristic scaled by vCPU hint.
                hourly_rate = 0.10 * max(4, _guess_vcpus(instance_type))
        gpu_instance = instance_type.startswith(("g5", "g6", "p3", "p4", "p5"))

        monthly_cost = hourly_rate * running_hours * DAYS_PER_MONTH
        result.total_monthly_compute_spend += monthly_cost

        # ---- Flag dead capital ---------------------------------------------
        idle_hours = _estimate_idle_hours(avg_cpu, avg_gpu, running_hours)
        peak_util = max(avg_gpu, avg_cpu)

        severity: str | None = None
        if idle_hours >= 12 or (running_hours >= 20 and peak_util < IDLE_UTILIZATION_PCT):
            severity = "critical" if (gpu_instance or hourly_rate >= 2.0) else "high"
        elif running_hours >= 16 and peak_util < SEVERE_UNDER_UTILIZATION_PCT:
            severity = "medium"

        if severity is None:
            result.healthy_instances.append(instance_id)
            continue

        # Recoverable spend: the fraction of hours that are idle (capped at
        # 85% — some baseline cost, e.g. cold starts/managed overhead, stays).
        waste_fraction = min(0.85, idle_hours / running_hours if running_hours else 0.0)
        monthly_waste = monthly_cost * waste_fraction

        finding = InstanceFinding(
            instance_id=instance_id,
            instance_type=instance_type,
            hourly_rate=hourly_rate,
            avg_cpu_utilization_pct=avg_cpu,
            avg_gpu_utilization_pct=avg_gpu,
            running_hours_per_day=running_hours,
            idle_hours_per_day=idle_hours,
            monthly_cost=round(monthly_cost, 2),
            monthly_waste=round(monthly_waste, 2),
            severity=severity,
            recommendation="",
        )
        finding.recommendation = _recommendation(finding, gpu_instance)
        result.findings.append(finding)
        result.total_monthly_waste += monthly_waste

    result.findings.sort(key=lambda f: f.monthly_waste, reverse=True)
    result.total_monthly_compute_spend = round(result.total_monthly_compute_spend, 2)
    result.total_monthly_waste = round(result.total_monthly_waste, 2)
    return result


def _guess_vcpus(instance_type: str) -> int:
    """Best-effort vCPU guess from an instance SKU for the pricing fallback."""
    parts = instance_type.split(".")
    if len(parts) == 2 and parts[1].endswith("xlarge"):
        size = parts[1][:-len("xlarge")]
        multipliers = {"": 4, "2": 8, "4": 16, "8": 32, "9": 36, "12": 48,
                       "16": 64, "24": 96, "48": 192}
        return multipliers.get(size, 8)
    return 8