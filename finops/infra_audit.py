"""Infrastructure audit: idle and oversized GPU / EC2 instances.

Two modes, chosen per instance:

**Time-series (preferred, ``--utilization``).** Hourly CPU/GPU samples (e.g.
exported from CloudWatch with ``scripts/export_cloudwatch.py``). We measure
which hours of the week an instance is genuinely idle, derive a concrete
stop/start schedule, count how many *busy* hours that schedule would
interrupt, and size the instance against its p95 load while active.

**Average (fallback, inventory only).** One average utilization number per
instance. Idle time is estimated from the average, so every figure in this
mode is labelled *estimated*.

Savings never double count: schedule savings come first, right-sizing is
priced only on the hours that remain.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Set, Tuple

from finops.ingest.tabular import IngestReport, read_rows
from finops.llm_audit import DAYS_PER_MONTH, _parse_ts
from finops.models import Confidence
from finops.pricing import InstancePrice, PriceBook

__all__ = ["InstanceFinding", "InfraAuditResult", "ScheduleWindow", "analyze_infra",
           "load_inventory", "load_utilization", "Thresholds"]

GPU_FAMILIES = ("g4", "g5", "g6", "p3", "p4", "p5", "p6", "inf", "trn")
WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


@dataclass(frozen=True)
class Thresholds:
    idle_gpu_pct: float = 5.0        # GPU instance hour is idle below this GPU util...
    idle_cpu_pct_gpu_host: float = 10.0   # ...and below this CPU util
    idle_cpu_pct: float = 5.0        # non-GPU hour is idle below this CPU util
    slot_idle_fraction: float = 0.9  # an hour-of-week is stoppable if idle ≥90% of weeks
    min_window_hours: int = 4        # ignore stop windows shorter than this
    target_peak_util: float = 70.0   # right-size so p95 active load lands near this
    min_days_measured: float = 7.0   # less data than this => "estimated"
    working_load_pct: float = 60.0   # average-mode heuristic only


@dataclass
class ScheduleWindow:
    weekday_stop_hour: Optional[int]     # UTC hour to stop on weekdays (None = no nightly stop)
    weekday_start_hour: Optional[int]
    weekend_off: bool
    hours_saved_per_week: float
    busy_hours_in_window_per_week: float
    slots: Tuple[Tuple[int, int], ...] = ()   # (weekday, hour) UTC slots kept stopped
    every_night: bool = False                 # nightly window applies on weekends too

    def transitions(self) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """(stop events, start events) as (weekday, hour) UTC, derived from the
        stopped-slot ring so every stop has a matching start."""
        on = set(self.slots)
        ring = [(d, h) for d in range(7) for h in range(24)]
        stops, starts = [], []
        for i, slot in enumerate(ring):
            prev = ring[i - 1]
            if slot in on and prev not in on:
                stops.append(slot)
            elif slot not in on and prev in on:
                starts.append(slot)
        return stops, starts

    def describe(self) -> str:
        parts = []
        if self.weekday_stop_hour is not None:
            parts.append(f"stop {self.weekday_stop_hour:02d}:00→{self.weekday_start_hour:02d}:00 "
                         "UTC " + ("every night" if self.every_night else "on weeknights"))
        if self.weekend_off:
            parts.append("off Sat–Sun")
        return "; ".join(parts) or "no safe window"


@dataclass
class InstanceFinding:
    instance_id: str
    instance_type: str
    name: str = ""
    team: str = ""
    gpu: bool = False
    hourly_rate: float = 0.0
    rate_source: str = ""
    running_hours_per_day: float = 24.0
    monthly_cost: float = 0.0
    method: str = "average"               # "timeseries" | "average"
    days_observed: float = 0.0
    idle_pct: float = 0.0                 # share of running hours that were idle
    p95_cpu_active: float = 0.0
    p95_gpu_active: float = 0.0
    zombie: bool = False
    schedule: Optional[ScheduleWindow] = None
    schedule_savings: float = 0.0
    rightsize_to: Optional[str] = None
    rightsize_savings: float = 0.0
    confidence: Confidence = Confidence.ESTIMATED
    severity: str = "ok"                  # critical | high | medium | ok
    recommendation: str = ""
    caveats: List[str] = field(default_factory=list)

    @property
    def monthly_savings(self) -> float:
        return self.schedule_savings + self.rightsize_savings

    @property
    def overall_confidence(self) -> Confidence:
        """Right-sizing is always an estimate (GPU memory isn't visible), so any
        finding that includes it is at most 'estimated'."""
        return Confidence.ESTIMATED if self.rightsize_savings > 0 else self.confidence


@dataclass
class InfraAuditResult:
    instances_audited: int = 0
    monthly_compute_spend: float = 0.0
    findings: List[InstanceFinding] = field(default_factory=list)
    healthy: List[str] = field(default_factory=list)
    unpriced: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def flagged(self) -> List[InstanceFinding]:
        return [f for f in self.findings if f.severity != "ok"]

    @property
    def monthly_savings(self) -> float:
        return sum(f.monthly_savings for f in self.flagged)

    @property
    def annual_savings(self) -> float:
        return self.monthly_savings * 12.0

    def savings_by_confidence(self) -> Dict[str, float]:
        out = {c.value: 0.0 for c in Confidence}
        for f in self.flagged:
            out[f.confidence.value] += f.schedule_savings
            out[Confidence.ESTIMATED.value] += f.rightsize_savings   # GPU memory unseen
        return out


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _f(row: Dict[str, object], *names: str) -> Optional[float]:
    """First parseable float among *names*; empty cells -> None (never crash)."""
    for name in names:
        value = row.get(name)
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        try:
            return float(str(value).replace("%", "").strip())
        except ValueError:
            continue
    return None


def _lower_keys(row: Dict[str, object]) -> Dict[str, object]:
    return {str(k).strip().lower(): v for k, v in row.items()}


def load_inventory(path: str) -> Tuple[List[Dict[str, object]], IngestReport]:
    report = IngestReport(path=path, fmt="inventory")
    out: List[Dict[str, object]] = []
    for raw in read_rows(path):
        report.rows_read += 1
        if not isinstance(raw, dict):
            report.skip("not an object")
            continue
        row = _lower_keys(raw)
        iid = str(row.get("instance_id") or row.get("instanceid") or "").strip()
        itype = str(row.get("instance_type") or row.get("instancetype") or "").strip().lower()
        if not iid or not itype:
            report.skip("missing instance_id or instance_type")
            continue
        out.append({
            "instance_id": iid, "instance_type": itype,
            "name": str(row.get("name") or ""), "team": str(row.get("team") or ""),
            "avg_cpu": _f(row, "avg_cpu_utilization_pct", "avg_cpu", "cpu_pct"),
            "avg_gpu": _f(row, "avg_gpu_utilization_pct", "avg_gpu", "gpu_pct"),
            "running_hours": _f(row, "running_hours_per_day", "hours_per_day"),
            "hourly_rate": _f(row, "hourly_rate", "hourly_cost"),
        })
    report.records = len(out)
    return out, report


def load_utilization(path: str) -> Tuple[Dict[str, List[Tuple[datetime, float, Optional[float]]]],
                                         Dict[str, str], IngestReport]:
    """Load hourly samples: instance_id, timestamp, cpu_pct[, gpu_pct][, instance_type]."""
    report = IngestReport(path=path, fmt="utilization")
    series: Dict[str, List[Tuple[datetime, float, Optional[float]]]] = defaultdict(list)
    types: Dict[str, str] = {}
    for raw in read_rows(path):
        report.rows_read += 1
        if not isinstance(raw, dict):
            report.skip("not an object")
            continue
        row = _lower_keys(raw)
        iid = str(row.get("instance_id") or "").strip()
        ts = _parse_ts(str(row.get("timestamp") or row.get("time") or ""))
        cpu = _f(row, "cpu_pct", "cpu_utilization", "cpuutilization", "cpu")
        gpu = _f(row, "gpu_pct", "gpu_utilization", "nvidia_smi_utilization_gpu", "gpu")
        if not iid or ts is None or cpu is None:
            report.skip("missing instance_id, timestamp or cpu_pct")
            continue
        series[iid].append((ts, cpu, gpu))
        if row.get("instance_type"):
            types[iid] = str(row["instance_type"]).strip().lower()
        report.records += 1
    return dict(series), types, report


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def _is_gpu(instance_type: str, price: Optional[InstancePrice]) -> bool:
    if price and price.gpus is not None:
        return price.gpus > 0
    return instance_type.startswith(GPU_FAMILIES)


def _p95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(math.ceil(0.95 * len(ordered))) - 1)]


def _rightsize(itype: str, price: Optional[InstancePrice], book: PriceBook,
               p95_cpu: float, p95_gpu: float, gpu: bool,
               th: Thresholds) -> Optional[InstancePrice]:
    """Cheapest same-family size that still fits p95 active load at the target
    utilization. Requires vCPU/GPU counts in the price book."""
    if not price or price.vcpus is None:
        return None
    need_vcpus = math.ceil(price.vcpus * p95_cpu / th.target_peak_util)
    need_gpus = (math.ceil((price.gpus or 0) * p95_gpu / th.target_peak_util) if gpu else 0)
    best = None
    for cand in book.siblings(itype):
        if cand.name == price.name or cand.hourly >= price.hourly or cand.vcpus is None:
            continue
        if cand.vcpus < max(1, need_vcpus):
            continue
        if gpu and (cand.gpus or 0) < max(1, need_gpus):
            continue
        if best is None or cand.hourly < best.hourly:
            best = cand
    return best


def _longest_circular_run(hours: Set[int]) -> Tuple[Optional[int], int]:
    """Longest run of consecutive hours (wrapping midnight). Returns (start, length)."""
    if not hours:
        return None, 0
    if len(hours) == 24:
        return 0, 24
    best_start, best_len = None, 0
    for h in hours:
        if (h - 1) % 24 in hours:
            continue                      # not the start of a run
        length = 0
        while (h + length) % 24 in hours:
            length += 1
        if length > best_len:
            best_start, best_len = h, length
    return best_start, best_len


def _schedule_from_profile(idle: Dict[Tuple[int, int], List[int]], weeks: float,
                           th: Thresholds) -> Optional[ScheduleWindow]:
    """Derive a weeknight stop window + weekend shutdown from an hour-of-week
    idle profile. ``idle[(weekday, hour)] = [idle_samples, total_samples]``."""
    def stoppable(slot: Tuple[int, int]) -> bool:
        idle_n, total = idle.get(slot, (0, 0))
        return total == 0 or idle_n / total >= th.slot_idle_fraction

    weekend_off = all(stoppable((d, h)) for d in (5, 6) for h in range(24))
    # Nightly window: hours stoppable on every day it applies to (weekdays
    # only if the weekend is off anyway, otherwise all seven days).
    nights = range(5) if weekend_off else range(7)
    nightly_hours = {h for h in range(24) if all(stoppable((d, h)) for d in nights)}
    start, length = _longest_circular_run(nightly_hours)
    if length >= 24:
        return None                           # idle all the time -> zombie, handled elsewhere
    if length < th.min_window_hours:
        start, length = None, 0

    stop_slots: Set[Tuple[int, int]] = set()
    if start is not None:
        for d in nights:
            for k in range(length):
                hour = start + k
                stop_slots.add(((d + hour // 24) % 7, hour % 24))
    if weekend_off:
        stop_slots.update((d, h) for d in (5, 6) for h in range(24))
        if start is not None and start + length > 24:
            # Sunday night's window spills into Monday morning before the
            # weekday start fires; those Monday hours are stopped too.
            stop_slots.update((0, h) for h in range(0, (start + length) % 24))
    if not stop_slots:
        return None
    saved = sum(idle.get(s, (0, 0))[1] for s in stop_slots) / max(weeks, 1e-9)
    busy = sum(idle.get(s, (0, 0))[1] - idle.get(s, (0, 0))[0]
               for s in stop_slots) / max(weeks, 1e-9)
    if saved <= 0:
        return None
    return ScheduleWindow(start, (start + length) % 24 if start is not None else None,
                          weekend_off, saved, busy, tuple(sorted(stop_slots)),
                          every_night=not weekend_off)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyze_infra(inventory: Sequence[Dict[str, object]], book: PriceBook,
                  utilization: Optional[Dict[str, List[Tuple[datetime, float, Optional[float]]]]] = None,
                  util_types: Optional[Dict[str, str]] = None,
                  thresholds: Thresholds = Thresholds()) -> InfraAuditResult:
    """Audit instances. *inventory* rows come from :func:`load_inventory`;
    instances present only in *utilization* are audited too if their type is
    known (``instance_type`` column)."""
    th = thresholds
    utilization = utilization or {}
    util_types = util_types or {}
    rows: Dict[str, Dict[str, object]] = {str(r["instance_id"]): dict(r) for r in inventory}
    for iid, itype in util_types.items():
        rows.setdefault(iid, {"instance_id": iid, "instance_type": itype, "name": "", "team": "",
                              "avg_cpu": None, "avg_gpu": None, "running_hours": None,
                              "hourly_rate": None})
    missing_type = [i for i in utilization if i not in rows]
    if missing_type:
        rows_note = ", ".join(missing_type[:5])
        result_warn = (f"{len(missing_type)} instance(s) in the utilization file have no "
                       f"instance_type ({rows_note}); add them to the inventory to audit them.")
    else:
        result_warn = ""

    result = InfraAuditResult(instances_audited=len(rows))
    if result_warn:
        result.warnings.append(result_warn)

    for iid, row in rows.items():
        itype = str(row["instance_type"])
        price = book.instance(itype)
        rate = row.get("hourly_rate")
        if rate is not None:
            rate_source = "inventory"
        elif price is not None:
            rate, rate_source = price.hourly, "price book (on-demand list)"
        else:
            result.unpriced.append(f"{iid} ({itype})")
            result.instances_audited -= 1
            continue
        gpu = _is_gpu(itype, price)
        f = InstanceFinding(instance_id=iid, instance_type=itype, name=str(row.get("name") or ""),
                            team=str(row.get("team") or ""), gpu=gpu, hourly_rate=float(rate),
                            rate_source=rate_source)

        samples = sorted(utilization.get(iid, []))
        if len(samples) >= 24:
            _analyze_timeseries(f, samples, price, book, th)
        else:
            if not _analyze_average(f, row, price, book, th):
                result.warnings.append(
                    f"{iid}: no utilization data (need avg_cpu_utilization_pct or hourly samples); skipped.")
                result.instances_audited -= 1
                continue
        result.monthly_compute_spend += f.monthly_cost
        _grade(f)
        result.findings.append(f)
        if f.severity == "ok":
            result.healthy.append(iid)

    result.findings.sort(key=lambda x: x.monthly_savings, reverse=True)
    if result.unpriced:
        result.warnings.append(
            f"No hourly rate for {len(result.unpriced)} instance(s): "
            f"{', '.join(result.unpriced[:5])}. Add hourly_rate to the inventory.")
    return result


def _analyze_timeseries(f: InstanceFinding, samples, price, book, th: Thresholds) -> None:
    f.method = "timeseries"
    # Sample interval: median gap, clamped to [5 min, 1 day].
    gaps = sorted((b[0] - a[0]).total_seconds() for a, b in zip(samples, samples[1:])
                  if b[0] > a[0])
    interval_h = min(24.0, max(1 / 12, (gaps[len(gaps) // 2] if gaps else 3600) / 3600))
    span_days = ((samples[-1][0] - samples[0][0]).total_seconds() / 3600 + interval_h) / 24
    f.days_observed = span_days
    weeks = span_days / 7.0
    running_hours = len(samples) * interval_h
    f.running_hours_per_day = min(24.0, running_hours / span_days)
    f.monthly_cost = f.hourly_rate * f.running_hours_per_day * DAYS_PER_MONTH

    profile: Dict[Tuple[int, int], List[int]] = defaultdict(lambda: [0, 0])
    active_cpu: List[float] = []
    active_gpu: List[float] = []
    idle_n = 0
    for ts, cpu, gpu in samples:
        if f.gpu:
            is_idle = (gpu if gpu is not None else 0.0) < th.idle_gpu_pct and cpu < th.idle_cpu_pct_gpu_host
        else:
            is_idle = cpu < th.idle_cpu_pct
        slot = profile[(ts.weekday(), ts.hour)]
        slot[1] += interval_h
        if is_idle:
            slot[0] += interval_h
            idle_n += 1
        else:
            active_cpu.append(cpu)
            active_gpu.append(gpu or 0.0)
    f.idle_pct = 100.0 * idle_n / len(samples)
    f.p95_cpu_active = _p95(active_cpu)
    f.p95_gpu_active = _p95(active_gpu)
    measured = span_days >= th.min_days_measured
    f.confidence = Confidence.MEASURED if measured else Confidence.ESTIMATED
    if not measured:
        f.caveats.append(f"only {span_days:.1f} days of data; collect ≥{th.min_days_measured:.0f} "
                         "days for a measured result")
    if any(g is None for _, _, g in samples) and f.gpu:
        f.caveats.append("GPU metric missing for some hours (install the CloudWatch agent "
                         "NVIDIA plugin); those hours judged on CPU only")

    if f.idle_pct >= 97.0:
        f.zombie = True
        f.schedule_savings = f.monthly_cost
        f.recommendation = (f"Idle {f.idle_pct:.0f}% of the {span_days:.0f} days observed: "
                            "stop it (snapshot first) or confirm an owner.")
        return

    sched = _schedule_from_profile(dict(profile), weeks, th)
    hours_per_month = f.running_hours_per_day * DAYS_PER_MONTH
    if sched:
        f.schedule = sched
        f.schedule_savings = f.hourly_rate * sched.hours_saved_per_week * DAYS_PER_MONTH / 7.0
        if sched.busy_hours_in_window_per_week > 0.5:
            f.caveats.append(f"schedule would interrupt ~{sched.busy_hours_in_window_per_week:.1f} "
                             "busy hours/week; confirm those jobs can move")
        hours_per_month -= sched.hours_saved_per_week * DAYS_PER_MONTH / 7.0

    target = _rightsize(f.instance_type, price, book, f.p95_cpu_active, f.p95_gpu_active, f.gpu, th)
    if target:
        f.rightsize_to = target.name
        f.rightsize_savings = max(0.0, (f.hourly_rate - target.hourly) * hours_per_month)
        if f.rate_source == "inventory" and price:
            # Inventory rate may be discounted; scale the target by the same ratio.
            f.rightsize_savings = max(0.0, f.hourly_rate * (1 - target.hourly / price.hourly)
                                      * hours_per_month)
        if f.gpu:
            f.caveats.append("check GPU memory: the model must still fit on the smaller instance")
    f.recommendation = _recommendation(f)


def _analyze_average(f: InstanceFinding, row, price, book, th: Thresholds) -> bool:
    avg_cpu = row.get("avg_cpu")
    avg_gpu = row.get("avg_gpu")
    if avg_cpu is None and avg_gpu is None:
        return False
    avg_cpu = float(avg_cpu or 0.0)
    avg_gpu = float(avg_gpu or 0.0)
    hours = row.get("running_hours")
    if hours is None:
        hours = 24.0
        f.caveats.append("running_hours_per_day missing; assumed 24")
    f.method = "average"
    f.confidence = Confidence.ESTIMATED
    f.running_hours_per_day = min(24.0, max(0.0, float(hours)))
    f.monthly_cost = f.hourly_rate * f.running_hours_per_day * DAYS_PER_MONTH
    util = max(avg_gpu, avg_cpu) if f.gpu else avg_cpu
    # Invert the average into an implied idle share, assuming the instance runs
    # at ~working_load_pct when busy. Deliberately conservative.
    idle_fraction = min(1.0, max(0.0, 1.0 - util / th.working_load_pct))
    f.idle_pct = 100.0 * idle_fraction
    f.p95_cpu_active = min(100.0, avg_cpu / max(1e-9, 1 - idle_fraction)) if idle_fraction < 1 else 0.0
    f.p95_gpu_active = min(100.0, avg_gpu / max(1e-9, 1 - idle_fraction)) if idle_fraction < 1 else 0.0
    f.caveats.append("averages only: idle time is inferred, not measured; upload hourly "
                     "utilization for a measured schedule")
    idle_hours = idle_fraction * f.running_hours_per_day
    if idle_hours >= 4 and f.running_hours_per_day >= 12:
        recoverable = min(0.85, idle_hours / f.running_hours_per_day)
        f.schedule_savings = f.monthly_cost * recoverable
    remaining = f.monthly_cost - f.schedule_savings
    target = _rightsize(f.instance_type, price, book, f.p95_cpu_active, f.p95_gpu_active, f.gpu, th)
    if target and price:
        f.rightsize_to = target.name
        f.rightsize_savings = max(0.0, remaining * (1 - target.hourly / price.hourly))
        if f.gpu:
            f.caveats.append("check GPU memory: the model must still fit on the smaller instance")
    f.recommendation = _recommendation(f)
    return True


def _recommendation(f: InstanceFinding) -> str:
    parts = []
    if f.schedule:
        parts.append(f"Schedule: {f.schedule.describe()} "
                     f"(saves {f.schedule.hours_saved_per_week:.0f} h/week).")
    elif f.schedule_savings > 0:
        parts.append(f"~{f.idle_pct:.0f}% of running time looks idle: add auto-stop "
                     + ("or move to serverless/on-demand inference." if f.gpu else
                        "outside working hours."))
    if f.rightsize_to:
        parts.append(f"Right-size {f.instance_type} → {f.rightsize_to} "
                     f"(p95 active load: CPU {f.p95_cpu_active:.0f}%"
                     + (f", GPU {f.p95_gpu_active:.0f}%" if f.gpu else "") + ").")
    return " ".join(parts)


def _grade(f: InstanceFinding) -> None:
    s = f.monthly_savings
    if s < 20 or s < 0.05 * max(f.monthly_cost, 1e-9):
        f.severity = "ok"
    elif f.zombie or s >= 1000:
        f.severity = "critical"
    elif s >= 200:
        f.severity = "high"
    else:
        f.severity = "medium"
