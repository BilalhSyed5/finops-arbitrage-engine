"""LLM spend audit: run-rate, attribution, and four stacked savings levers.

Levers are applied per record **in sequence**, each on the cost left by the
previous one, so savings never double count and always sum exactly to
``current - optimized``:

1. **Model refresh** -- a newer model at the same capability tier that is
   cheaper for this traffic (e.g. Sonnet 4.5 -> Sonnet 5.5). Applies to all
   task types. Confidence: *measured* (pure price math; regression-test it).
2. **Task routing** -- move routable task types (classification, extraction,
   ...) to the policy's cheaper model, *only if it is actually cheaper* for
   this traffic. A failed quality proof blocks the route; a passed proof
   upgrades the confidence to *verified*. Otherwise *estimated*.
   Refresh and routing are alternatives: the cheaper of the two wins.
3. **Prompt caching** -- measured from shared prompt prefixes when prompts are
   logged; otherwise an explicit, conservative assumption (*estimated*).
4. **Batch API** -- traffic the policy marks as non-interactive (*measured*).
"""

from __future__ import annotations

import dataclasses
import fnmatch
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from finops.models import Confidence, TaskSource, UsageRecord
from finops.policy import Policy
from finops.pricing import ModelPrice, PriceBook, normalize_model_name

__all__ = ["Agg", "LeverLine", "LLMAuditResult", "analyze_llm_spend",
           "annotate_shared_prefixes", "infer_window_days", "route_candidates", "LEVERS", "DAYS_PER_MONTH"]

DAYS_PER_MONTH = 30.4
LEVERS = ("model_refresh", "routing", "caching", "batch")
_CONF_RANK = {Confidence.ESTIMATED: 0, Confidence.MEASURED: 1, Confidence.VERIFIED: 2}

ProofIndex = Dict[Tuple[str, str], str]   # (task_type, candidate model) -> pass|fail|inconclusive


@dataclass
class Agg:
    """Monthly spend for one slice (model, team, feature, ...)."""

    current: float = 0.0
    optimized: float = 0.0
    requests: float = 0.0
    input_tokens: float = 0.0
    output_tokens: float = 0.0

    @property
    def savings(self) -> float:
        return self.current - self.optimized


@dataclass
class LeverLine:
    """Aggregated savings for one (lever, model, target, task, feature) group."""

    lever: str
    model: str
    target: str
    task_type: str
    feature: str
    affected_monthly_cost: float = 0.0
    monthly_savings: float = 0.0
    requests_per_month: float = 0.0
    confidence: Confidence = Confidence.VERIFIED
    method: str = ""
    task_sources: Set[str] = field(default_factory=set)


@dataclass
class LLMAuditResult:
    window_days: float = 0.0
    window_inferred: bool = True
    sample_rate: float = 1.0
    records_analyzed: int = 0
    records_priced: int = 0
    monthly_current: float = 0.0
    monthly_optimized: float = 0.0
    lever_totals: Dict[str, float] = field(default_factory=lambda: {k: 0.0 for k in LEVERS})
    savings_by_confidence: Dict[str, float] = field(
        default_factory=lambda: {c.value: 0.0 for c in Confidence})
    lever_lines: List[LeverLine] = field(default_factory=list)
    by_model: Dict[str, Agg] = field(default_factory=dict)
    by_provider: Dict[str, Agg] = field(default_factory=dict)
    by_task: Dict[str, Agg] = field(default_factory=dict)
    by_team: Dict[str, Agg] = field(default_factory=dict)
    by_feature: Dict[str, Agg] = field(default_factory=dict)
    by_customer: Dict[str, Agg] = field(default_factory=dict)
    by_project: Dict[str, Agg] = field(default_factory=dict)
    spend_by_task_source: Dict[str, float] = field(default_factory=dict)
    coverage: Dict[str, float] = field(default_factory=dict)
    unpriced: Dict[str, Dict[str, float]] = field(default_factory=dict)
    upcoming_price_changes: List[Dict[str, object]] = field(default_factory=list)
    blocked_by_proof: Dict[str, float] = field(default_factory=dict)
    reconciliation: Optional[Dict[str, float]] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def monthly_savings(self) -> float:
        return self.monthly_current - self.monthly_optimized

    @property
    def annual_savings(self) -> float:
        return self.monthly_savings * 12.0

    @property
    def savings_pct(self) -> float:
        return 100.0 * self.monthly_savings / self.monthly_current if self.monthly_current else 0.0


# ---------------------------------------------------------------------------
# Time window
# ---------------------------------------------------------------------------

def _parse_ts(value: str) -> Optional[datetime]:
    if not value:
        return None
    text = str(value).strip()
    try:
        num = float(text)
        if num > 1e11:      # epoch milliseconds
            num /= 1000.0
        return datetime.fromtimestamp(num, tz=timezone.utc).replace(tzinfo=None)
    except ValueError:
        pass
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def infer_window_days(records: Iterable[UsageRecord]) -> Optional[float]:
    """Days covered by the data: earliest start to latest end (bucket end if
    the source has one). ``None`` if timestamps are unusable."""
    starts: List[datetime] = []
    ends: List[datetime] = []
    for rec in records:
        start = _parse_ts(rec.timestamp)
        if start:
            starts.append(start)
            end = _parse_ts(rec.period_end)
            if end is None:
                # A date-only stamp ("2026-11-03") is a daily bucket: it covers the whole day.
                end = start + timedelta(days=1) if len(rec.timestamp.strip()) == 10 else start
            ends.append(end)
    if not starts:
        return None
    span = (max(ends) - min(starts)).total_seconds() / 86400.0
    return span if span > 0 else None


# ---------------------------------------------------------------------------
# Prompt-prefix analysis (caching lever)
# ---------------------------------------------------------------------------

def _common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def annotate_shared_prefixes(records: Sequence[UsageRecord], pricebook: PriceBook,
                             chars_per_token: float = 4.0) -> int:
    """Estimate, per logged prompt, how many leading tokens it shares with
    other prompts to the same model+feature/task (i.e. what a prompt cache
    could serve). Stores ``extra['shared_prefix_tokens']``. Returns how many
    records were annotated.

    Sorting prompts puts the closest prefix-neighbours next to each other, so
    the longest common prefix with either neighbour is exact for each prompt.
    """
    groups: Dict[Tuple[str, str], List[UsageRecord]] = defaultdict(list)
    for rec in records:
        if rec.prompt and len(rec.prompt) > 200:
            price = pricebook.resolve(rec.model)
            key = (price.name if price else rec.model, rec.feature or rec.task_type)
            groups[key].append(rec)
    annotated = 0
    for group in groups.values():
        if len(group) < 2:
            continue
        group.sort(key=lambda r: r.prompt or "")
        for i, rec in enumerate(group):
            best = 0
            for j in (i - 1, i + 1):
                if 0 <= j < len(group):
                    best = max(best, _common_prefix_len(rec.prompt or "", group[j].prompt or ""))
            if best:
                rec.extra["shared_prefix_tokens"] = best / chars_per_token
                annotated += 1
    return annotated


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def _cost_on(target: ModelPrice, rec: UsageRecord, same_model: bool,
             cached: Optional[float] = None, writes: Optional[float] = None) -> float:
    """Cost of *rec*'s traffic if served by *target*."""
    keep_cache = same_model or target.supports_caching
    return target.cost(
        rec.input_tokens, rec.output_tokens,
        cached_input_tokens=(cached if cached is not None
                             else (rec.cached_input_tokens if keep_cache else 0)),
        cache_write_tokens=(writes if writes is not None
                            else (rec.cache_write_tokens if same_model or target.cache_write else 0)),
        requests=rec.requests, batch=rec.batch,
        long_context=rec.long_context if same_model else None)


def route_candidates(price: ModelPrice, task_type: str, policy: Policy) -> List[str]:
    """Models the policy allows *task_type* traffic on *price* to move to.
    Shared by the audit and by proof mode so both evaluate the same routes."""
    targets = policy.targets_for(task_type)
    if not targets:
        return []
    if policy.cross_provider:
        names = sorted(set(targets.values()))
    else:
        pick = targets.get(price.provider) or (
            targets.get("default") if price.provider == "unknown" else None)
        names = [pick] if pick else []
    return [n for n in names if n != price.name]


def _batch_eligible(rec: UsageRecord, policy: Policy) -> bool:
    if rec.batch:
        return False
    if rec.extra.get("batch_candidate"):
        return True
    if rec.task_type in policy.batch_tasks:
        return True
    feature = rec.feature.lower()
    return bool(feature) and any(fnmatch.fnmatchcase(feature, p)
                                 for p in policy.batch_feature_patterns)


def analyze_llm_spend(records: Sequence[UsageRecord], pricebook: PriceBook, policy: Policy,
                      *, sample_days: Optional[float] = None, sample_rate: float = 1.0,
                      proof: Optional[ProofIndex] = None,
                      today: Optional[date] = None) -> LLMAuditResult:
    """Analyze LLM usage and quantify savings levers (monthly USD).

    Args:
        records: normalized, enriched usage records.
        pricebook / policy: pricing snapshot and lever configuration.
        sample_days: explicit length of the data window; inferred otherwise.
        sample_rate: fraction of traffic present in the data (e.g. 0.05 for a
            5% sampled request log). Used to scale up to full volume.
        proof: quality-eval verdicts from ``finops proof``.
    """
    proof = proof or {}
    today = today or date.today()
    result = LLMAuditResult(records_analyzed=len(records), sample_rate=sample_rate)
    if not records:
        return result

    # ---- Monthly scale -----------------------------------------------------
    if sample_days and sample_days > 0:
        result.window_days, result.window_inferred = sample_days, False
    else:
        inferred = infer_window_days(records)
        if inferred is None:
            result.window_days = DAYS_PER_MONTH
            result.warnings.append(
                "Timestamps missing or unparseable: assumed the data covers one month. "
                "Pass --sample-days to set the window explicitly.")
        else:
            result.window_days = max(inferred, 1.0 / 24.0)
            if inferred < 1.0:
                result.warnings.append(
                    f"Data covers only {inferred * 24:.1f} hours; monthly figures are a "
                    "straight-line extrapolation. Use ≥7 days of data for a reliable audit.")
    if not 0 < sample_rate <= 1:
        raise ValueError("sample_rate must be in (0, 1].")
    scale = DAYS_PER_MONTH / result.window_days / sample_rate

    # ---- Caching inputs ----------------------------------------------------
    cfg = policy.caching
    min_prompt = float(cfg.get("min_prompt_tokens", 1024))
    min_shared = float(cfg.get("min_shared_prefix_tokens", 1024))
    assumed_frac = float(cfg.get("assumed_cacheable_fraction", 0.3))
    hit_rate = float(cfg.get("assumed_hit_rate", 0.8))
    min_reqs = float(cfg.get("min_requests_for_assumption", 1000))
    annotate_shared_prefixes(records, pricebook, float(cfg.get("chars_per_token", 4.0)))

    group_requests: Dict[Tuple[str, str], float] = defaultdict(float)
    for rec in records:
        group_requests[(rec.model, rec.feature or rec.task_type)] += rec.requests * scale

    # ---- Per-record pass ---------------------------------------------------
    lines: Dict[Tuple[str, str, str, str, str], LeverLine] = {}
    slices = (("by_model", None), ("by_provider", None), ("by_task", None),
              ("by_team", None), ("by_feature", None), ("by_customer", None),
              ("by_project", None))
    for name, _ in slices:
        setattr(result, name, defaultdict(Agg))
    task_src: Dict[str, float] = defaultdict(float)
    covered = {"team": 0.0, "feature": 0.0, "customer": 0.0}
    upcoming: Dict[str, Dict[str, object]] = {}
    recon_actual = recon_computed = 0.0
    recon_n = 0

    def add_line(lever: str, rec: UsageRecord, model: str, target: str,
                 before: float, saved: float, conf: Confidence, method: str) -> None:
        key = (lever, model, target, rec.task_type, rec.feature)
        line = lines.get(key)
        if line is None:
            line = lines[key] = LeverLine(lever, model, target, rec.task_type, rec.feature,
                                          method=method)
        line.affected_monthly_cost += before * scale
        line.monthly_savings += saved * scale
        line.requests_per_month += rec.requests * scale
        if _CONF_RANK[conf] < _CONF_RANK[line.confidence]:
            line.confidence = conf
        line.task_sources.add(rec.task_source.value)
        result.lever_totals[lever] += saved * scale
        result.savings_by_confidence[conf.value] += saved * scale

    for rec in records:
        price = pricebook.resolve(rec.model)
        if price is None:
            u = result.unpriced.setdefault(rec.model, {"input_tokens": 0.0, "output_tokens": 0.0,
                                                       "requests": 0.0})
            u["input_tokens"] += rec.input_tokens * scale
            u["output_tokens"] += rec.output_tokens * scale
            u["requests"] += rec.requests * scale
            continue
        result.records_priced += 1
        # Show the model as the customer names it ("claude-sonnet-4-5"), not the
        # price-book entry it resolved to ("claude-sonnet-4-6" alias group).
        shown = normalize_model_name(rec.model)[-1]

        current = _cost_on(price, rec, same_model=True)
        cost = current
        model = price

        if rec.actual_spend is not None:
            recon_actual += rec.actual_spend
            recon_computed += current
            recon_n += 1

        # ---- Levers 1+2: refresh vs routing (cheaper wins) ------------------
        options: List[Tuple[str, ModelPrice, float, Confidence, str]] = []
        if policy.model_refresh and price.successor:
            succ = pricebook.get(price.successor)
            if succ:
                c = _cost_on(succ, rec, same_model=False)
                if c < current * 0.995:
                    options.append(("model_refresh", succ, c, Confidence.MEASURED,
                                    "same-tier successor; regression-test before switching"))
        for name in route_candidates(price, rec.task_type, policy):
            tgt = pricebook.get(name)
            if tgt is None or tgt.name == price.name:
                continue
            c = _cost_on(tgt, rec, same_model=False)
            if c >= current * 0.995:
                continue          # never "route" to something that isn't cheaper
            verdict = proof.get((rec.task_type, tgt.name))
            if verdict == "fail":
                label = f"{rec.task_type}: {shown} → {tgt.name}"
                result.blocked_by_proof[label] = (
                    result.blocked_by_proof.get(label, 0.0) + (current - c) * scale)
                continue
            conf = Confidence.VERIFIED if verdict == "pass" else Confidence.ESTIMATED
            method = ("quality proof passed" if verdict == "pass"
                      else "pending quality proof (run `finops proof`)")
            if rec.task_source == TaskSource.INFERRED:
                # The proof covered traffic whose task type was known; inferred
                # traffic might be something else, so it can't inherit 'verified'.
                conf = Confidence.ESTIMATED
                method += "; task type inferred"
            options.append(("routing", tgt, c, conf, method))
        if options:
            lever, tgt, c, conf, method = min(options, key=lambda o: o[2])
            add_line(lever, rec, shown, tgt.name, current, current - c, conf, method)
            cost, model = c, tgt
        same = model is price

        # ---- Lever 3: prompt caching ---------------------------------------
        if model.supports_caching and rec.input_tokens > 0:
            frac, conf, method = 0.0, Confidence.ESTIMATED, ""
            avg_in = rec.input_tokens / rec.requests if rec.requests > 0 else None
            shared = rec.extra.get("shared_prefix_tokens")
            if shared and avg_in and shared >= min_shared:
                frac, conf = min(1.0, shared / avg_in), Confidence.MEASURED
                method = "shared prompt prefix measured from logged prompts"
            elif (avg_in and avg_in >= min_prompt
                  and group_requests[(rec.model, rec.feature or rec.task_type)] >= min_reqs
                  and (rec.cache_reported or model.provider != "openai")
                  and rec.cached_input_tokens < 0.05 * rec.input_tokens):
                # OpenAI caches automatically; only flag it when the data shows
                # it isn't happening. Other providers need explicit cache markers.
                frac = assumed_frac
                method = (f"assumes {assumed_frac:.0%} of each long prompt is a stable "
                          "prefix; log prompts to measure it")
            if frac > 0:
                cacheable = frac * rec.input_tokens
                existing = rec.cached_input_tokens if (same or model.supports_caching) else 0
                new_cached = max(existing, cacheable * hit_rate)
                writes = (cacheable * (1 - hit_rate)
                          if model.cache_write and model.cache_write > model.input else 0.0)
                c = _cost_on(model, rec, same_model=same, cached=new_cached,
                             writes=max(writes, rec.cache_write_tokens if same else 0))
                if c < cost * 0.995:
                    add_line("caching", rec, shown if same else model.name, "", cost, cost - c,
                             conf, method)
                    cost = c

        # ---- Lever 4: batch ---------------------------------------------------
        if model.supports_batch and _batch_eligible(rec, policy):
            c = cost * (1.0 - float(model.batch_discount or 0))
            add_line("batch", rec, shown if same else model.name, "", cost, cost - c, Confidence.MEASURED,
                     "non-interactive traffic; move to the provider's Batch API")
            cost = c

        # ---- Aggregation -------------------------------------------------------
        cur_m, opt_m = current * scale, cost * scale
        for attr, key in (("by_model", shown), ("by_provider", price.provider),
                          ("by_task", rec.task_type or "unknown"),
                          ("by_team", rec.team), ("by_feature", rec.feature),
                          ("by_customer", rec.customer_id), ("by_project", rec.project)):
            if not key:
                continue
            agg = getattr(result, attr)[key]
            agg.current += cur_m
            agg.optimized += opt_m
            agg.requests += rec.requests * scale
            agg.input_tokens += rec.input_tokens * scale
            agg.output_tokens += rec.output_tokens * scale
        result.monthly_current += cur_m
        result.monthly_optimized += opt_m
        task_src[rec.task_source.value] += cur_m
        covered["team"] += cur_m if rec.team else 0.0
        covered["feature"] += cur_m if rec.feature else 0.0
        covered["customer"] += cur_m if rec.customer_id else 0.0

        # ---- Announced price changes (status quo) ---------------------------
        sc = price.scheduled_change
        if sc:
            try:
                effective = date.fromisoformat(sc.effective)
            except ValueError:
                effective = None
            if effective and effective > today:
                future = dataclasses.replace(
                    price, input=sc.input, output=sc.output,
                    cached_input=sc.cached_input if sc.cached_input is not None
                    else price.cached_input, scheduled_change=None)
                delta = (_cost_on(future, rec, same_model=True) - current) * scale
                entry = upcoming.setdefault(price.name, {
                    "model": price.name, "effective": sc.effective, "reason": sc.reason,
                    "monthly_delta": 0.0, "current_monthly": 0.0})
                entry["monthly_delta"] = float(entry["monthly_delta"]) + delta
                entry["current_monthly"] = float(entry["current_monthly"]) + cur_m

    # ---- Finalize ----------------------------------------------------------
    for name, _ in slices:
        setattr(result, name, dict(getattr(result, name)))
    result.lever_lines = sorted(lines.values(), key=lambda l: l.monthly_savings, reverse=True)
    result.spend_by_task_source = dict(task_src)
    total = result.monthly_current or 1.0
    result.coverage = {k: 100.0 * v / total for k, v in covered.items()}
    result.upcoming_price_changes = sorted(upcoming.values(),
                                           key=lambda e: -float(e["monthly_delta"]))

    if recon_n:
        variance = (100.0 * (recon_computed - recon_actual) / recon_actual
                    if recon_actual else 0.0)
        result.reconciliation = {"records": recon_n, "actual": recon_actual,
                                 "computed_list_price": recon_computed,
                                 "variance_pct": variance}
        if abs(variance) > 15:
            result.warnings.append(
                f"List-price math differs from billed spend by {variance:+.0f}% on "
                f"{recon_n:,} records. If you have negotiated rates, put them in a "
                "--pricing-config override so savings use your real prices.")
    if result.unpriced:
        unpriced_tokens = sum(v["input_tokens"] + v["output_tokens"]
                              for v in result.unpriced.values())
        result.warnings.append(
            f"{len(result.unpriced)} model(s) have no price ({', '.join(sorted(result.unpriced)[:5])}"
            f"{'…' if len(result.unpriced) > 5 else ''}; {unpriced_tokens:,.0f} tokens/mo). "
            "They are excluded from all dollar figures — add them to a pricing override.")
    unknown_share = task_src.get(TaskSource.UNKNOWN.value, 0.0) / total
    if unknown_share > 0.25:
        result.warnings.append(
            f"{unknown_share:.0%} of spend has no task type, so routing can't be assessed for "
            "it. Tag calls (LiteLLM metadata) or add an --attribution map.")
    return result
