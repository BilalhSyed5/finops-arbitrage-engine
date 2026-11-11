"""Turn findings into a ranked action plan and ready-to-apply artifacts.

Artifacts (written with ``--actions-dir``):

``litellm_router.yaml``  LiteLLM proxy config: one alias per routed task
                         (``finops-classification`` ...) pointing at the
                         cheaper model, with the current model as fallback.
                         Unverified routes are emitted commented out.
``ec2_schedules.sh``     AWS EventBridge Scheduler commands that stop/start
                         instances on the measured idle windows, plus
                         commented right-sizing commands. Dry-run by default.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from finops.infra_audit import InfraAuditResult, WEEKDAYS
from finops.llm_audit import LLMAuditResult
from finops.models import Confidence
from finops.pricing import PriceBook

__all__ = ["Action", "build_actions", "render_litellm_config", "render_ec2_script",
           "write_artifacts"]

_CONF_RANK = {Confidence.ESTIMATED.value: 0, Confidence.MEASURED.value: 1,
              Confidence.VERIFIED.value: 2}
LITELLM_PREFIX = {"openai": "openai", "anthropic": "anthropic", "google": "gemini",
                  "open-weight": "groq"}
API_KEY_ENV = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
               "google": "GEMINI_API_KEY", "open-weight": "GROQ_API_KEY"}


@dataclass
class Action:
    category: str          # routing | model_refresh | caching | batch | schedule | rightsize | shutdown
    title: str
    monthly_savings: float
    confidence: str
    effort: str
    detail: str = ""
    evidence: List[str] = field(default_factory=list)

    @property
    def annual_savings(self) -> float:
        return self.monthly_savings * 12.0


_EFFORT = {
    "routing": "config change at the gateway (alias swap)",
    "model_refresh": "config change + regression test",
    "caching": "prompt restructuring / cache markers",
    "batch": "code change (async Batch API job)",
    "schedule": "infra change (EventBridge Scheduler)",
    "rightsize": "infra change (stop → resize → start)",
    "shutdown": "infra change (snapshot + stop)",
}


def build_actions(llm: Optional[LLMAuditResult], infra: Optional[InfraAuditResult],
                  min_monthly: float = 1.0) -> List[Action]:
    """Group lever lines into human-sized actions, ranked by monthly savings."""
    actions: List[Action] = []
    if llm:
        groups: Dict[Tuple[str, str, str, str, bool], List] = defaultdict(list)
        for line in llm.lever_lines:
            inferred_only = line.task_sources == {"inferred"}
            if line.lever in ("routing", "model_refresh"):
                key = (line.lever, line.model, line.target,
                       line.task_type if line.lever == "routing" else "", inferred_only)
            else:
                key = (line.lever, line.model, "", line.feature or line.task_type, inferred_only)
            groups[key].append(line)
        for (lever, model, target, scope, inferred_only), lines in groups.items():
            savings = sum(l.monthly_savings for l in lines)
            if savings < min_monthly:
                continue
            affected = sum(l.affected_monthly_cost for l in lines)
            reqs = sum(l.requests_per_month for l in lines)
            conf = min((l.confidence.value for l in lines), key=lambda c: _CONF_RANK[c])
            methods = sorted({l.method for l in lines if l.method})
            if lever == "routing" and inferred_only:
                title = (f"Tag, then route untagged {model} calls that look like `{scope}` "
                         f"to {target}")
            elif lever == "routing":
                title = f"Route `{scope}` calls from {model} to {target}"
            elif lever == "model_refresh":
                title = f"Upgrade {model} → {target} (newer, cheaper, same tier)"
            elif lever == "caching":
                title = f"Enable prompt caching for `{scope}` on {model}"
            else:
                title = f"Move `{scope}` on {model} to the Batch API"
            evidence = [f"${affected:,.0f}/mo of traffic affected", f"{reqs:,.0f} requests/mo"]
            actions.append(Action(lever, title, savings, conf, _EFFORT[lever],
                                  "; ".join(methods), evidence))
    if infra:
        for f in infra.flagged:
            label = f.name or f.instance_id
            if f.zombie:
                actions.append(Action("shutdown", f"Stop idle instance {label} ({f.instance_type})",
                                      f.schedule_savings, f.confidence.value, _EFFORT["shutdown"],
                                      f.recommendation, [f"idle {f.idle_pct:.0f}% over "
                                                         f"{f.days_observed:.0f} days"]))
                continue
            if f.schedule_savings >= min_monthly:
                what = (f.schedule.describe() if f.schedule else "auto-stop during idle hours")
                actions.append(Action("schedule", f"Schedule {label} ({f.instance_type}): {what}",
                                      f.schedule_savings, f.confidence.value, _EFFORT["schedule"],
                                      f.recommendation, f.caveats[:]))
            if f.rightsize_to and f.rightsize_savings >= min_monthly:
                actions.append(Action("rightsize",
                                      f"Right-size {label}: {f.instance_type} → {f.rightsize_to}",
                                      f.rightsize_savings, Confidence.ESTIMATED.value,
                                      _EFFORT["rightsize"], f.recommendation, f.caveats[:]))
    actions.sort(key=lambda a: a.monthly_savings, reverse=True)
    return actions


# ---------------------------------------------------------------------------
# LiteLLM config
# ---------------------------------------------------------------------------

def _yaml_str(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _litellm_model(book: PriceBook, name: str) -> Tuple[str, str]:
    price = book.get(name)
    provider = price.provider if price else "openai"
    return f"{LITELLM_PREFIX.get(provider, provider)}/{name}", API_KEY_ENV.get(provider, "API_KEY")


def render_litellm_config(llm: LLMAuditResult, book: PriceBook) -> str:
    """LiteLLM proxy config implementing the recommended routes.

    Each routed task gets an alias ``finops-<task>`` served by the cheaper
    model, with the current model as fallback. The application switches by
    sending ``model="finops-<task>"`` for that call site. Routes without a
    passing quality proof are written commented out.
    """
    by_task: Dict[str, Dict[Tuple[str, str], List]] = defaultdict(lambda: defaultdict(list))
    refresh: Dict[Tuple[str, str], float] = defaultdict(float)
    for line in llm.lever_lines:
        if line.lever == "routing":
            by_task[line.task_type][(line.model, line.target)].append(line)
        elif line.lever == "model_refresh":
            refresh[(line.model, line.target)] += line.monthly_savings

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out = [
        "# LiteLLM proxy config generated by finops-arbitrage-engine " + stamp,
        "# Docs: https://docs.litellm.ai/docs/proxy/configs",
        "#",
        "# How to use: merge into your proxy config, then point each call site at its",
        "# alias, e.g. model=\"finops-classification\". The current model stays as the",
        "# fallback, so a failure on the cheaper model degrades to today's behaviour.",
        "# Routes marked UNVERIFIED have not passed `finops proof`; they are commented",
        "# out until they do.",
        "",
        "model_list:",
    ]
    fallbacks: List[Tuple[str, str]] = []
    emitted = False
    for task in sorted(by_task):
        routes = by_task[task]
        (src, tgt), lines = max(routes.items(),
                                key=lambda kv: sum(l.monthly_savings for l in kv[1]))
        savings = sum(l.monthly_savings for l in lines)
        # Untagged traffic whose task was inferred won't call the alias anyway
        # (it has to be tagged first), so it doesn't decide verification.
        tagged = [l for l in lines if l.task_sources - {"inferred"}] or lines
        verified = all(l.confidence == Confidence.VERIFIED for l in tagged)
        alias = "finops-" + re.sub(r"[^a-z0-9_-]+", "-", task.lower())
        tgt_model, tgt_key = _litellm_model(book, tgt)
        src_model, src_key = _litellm_model(book, src)
        prefix = "" if verified else "# "
        status = "VERIFIED by quality proof" if verified else "UNVERIFIED - run `finops proof` first"
        out += [f"  # {task}: {src} → {tgt}, saves ~${savings:,.0f}/mo ({status})",
                f"{prefix}  - model_name: {alias}",
                f"{prefix}    litellm_params:",
                f"{prefix}      model: {_yaml_str(tgt_model)}",
                f"{prefix}      api_key: os.environ/{tgt_key}",
                f"{prefix}  - model_name: {alias}-fallback",
                f"{prefix}    litellm_params:",
                f"{prefix}      model: {_yaml_str(src_model)}",
                f"{prefix}      api_key: os.environ/{src_key}"]
        others = [f"{s} → {t}" for (s, t) in routes if (s, t) != (src, tgt)]
        if others:
            out.append(f"  # also routable for {task}: {', '.join(others)}")
        if verified:
            fallbacks.append((alias, alias + "-fallback"))
            emitted = True
    if refresh:
        out += ["", "  # Model refresh: serve existing model names from their cheaper successors.",
                "  # Uncomment after a regression test; apps need no code change."]
        for (src, tgt), saved in sorted(refresh.items(), key=lambda kv: -kv[1]):
            tgt_model, key = _litellm_model(book, tgt)
            out += [f"  # {src} → {tgt}: ~${saved:,.0f}/mo",
                    f"  # - model_name: {src}",
                    f"  #   litellm_params:",
                    f"  #     model: {_yaml_str(tgt_model)}",
                    f"  #     api_key: os.environ/{key}"]
    if not emitted:
        out.append("  []  # no verified routes yet")
    out += ["", "router_settings:"]
    if fallbacks:
        out.append("  fallbacks:")
        for alias, fb in fallbacks:
            out.append(f"    - {alias}: [{fb}]")
    else:
        out.append("  fallbacks: []")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# EC2 schedule script
# ---------------------------------------------------------------------------

def _cron_groups(events: List[Tuple[int, int]]) -> Dict[int, List[str]]:
    by_hour: Dict[int, List[str]] = defaultdict(list)
    for day, hour in events:
        by_hour[hour].append(WEEKDAYS[day].upper())
    return by_hour


def render_ec2_script(infra: InfraAuditResult) -> str:
    """Bash script with EventBridge Scheduler commands. Prints commands by
    default; set DRY_RUN=0 to execute."""
    lines = [
        "#!/usr/bin/env bash",
        "# EC2 schedules generated by finops-arbitrage-engine.",
        "# Review before running. Prints the commands by default; DRY_RUN=0 executes them.",
        "# Needs: aws CLI v2, and ROLE_ARN = an IAM role EventBridge Scheduler can assume",
        "# with ec2:StartInstances / ec2:StopInstances on these instances.",
        "set -euo pipefail",
        'DRY_RUN="${DRY_RUN:-1}"',
        'ROLE_ARN="${ROLE_ARN:?set ROLE_ARN to the scheduler execution role ARN}"',
        'TZ_NAME="UTC"   # windows below were measured in UTC',
        "",
        'run() { if [ "$DRY_RUN" = "1" ]; then printf "%q " "$@"; echo; else "$@"; fi; }',
        "",
        "schedule() {  # name cron-expression action instance_id",
        '  local name="$1" expr="$2" action="$3" iid="$4" target',
        """  target=$(printf '{"Arn":"arn:aws:scheduler:::aws-sdk:ec2:%s","RoleArn":"%s","Input":"{\\\\"InstanceIds\\\\":[\\\\"%s\\\\"]}"}' "$action" "$ROLE_ARN" "$iid")""",
        '  run aws scheduler create-schedule --name "$name" \\',
        '    --schedule-expression "$expr" --schedule-expression-timezone "$TZ_NAME" \\',
        '    --flexible-time-window Mode=OFF --target "$target"',
        "}",
        "",
    ]
    any_cmd = False
    for f in infra.flagged:
        header = (f"# {f.instance_id} {f.name} ({f.instance_type}) — saves "
                  f"~${f.monthly_savings:,.0f}/mo [{f.overall_confidence.value}]")
        if f.schedule and f.schedule.slots:
            stops, starts = f.schedule.transitions()
            lines.append(header)
            lines.append(f"# window: {f.schedule.describe()}")
            for c in f.caveats:
                lines.append(f"# caveat: {c}")
            for kind, events, action in (("stop", stops, "stopInstances"),
                                         ("start", starts, "startInstances")):
                for hour, days in sorted(_cron_groups(events).items()):
                    name = f"finops-{kind}-{f.instance_id}-{hour:02d}"
                    expr = f"cron(0 {hour} ? * {','.join(days)} *)"
                    lines.append(f'schedule "{name}" "{expr}" {action} {f.instance_id}')
                    any_cmd = True
            lines.append("")
        elif f.zombie:
            lines += [header, f"# idle {f.idle_pct:.0f}% of observed time. Snapshot, then stop:",
                      f"# run aws ec2 create-image --instance-id {f.instance_id} "
                      f"--name finops-backup-{f.instance_id} --no-reboot",
                      f"# run aws ec2 stop-instances --instance-ids {f.instance_id}", ""]
        if f.rightsize_to:
            if not (f.schedule and f.schedule.slots) and not f.zombie:
                lines.append(header)
            lines += [f"# right-size {f.instance_id}: {f.instance_type} → {f.rightsize_to} "
                      f"(~${f.rightsize_savings:,.0f}/mo). Needs a stop; do it in a window:",
                      f"# run aws ec2 stop-instances --instance-ids {f.instance_id}",
                      f"# run aws ec2 wait instance-stopped --instance-ids {f.instance_id}",
                      f"# run aws ec2 modify-instance-attribute --instance-id {f.instance_id} "
                      f"--instance-type Value={f.rightsize_to}",
                      f"# run aws ec2 start-instances --instance-ids {f.instance_id}", ""]
    if not any_cmd:
        lines.append("echo 'No measured schedules to create (upload hourly utilization to get them).'")
    return "\n".join(lines) + "\n"


def write_artifacts(out_dir: str, llm: Optional[LLMAuditResult],
                    infra: Optional[InfraAuditResult], book: PriceBook) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    written: List[str] = []
    if llm and llm.lever_lines:
        path = os.path.join(out_dir, "litellm_router.yaml")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(render_litellm_config(llm, book))
        written.append(path)
    if infra and infra.flagged:
        path = os.path.join(out_dir, "ec2_schedules.sh")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(render_ec2_script(infra))
        try:
            os.chmod(path, 0o755)
        except OSError:
            pass
        written.append(path)
    return written
