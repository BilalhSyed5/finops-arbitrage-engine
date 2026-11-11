"""Enrichment: attribution mapping and task-type inference.

Real usage exports rarely say *what* a call was for. Two mechanisms close the
gap, in priority order:

1. **Attribution map** (``--attribution map.json``): ordered glob rules that
   map identifiers the provider *does* export (project id, API key, model,
   workspace) to team / feature / customer / task_type. Most companies
   already use one project or key per feature, so a 10-line map usually
   attributes the majority of spend.
2. **Inference** (fallback): keyword rules over logged prompt text, then a
   conservative token-shape rule. Inferred tasks are always labelled as such
   and any savings that depend on them are marked *estimated*.

Example map::

    {"rules": [
      {"match": {"project": "proj_support*"},
       "set":   {"team": "support", "feature": "ticket-triage",
                 "task_type": "classification"}},
      {"match": {"api_key": "key_batch_*"}, "set": {"feature": "nightly-enrichment",
                                                     "batch_candidate": true}}
    ]}
"""

from __future__ import annotations

import fnmatch
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from finops.ingest.tabular import DataLoadError
from finops.models import TaskSource, UsageRecord

__all__ = ["AttributionMap", "load_attribution_map", "apply_attribution", "infer_tasks"]

_MATCHABLE = ("model", "project", "api_key", "team", "feature", "customer_id",
              "source", "task_type")
_SETTABLE = ("team", "feature", "customer_id", "task_type", "batch_candidate")


@dataclass
class AttributionRule:
    match: Dict[str, str]
    set: Dict[str, Any]
    override: bool = False
    hits: int = 0


@dataclass
class AttributionMap:
    rules: List[AttributionRule] = field(default_factory=list)
    path: str = ""


def load_attribution_map(path: str) -> AttributionMap:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError as exc:
        raise DataLoadError(f"Attribution map not found: '{path}'") from exc
    except json.JSONDecodeError as exc:
        raise DataLoadError(f"Invalid JSON in attribution map '{path}': {exc}") from exc
    rules: List[AttributionRule] = []
    for idx, raw in enumerate(data.get("rules", []) if isinstance(data, dict) else []):
        match = raw.get("match") or {}
        sets = raw.get("set") or {}
        bad = [k for k in match if k not in _MATCHABLE and not k.startswith("extra.")]
        bad += [k for k in sets if k not in _SETTABLE]
        if not match or not sets or bad:
            raise DataLoadError(
                f"Attribution rule #{idx + 1} in '{path}' is invalid"
                + (f" (unknown keys: {bad})" if bad else " (needs 'match' and 'set')")
                + f". Match keys: {list(_MATCHABLE)}; set keys: {list(_SETTABLE)}.")
        rules.append(AttributionRule({k: str(v).lower() for k, v in match.items()},
                                     dict(sets), bool(raw.get("override"))))
    if not rules:
        raise DataLoadError(f"Attribution map '{path}' has no rules.")
    return AttributionMap(rules=rules, path=path)


def _field_value(rec: UsageRecord, key: str) -> str:
    if key.startswith("extra."):
        return str(rec.extra.get(key[6:], "")).lower()
    return str(getattr(rec, key, "") or "").lower()


def apply_attribution(records: Sequence[UsageRecord], amap: AttributionMap) -> None:
    """Apply the first matching rule to each record (in place)."""
    for rec in records:
        for rule in amap.rules:
            if all(fnmatch.fnmatchcase(_field_value(rec, k), pattern)
                   for k, pattern in rule.match.items()):
                rule.hits += 1
                for key, value in rule.set.items():
                    if key == "batch_candidate":
                        rec.extra["batch_candidate"] = bool(value)
                        continue
                    if rule.override or not getattr(rec, key):
                        setattr(rec, key, str(value).strip().lower()
                                if key == "task_type" else str(value).strip())
                        if key == "task_type":
                            rec.task_source = TaskSource.MAPPED
                break


# ---------------------------------------------------------------------------
# Task inference
# ---------------------------------------------------------------------------

# Ordered: the first rule whose pattern appears in the prompt wins. Patterns
# look at instructions, so they're matched on the first ~3000 characters.
_PROMPT_RULES: Tuple[Tuple[str, re.Pattern], ...] = tuple(
    (task, re.compile(pattern, re.IGNORECASE)) for task, pattern in (
        ("code_gen", r"\b(write|implement|refactor|fix) (a |the )?(function|class|method|bug|test)|```"),
        ("classification", r"\b(classify|categori[sz]e|which category|sentiment|"
                           r"is this (email |message )?spam|respond with (only )?one of|label (this|the))\b"),
        ("extraction", r"\b(extract|pull out|parse)\b.*\b(fields?|json|entities|values)\b|"
                       r"\breturn (only )?(valid )?json\b"),
        ("summarization", r"\b(summari[sz]e|summary|tl;?dr|condense)\b"),
        ("translation", r"\btranslate\b"),
        ("tagging", r"\b(tag|keywords?) (this|the)\b|\bgenerate tags\b"),
    ))


def _infer_from_prompt(prompt: str) -> Optional[str]:
    head = prompt[:3000]
    for task, pattern in _PROMPT_RULES:
        if pattern.search(head):
            return task
    return None


def infer_tasks(records: Sequence[UsageRecord]) -> Dict[str, int]:
    """Fill in missing task types (in place). Returns counts by method."""
    stats = {"prompt": 0, "shape": 0, "unknown": 0}
    for rec in records:
        if rec.task_type:
            continue
        task = _infer_from_prompt(rec.prompt) if rec.prompt else None
        if task:
            stats["prompt"] += 1
        elif rec.requests > 0 and rec.output_tokens / rec.requests <= 8 \
                and rec.input_tokens / rec.requests >= 100:
            # A handful of output tokens per call on a real prompt is almost
            # always a label/score/yes-no: classify-shaped traffic.
            task = "classification"
            stats["shape"] += 1
        if task:
            rec.task_type = task
            rec.task_source = TaskSource.INFERRED
        else:
            rec.task_type = "unknown"
            rec.task_source = TaskSource.UNKNOWN
            stats["unknown"] += 1
    return stats
