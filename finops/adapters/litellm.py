"""LiteLLM proxy adapter.

Converts LiteLLM proxy spend logs (from the ``litellm_spend_logs`` table, the
``/spend/logs`` endpoint, or a JSON/JSONL export) into the engine's neutral
record schema.

LiteLLM log entries look like::

    {
      "request_id": "chatcmpl-...",
      "model": "openai/gpt-4o",
      "prompt_tokens": 1240,
      "completion_tokens": 86,
      "spend": 0.0034,
      "startTime": "2025-11-03T09:14:00Z",
      "end_user": "customer-42",
      "api_key": "sk-...",
      "metadata": {"task_type": "classification", "feature": "inbox-triage"}
    }

Mapping notes
-------------
* ``model``            -> ``model_used`` (provider prefixes like
  ``openai/gpt-4o`` are handled downstream by ``resolve_pricing``)
* ``prompt_tokens``    -> ``input_tokens``
* ``completion_tokens``-> ``output_tokens``
* ``startTime`` / ``date`` -> ``timestamp``
* ``metadata.task_type``   -> ``task_type`` (falls back to ``untagged``)
* ``metadata.team`` / ``metadata.feature`` / ``end_user`` ->
  attribution fields for unit-economics rollups
* ``spend`` is *not* used for pricing (list-price math keeps the report
  consistent) but is preserved on records as ``actual_spend`` for future
  reconciliation against provider invoices.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

__all__ = ["load_litellm_records"]


def _normalize_entry(entry: Dict[str, Any]) -> Dict[str, Any] | None:
    """Map one LiteLLM log entry to the neutral schema; None if unusable."""
    model = entry.get("model") or entry.get("model_id")
    prompt = entry.get("prompt_tokens", entry.get("input_tokens"))
    completion = entry.get("completion_tokens", entry.get("output_tokens"))
    if not model or prompt is None or completion is None:
        return None

    metadata = entry.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}

    tags = metadata.get("tags") or metadata.get("tags_to_save") or []
    if isinstance(tags, str):
        tags = [tags]

    def _first_tag(prefix: str) -> str:
        for tag in tags:
            if isinstance(tag, str) and tag.lower().startswith(prefix):
                return tag.split(":", 1)[1].strip()
        return ""

    return {
        "timestamp": str(entry.get("startTime") or entry.get("date") or ""),
        "model_used": str(model),
        "input_tokens": int(float(prompt)),
        "output_tokens": int(float(completion)),
        "task_type": str(
            metadata.get("task_type")
            or (metadata.get("litellm_metadata", {}) or {}).get("task_type")
            or _first_tag("task_type:")
            or "untagged"
        ).strip().lower(),
        "team": str(
            metadata.get("team") or _first_tag("team:") or ""
        ).strip(),
        "feature": str(
            metadata.get("feature") or _first_tag("feature:") or ""
        ).strip(),
        "customer_id": str(
            entry.get("end_user") or metadata.get("customer_id") or ""
        ).strip(),
        "actual_spend": entry.get("spend"),
    }


def load_litellm_records(path: str) -> List[Dict[str, Any]]:
    """Load a LiteLLM spend-log export (JSON array or JSONL) into neutral rows.

    Args:
        path: Path to a ``.json`` (array) or ``.jsonl`` (one entry per line) file.

    Returns:
        Rows ready for ``cost_calculator.analyze_api_costs``.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the file parses but contains no usable entries.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"LiteLLM log file not found: '{path}'")

    entries: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8-sig") as handle:
            head = handle.read(1)
            handle.seek(0)
            if head == "[":  # JSON array export
                entries = json.load(handle)
            else:  # JSONL (one object per line)
                for line in handle:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in '{path}': {exc}") from exc
    except OSError as exc:
        raise ValueError(f"Cannot read '{path}': {exc}") from exc

    rows: List[Dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        row = _normalize_entry(entry)
        if row is not None:
            rows.append(row)

    if not rows:
        raise ValueError(
            f"No usable LiteLLM entries in '{path}' (need model + token counts).")
    return rows