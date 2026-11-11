"""Source adapters: turn each provider/gateway export into UsageRecords.

Supported formats (auto-detected):

``generic``          CSV/JSON/JSONL with flexible column names
                     (model|model_used, input_tokens|prompt_tokens, ...).
``litellm``          LiteLLM proxy spend logs (``/spend/logs`` or the
                     ``LiteLLM_SpendLogs`` table export).
``openai-usage``     OpenAI Usage API, completions endpoint
                     (``GET /v1/organization/usage/completions``), raw pages.
``anthropic-usage``  Anthropic Admin API usage report
                     (``GET /v1/organizations/usage_report/messages``).

Every adapter is tolerant: bad rows are skipped and counted in the
:class:`IngestReport`, never silently dropped and never fatal on their own.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from finops.ingest.tabular import DataLoadError, IngestReport, read_json_document, read_rows
from finops.models import TaskSource, UsageRecord

__all__ = ["load_usage_file", "detect_format", "FORMATS", "flatten_messages"]

FORMATS = ("auto", "generic", "litellm", "openai-usage", "anthropic-usage")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _ci_get(row: Dict[str, Any], *names: str) -> Any:
    """Case-insensitive lookup of the first present, non-empty key."""
    lowered = {str(k).strip().lower(): v for k, v in row.items()}
    for name in names:
        value = lowered.get(name)
        if value not in (None, ""):
            return value
    return None


def _num(value: Any, field: str) -> float:
    if value is None or value == "":
        raise ValueError(f"missing {field}")
    try:
        out = float(str(value).replace(",", "")) if isinstance(value, str) else float(value)
    except (TypeError, ValueError):
        raise ValueError(f"non-numeric {field}")
    if out < 0 or out != out:  # negative or NaN
        raise ValueError(f"invalid {field}")
    return out


def _opt_num(value: Any) -> Optional[float]:
    try:
        return _num(value, "x")
    except ValueError:
        return None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "y", "batch")


def _iso_from_epoch(value: Any) -> str:
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OSError):
        return ""


def flatten_messages(value: Any) -> Optional[str]:
    """Flatten a prompt given as text, an OpenAI-style ``messages`` list, or
    a list of content parts into plain text (``role: content`` lines)."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if "messages" in value:
            return flatten_messages(value["messages"])
        if "content" in value:
            return flatten_messages(value["content"])
        return None
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            if isinstance(item, dict) and "role" in item:
                text = flatten_messages(item.get("content"))
                if text:
                    parts.append(f"{item['role']}: {text}")
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts) if parts else None
    return None


def _response_text(value: Any) -> Optional[str]:
    """Extract assistant text from a raw string or an OpenAI/Anthropic
    response object."""
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, dict):
        choices = value.get("choices")
        if isinstance(choices, list) and choices:
            msg = choices[0].get("message") or {}
            return flatten_messages(msg.get("content")) or choices[0].get("text")
        if isinstance(value.get("content"), list):  # Anthropic Messages response
            return flatten_messages(value["content"])
    return None


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def _first_dict(rows: Iterable[Any]) -> Dict[str, Any]:
    for row in rows:
        if isinstance(row, dict):
            return row
    return {}


def _usage_pages(doc: Any) -> Optional[List[Dict[str, Any]]]:
    """Return bucket list if *doc* looks like provider usage-API pages."""
    pages = doc if isinstance(doc, list) else [doc]
    buckets: List[Dict[str, Any]] = []
    for page in pages:
        if not isinstance(page, dict):
            return None
        data = page.get("data")
        if isinstance(data, list) and data and isinstance(data[0], dict) and "results" in data[0]:
            buckets.extend(data)
        elif "results" in page and ("start_time" in page or "starting_at" in page):
            buckets.append(page)   # a bare bucket
        else:
            return None
    return buckets or None


def detect_format(path: str) -> str:
    """Guess the export format from the file's content."""
    if os.path.splitext(path)[1].lower() in (".csv", ".tsv"):
        return "generic"
    doc = read_json_document(path)
    buckets = _usage_pages(doc)
    if buckets:
        sample = _first_dict(r for b in buckets for r in b.get("results", []))
        if "uncached_input_tokens" in sample or "cache_read_input_tokens" in sample:
            return "anthropic-usage"
        return "openai-usage"
    rows = doc if isinstance(doc, list) else (doc.get("data") or doc.get("logs") or [])
    sample = _first_dict(rows)
    keys = {str(k).lower() for k in sample}
    if {"prompt_tokens", "completion_tokens"} <= keys and (
            "spend" in keys or "starttime" in keys or "request_id" in keys):
        return "litellm"
    return "generic"


# ---------------------------------------------------------------------------
# Generic CSV / JSON
# ---------------------------------------------------------------------------

_MODEL = ("model", "model_used", "model_name", "model_id")
_INPUT = ("input_tokens", "prompt_tokens", "tokens_in", "input")
_OUTPUT = ("output_tokens", "completion_tokens", "tokens_out", "output")
_CACHED = ("cached_input_tokens", "cached_tokens", "cache_read_input_tokens",
           "input_cached_tokens")
_CACHE_WRITE = ("cache_write_tokens", "cache_creation_input_tokens")
_TIME = ("timestamp", "time", "created_at", "start_time", "starttime", "date")


def _generic_row(row: Dict[str, Any]) -> UsageRecord:
    model = _ci_get(row, *_MODEL)
    if not model:
        raise ValueError("missing model")
    task = _ci_get(row, "task_type", "task")
    raw_prompt = _ci_get(row, "messages", "prompt", "input_text")
    prompt = flatten_messages(raw_prompt)
    rec = UsageRecord(
        model=str(model),
        input_tokens=int(_num(_ci_get(row, *_INPUT), "input_tokens")),
        output_tokens=int(_num(_ci_get(row, *_OUTPUT), "output_tokens")),
        cached_input_tokens=int(_opt_num(_ci_get(row, *_CACHED)) or 0),
        cache_write_tokens=int(_opt_num(_ci_get(row, *_CACHE_WRITE)) or 0),
        cache_reported=any(str(k).strip().lower() in _CACHED for k in row),
        timestamp=str(_ci_get(row, *_TIME) or ""),
        requests=int(_opt_num(_ci_get(row, "requests", "num_requests", "request_count")) or 1),
        batch=_truthy(_ci_get(row, "batch", "is_batch") or False),
        task_type=str(task).strip().lower() if task else "",
        task_source=TaskSource.PROVIDED if task else TaskSource.UNKNOWN,
        team=str(_ci_get(row, "team") or "").strip(),
        feature=str(_ci_get(row, "feature") or "").strip(),
        customer_id=str(_ci_get(row, "customer_id", "customer", "end_user") or "").strip(),
        project=str(_ci_get(row, "project", "project_id") or "").strip(),
        api_key=str(_ci_get(row, "api_key_id", "api_key_alias") or "").strip(),
        request_id=str(_ci_get(row, "request_id", "id") or ""),
        actual_spend=_opt_num(_ci_get(row, "actual_spend", "spend", "cost", "cost_usd")),
        prompt=prompt,
        response=_response_text(_ci_get(row, "response", "output_text", "completion")),
        source="generic",
    )
    if isinstance(raw_prompt, list):
        rec.extra["messages"] = raw_prompt   # kept verbatim for faithful replay
    return rec


# ---------------------------------------------------------------------------
# LiteLLM
# ---------------------------------------------------------------------------

def _litellm_row(entry: Dict[str, Any]) -> UsageRecord:
    model = entry.get("model") or entry.get("model_id") or entry.get("model_group")
    if not model:
        raise ValueError("missing model")
    metadata = entry.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    tags = (entry.get("request_tags") or metadata.get("tags")
            or metadata.get("tags_to_save") or [])
    if isinstance(tags, str):
        tags = [tags]

    def tag(prefix: str) -> str:
        for t in tags:
            if isinstance(t, str) and t.lower().startswith(prefix + ":"):
                return t.split(":", 1)[1].strip()
        return ""

    task = (metadata.get("task_type")
            or (metadata.get("litellm_metadata") or {}).get("task_type")
            or tag("task_type") or tag("task"))
    usage = metadata.get("usage_object") or {}
    cached = (_opt_num((usage.get("prompt_tokens_details") or {}).get("cached_tokens"))
              or _opt_num(usage.get("cache_read_input_tokens")) or 0)
    writes = _opt_num(usage.get("cache_creation_input_tokens")) or 0
    rec = UsageRecord(
        model=str(model),
        input_tokens=int(_num(entry.get("prompt_tokens", entry.get("input_tokens")), "prompt_tokens")),
        output_tokens=int(_num(entry.get("completion_tokens", entry.get("output_tokens")),
                               "completion_tokens")),
        cached_input_tokens=int(cached),
        cache_write_tokens=int(writes),
        cache_reported=bool(usage),
        timestamp=str(entry.get("startTime") or entry.get("start_time") or entry.get("date") or ""),
        task_type=str(task).strip().lower() if task else "",
        task_source=TaskSource.PROVIDED if task else TaskSource.UNKNOWN,
        team=str(metadata.get("team") or metadata.get("user_api_key_team_alias")
                 or tag("team") or "").strip(),
        feature=str(metadata.get("feature") or tag("feature") or "").strip(),
        customer_id=str(entry.get("end_user") or metadata.get("customer_id")
                        or tag("customer") or "").strip(),
        api_key=str(metadata.get("user_api_key_alias") or entry.get("api_key") or "")[:64],
        request_id=str(entry.get("request_id") or ""),
        actual_spend=_opt_num(entry.get("spend")),
        prompt=flatten_messages(entry.get("messages") or entry.get("prompt")),
        response=_response_text(entry.get("response")),
        source="litellm",
    )
    if isinstance(entry.get("messages"), list):
        rec.extra["messages"] = entry["messages"]
    return rec


# ---------------------------------------------------------------------------
# Provider usage APIs (aggregated buckets)
# ---------------------------------------------------------------------------

def _openai_rows(buckets: List[Dict[str, Any]], report: IngestReport) -> List[UsageRecord]:
    out: List[UsageRecord] = []
    for bucket in buckets:
        start = _iso_from_epoch(bucket.get("start_time"))
        end = _iso_from_epoch(bucket.get("end_time"))
        for res in bucket.get("results") or []:
            report.rows_read += 1
            try:
                model = res.get("model")
                if not model:
                    # Usage API omits model unless group_by=model was requested.
                    raise ValueError("no model (export with group_by=model)")
                out.append(UsageRecord(
                    model=str(model),
                    input_tokens=int(_num(res.get("input_tokens"), "input_tokens")),
                    output_tokens=int(_num(res.get("output_tokens"), "output_tokens")),
                    cached_input_tokens=int(_opt_num(res.get("input_cached_tokens")) or 0),
                    cache_reported=True,
                    requests=int(_opt_num(res.get("num_model_requests")) or 0),
                    batch=bool(res.get("batch")),
                    timestamp=start, period_end=end,
                    project=str(res.get("project_id") or ""),
                    api_key=str(res.get("api_key_id") or ""),
                    customer_id=str(res.get("user_id") or ""),
                    source="openai-usage",
                ))
            except (ValueError, TypeError) as exc:
                report.skip(str(exc))
    return out


def _anthropic_rows(buckets: List[Dict[str, Any]], report: IngestReport) -> List[UsageRecord]:
    out: List[UsageRecord] = []
    for bucket in buckets:
        start = str(bucket.get("starting_at") or "")
        end = str(bucket.get("ending_at") or "")
        for res in bucket.get("results") or []:
            report.rows_read += 1
            try:
                model = res.get("model")
                if not model:
                    raise ValueError("no model (export with group_by[]=model)")
                uncached = _num(res.get("uncached_input_tokens", 0), "uncached_input_tokens")
                cache_read = _opt_num(res.get("cache_read_input_tokens")) or 0
                creation = res.get("cache_creation") or {}
                writes = sum(_opt_num(v) or 0 for v in creation.values()) if isinstance(
                    creation, dict) else (_opt_num(creation) or 0)
                ctx = str(res.get("context_window") or "").lower()
                out.append(UsageRecord(
                    model=str(model),
                    input_tokens=int(uncached + cache_read + writes),
                    output_tokens=int(_num(res.get("output_tokens"), "output_tokens")),
                    cached_input_tokens=int(cache_read),
                    cache_write_tokens=int(writes),
                    cache_reported=True,
                    requests=0,  # not reported by this endpoint
                    batch=str(res.get("service_tier") or "").lower() == "batch",
                    long_context=(None if not ctx else not ctx.startswith("0-")),
                    timestamp=start, period_end=end,
                    project=str(res.get("workspace_id") or ""),
                    api_key=str(res.get("api_key_id") or ""),
                    source="anthropic-usage",
                ))
            except (ValueError, TypeError) as exc:
                report.skip(str(exc))
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _row_loop(rows: List[Any], fn: Callable[[Dict[str, Any]], UsageRecord],
              report: IngestReport) -> List[UsageRecord]:
    out: List[UsageRecord] = []
    for row in rows:
        report.rows_read += 1
        if not isinstance(row, dict):
            report.skip("not an object")
            continue
        try:
            out.append(fn(row))
        except (ValueError, TypeError) as exc:
            report.skip(str(exc))
    return out


def load_usage_file(path: str, fmt: str = "auto") -> Tuple[List[UsageRecord], IngestReport]:
    """Load one usage export into normalized records.

    Raises:
        DataLoadError: unreadable file, unknown format, or zero usable rows.
    """
    if fmt not in FORMATS:
        raise DataLoadError(f"Unknown format '{fmt}'. Choose one of {', '.join(FORMATS)}.")
    if fmt == "auto":
        try:
            fmt = detect_format(path)
        except ValueError as exc:  # JSONDecodeError subclasses ValueError
            raise DataLoadError(f"Invalid JSON in '{path}': {exc}") from exc
    report = IngestReport(path=path, fmt=fmt)

    if fmt in ("openai-usage", "anthropic-usage"):
        buckets = _usage_pages(read_json_document(path))
        if not buckets:
            raise DataLoadError(f"'{path}' is not a {fmt} export (no data[].results).")
        records = (_openai_rows if fmt == "openai-usage" else _anthropic_rows)(buckets, report)
    else:
        rows = read_rows(path)
        records = _row_loop(rows, _litellm_row if fmt == "litellm" else _generic_row, report)

    report.records = len(records)
    if not records:
        reasons = ", ".join(r for r, _ in report.skip_reasons.most_common(3)) or "no rows"
        raise DataLoadError(f"No usable records in '{path}' ({reasons}).")
    return records, report
