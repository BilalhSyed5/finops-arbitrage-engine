"""Low-level file reading shared by every adapter.

Reads CSV, TSV, JSON (array, or object wrapping an array) and JSONL into a
list of ``dict`` rows. Format is chosen by content, not just extension, so a
``.json`` file that is really JSONL still loads.
"""

from __future__ import annotations

import csv
import io
import json
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = ["DataLoadError", "IngestReport", "read_rows", "read_json_document"]

_WRAPPER_KEYS = ("records", "data", "logs", "instances", "items", "results")


class DataLoadError(Exception):
    """An input file cannot be read, or contains nothing usable."""


@dataclass
class IngestReport:
    """What happened while loading one file. Surfaced in the report so users
    can see exactly how much of their data was used."""

    path: str
    fmt: str = ""
    rows_read: int = 0
    records: int = 0
    skipped: int = 0
    skip_reasons: Counter = field(default_factory=Counter)
    warnings: List[str] = field(default_factory=list)

    def skip(self, reason: str) -> None:
        self.skipped += 1
        self.skip_reasons[reason] += 1

    def summary(self) -> str:
        text = (f"{os.path.basename(self.path)} [{self.fmt}]: "
                f"{self.records:,} records from {self.rows_read:,} rows")
        if self.skipped:
            top = ", ".join(f"{r} ×{n}" for r, n in self.skip_reasons.most_common(3))
            text += f"; skipped {self.skipped:,} ({top})"
        return text


def _open_text(path: str) -> str:
    if not os.path.isfile(path):
        raise DataLoadError(f"Input file not found: '{path}'")
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as handle:
            return handle.read()
    except (OSError, UnicodeDecodeError) as exc:
        raise DataLoadError(f"Cannot read '{path}': {exc}") from exc


def read_json_document(path: str) -> Any:
    """Parse a JSON or JSONL file and return the raw document.

    JSONL becomes a list of objects. Raises :class:`DataLoadError` with the
    line number on malformed JSONL.
    """
    text = _open_text(path)
    stripped = text.lstrip()
    if not stripped:
        raise DataLoadError(f"Input file '{path}' is empty.")
    if stripped[0] in "[{":
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            if stripped[0] == "[":
                raise
            # Fall through: probably JSONL whose first line is an object.
    rows: List[Any] = []
    for lineno, line in enumerate(stripped.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise DataLoadError(f"Invalid JSON on line {lineno} of '{path}': {exc}") from exc
    return rows


def _unwrap(doc: Any, path: str) -> List[Dict[str, Any]]:
    if isinstance(doc, dict):
        for key in _WRAPPER_KEYS:
            if isinstance(doc.get(key), list):
                doc = doc[key]
                break
        else:
            raise DataLoadError(
                f"'{path}': JSON object has no list under any of {list(_WRAPPER_KEYS)}.")
    if not isinstance(doc, list):
        raise DataLoadError(f"'{path}': expected a list of records.")
    return doc


def read_rows(path: str, fmt: Optional[str] = None) -> List[Dict[str, Any]]:
    """Read *path* into a list of dict rows (non-dict entries are kept so the
    caller can count them as malformed)."""
    ext = os.path.splitext(path)[1].lower()
    fmt = fmt or ("csv" if ext in (".csv", ".tsv", ".txt") else "json")
    if fmt == "csv":
        text = _open_text(path)
        if not text.strip():
            raise DataLoadError(f"Input file '{path}' is empty.")
        dialect: Any = "excel-tab" if ext == ".tsv" else "excel"
        reader = csv.DictReader(io.StringIO(text), dialect=dialect)
        if not reader.fieldnames:
            raise DataLoadError(f"'{path}': CSV has no header row.")
        rows = []
        for row in reader:
            # Normalize headers: " Model_Used " -> "model_used"; drop overflow cols.
            rows.append({(k or "").strip().lower(): (v.strip() if isinstance(v, str) else v)
                         for k, v in row.items() if k is not None})
        return rows
    try:
        doc = read_json_document(path)
    except json.JSONDecodeError as exc:
        raise DataLoadError(f"Invalid JSON in '{path}': {exc}") from exc
    return _unwrap(doc, path)
