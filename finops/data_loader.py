"""Tolerant, well-validated data ingestion for CSV and JSON inputs.

Both audit modules consume "records" (list of dicts). This module centralizes
parsing so that malformed rows are reported clearly instead of crashing deep
inside the analysis logic.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from typing import Any, Dict, List, Sequence

__all__ = ["load_records", "DataLoadError"]


class DataLoadError(Exception):
    """Raised when an input file cannot be read or parsed."""


def _coerce(value: Any, cast, field: str) -> Any:
    """Best-effort numeric coercion; returns None for empty/unparseable values."""
    if value is None:
        return None
    if isinstance(value, cast):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return cast(text)
    except (ValueError, TypeError):
        raise DataLoadError(
            f"Field '{field}' has non-numeric value '{value}'."
        )


def _normalize_row(row: Dict[str, Any], required: Sequence[str], row_no: int) -> Dict[str, Any]:
    """Validate a single record against required fields and normalize key casing."""
    normalized = {k.strip().lower(): (v.strip() if isinstance(v, str) else v)
                  for k, v in row.items() if k is not None}
    for field in required:
        if field not in normalized or normalized[field] in (None, ""):
            raise DataLoadError(f"Row {row_no}: missing required field '{field}'.")
    return normalized


def load_records(path: str, required_fields: Sequence[str], numeric_fields: Sequence[str] = ()) -> List[Dict[str, Any]]:
    """Load CSV or JSON records from *path* with strict validation.

    Args:
        path: Path to a ``.csv`` or ``.json`` file.
        required_fields: Fields that must be present and non-empty in every row.
        numeric_fields: Fields that must parse to float.

    Returns:
        A list of normalized record dicts.

    Raises:
        DataLoadError: If the file is missing, unreadable, malformed, or empty.
    """
    if not os.path.isfile(path):
        raise DataLoadError(f"Input file not found: '{path}'")

    ext = os.path.splitext(path)[1].lower()
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as handle:
            if ext == ".json":
                raw = json.load(handle)
                if isinstance(raw, dict):  # allow {"records": [...]} wrappers
                    for key in ("records", "data", "instances", "logs"):
                        if key in raw and isinstance(raw[key], list):
                            raw = raw[key]
                            break
                if not isinstance(raw, list):
                    raise DataLoadError(
                        "JSON input must be a list of records (or an object "
                        "containing a 'records'/'data' list)."
                    )
                rows = raw
            else:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None:
                    raise DataLoadError("CSV file has no header row.")
                rows = list(reader)
    except DataLoadError:
        raise
    except json.JSONDecodeError as exc:
        raise DataLoadError(f"Invalid JSON in '{path}': {exc}") from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise DataLoadError(f"Cannot read '{path}': {exc}") from exc

    if not rows:
        raise DataLoadError(f"Input file '{path}' contains no records.")

    records: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows, start=2):  # header is line 1 in CSVs
        try:
            rec = _normalize_row(row, required_fields, idx)
            for field in numeric_fields:
                rec[field] = _coerce(rec[field], float, field)
                if rec[field] is None:
                    raise DataLoadError(f"Row {idx}: missing numeric field '{field}'.")
            records.append(rec)
        except DataLoadError as exc:
            # Tolerant mode: warn and skip bad rows rather than aborting the audit.
            print(f"[warn] Skipping malformed record: {exc}", file=sys.stderr)

    if not records:
        raise DataLoadError(
            f"All records in '{path}' were malformed; nothing to audit."
        )
    return records