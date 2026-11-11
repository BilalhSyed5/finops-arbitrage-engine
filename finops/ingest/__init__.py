"""Data ingestion: provider/gateway exports -> normalized UsageRecords."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from finops.ingest.adapters import FORMATS, detect_format, flatten_messages, load_usage_file
from finops.ingest.enrich import (AttributionMap, apply_attribution, infer_tasks,
                                  load_attribution_map)
from finops.ingest.tabular import DataLoadError, IngestReport, read_rows
from finops.models import UsageRecord

__all__ = [
    "DataLoadError", "IngestReport", "FORMATS", "detect_format", "load_usage_file",
    "load_usage", "load_attribution_map", "apply_attribution", "infer_tasks",
    "read_rows", "flatten_messages", "AttributionMap",
]


def load_usage(paths: Sequence[str], fmt: str = "auto",
               attribution: Optional[AttributionMap] = None,
               ) -> Tuple[List[UsageRecord], List[IngestReport], dict]:
    """Load and enrich one or more usage files.

    Returns ``(records, per-file reports, task-inference stats)``.
    """
    records: List[UsageRecord] = []
    reports: List[IngestReport] = []
    for path in paths:
        recs, report = load_usage_file(path, fmt)
        records.extend(recs)
        reports.append(report)
    if attribution:
        apply_attribution(records, attribution)
    stats = infer_tasks(records)
    return records, reports, stats
