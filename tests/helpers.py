"""Shared fixtures for the test suite (stdlib unittest only)."""

import os
import tempfile
from datetime import date

from finops.models import TaskSource, UsageRecord
from finops.policy import load_policy
from finops.pricing import load_pricebook

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE = os.path.join(ROOT, "sample_data")
TODAY = date(2026, 11, 11)
BOOK = load_pricebook()
POLICY = load_policy(None, BOOK)


def rec(model, inp, out, task="", ts="2026-11-01T00:00:00", **kw):
    """Build a UsageRecord; task given => provided task type."""
    src = kw.pop("task_source", TaskSource.PROVIDED if task else TaskSource.UNKNOWN)
    return UsageRecord(model=model, input_tokens=inp, output_tokens=out, timestamp=ts,
                       task_type=task, task_source=src, **kw)


def write_tmp(text, suffix):
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path
