"""Core data types shared by every module.

Everything the engine ingests is normalized into :class:`UsageRecord` (LLM
usage) so the analysis code never has to know which tool produced the data.

Token accounting convention (important, providers differ):

* ``input_tokens``        -- ALL prompt tokens, including cached reads and
                             cache writes.
* ``cached_input_tokens`` -- the subset of ``input_tokens`` served from a
                             prompt cache (billed at the cached-read rate).
* ``cache_write_tokens``  -- the subset of ``input_tokens`` written to a
                             cache (billed at the cache-write rate).
* ``output_tokens``       -- completion tokens, including reasoning tokens.

A record may represent one request or an aggregated bucket of ``requests``
calls (e.g. one row of the OpenAI Usage API). ``requests == 0`` means the
source did not report a request count (e.g. Anthropic usage reports).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

__all__ = ["UsageRecord", "Confidence", "TaskSource"]


class Confidence(str, Enum):
    """How a dollar figure was derived. Shown next to every number we report.

    VERIFIED  -- backed by a quality eval on the customer's own traffic.
    MEASURED  -- computed directly from the customer's data at list prices.
    ESTIMATED -- depends on an assumption (documented in the report).
    """

    VERIFIED = "verified"
    MEASURED = "measured"
    ESTIMATED = "estimated"


class TaskSource(str, Enum):
    PROVIDED = "provided"      # task_type column / metadata on the record
    MAPPED = "mapped"          # attribution map rule
    INFERRED = "inferred"      # heuristic (prompt keywords or token shape)
    UNKNOWN = "unknown"


@dataclass
class UsageRecord:
    """One normalized LLM usage record (a request or an aggregated bucket)."""

    model: str
    input_tokens: int
    output_tokens: int
    timestamp: str = ""
    period_end: str = ""            # bucket end time, when the source has one
    requests: int = 1
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    cache_reported: bool = False          # source reports cache usage at all
    batch: bool = False
    long_context: Optional[bool] = None   # explicit long-prompt flag, if known
    task_type: str = ""
    task_source: TaskSource = TaskSource.UNKNOWN
    team: str = ""
    feature: str = ""
    customer_id: str = ""
    project: str = ""
    api_key: str = ""
    request_id: str = ""
    actual_spend: Optional[float] = None   # what the provider/gateway billed
    prompt: Optional[str] = None           # flattened prompt text, if logged
    response: Optional[str] = None         # model output text, if logged
    source: str = ""                       # which adapter produced it
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def uncached_input_tokens(self) -> int:
        return max(0, self.input_tokens - self.cached_input_tokens
                   - self.cache_write_tokens)

    @property
    def avg_input_per_request(self) -> float:
        return self.input_tokens / max(1, self.requests)
