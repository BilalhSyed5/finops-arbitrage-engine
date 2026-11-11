"""Versioned, config-driven price book for LLM tokens and EC2 instances.

Design goals
------------
* **Correct model resolution.** Real logs contain names like
  ``openai/gpt-4o-mini-2024-07-18``, ``anthropic.claude-sonnet-4-5-20250929-v1:0``
  (Bedrock) or ``claude-sonnet-4-5@20250929`` (Vertex). We normalize, then
  match exact names/aliases first and fall back to the *longest* known
  prefix. A shorter name never captures a longer one
  (``gpt-4o`` will not swallow ``gpt-4o-mini``).
* **Unknown models are never silently priced.** They are reported as
  unpriced volume so the user can add a price, instead of inflating both the
  bill and the "savings".
* **Every rate that matters is modeled:** cached reads, cache writes, Batch
  API discounts, long-prompt surcharges and announced price changes.
* **Reproducible.** Defaults ship in ``finops/data``; a ``--pricing-config``
  directory merges over them. Each snapshot carries an ``as_of`` date and the
  report warns when it is stale.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Tuple

__all__ = [
    "ModelPrice", "InstancePrice", "PriceBook", "load_pricebook",
    "DEFAULT_DATA_DIR", "normalize_model_name", "PricingError",
]

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
STALE_AFTER_DAYS = 45


class PricingError(Exception):
    """Raised when a pricing file is present but invalid."""


@dataclass(frozen=True)
class ScheduledChange:
    effective: str
    input: float
    output: float
    cached_input: Optional[float] = None
    reason: str = ""


@dataclass(frozen=True)
class ModelPrice:
    """Token prices for one model, USD per 1M tokens."""

    name: str
    provider: str
    tier: str                       # "frontier" | "mid" | "budget"
    input: float
    output: float
    cached_input: Optional[float] = None
    cache_write: Optional[float] = None
    batch_discount: Optional[float] = None   # e.g. 0.5 => batch costs 50% less
    long_context_threshold: Optional[int] = None
    long_context_input_mult: float = 1.0
    long_context_output_mult: float = 1.0
    successor: Optional[str] = None
    status: str = "current"
    notes: str = ""
    aliases: Tuple[str, ...] = ()
    scheduled_change: Optional[ScheduledChange] = None

    @property
    def cached_rate(self) -> float:
        """Cached-read rate; falls back to the full input rate if unpublished
        (conservative: we never assume a discount the provider doesn't offer)."""
        return self.cached_input if self.cached_input is not None else self.input

    @property
    def cache_write_rate(self) -> float:
        return self.cache_write if self.cache_write is not None else self.input

    @property
    def supports_batch(self) -> bool:
        return bool(self.batch_discount)

    @property
    def supports_caching(self) -> bool:
        return self.cached_input is not None and self.cached_input < self.input

    def cost(self, input_tokens: float, output_tokens: float, *,
             cached_input_tokens: float = 0.0, cache_write_tokens: float = 0.0,
             requests: int = 1, batch: bool = False,
             long_context: Optional[bool] = None) -> float:
        """USD cost of a request (or a bucket of ``requests`` requests).

        ``input_tokens`` includes cached and cache-write tokens (see
        :mod:`finops.models`). The long-context surcharge is applied when the
        *average* input per request exceeds the model's threshold, or when
        ``long_context`` says so explicitly (some usage exports report the
        context bucket but not the request count; ``requests=0`` = unknown).
        """
        cached = min(max(0.0, cached_input_tokens), input_tokens)
        writes = min(max(0.0, cache_write_tokens), input_tokens - cached)
        fresh = max(0.0, input_tokens - cached - writes)

        if long_context is None:
            long_context = bool(
                self.long_context_threshold and requests > 0
                and input_tokens / requests > self.long_context_threshold)
        in_mult = out_mult = 1.0
        if long_context and self.long_context_threshold:
            in_mult = self.long_context_input_mult
            out_mult = self.long_context_output_mult

        usd = (fresh * self.input * in_mult
               + cached * self.cached_rate * in_mult
               + writes * self.cache_write_rate * in_mult
               + output_tokens * self.output * out_mult) / 1_000_000.0
        if batch and self.batch_discount:
            usd *= (1.0 - self.batch_discount)
        return usd


@dataclass(frozen=True)
class InstancePrice:
    name: str
    hourly: float
    vcpus: Optional[int] = None
    gpus: Optional[int] = None

    @property
    def family(self) -> str:
        return self.name.split(".", 1)[0]


@dataclass
class PriceBook:
    """A pricing snapshot: model token prices + EC2 hourly rates."""

    models: Dict[str, ModelPrice] = field(default_factory=dict)
    instances: Dict[str, InstancePrice] = field(default_factory=dict)
    as_of: str = ""
    sources: List[str] = field(default_factory=list)
    loaded_from: List[str] = field(default_factory=list)
    _alias_index: Dict[str, str] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------ models
    def _rebuild_index(self) -> None:
        index: Dict[str, str] = {}
        for name, price in self.models.items():
            index[name] = name
            for alias in price.aliases:
                index.setdefault(alias.lower(), name)
        self._alias_index = index

    def get(self, name: str) -> Optional[ModelPrice]:
        """Exact lookup by canonical name or alias (no fuzzy matching)."""
        if not self._alias_index:
            self._rebuild_index()
        canonical = self._alias_index.get(name.lower())
        return self.models.get(canonical) if canonical else None

    def resolve(self, model: str) -> Optional[ModelPrice]:
        """Resolve a raw model string from a log to a price, or ``None``."""
        if not self._alias_index:
            self._rebuild_index()
        for candidate in normalize_model_name(model):
            hit = self._alias_index.get(candidate)
            if hit:
                return self.models[hit]
        # Longest-prefix fallback: "gpt-6-luna-experimental" -> "gpt-6-luna".
        # The remainder must not start with a digit: "claude-opus-5-6" is a
        # *different, newer model*, not a variant of "claude-opus-5".
        base = normalize_model_name(model)[-1]
        best: Optional[str] = None
        for key in self._alias_index:
            if base.startswith(key + "-") and not base[len(key) + 1:][:1].isdigit():
                if best is None or len(key) > len(best):
                    best = key
        return self.models[self._alias_index[best]] if best else None

    def cheapest_in(self, provider: str, tier: str) -> Optional[ModelPrice]:
        pool = [m for m in self.models.values()
                if m.provider == provider and m.tier == tier and m.status != "legacy"]
        return min(pool, key=lambda m: m.input + m.output, default=None)

    # ------------------------------------------------------------- instances
    def instance(self, instance_type: str) -> Optional[InstancePrice]:
        return self.instances.get(instance_type.strip().lower())

    def siblings(self, instance_type: str) -> List[InstancePrice]:
        """Other sizes in the same family, cheapest first."""
        family = instance_type.split(".", 1)[0].lower()
        return sorted((p for p in self.instances.values() if p.family == family),
                      key=lambda p: p.hourly)

    # ------------------------------------------------------------- freshness
    def age_days(self, today: Optional[date] = None) -> Optional[int]:
        try:
            snap = date.fromisoformat(self.as_of)
        except ValueError:
            return None
        return ((today or date.today()) - snap).days

    def is_stale(self, today: Optional[date] = None) -> bool:
        age = self.age_days(today)
        return age is not None and age > STALE_AFTER_DAYS


# ---------------------------------------------------------------------------
# Model-name normalization
# ---------------------------------------------------------------------------

_DATE_SUFFIX = re.compile(
    r"(?:[-@](?:\d{4}-\d{2}-\d{2}|\d{8}|\d{6}|\d{4})|-latest)$")
_BEDROCK_VERSION = re.compile(r"-v\d+(?::\d+)?$")
_REGION_PREFIX = re.compile(r"^(?:us|eu|apac|global|jp|au)\.")
_VENDOR_PREFIX = re.compile(r"^(?:anthropic|openai|google|meta|mistral|amazon)\.")


def normalize_model_name(raw: str) -> List[str]:
    """Return lookup candidates for a raw model string, most specific first.

    >>> normalize_model_name("openai/gpt-4o-mini-2024-07-18")
    ['gpt-4o-mini-2024-07-18', 'gpt-4o-mini']
    >>> normalize_model_name("us.anthropic.claude-sonnet-4-5-20250929-v1:0")[-1]
    'claude-sonnet-4-5'
    """
    key = str(raw).strip().lower()
    # Provider/router prefixes: "openai/gpt-4o", "bedrock/anthropic.claude-...",
    # "models/gemini-2.5-pro", "openrouter/anthropic/claude-...".
    if "/" in key:
        key = key.rsplit("/", 1)[-1]
    key = _REGION_PREFIX.sub("", key)
    key = _VENDOR_PREFIX.sub("", key)
    candidates = [key]
    stripped = _BEDROCK_VERSION.sub("", key)
    # Strip date/snapshot suffixes repeatedly ("...-20250929-v1:0").
    for _ in range(3):
        nxt = _DATE_SUFFIX.sub("", stripped)
        if nxt == stripped:
            break
        stripped = nxt
    if stripped != key:
        candidates.append(stripped)
    # Dot/hyphen version spelling: "claude-opus-5.5" <-> "claude-opus-5-5".
    if stripped.startswith("claude-"):
        hyphenated = re.sub(r"(\d)\.(\d)", r"\1-\2", stripped)
        if hyphenated not in candidates:
            candidates.append(hyphenated)
    return candidates


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _read_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise PricingError(f"Invalid JSON in pricing file '{path}': {exc}") from exc
    if not isinstance(data, dict):
        raise PricingError(f"Pricing file '{path}' must contain a JSON object.")
    return data


def _parse_model(name: str, spec: dict, path: str) -> ModelPrice:
    try:
        lc = spec.get("long_context") or {}
        sc = spec.get("scheduled_change")
        return ModelPrice(
            name=name,
            provider=str(spec.get("provider", "unknown")).lower(),
            tier=str(spec.get("tier", "frontier")).lower(),
            input=float(spec["input"]),
            output=float(spec["output"]),
            cached_input=(float(spec["cached_input"])
                          if spec.get("cached_input") is not None else None),
            cache_write=(float(spec["cache_write"])
                         if spec.get("cache_write") is not None else None),
            batch_discount=(float(spec["batch_discount"])
                            if spec.get("batch_discount") else None),
            long_context_threshold=int(lc["threshold"]) if lc.get("threshold") else None,
            long_context_input_mult=float(lc.get("input_multiplier", 1.0)),
            long_context_output_mult=float(lc.get("output_multiplier", 1.0)),
            successor=(str(spec["successor"]).lower() if spec.get("successor") else None),
            status=str(spec.get("status", "current")),
            notes=str(spec.get("notes", "")),
            aliases=tuple(str(a).lower() for a in spec.get("aliases", [])),
            scheduled_change=(ScheduledChange(
                effective=str(sc["effective"]), input=float(sc["input"]),
                output=float(sc["output"]),
                cached_input=(float(sc["cached_input"])
                              if sc.get("cached_input") is not None else None),
                reason=str(sc.get("reason", ""))) if sc else None),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise PricingError(
            f"{path}: model '{name}' is malformed ({exc!r}); "
            "expected at least {\"input\": <float>, \"output\": <float>}.") from exc


def _models_from_file(data: dict, path: str) -> Dict[str, ModelPrice]:
    models: Dict[str, ModelPrice] = {}
    if "models" in data:
        for name, spec in data["models"].items():
            models[name.lower()] = _parse_model(name.lower(), spec, path)
    # Legacy v1 format: {"frontier": {"gpt-4o": [in, out]}, "budget": {...}}
    for tier_key, tier in (("frontier", "frontier"), ("budget", "budget")):
        for name, pair in (data.get(tier_key) or {}).items():
            if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
                raise PricingError(
                    f"{path}: legacy entry '{name}' must be [input, output].")
            models[name.lower()] = ModelPrice(
                name=name.lower(), provider="unknown", tier=tier,
                input=float(pair[0]), output=float(pair[1]))
    return models


def _instances_from_file(data: dict, path: str) -> Dict[str, InstancePrice]:
    out: Dict[str, InstancePrice] = {}
    for name, spec in (data.get("instances") or {}).items():
        key = name.strip().lower()
        try:
            if isinstance(spec, (int, float)):
                out[key] = InstancePrice(key, float(spec))
            else:
                out[key] = InstancePrice(
                    key, float(spec["hourly"]),
                    int(spec["vcpus"]) if spec.get("vcpus") is not None else None,
                    int(spec["gpus"]) if spec.get("gpus") is not None else None)
        except (KeyError, TypeError, ValueError) as exc:
            raise PricingError(f"{path}: instance '{name}' is malformed ({exc!r}).") from exc
    return out


def load_pricebook(override_dir: Optional[str] = None) -> PriceBook:
    """Load the bundled price book, then merge ``override_dir`` files over it.

    Override files use the same names (``model_pricing.json``,
    ``ec2_pricing.json``) and only need the entries you want to change
    (e.g. your negotiated rates).

    Raises:
        PricingError: if a pricing file exists but is malformed. A bad price
            file must stop the audit; silently falling back to defaults would
            produce a confident report with the wrong numbers.
    """
    book = PriceBook()
    dirs = [DEFAULT_DATA_DIR]
    if override_dir:
        if not os.path.isdir(override_dir):
            raise PricingError(f"Pricing config directory not found: '{override_dir}'")
        dirs.append(override_dir)

    for directory in dirs:
        model_path = os.path.join(directory, "model_pricing.json")
        if os.path.isfile(model_path):
            data = _read_json(model_path)
            book.models.update(_models_from_file(data, model_path))
            meta = data.get("_meta", {})
            if meta.get("as_of"):
                book.as_of = str(meta["as_of"])
            book.sources.extend(meta.get("sources", []))
            book.loaded_from.append(model_path)
        ec2_path = os.path.join(directory, "ec2_pricing.json")
        if os.path.isfile(ec2_path):
            data = _read_json(ec2_path)
            book.instances.update(_instances_from_file(data, ec2_path))
            book.loaded_from.append(ec2_path)

    # Validate successor links so a typo can't silently disable a lever.
    for price in book.models.values():
        if price.successor and price.successor not in book.models:
            raise PricingError(
                f"Model '{price.name}' names unknown successor '{price.successor}'.")
    book._rebuild_index()
    return book
