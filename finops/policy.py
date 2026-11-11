"""Optimization policy: which levers are allowed and how they're tuned.

Shipped defaults live in ``finops/data/routing_policy.json``. A user policy
file (``--policy``) is deep-merged over them, so a team can e.g. allow
cross-provider routing or add a task type without copying the whole file.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from finops.pricing import DEFAULT_DATA_DIR, PriceBook, PricingError

__all__ = ["Policy", "load_policy"]


@dataclass
class Policy:
    routable_tasks: Dict[str, Dict[str, str]] = field(default_factory=dict)
    cross_provider: bool = False
    model_refresh: bool = True
    caching: Dict[str, float] = field(default_factory=dict)
    batch_tasks: List[str] = field(default_factory=list)
    batch_feature_patterns: List[str] = field(default_factory=list)
    proof_pass_threshold: float = 0.9
    proof_min_samples: int = 30
    source: List[str] = field(default_factory=list)

    def targets_for(self, task: str) -> Dict[str, str]:
        return self.routable_tasks.get(task, {})

    def cache_param(self, key: str) -> float:
        return float(self.caching[key])


def _deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_policy(path: Optional[str] = None, pricebook: Optional[PriceBook] = None) -> Policy:
    """Load the default policy, merge *path* over it, and validate targets
    against *pricebook* (an unpriced target would silently disable a lever)."""
    files = [os.path.join(DEFAULT_DATA_DIR, "routing_policy.json")]
    if path:
        files.append(path)
    merged: Dict[str, Any] = {}
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as handle:
                merged = _deep_merge(merged, json.load(handle))
        except FileNotFoundError as exc:
            raise PricingError(f"Policy file not found: '{fp}'") from exc
        except json.JSONDecodeError as exc:
            raise PricingError(f"Invalid JSON in policy file '{fp}': {exc}") from exc

    routable: Dict[str, Dict[str, str]] = {}
    for task, spec in (merged.get("routable_tasks") or {}).items():
        if spec is None:          # allow a user policy to remove a task: "summarization": null
            continue
        targets = {k.lower(): str(v).lower() for k, v in (spec.get("targets") or {}).items()}
        if pricebook:
            for provider, model in targets.items():
                if not pricebook.get(model):
                    raise PricingError(
                        f"Policy routes '{task}' ({provider}) to '{model}', "
                        "which has no price in the price book.")
        routable[task.lower()] = targets

    batch = merged.get("batch") or {}
    proof = merged.get("proof") or {}
    return Policy(
        routable_tasks=routable,
        cross_provider=bool(merged.get("cross_provider", False)),
        model_refresh=bool(merged.get("model_refresh", True)),
        caching=dict(merged.get("caching") or {}),
        batch_tasks=[t.lower() for t in batch.get("tasks", [])],
        batch_feature_patterns=[p.lower() for p in batch.get("feature_patterns", [])],
        proof_pass_threshold=float(proof.get("pass_threshold", 0.9)),
        proof_min_samples=int(proof.get("min_samples", 30)),
        source=files,
    )
