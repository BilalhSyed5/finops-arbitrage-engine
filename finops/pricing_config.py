"""Versioned, config-driven pricing.

Pricing tables live in JSON files (default: ``config/``) instead of hardcoded
Python so that price refreshes are a data change, not a code change, and so
reports are reproducible: commit a pricing snapshot alongside your logs.

File format
-----------
``model_pricing.json``::

    {
      "frontier": {"gpt-4o": [2.50, 10.00]},
      "budget":   {"llama-3.1-8b-instruct": [0.05, 0.08]}
    }

``ec2_pricing.json``::

    {"instances": {"g5.12xlarge": 5.672}}

Prices are USD per 1M tokens as ``[input, output]`` and USD per hour for EC2.
Any file present in the config directory *merges over* the built-in defaults,
so you can override a handful of SKUs without copying whole tables.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Tuple

__all__ = ["Pricing", "load_pricing", "DEFAULT_CONFIG_DIR"]

DEFAULT_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")

# --- Built-in fallbacks (used when no config file overrides them) ----------
DEFAULT_FRONTIER_PRICING: Dict[str, Tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "o1": (15.00, 60.00),
    "o3-mini": (1.10, 4.40),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-7-sonnet": (3.00, 15.00),
    "claude-3-5-haiku": (0.80, 4.00),
    "gemini-1.5-pro": (1.25, 5.00),
    "gemini-1.5-flash": (0.075, 0.30),
}

DEFAULT_BUDGET_PRICING: Dict[str, Tuple[float, float]] = {
    "llama-3.1-8b-instruct": (0.05, 0.08),
    "llama-3.1-70b-instruct": (0.88, 0.88),
    "llama-3.3-70b-instruct": (0.88, 0.88),
    "mixtral-8x7b-instruct": (0.24, 0.24),
    "qwen-2.5-72b-instruct": (1.20, 1.20),
    "deepseek-v3": (0.27, 1.10),
}

DEFAULT_EC2_PRICING: Dict[str, float] = {
    "m5.xlarge": 0.192, "m5.2xlarge": 0.384, "m5.4xlarge": 0.768,
    "m5.12xlarge": 2.304, "m6i.4xlarge": 0.768, "m6i.8xlarge": 1.536,
    "t3.large": 0.0832, "t3.xlarge": 0.1664, "t3.2xlarge": 0.3328,
    "c5.2xlarge": 0.34, "c5.4xlarge": 0.68, "c5.9xlarge": 1.53,
    "r5.xlarge": 0.252, "r5.2xlarge": 0.504, "r5.4xlarge": 1.008,
    "g5.xlarge": 1.006, "g5.2xlarge": 1.212, "g5.4xlarge": 2.03,
    "g5.12xlarge": 5.672, "g5.48xlarge": 22.33,
    "g6.xlarge": 0.804, "g6.12xlarge": 4.715,
    "p4d.24xlarge": 32.77, "p4de.24xlarge": 40.96,
    "p5.48xlarge": 98.32,
    "p3.2xlarge": 3.06, "p3.8xlarge": 12.24,
}


@dataclass
class Pricing:
    """Immutable snapshot of pricing tables used by the audit modules."""

    frontier: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: dict(DEFAULT_FRONTIER_PRICING))
    budget: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: dict(DEFAULT_BUDGET_PRICING))
    ec2_hourly: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_EC2_PRICING))
    source: str = "built-in defaults"


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _merge_model_table(base: Dict[str, Tuple[float, float]],
                       overrides: dict, label: str) -> Dict[str, Tuple[float, float]]:
    merged = dict(base)
    for name, price in (overrides or {}).items():
        if isinstance(price, (list, tuple)) and len(price) == 2:
            merged[str(name).lower()] = (float(price[0]), float(price[1]))
        else:
            print(f"[warn] pricing config: ignoring malformed {label} entry "
                  f"'{name}' (expected [input, output]).")
    return merged


def load_pricing(config_dir: str | None = None) -> Pricing:
    """Build a :class:`Pricing` snapshot.

    Args:
        config_dir: Directory containing ``model_pricing.json`` and/or
            ``ec2_pricing.json``. Defaults to the project's ``config/`` dir.
            Missing files are fine — built-in defaults apply.

    Returns:
        A merged :class:`Pricing` snapshot.
    """
    config_dir = config_dir or DEFAULT_CONFIG_DIR
    pricing = Pricing()

    model_path = os.path.join(config_dir, "model_pricing.json")
    if os.path.isfile(model_path):
        try:
            data = _read_json(model_path)
            pricing.frontier = _merge_model_table(pricing.frontier, data.get("frontier"), "frontier")
            pricing.budget = _merge_model_table(pricing.budget, data.get("budget"), "budget")
            pricing.source = model_path
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[warn] Could not read {model_path}: {exc}. Using defaults.")

    ec2_path = os.path.join(config_dir, "ec2_pricing.json")
    if os.path.isfile(ec2_path):
        try:
            data = _read_json(ec2_path)
            overrides = data.get("instances", {})
            merged = dict(pricing.ec2_hourly)
            for name, rate in overrides.items():
                merged[str(name).lower()] = float(rate)
            pricing.ec2_hourly = merged
            pricing.source = ec2_path
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
            print(f"[warn] Could not read {ec2_path}: {exc}. Using defaults.")

    return pricing