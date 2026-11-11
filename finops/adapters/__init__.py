"""Data-source adapters.

Each adapter converts a third-party log format into the engine's neutral
record schema so ``cost_calculator.analyze_api_costs`` can consume it::

    timestamp, model_used, input_tokens, output_tokens, task_type
    (+ optional attribution: team, feature, customer_id)
"""

from finops.adapters import litellm

__all__ = ["litellm"]