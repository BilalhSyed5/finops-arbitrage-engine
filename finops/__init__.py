"""FinOps Arbitrage Engine.

A lightweight diagnostic toolkit that ingests LLM API usage logs and cloud
infrastructure metrics to quantify capital waste and compute arbitrage savings.

Modules:
    cost_calculator       -- Module A: Model ROI & arbitrage calculator.
    infrastructure_audit  -- Module B: Idle GPU / EC2 right-sizing audit.
    report_generator      -- Executive-ready Markdown report builder.
    data_loader           -- Tolerant CSV/JSON ingestion with strict validation.
"""

from finops import cost_calculator, infrastructure_audit, report_generator

__all__ = ["cost_calculator", "infrastructure_audit", "report_generator"]
__version__ = "1.0.0"