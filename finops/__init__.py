"""FinOps Arbitrage Engine.

Turns raw LLM usage exports and cloud utilization data into a dollarized,
confidence-labelled savings plan plus ready-to-apply configuration.

Main entry points:
    finops.ingest.load_usage            -- load + enrich usage exports
    finops.llm_audit.analyze_llm_spend  -- LLM run-rate and savings levers
    finops.infra_audit.analyze_infra    -- idle / oversized GPU & EC2 audit
    finops.proof.run_proof              -- quality eval for proposed routes
    finops.report                       -- Markdown + JSON reports
"""

__version__ = "2.0.0"
