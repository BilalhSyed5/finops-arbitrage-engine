#!/usr/bin/env python3
"""FinOps Arbitrage Engine — CLI entrypoint (also installed as ``finops``).

    python finops_audit.py audit --demo
    python finops_audit.py audit --usage litellm_spend_logs.json --inventory inv.csv
    python finops_audit.py proof --usage requests.jsonl --base-url http://localhost:4000
    python finops_audit.py pricing gpt-4o-mini claude-sonnet-4-5

See README.md for every input format and flag.
"""

import sys

from finops.cli import main

if __name__ == "__main__":
    sys.exit(main())
