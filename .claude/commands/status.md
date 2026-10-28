---
description: Team progress overview - checklist, reports, results and blockers
---
Give a short team status report for this project.

1. Read the Status checklist in @CLAUDE.md.
2. Check which pipeline files exist (`src/*.py`, `0*_*.py`, `results/results.csv`, `submission.csv`).
3. For each file in `reports/`, read its Status and Last updated date.
4. If `results/results.csv` exists, list each model with its accuracy mean ± std.

Reply with:
- A table: Role | Deliverables done | Report status | Last updated
- The current best model and score (from results.csv only)
- Blockers: work waiting on another role, e.g. models waiting for `src/features.py`
- Checklist items that look done but aren't ticked. Offer to tick them; don't edit without asking.
