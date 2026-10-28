---
description: (Role 2) Build or extend 01_explore.py - the 8-section EDA with conclusions and charts
argument-hint: [section number 1-8 to work on - default all]
---
I'm Role 2. Build `01_explore.py` together with me, following the 8-section EDA checklist in my role section of @CLAUDE.md. Focus on: $ARGUMENTS

Requirements:
- Use `src/config.py` paths and constants if they exist (otherwise build paths from the script's folder and note this for Role 1).
- Each section prints a plain-English conclusion (`-> ...`), and the conclusions are saved to `results/eda_findings.txt`.
- Charts go to `plots/` with numbered names, a title that states the finding, and the team colours from CLAUDE.md.
- When ranking features, ignore tiny groups (fewer than about 30 passengers).
- Section 7 must check travel-group overlap between train.csv and test.csv, because it justifies the GroupKFold rule.
- Never modify train.csv or test.csv.

Explain each section as we write it. Run the script, and only state findings that appear in its output. Then use /update-report 2.
