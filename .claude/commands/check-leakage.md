---
description: Review code for data leakage, unfair comparison and shared-rule violations
argument-hint: [file(s) to review - default all src/ and numbered scripts]
---
Use the **leakage-reviewer** subagent to review: $ARGUMENTS

If no files were given, review `src/*.py` and every numbered script (`0*_*.py`) that exists.

Then summarise for me:
1. The verdict (no leakage found / N issues)
2. Each issue: file:line, why it matters, and the fix
3. For each issue, which role owns the file (CLAUDE.md > Project structure)

Don't fix anything automatically. Ask me which fixes to apply, and only change files my role owns.
