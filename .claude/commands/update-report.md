---
description: Update a role's report in reports/ from the real script output (for the slide maker)
argument-hint: [role number 1-5]
---
Update the role report for **Role $ARGUMENTS**. If no role was given, work it out from the files changed in this session, or ask me.

Use the **report-writer** subagent. Tell it which role, which scripts belong to that role (see CLAUDE.md > Project structure), and to follow the template and rules in CLAUDE.md > Role reports.

When it finishes, show me:
- What changed in the report
- Any section left empty and why
- Numbers that need a script re-run before they can be trusted
