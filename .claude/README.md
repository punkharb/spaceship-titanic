# .claude – Claude Code setup for the Spaceship Titanic team

Opened automatically by Claude Code (terminal or VS Code) when you work in this folder.
Project rules live in `../CLAUDE.md`; this folder adds commands, helpers and safety checks.

## Start here
`/start-role <1-5>`: shows your tasks, progress and next step.

## Slash commands
| Command | Who | What |
|---|---|---|
| `/start-role <n>` | everyone | Begin a session as your role |
| `/status` | everyone | Team progress, best model so far, blockers |
| `/update-report [n]` | everyone | Update your `reports/` file from real script output |
| `/check-leakage [files]` | everyone | Review code for leakage and unfair comparison |
| `/explain <topic>` | everyone | Plain-English explanation and likely Q&A |
| `/setup-project` | Role 1 | `requirements.txt`, `src/config.py`, `src/evaluation.py` |
| `/run-pipeline` | Role 1 | Run 01→07 and check the outputs |
| `/final-model [model]` | Role 1 | `07_final_model.py` + `submission.csv` |
| `/eda [section]` | Role 2 | `01_explore.py` |
| `/features` | Role 2 | `src/features.py` + `02_prepare.py` |
| `/baseline` | Role 2 | `03_baseline.py` |
| `/train-model <random_forest\|lightgbm>` | Roles 3, 4 | Build, tune and evaluate a model |
| `/compare-models` | Roles 5 + 1 | `06_compare.py` |
| `/create-reports` | Role 5 | Create the 5 report files from the template |
| `/slide-plan` | Role 5 | Slide plan from the reports, checked against the results |
| `/qa-prep` | Role 5 | `notes/qa_prep.md` |

## Subagents (`agents/`)
- **leakage-reviewer**: read-only check for leakage, bias and rule violations.
- **report-writer**: fills a role report from real script output.
- **slide-checker**: read-only check that slides and reports match `results.csv` and `plots/`.

## Safety (`settings.json` + `hooks/`)
- `train.csv` / `test.csv` can't be edited, overwritten, moved or deleted (permission rules + `hooks/protect_raw_data.py`).
- After a pipeline file is edited, `hooks/remind_report.py` reminds Claude to run it and update the owner's report.
- `python`, `pip install` and read-only `git` commands are pre-approved; `rm -rf`, `git push --force` and `git reset --hard` are blocked.

## Requirements
- Python on your PATH as `python` (the hooks use it). Check with `python --version`.
- Commit this folder to Git so everyone shares it. Personal overrides go in `.claude/settings.local.json` (git-ignored).
