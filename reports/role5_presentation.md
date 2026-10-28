# Role 5 – Analysis & presentation (Bard)
Last updated: 2026-10-28 · Status: In progress

## 1. Slide summary
3–5 bullets a slide could show as is, each one finding or result.

## 2. What I did
Numbered steps in the order done, each with the script/function that does it.

1. Created the five role reports in `reports/` from the CLAUDE.md template, each with its role's deliverables as a checklist (`/create-reports`).

## 3. Key decisions and why
| Decision | Why | Evidence (number, chart or source) |
|---|---|---|

## 4. Results
Numbers copied from the script output or results/results.csv (mean ± std where available).

## 5. Charts for the slides
| File in plots/ | One-line caption (the finding) | Suggested slide |
|---|---|---|

## 6. Bias and leakage checks
What I did to keep the result fair (e.g. GroupKFold, fitting inside Pipeline, same seed).

## 7. Problems and how I fixed them
Short bullets: issue → fix. These make good "lessons learned" slides.

## 8. Likely Q&A questions
| Question | Short answer |
|---|---|

## 9. Open items / handover
What's not done yet and who needs what from me.

- [x] Create the report template files for all 5 roles in `reports/`
- [ ] Start the slides: problem → data → EDA findings → pipeline diagram → how we avoided bias → model comparison → best model (confusion matrix + feature importance) → limitations & next steps
- [ ] Build slides from `reports/`: read reports regularly, ask owners when a section is empty or a number doesn't match `results.csv`
- [ ] Keep this report up to date: slide plan with source reports, missing content, rehearsal notes (timing, who presents what)
- [ ] `notes/qa_prep.md`: collect section 8 from every report + likely questions ("Why GroupKFold?", "Why accuracy?", "Why drop Name?", "How did you prevent leakage?", "Why this model?"); make sure every member rehearses them
- [ ] `06_compare.py` (with Role 1): comparison table and chart from `results.csv` (mean ± std across folds)
- [ ] Teamwork evidence: task board and `notes/meetings.md` (date, attendees, decisions, next actions)
- [ ] At least one full timed rehearsal with the whole team

## Slide plan
| Slide # | Title | Source report | Owner | Status |
|---|---|---|---|---|
| 1 | Problem: who was Transported? | CLAUDE.md (task description) | Bard | Not started |
| 2 | Data overview | `role2_data_baseline.md` | Champ | Waiting on content |
| 3 | EDA findings | `role2_data_baseline.md` | Champ | Waiting on content |
| 4 | Data cleaning and features | `role2_data_baseline.md` | Champ | Waiting on content |
| 5 | Pipeline overview diagram | `role1_lead.md` | Pun | Waiting on content |
| 6 | How we avoided bias (GroupKFold, no leakage) | `role1_lead.md` (+ section 6 of every report) | Pun | Waiting on content |
| 7 | Baseline results (Dummy, Logistic Regression) | `role2_data_baseline.md` | Champ | Waiting on content |
| 8 | Random Forest: tuning, results, feature importance | `role3_random_forest.md` | Pat | Waiting on content |
| 9 | LightGBM: tuning, results, vs Random Forest | `role4_lightgbm.md` | Get | Waiting on content |
| 10 | Model comparison (mean ± std) | `role5_presentation.md` (`06_compare.py`, `results.csv`) | Bard + Pun | Waiting on content |
| 11 | Best model: confusion matrix + feature importance | `role1_lead.md`, `role3_random_forest.md` / `role4_lightgbm.md` | Pun | Waiting on content |
| 12 | Final model and submission | `role1_lead.md` | Pun | Waiting on content |
| 13 | Limitations and next steps | all reports (sections 7 and 9) | Bard | Not started |
