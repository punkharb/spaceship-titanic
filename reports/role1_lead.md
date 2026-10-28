# Role 1 – Lead / integrator (Pun)
Last updated: 2026-10-28 · Status: In progress

## 1. Slide summary
3–5 bullets a slide could show as is, each one finding or result.

## 2. What I did
Numbered steps in the order done, each with the script/function that does it.

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

- [ ] Set up the shared repo/folder, `requirements.txt` (pinned versions), the `src/` package and `src/config.py`
- [ ] Keep CLAUDE.md up to date
- [ ] `src/evaluation.py`: fixed `GroupKFold` split (`get_groups()`, `get_cv()`), `evaluate()` and logging to `results/results.csv`
- [ ] Review every merge for leakage (rule 5), reproducibility and shared-rule compliance, then merge
- [ ] `07_final_model.py`: retrain the chosen model on all of train.csv with its best parameters, predict test.csv, write `submission.csv` (`PassengerId, Transported` as True/False, 4,277 rows)
- [ ] Run the whole pipeline 01 → 07 from a clean state before the deadline
- [ ] Pair with Role 5 to check the technical slides are correct
- [ ] Keep this report up to date: setup, validation split and `evaluate()` (how and why), final model choice, submission, merge/review notes
