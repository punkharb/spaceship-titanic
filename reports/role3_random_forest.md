# Role 3 – Model B: Random Forest (Pat)
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

- [ ] `04_random_forest.py`: `Pipeline(build_preprocessor(), RandomForestClassifier(random_state=SEED))`
- [ ] Tuning: `RandomizedSearchCV` with `cv=get_cv()` and `groups`, searching `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`
- [ ] Save best parameters to `results/best_params_random_forest.json`
- [ ] Log the tuned result with `evaluate()` (row in `results.csv`)
- [ ] Overfitting check: training vs validation accuracy, explain any gap
- [ ] Permutation importance chart → `plots/`
- [ ] Keep this report up to date: search ranges and best parameters, scores before/after tuning, overfitting check, top features with chart
