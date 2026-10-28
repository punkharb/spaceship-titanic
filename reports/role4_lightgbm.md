# Role 4 – Model C: LightGBM (Get)
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

- [ ] `05_lightgbm.py`: `Pipeline(build_preprocessor(), LGBMClassifier(random_state=SEED, verbose=-1))`
- [ ] Tuning on the same `get_cv()` folds: `learning_rate`, `num_leaves`/`max_depth`, `n_estimators`, `min_child_samples`, `subsample`, `colsample_bytree` (modest search, sensible ranges)
- [ ] Overfitting check: training vs validation score
- [ ] Save best parameters to `results/best_params_lightgbm.json`
- [ ] Log the tuned result with `evaluate()` (row in `results.csv`)
- [ ] Explain why LightGBM beats or loses to Random Forest
- [ ] Keep this report up to date: search ranges and best parameters, scores before/after tuning, overfitting check, comparison with Random Forest
