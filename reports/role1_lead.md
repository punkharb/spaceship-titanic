# Role 1 – Lead / integrator (Pun)
Last updated: 2026-11-20 · Status: Done

## 1. Slide summary
- **Final model: LightGBM**, CV accuracy **0.8142 ± 0.0112** (ROC-AUC 0.9025), retrained on all 8,693 rows → `submission.csv` (4,277 rows).
- Fold by fold (`06b_paired_check.py`), LightGBM **clearly** beats Random Forest (+0.82 pts, better on all 5 folds) and fits ~12× faster. CatBoost (0.8177) is +0.35 pts ahead of LightGBM but **not** a clear win (needs > 0.36 pts), so LightGBM stays the final model.
- The whole pipeline 01→07 (including 06_compare) re-runs from a clean checkout in ~196 s with **identical** scores, parameters, comparison table, 13 plots and submission.
- Every model is validated on the **same 5 GroupKFold folds**, grouped by travel group (`gggg` of `PassengerId`).
- Every model is scored by **one function**, `evaluate()`: accuracy, precision, recall, F1, ROC-AUC (mean ± std) and fit time, logged to `results/results.csv`.
- Preprocessing is refitted **inside each fold**, so validation rows never influence training.
- One config file (`src/config.py`) fixes paths, target, `SEED = 42` and `N_SPLITS = 5` for the whole team.

- **Nested CV (issue #16):** the honest score for tuned LightGBM is **0.8109 ± 0.0099** (0.3 points below the tuned 0.8142); Random Forest 0.8025 ± 0.0133 (0.4 below). LightGBM still leads, and the ranking does not change.

## 2. What I did
1. Created folders `src/`, `plots/`, `results/`, `reports/`, `slides/`, `notes/` and `.gitignore`.
2. `requirements.txt`: pinned pandas 2.3.3, numpy 2.5.2, scikit-learn 1.9.0, matplotlib 3.11.2, lightgbm 4.7.0 (Python 3.14.4).
3. `src/config.py`: `DATA_DIR`, `TRAIN_PATH`, `TEST_PATH`, `PLOT_DIR`, `RESULTS_DIR`, `TARGET`, `ID_COL`, `SEED`, `N_SPLITS`.
4. `src/evaluation.py`: `get_groups()`, `get_cv()`, `evaluate()`.
5. Self-test `python -m src.evaluation`, written before the code (test first), passes.
6. Added the Kaggle data (`train.csv`, `test.csv`, `sample_submission.csv`) and a `.gitattributes` so git keeps them byte-identical to the download.
7. `07_final_model.py`: chooses the final model from `results.csv`, refits it on all of train.csv with its saved best parameters, writes and checks `submission.csv`. Self-test `python 07_final_model.py --self-test` passes.
8. Post-merge leakage review of Roles 2–4 (`src/features.py`, 01–05) by independent reviewers: no leakage found (details in section 6).
9. Ran `python 07_final_model.py` → LightGBM chosen, `submission.csv` written and checked against Kaggle's `sample_submission.csv`.
10. Clean run: fresh checkout, all outputs deleted, 01→07 run in order → all exit 0, results identical to the committed ones.
11. Final clean run on 2026-10-31 after `06_compare.py` was merged: all 7 scripts exit 0; `results.csv` metrics, both `best_params` files, `model_comparison.csv`, `eda_findings.txt`, all 13 plots and `submission.csv` (byte for byte) identical to `main`.

12. `05b_nested_cv.py` (issue #16): for each of the 5 shared outer folds, re-runs 04/05's exact random search (same space, 30 settings, seed) on the other 4 folds with inner folds grouped by travel group, refits the best setting there and scores it on the unseen outer fold. Logged with `evaluate()` as `random_forest_nested` / `lightgbm_nested`. `06_compare.py` shows these rows but never treats them as rival models.
13. `06b_paired_check.py` (after PR #28 added CatBoost): refits LightGBM, Random Forest and CatBoost with their saved best params on the 5 shared folds and compares each challenger with LightGBM fold by fold (clear win only if mean gain > 2 × standard error of the per-fold gains, the rule from `02b`).
14. Made two extra Kaggle files on request, not committed: `submission_lightgbm.csv` (copy of `submission.csv`) and `submission_catboost.csv` (CatBoost with its saved params, built with `07`'s own functions). The two agree on 4,037 of 4,277 passengers (94.4%). The final model is still chosen by CV, not by Kaggle score.

## 3. Key decisions and why
| Decision | Why | Evidence (number, chart or source) |
|---|---|---|
| GroupKFold by travel group | Group members share circumstances; a random split would put relatives in both train and validation and inflate scores | CLAUDE.md rule 2; Role 2 EDA to confirm no group overlap train/test |
| GroupKFold without shuffle | Folds depend only on the data, so identical for every role and run | `get_cv()` |
| Paths built from `config.py` location | Works on every team member's computer, from any folder | `src/config.py` |
| Out-of-fold confusion matrix from the same 5 fold models | Each row predicted by a model that never saw it → honest error picture; no second round of training | `evaluate()` (`return_estimator`, `return_indices`) |
| One row per model in results.csv (re-run replaces) | Re-runs before merging must not leave duplicates; comparison can't pick a stale or best-of-N row | Self-test: 2 runs of "dummy" → 1 row |
| Final model = best tuned model by CV mean accuracy | Accuracy is the Kaggle metric; only models with a `best_params_*.json` are candidates, so baselines are never shipped | `choose_model()` in `07_final_model.py` |
| Flag when top two are within one std | A gap smaller than fold-to-fold noise isn't a real win; the simpler/faster model is then defensible | Printed note in `07_final_model.py` |
| Decide model switches fold by fold (2 × SE rule) | All models share the same folds, so fold difficulty cancels; the per-fold gain's spread is the right noise measure, and the one-std rule is too cautious | `06b_paired_check.py`: LightGBM vs RF clear (+0.82, 5/5), CatBoost vs LightGBM not clear (+0.35 vs 0.36 needed) |
| Refit once on all train.csv, no re-tuning | More data for the final fit; tuning already happened in CV; test.csv never used for choices | `main()` in `07_final_model.py` |
| Check submission before writing | Columns, 4,277 rows, True/False values and ids matching test.csv row for row | `check_submission()` |
| Fail loudly on bad target or failed fold | Otherwise sklearn logs NaN scores silently | `error_score="raise"`, bool check on `y` |
| `fit_time_s` = mean fit time per fold | Comparable across models regardless of tuning | `evaluate()` |
| `evaluate()` also returns train accuracy (not logged) | Roles 3 and 4 need it for the overfitting check | `train_accuracy_mean` in returned dict |

| Nested CV as a separate script, reusing 04/05's search unchanged | Gives an unbiased score without touching Roles 3–4's files; identical search keeps the comparison fair | `05b_nested_cv.py` imports `PARAM_SPACE`, `N_ITER`, `make_pipeline()` |
| Inner folds grouped by travel group | Same leakage rule as the outer folds; groups read from `PassengerId` of the rows each outer fold hands over | `GroupedSearch.fit()`; review confirmed inner search saw only outer-training rows |
| Nested rows logged but excluded from the model verdict | They re-score an existing model; they answer "how good is tune-then-predict", not "which model" | `_candidates()` in `06_compare.py` |

## 4. Results
- Self-test (synthetic data, 20 groups × 4 people, 75% transported): DummyClassifier accuracy 0.7500 ± 0.0000, ROC-AUC 0.5000 — matches the known answers.
- Data added: train.csv 8,693 rows × 14 columns, 6,217 travel groups; test.csv 4,277 rows. `Transported` loads as bool, as `evaluate()` requires.

**Tuned candidates for the final model** (from `results/results.csv`, printed by `07_final_model.py`):

| Model | Accuracy (mean ± std) | ROC-AUC | Fit time per fold |
|---|---|---|---|
| lightgbm | 0.8142 ± 0.0112 | 0.9025 | 0.17 s |
| random_forest | 0.8060 ± 0.0131 | 0.8950 | 2.05 s |

- Final model: **lightgbm** with `subsample_freq=1, subsample=1.0, num_leaves=15, n_estimators=500, min_child_samples=20, max_depth=4, learning_rate=0.03, colsample_bytree=0.8, random_state=42`.
- Submission check: 4,277 rows, columns `PassengerId, Transported`, True/False with no gaps, ids identical and in the same order as `sample_submission.csv`.
- Predicted transported share on test: **51.5%** (2,201 True / 2,076 False) vs 50.4% in train → no drift towards one class.
- Final clean run times: 01 6 s, 02 4 s, 03 5 s, 04 118 s, 05 56 s, 06 2 s, 07 5 s (~196 s total).

**Nested CV vs tuned score** (`python 05b_nested_cv.py`, ~7.5 min; scores in `results/results.csv`):

| Model | Tuned (same folds) | Nested CV (honest) | Optimism | Default (untuned) |
|---|---|---|---|---|
| LightGBM | 0.8142 ± 0.0112 | 0.8109 ± 0.0099 | 0.3 points | 0.8106 ± 0.0121 |
| Random Forest | 0.8060 ± 0.0131 | 0.8025 ± 0.0133 | 0.4 points | 0.8005 ± 0.0128 |

- Nested ROC-AUC: LightGBM 0.9042, Random Forest 0.8944.
- Nested LightGBM (0.8109) is about the same as default LightGBM (0.8106): once the best-of-30 luck is removed, tuning adds almost nothing.
- Nested `fit_time_s` (LightGBM 35.75 s, RF 53.24 s per outer fold) includes the whole inner search, so it is not comparable with the other rows.

**Fold-by-fold check** (`python 06b_paired_check.py`, accuracy per shared fold, saved best params):

| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|---|---|---|---|---|---|---|
| lightgbm (final) | 0.8281 | 0.8217 | 0.8189 | 0.7986 | 0.8038 | 0.8142 |
| random_forest | 0.8194 | 0.8212 | 0.8074 | 0.7894 | 0.7929 | 0.8060 |
| catboost | 0.8321 | 0.8269 | 0.8154 | 0.8055 | 0.8084 | 0.8177 |

- Random Forest vs LightGBM: −0.82 pts (per fold −0.86, −0.06, −1.15, −0.92, −1.09; better on 0/5); threshold 0.39 → LightGBM is clearly better.
- CatBoost vs LightGBM: +0.35 pts (per fold +0.40, +0.52, −0.35, +0.69, +0.46; better on 4/5); threshold 0.36 → not a clear win, LightGBM stays.

## 5. Charts for the slides
| File in plots/ | One-line caption (the finding) | Suggested slide |
|---|---|---|
| – | none yet | – |

## 6. Bias and leakage checks
- One fixed split (`get_cv()`) for all models → fair comparison.
- `evaluate()` takes a whole Pipeline, so imputer/scaler/encoder are fitted on training folds only.
- Self-test writes to a temp folder, so `results/results.csv` only holds real runs.
- Final model chosen only from CV scores on train.csv; test.csv is read after the choice and only passed to `predict()`.
- Post-merge review of Role 2 (`features.py`, 01–03): `add_features()` is row-wise plus within-file group counts (02 checks it gives identical values on every fold); all learned steps sit in `build_preprocessor()`; EDA confirms **0** travel groups in both train (6,217) and test (3,063).
- Post-merge review of Roles 3–4 (04, 05): tuning only via `RandomizedSearchCV(cv=get_cv(), groups=...)`, same budget (30 settings) and seed for both; scores from `evaluate()` not `best_score_`; permutation importance on held-out folds; test.csv never read.
- Tuned scores are scored on the same folds used for tuning, so they are slightly optimistic. Nested CV (`05b_nested_cv.py`) measures this: 0.3 points for LightGBM, 0.4 for Random Forest.

## 7. Problems and how I fixed them
- lightgbm and matplotlib not installed on lead's machine → installed with pip and pinned; other pins unchanged, self-test still passes.
- train.csv / test.csv missing from repo → self-test uses synthetic data with known answers.
- Review of `07_final_model.py` (no leakage; stand-in run refitted once on 8,693 rows, wrote 4,277 rows, reproducible) flagged: exact model names not in the contract, saved params could differ from the scored ones, `prep__` keys and `verbose` crashed the model, ties broken by row order, unclear error for empty results.csv → all fixed and covered by the self-test; naming and `X` rule added to the CLAUDE.md contract.
- Raw CSVs were being converted to CRLF on checkout → `.gitattributes` `*.csv -text`; files now byte-identical to the Kaggle download.
- Independent leakage review (no leakage found) flagged: duplicate rows on re-run, NaN scores logged silently, missing header on empty results.csv, every model trained twice → all four fixed, each covered by the self-test or the rewrite of the logging.

## 8. Likely Q&A questions
| Question | Short answer |
|---|---|
| Why GroupKFold, not random KFold? | Travel-group members are similar; a random split leaks group info into validation and inflates scores. |
| How do you prevent leakage in preprocessing? | All preprocessing is inside the Pipeline, which `cross_validate` refits on each training fold. |
| Why one `evaluate()` for everyone? | Same folds, same metrics, same logging → numbers are comparable and traceable to `results.csv`. |
| Why retrain on all of train.csv at the end? | CV already chose the model and parameters; the final fit uses all 8,693 rows for the best possible model, with no further tuning. |
| How do you know test.csv didn't influence the model? | It is only read after the model is chosen and fitted, and is only passed to `predict()`. |
| Why LightGBM? | It clearly beats Random Forest fold by fold (+0.82 pts, better on all 5 folds) and fits about 12× faster. CatBoost is +0.35 pts ahead of it, but that is not a clear win (needs > 0.36 pts, and fold 3 goes the other way), and LightGBM fits about 15× faster than CatBoost. |
| Why compare fold by fold instead of by ± std? | Every model uses the same 5 folds, so some folds are hard for all of them. Comparing per fold cancels that, and the spread of the per-fold gains is the right noise measure. |
| Did you submit more than one model to Kaggle? | We made a CatBoost file for reference, but the final model was fixed by CV before looking at Kaggle; choosing by leaderboard score would use test.csv for model choice. |
| Why tuned LightGBM, not the default? | Tuning added only 0.4 points (0.8142 vs 0.8106), within one std, so the default would be about as good. `07_final_model.py` only ships tuned models (those with saved best params), and the gap is honest to report as "not clearly real". |
| Is the tuned score optimistic? | Slightly. Nested CV, where each outer fold is scored by a model tuned without it, gives LightGBM 0.8109 vs 0.8142 tuned (0.3 points) and Random Forest 0.8025 vs 0.8060 (0.4 points). |
| What is nested CV? | Tuning is repeated inside each outer training part, so the fold used for scoring never influenced the choice of settings. It costs about 5× the fits (750 per model). |
| Does nested CV explain the CV → Kaggle gap? | Only part of it: about 0.3 points of the ~1 point gap. The rest is most likely ordinary sampling difference between our folds and Kaggle's test set; we cannot check that without Kaggle's labels. |
| Is the pipeline reproducible? | Yes: a clean checkout re-run of 01→07 gave identical scores, parameters, plots and submission. |
| Why report mean ± std? | Std across folds shows how stable a score is; a small difference within the std is not a real win. |

## 9. Open items / handover
- `06_compare.py` + comparison chart: joint with Role 5 (Bard), not started.
- Check the technical slides with Role 5; rehearsal with the whole team.
- Review follow-ups for owners (low severity, no leakage): **Role 4** – `05_lightgbm.py`:167 prints `≈`, which crashes on cp1252/cp874 consoles (use `~` like 04); report line 88 "cancels out" → "similar for both". **Roles 3 & 4** – add `error_score="raise"` to `RandomizedSearchCV`. **Role 2** – be ready to justify `CHILD_AGE = 13` (chosen from EDA, not the target).
- Re-run 01→07 only if a script changes again. `05b_nested_cv.py` is slow (~7.5 min) and only adds analysis rows; it does not change the final model or `submission.csv`.
- Tell Role 5: slide 10 can now show the nested scores (chart 12 includes them).
