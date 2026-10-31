# Role 1 – Lead / integrator (Pun)
Last updated: 2026-10-31 · Status: Done

## 1. Slide summary
- **Final model: LightGBM**, CV accuracy **0.8142 ± 0.0112** (ROC-AUC 0.9025), retrained on all 8,693 rows → `submission.csv` (4,277 rows).
- LightGBM vs Random Forest (0.8060 ± 0.0131) is within one std, but LightGBM is ~12× faster to fit (0.17 s vs 2.05 s per fold) → best on both counts.
- The whole pipeline 01→07 (including 06_compare) re-runs from a clean checkout in ~196 s with **identical** scores, parameters, comparison table, 13 plots and submission.
- Every model is validated on the **same 5 GroupKFold folds**, grouped by travel group (`gggg` of `PassengerId`).
- Every model is scored by **one function**, `evaluate()`: accuracy, precision, recall, F1, ROC-AUC (mean ± std) and fit time, logged to `results/results.csv`.
- Preprocessing is refitted **inside each fold**, so validation rows never influence training.
- One config file (`src/config.py`) fixes paths, target, `SEED = 42` and `N_SPLITS = 5` for the whole team.

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
| Refit once on all train.csv, no re-tuning | More data for the final fit; tuning already happened in CV; test.csv never used for choices | `main()` in `07_final_model.py` |
| Check submission before writing | Columns, 4,277 rows, True/False values and ids matching test.csv row for row | `check_submission()` |
| Fail loudly on bad target or failed fold | Otherwise sklearn logs NaN scores silently | `error_score="raise"`, bool check on `y` |
| `fit_time_s` = mean fit time per fold | Comparable across models regardless of tuning | `evaluate()` |
| `evaluate()` also returns train accuracy (not logged) | Roles 3 and 4 need it for the overfitting check | `train_accuracy_mean` in returned dict |

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
- Known, disclosed limitation: tuned scores are scored on the same folds used for tuning, so they are slightly optimistic (both reports say so). Nested CV would remove this at ~5× the cost.

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
| Why LightGBM? | Highest CV accuracy (0.8142) and about 12× faster to fit than Random Forest (0.17 s vs 2.05 s per fold). Its 0.8-point lead over Random Forest is within one std, so speed backs the choice too. |
| Why tuned LightGBM, not the default? | Tuning added only 0.4 points (0.8142 vs 0.8106), within one std, so the default would be about as good. `07_final_model.py` only ships tuned models (those with saved best params), and the gap is honest to report as "not clearly real". |
| Is the tuned score optimistic? | Slightly: the best of 30 settings is scored on the same folds it was picked on. Both models had the same budget and folds; nested CV would fix it. |
| Is the pipeline reproducible? | Yes: a clean checkout re-run of 01→07 gave identical scores, parameters, plots and submission. |
| Why report mean ± std? | Std across folds shows how stable a score is; a small difference within the std is not a real win. |

## 9. Open items / handover
- `06_compare.py` + comparison chart: joint with Role 5 (Bard), not started.
- Check the technical slides with Role 5; rehearsal with the whole team.
- Review follow-ups for owners (low severity, no leakage): **Role 4** – `05_lightgbm.py`:167 prints `≈`, which crashes on cp1252/cp874 consoles (use `~` like 04); report line 88 "cancels out" → "similar for both". **Roles 3 & 4** – add `error_score="raise"` to `RandomizedSearchCV`. **Role 2** – be ready to justify `CHILD_AGE = 13` (chosen from EDA, not the target).
- Re-run 01→07 only if a script changes again; the last clean run matches `main`.
