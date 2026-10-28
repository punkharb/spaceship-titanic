# Role 1 – Lead / integrator (Pun)
Last updated: 2026-10-28 · Status: In progress

## 1. Slide summary
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
7. `07_final_model.py`: chooses the final model from `results.csv`, refits it on all of train.csv with its saved best parameters, writes and checks `submission.csv`. Self-test `python 07_final_model.py --self-test` passes; the real run waits for `src/features.py` and the tuned models.

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
- No real model results yet.

## 5. Charts for the slides
| File in plots/ | One-line caption (the finding) | Suggested slide |
|---|---|---|
| – | none yet | – |

## 6. Bias and leakage checks
- One fixed split (`get_cv()`) for all models → fair comparison.
- `evaluate()` takes a whole Pipeline, so imputer/scaler/encoder are fitted on training folds only.
- Self-test writes to a temp folder, so `results/results.csv` only holds real runs.

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
| Why report mean ± std? | Std across folds shows how stable a score is; a small difference within the std is not a real win. |

## 9. Open items / handover
- Review and merge Role 2's `features.py` and the Role 3/4 model scripts.
- Then run `python 07_final_model.py` to choose the final model and write `submission.csv`, and record the choice and the submission check here.
- Run 01→07 from a clean state before the deadline.
- **For Roles 3 and 4:** follow the naming contract in CLAUDE.md (under results.csv columns): tuned model logged as `name="random_forest"` / `"lightgbm"` with `params=best_params`, the same dict saved to `results/best_params_<name>.json` (search keys like `model__n_estimators` are fine), and `X = add_features(df).drop(columns=[TARGET])`.
