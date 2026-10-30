# Role 4 – Model C: LightGBM (Get)
Last updated: 2026-10-30 · Status: In progress (Random Forest comparison waits on Role 3)

## 1. Slide summary
- Tuned LightGBM reaches **81.42% ± 1.12** cross-validated accuracy (ROC-AUC 0.9025), **+4.01 points** above Logistic Regression (77.41%).
- Tuning added only **+0.36 points** over the default settings (81.06%), well within the ~1.2-point fold-to-fold noise: LightGBM's defaults were already good.
- Tuning **halved overfitting**: the train–validation gap fell from 8.6 to 4.0 points (simpler trees: 15 leaves, depth 4, small learning rate).
- The model relies most on **spending** (Spa, VRDeck, FoodCourt) and **CryoSleep**; GroupSize and VIP add almost nothing.
- The whole search (30 settings × 5 folds) took **25 seconds**.

## 2. What I did
1. Scored LightGBM with default settings using the shared `evaluate()` → row `lightgbm_default` (`05_lightgbm.py`, step 1).
2. Ran `RandomizedSearchCV` with 30 random settings on the shared `get_cv()` GroupKFold folds (`tune()`).
3. Saved the best settings to `results/best_params_lightgbm.json` and scored them with `evaluate()` → row `lightgbm` (the name `07_final_model.py` looks for).
4. Overfitting check: mean training-fold vs validation-fold accuracy, default vs tuned (`overfit_report()`).
5. Permutation importance on the validation folds only, averaged over the 5 folds (`permutation_importance_cv()`) → `plots/11_lightgbm_permutation_importance.png`.
6. Compared with the baselines in `results.csv` (`compare_with_others()`); Random Forest is added automatically once Role 3 logs it.
7. Checked: the script gives identical results on a second run; the saved params pass `07_final_model.py`'s `load_params()` check; leakage review found no leakage.

## 3. Key decisions and why
| Decision | Why | Evidence (number, chart or source) |
|---|---|---|
| `RandomizedSearchCV`, 30 settings, not a full grid | The full grid has 4·4·3·4·4·3·3 = 6,912 settings; 30 random ones find a good region fast. Efficiency is graded. | Search took 25 s; the top 5 settings all score 0.8123–0.8142, so more search would not help |
| Small value lists: learning_rate {0.01, 0.03, 0.05, 0.1}, n_estimators {100, 200, 300, 500}, num_leaves {7, 15, 31}, max_depth {-1, 4, 6, 8}, min_child_samples {10, 20, 40, 80}, subsample {0.7, 0.85, 1.0}, colsample_bytree {0.6, 0.8, 1.0} | Common sensible ranges around LightGBM's defaults; mostly towards *simpler* trees, because the default overfits (8.6-point gap) | Script output, step 4 |
| `subsample_freq = 1` kept in the search space | LightGBM ignores `subsample` unless `subsample_freq > 0`. Saving it with the params makes 07 train exactly the model we validated. | Leakage review: 07's model gives identical predictions to the CV model |
| Plain Python lists instead of scipy distributions | The chosen values must survive JSON unchanged, because 07 checks the json file against the params in `results.csv` | `load_params("lightgbm")` passes |
| `refit=False` in the search | `evaluate()` refits per fold, and the final fit on all data is Role 1's job (07); a refit here would be wasted time | – |
| Permutation importance on validation folds, shuffling original columns | Measures what helps on *unseen* passengers; one-hot columns are judged together (e.g. "Deck", not "Deck_F") | `plots/11_lightgbm_permutation_importance.png` |
| Names `lightgbm_default` / `lightgbm` | CLAUDE.md contract: 07 looks for `lightgbm` + `best_params_lightgbm.json` | `results/results.csv` |

## 4. Results
5-fold GroupKFold, mean (± std where logged), copied from `results/results.csv` (rounded to 4 decimals):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Fit time / fold |
|---|---|---|---|---|---|---|
| Dummy (most frequent) | 0.4963 ± 0.0141 | 0.3999 | 0.8000 | 0.5332 | 0.5000 | 0.0025 s |
| Logistic Regression | 0.7741 ± 0.0071 | 0.7668 | 0.7922 | 0.7792 | 0.8525 | 0.0448 s |
| LightGBM default | 0.8106 ± 0.0121 | 0.8080 | 0.8188 | 0.8132 | 0.9030 | 0.0994 s |
| **LightGBM tuned** | **0.8142 ± 0.0112** | 0.8088 | 0.8267 | 0.8176 | 0.9025 | 0.1794 s |

(Fit time is wall-clock and varies slightly between runs; all scores are identical between runs.)

**Best parameters** (`results/best_params_lightgbm.json`): learning_rate 0.03, n_estimators 500, num_leaves 15, max_depth 4, min_child_samples 20, subsample 1.0 (so row bagging is off), subsample_freq 1, colsample_bytree 0.8.

**Top 5 settings in the search** (validation accuracy): 0.8142 ± 0.0112, 0.8135 ± 0.0090, 0.8133 ± 0.0081, 0.8124 ± 0.0118, 0.8123 ± 0.0078. Many different settings score almost the same, so the result does not depend on one lucky setting.

**Overfitting check** (mean over 5 folds):
| | Train accuracy | Validation accuracy | Gap |
|---|---|---|---|
| Default | 0.8971 | 0.8106 | 8.6 points |
| Tuned | 0.8545 | 0.8142 | 4.0 points |

The default model (31 leaves, no depth limit) partly memorises the training folds. The tuned model uses smaller trees (15 leaves, depth 4) with a slower learning rate, so it fits the training folds less closely but does just as well on new passengers.

**Out-of-fold confusion matrix** (tuned; rows = true, columns = predicted):
| | Pred. not transported | Pred. transported |
|---|---|---|
| Not transported | 3,459 | 856 |
| Transported | 759 | 3,619 |

Errors are balanced between the two classes (856 false positives vs 759 false negatives).

**Permutation importance** (accuracy lost when shuffled, percentage points, mean ± std over folds): Spa_log 4.38 ± 0.70, VRDeck_log 3.85 ± 0.73, FoodCourt_log 3.76 ± 0.34, CryoSleep 2.37 ± 0.59, TotalSpend_log 2.13 ± 0.49, RoomService_log 2.09 ± 0.26, Deck 1.99 ± 0.73, CabinNum 1.95 ± 0.35, Side 1.37 ± 0.45, ShoppingMall_log 1.35 ± 0.36.

**Comparison**
- vs Logistic Regression: **+4.01 points** accuracy, larger than the fold-to-fold spread (std ≈ 1.12 points); ROC-AUC 0.9025 vs 0.8525. A likely reason: trees can capture interactions between features (e.g. spending only varies for awake passengers) and non-linear effects that a linear model can't.
- vs Random Forest: *pending — Role 3 has not logged `random_forest` yet.*

## 5. Charts for the slides
| File in plots/ | One-line caption (the finding) | Suggested slide |
|---|---|---|
| `plots/11_lightgbm_permutation_importance.png` | LightGBM relies most on spending (Spa, VRDeck, FoodCourt) and CryoSleep; GroupSize and VIP add almost nothing | LightGBM results / best model feature importance |

## 6. Bias and leakage checks
- Same `get_cv()` GroupKFold folds as every model, with `groups` passed to the search: no travel group is in both training and validation (confirmed for all 5 folds in the leakage review).
- All preprocessing is inside `Pipeline([("prep", build_preprocessor()), ("model", ...)])`, so imputation, scaling and encoding are fitted on training folds only.
- Tuning used train.csv only; test.csv is never read in `05_lightgbm.py`.
- `random_state=SEED` for LightGBM, the search and permutation importance; two runs gave identical scores, params and importances.
- Scored with the shared `evaluate()`; `07_final_model.py` rebuilds exactly the validated model (leakage review: identical predictions).
- **Honest caveat:** 0.8142 is the best of 30 settings, scored on the same folds that chose it, so it is slightly optimistic (probably a few tenths of a point). Expect about 0.81 on new data. Random Forest is tuned the same way on the same folds, so the RF vs LightGBM comparison stays fair. Nested CV would remove this bias at 5× the cost; we judged it unnecessary.
- The importance chart uses the same validation rows that chose the settings; the effect on the ranking is negligible.

## 7. Problems and how I fixed them
- PC had only Python 3.10 without packages; the team pins Python 3.14 versions → installed Python 3.14 and a project `.venv` with `requirements.txt`.
- Thai Windows console (cp874) can't print "±" → `UnicodeEncodeError` → run with `PYTHONUTF8=1` (affects every script; flagged to Role 1).
- LightGBM silently ignores `subsample` without `subsample_freq` → added `subsample_freq: [1]` to the search space so it is saved and reused by 07.
- First printout called a difference bigger than one std "a real difference" → reworded to "larger than the fold-to-fold spread", because it is a rule of thumb, not a test.

## 8. Likely Q&A questions
| Question | Short answer |
|---|---|
| What's the difference between LightGBM and Random Forest? | RF builds many deep trees independently and averages them; LightGBM builds small trees one after another, each correcting the previous trees' errors (boosting). |
| Why RandomizedSearchCV instead of GridSearchCV? | The full grid is 6,912 settings; 30 random ones took 25 s and the top 5 all score 0.812–0.814, so a bigger search wouldn't help. |
| Did tuning help? | Only +0.36 points accuracy, within noise. Its real benefit: the overfitting gap halved (8.6 → 4.0 points). |
| Is 81.42% an honest score? | Slightly optimistic: it's the best of 30 settings on the same folds. Expect ~81% on new data. RF has the same bias, so the comparison is fair. |
| Why `subsample_freq = 1`? | LightGBM ignores `subsample` unless bagging is switched on; saving it means the final model is exactly the validated one. |
| How do you know it's not overfitting? | Train 85.45% vs validation 81.42% (4.0-point gap), down from 8.6 points with default settings. |
| Why permutation importance and not LightGBM's built-in importance? | Built-in importance counts splits on training data; permutation measures accuracy lost on unseen validation passengers. |
| Why are the spending features so important? | Spending means the passenger was awake and active; EDA showed sleepers spend nothing and are transported far more often. |

## 9. Open items / handover
- [x] `05_lightgbm.py`: default + tuned LightGBM, overfitting check, importance chart
- [x] `results/best_params_lightgbm.json` saved; `lightgbm_default` and `lightgbm` rows in `results.csv`
- [x] Leakage review passed (no leakage)
- [ ] **Compare with Random Forest** once Role 3 logs `random_forest`: re-run `python 05_lightgbm.py` (step 6 prints it), then write why LightGBM beats or loses to RF here → then mark this report *Done*.
- For Role 1: the Thai console needs `PYTHONUTF8=1` to print "±"; `evaluate()` rewriting the CSV trims the last float digits of other rows (cosmetic diff only).
- For Role 3: plot number 10 is left free for the RF importance chart.
- For `results.csv` merge conflicts with Role 3's PR: re-run both scripts; `evaluate()` replaces only its own rows.
