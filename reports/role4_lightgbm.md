# Role 4 – Model C: LightGBM (Get)
Last updated: 2026-11-19 · Status: Done

## 1. Slide summary
- Tuned LightGBM reaches **81.42% ± 1.12** cross-validated accuracy (ROC-AUC 0.9025), **+4.01 points** above Logistic Regression (77.41%).
- Tuning added only **+0.36 points** over the default settings (81.06%), well within the ~1.2-point fold-to-fold noise: LightGBM's defaults were already good.
- Tuning **halved overfitting**: the train–validation gap fell from 8.6 to 4.0 points (simpler trees: 15 leaves, depth 4, small learning rate).
- The model relies most on **spending** (Spa, VRDeck, FoodCourt) and **CryoSleep**; GroupSize and VIP add almost nothing.
- vs tuned Random Forest (80.60%): LightGBM is **+0.82 points** ahead, but that is **within the fold-to-fold spread (~1.31 points)**, so the two are effectively tied. LightGBM has higher F1 and ROC-AUC and fits more than 10× faster per fold (0.17 s vs 2.05 s).
- The whole search (30 settings × 5 folds) took **25–27 seconds**.
- **Blend (issue #19):** a 50/50 soft-voting blend of tuned LightGBM + tuned RF scores **80.93% ± 1.29**, **0.49 points below LightGBM alone** (within noise). The two models agree on 93% of passengers, and where they disagree LightGBM is right more often, so averaging adds nothing. LightGBM stays the final model.

## 2. What I did
1. Scored LightGBM with default settings using the shared `evaluate()` → row `lightgbm_default` (`05_lightgbm.py`, step 1).
2. Ran `RandomizedSearchCV` with 30 random settings on the shared `get_cv()` GroupKFold folds (`tune()`).
3. Saved the best settings to `results/best_params_lightgbm.json` and scored them with `evaluate()` → row `lightgbm` (the name `07_final_model.py` looks for).
4. Overfitting check: mean training-fold vs validation-fold accuracy, default vs tuned (`overfit_report()`).
5. Permutation importance on the validation folds only, averaged over the 5 folds (`permutation_importance_cv()`) → `plots/11_lightgbm_permutation_importance.png`.
6. Compared with the baselines and Role 3's Random Forest in `results.csv` (`compare_with_others()`), re-run on up-to-date main after Role 3's merge.
7. Checked: the script gives identical results on a second run; the saved params pass `07_final_model.py`'s `load_params()` check; leakage review found no leakage.
8. Issue #19: built a 50/50 soft-voting blend of the tuned LightGBM and tuned Random Forest (`05c_blend.py`). Rebuilt both from their saved `best_params_*.json` (checked against `results.csv`), scored with `evaluate()` → row `lgbm_rf_blend`, ran an overfitting check and an out-of-fold agreement analysis, and compared against both models.

## 3. Key decisions and why
| Decision | Why | Evidence (number, chart or source) |
|---|---|---|
| `RandomizedSearchCV`, 30 settings, not a full grid | The full grid has 4·4·3·4·4·3·3 = 6,912 settings; 30 random ones find a good region fast. Efficiency is graded. | Search took 25–27 s (wall-clock, varies per run); the top 5 settings all score 0.8123–0.8142, so more search would not help |
| Small value lists: learning_rate {0.01, 0.03, 0.05, 0.1}, n_estimators {100, 200, 300, 500}, num_leaves {7, 15, 31}, max_depth {-1, 4, 6, 8}, min_child_samples {10, 20, 40, 80}, subsample {0.7, 0.85, 1.0}, colsample_bytree {0.6, 0.8, 1.0} | Common sensible ranges around LightGBM's defaults; mostly towards *simpler* trees, because the default overfits (8.6-point gap) | Script output, step 4 |
| `subsample_freq = 1` kept in the search space | LightGBM ignores `subsample` unless `subsample_freq > 0`. Saving it with the params makes 07 train exactly the model we validated. | Leakage review: 07's model gives identical predictions to the CV model |
| Plain Python lists instead of scipy distributions | The chosen values must survive JSON unchanged, because 07 checks the json file against the params in `results.csv` | `load_params("lightgbm")` passes |
| `refit=False` in the search | `evaluate()` refits per fold, and the final fit on all data is Role 1's job (07); a refit here would be wasted time | – |
| Permutation importance on validation folds, shuffling original columns | Measures what helps on *unseen* passengers; one-hot columns are judged together (e.g. "Deck", not "Deck_F") | `plots/11_lightgbm_permutation_importance.png` |
| Blend: fixed 50/50 weights, no weight tuning | Tuning the weights on the same folds would be another lucky pick (more optimistic score). With fixed weights the blend adds no selection of its own and keeps issue #19's tuning budget | Issue #19 rules; `05c_blend.py` docstring |
| Blend: rebuild both models from `best_params_*.json`, checked with 07's `load_params()` | Guarantees we blend exactly the models whose CV scores are on the slides | `05c_blend.py` step 1 |
| Blend: one shared `build_preprocessor()` before a `VotingClassifier(voting="soft")` | Same preparation step as every model (rule 4); soft voting uses the probabilities, so a confident model counts more than an unsure one | – |
| Names `lightgbm_default` / `lightgbm` | CLAUDE.md contract: 07 looks for `lightgbm` + `best_params_lightgbm.json` | `results/results.csv` |

## 4. Results
5-fold GroupKFold, mean (± std where logged), copied from `results/results.csv` (rounded to 4 decimals):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Fit time / fold |
|---|---|---|---|---|---|---|
| Dummy (most frequent) | 0.4963 ± 0.0141 | 0.3999 | 0.8000 | 0.5332 | 0.5000 | 0.0025 s |
| Logistic Regression | 0.7741 ± 0.0071 | 0.7668 | 0.7922 | 0.7792 | 0.8525 | 0.0448 s |
| Random Forest default (Role 3) | 0.8005 ± 0.0128 | 0.8270 | 0.7638 | 0.7940 | 0.8852 | 1.2509 s |
| Random Forest tuned (Role 3) | 0.8060 ± 0.0131 | 0.8129 | 0.7991 | 0.8057 | 0.8950 | 2.0473 s |
| LightGBM default | 0.8106 ± 0.0121 | 0.8080 | 0.8188 | 0.8132 | 0.9030 | 0.1002 s |
| **LightGBM tuned** | **0.8142 ± 0.0112** | 0.8088 | 0.8267 | 0.8176 | 0.9025 | 0.1735 s |

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
- vs Random Forest (both tuned; RF numbers from `results.csv` and `reports/role3_random_forest.md`):

| | LightGBM tuned | Random Forest tuned |
|---|---|---|
| Accuracy | **0.8142 ± 0.0112** | 0.8060 ± 0.0131 |
| Precision | 0.8088 | **0.8129** |
| Recall | **0.8267** | 0.7991 |
| F1 | **0.8176** | 0.8057 |
| ROC-AUC | **0.9025** | 0.8950 |
| Fit time / fold | **0.1735 s** | 2.0473 s |
| Train vs validation gap | **4.0 points** (0.8545 vs 0.8142) | 9.2 points (0.8978 vs 0.8060) |
| Best parameters | 15 leaves, depth 4, 500 trees, learning rate 0.03 | 200 trees, depth 20, min_samples_leaf 4, max_features sqrt |
| Top 4 features | Spa_log 4.38, VRDeck_log 3.85, FoodCourt_log 3.76, CryoSleep 2.37 | FoodCourt_log 3.92, Spa_log 2.39, VRDeck_log 2.23, CryoSleep 1.62 |

  Script output (step 6): "LightGBM vs random_forest: +0.82 points (within the fold-to-fold spread, std ≈ 1.31 points)".

**Why LightGBM vs Random Forest**
- **They are statistically very close.** LightGBM is +0.82 points ahead in accuracy, but the fold-to-fold spread is about 1.31 points, so we cannot claim LightGBM is clearly better. A different split of passengers could plausibly narrow or reverse the gap.
- **The comparison is fair.** Both were tuned the same way (`RandomizedSearchCV`, 30 settings) on the same `get_cv()` GroupKFold folds, with the same `build_preprocessor()` and scored with the same `evaluate()`. Both best scores share the same small optimistic bias (best of 30 on the folds that chose it), so it cancels out in the comparison.
- **Likely reasons LightGBM is slightly ahead** (plausible explanations, not proven):
  - Boosting builds each tree to correct the errors of the previous ones, while a Random Forest averages independent trees that each try to solve the whole problem. Boosting can keep reducing the remaining errors; averaging mainly reduces noise.
  - The tuned LightGBM uses many shallow trees (depth 4, 15 leaves) with a small learning rate (0.03), so each tree adds a small correction. This likely explains why it overfits less (4.0 vs 9.2 point gap) while still scoring at least as well.
  - LightGBM is faster because it bins numeric values into histograms and grows small trees, while the forest grows 200 deep trees (depth up to 20) on bootstrap samples.
- **Both models agree on what matters:** spending (Spa, VRDeck, FoodCourt) and CryoSleep are the top 4 features in both, which makes the finding robust to the model choice.
- **Practical tie-breaker:** with accuracy within noise, LightGBM also has higher F1 (0.8176 vs 0.8057) and ROC-AUC (0.9025 vs 0.8950), a smaller overfitting gap, and fits more than 10× faster per fold (0.1735 s vs 2.0473 s). The forest has slightly higher precision (0.8129 vs 0.8088), LightGBM higher recall (0.8267 vs 0.7991). The final choice is Role 1's (`07_final_model.py`).

**Soft-voting blend: LightGBM + Random Forest (issue #19, `05c_blend.py`)**

Each model's probability of "transported" is averaged 50/50; the blend predicts True when the average is above 0.5.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Fit time / fold |
|---|---|---|---|---|---|---|
| **LightGBM tuned** | **0.8142 ± 0.0112** | 0.8088 | **0.8267** | **0.8176** | **0.9025** | **0.1735 s** |
| Blend 50/50 (`lgbm_rf_blend`) | 0.8093 ± 0.0129 | 0.8078 | 0.8154 | 0.8114 | 0.9022 | 1.1861 s |
| Random Forest tuned | 0.8060 ± 0.0131 | **0.8129** | 0.7991 | 0.8057 | 0.8950 | 2.0473 s |

- Script output: "blend vs lightgbm: -0.49 points (within the fold-to-fold spread, std ~1.29 points)"; "blend vs random_forest: +0.32 points (within the fold-to-fold spread, std ~1.31 points)".
- **Overfitting:** train 0.8815 vs validation 0.8093 → **7.2-point gap**, between LightGBM (4.0) and RF (9.2). The deep RF trees bring their memorisation into the blend.
- **Out-of-fold confusion matrix** (rows = true, columns = predicted): not transported 3,466 / 849; transported 809 / 3,569.
- **Why the blend doesn't help** (out-of-fold, same folds):
  - The two models' probabilities correlate at **0.9648**. They predict different classes for only **607 of 8,693 passengers (7.0%)**.
  - Blending can only change those 607: where both models agree, their average is on the same side of 0.5.
  - On those 607, **LightGBM alone is right 55.8%**, Random Forest 44.2%, the blend 48.8%. Averaging in the weaker forest costs LightGBM some of its correct answers.
  - A blend helps when the models are about equally good *and* make different mistakes. Here LightGBM is the stronger model and the two mostly make the same mistakes.
- **Decision:** the blend does not beat the current best by more than one std (it is below it), so `07_final_model.py` stays on LightGBM (issue #19 rule).

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
- Blend: same `get_cv()` folds, preprocessing inside the Pipeline, `random_state=SEED` in both models, test.csv never read. The agreement figures use out-of-fold probabilities (`cross_val_predict` on `get_cv()`), so every passenger is judged by models that never saw them. The component params were chosen on these folds, so the blend inherits their small best-of-30 optimism, but it adds none of its own (fixed weights). Two runs gave identical scores.

## 7. Problems and how I fixed them
- PC had only Python 3.10 without packages; the team pins Python 3.14 versions → installed Python 3.14 and a project `.venv` with `requirements.txt`.
- Thai Windows console (cp874) can't print "±" → `UnicodeEncodeError` → run with `PYTHONUTF8=1` (affects every script; flagged to Role 1).
- LightGBM silently ignores `subsample` without `subsample_freq` → added `subsample_freq: [1]` to the search space so it is saved and reused by 07.
- Blend script: the first draft of the conclusion was written before seeing the numbers ("disagree on only…") → rewrote it to print what the data says (who is right where the models disagree).
- First printout called a difference bigger than one std "a real difference" → reworded to "larger than the fold-to-fold spread", because it is a rule of thumb, not a test.

## 8. Likely Q&A questions
| Question | Short answer |
|---|---|
| What's the difference between LightGBM and Random Forest? | RF builds many deep trees independently and averages them; LightGBM builds small trees one after another, each correcting the previous trees' errors (boosting). |
| Why RandomizedSearchCV instead of GridSearchCV? | The full grid is 6,912 settings; 30 random ones took 25–27 s and the top 5 all score 0.812–0.814, so a bigger search wouldn't help. |
| Did tuning help? | Only +0.36 points accuracy, within noise. Its real benefit: the overfitting gap halved (8.6 → 4.0 points). |
| Is 81.42% an honest score? | Slightly optimistic: it's the best of 30 settings on the same folds. Expect ~81% on new data. RF has the same bias, so the comparison is fair. |
| Is LightGBM really better than Random Forest? | Not clearly. It is +0.82 points ahead (81.42% vs 80.60%), but the fold-to-fold spread is ~1.31 points, so the difference is within noise. Both were tuned the same way on the same folds, so the comparison is fair. |
| Why choose LightGBM if the difference is within noise? | The final choice is Role 1's in `07_final_model.py`. When accuracy ties, a defensible tie-breaker is the rest: LightGBM has higher F1 (0.8176 vs 0.8057) and ROC-AUC (0.9025 vs 0.8950), a smaller overfitting gap (4.0 vs 9.2 points), and fits more than 10× faster (0.17 s vs 2.05 s per fold). |
| Why `subsample_freq = 1`? | LightGBM ignores `subsample` unless bagging is switched on; saving it means the final model is exactly the validated one. |
| How do you know it's not overfitting? | Train 85.45% vs validation 81.42% (4.0-point gap), down from 8.6 points with default settings. |
| Why permutation importance and not LightGBM's built-in importance? | Built-in importance counts splits on training data; permutation measures accuracy lost on unseen validation passengers. |
| What is soft voting? | Each model gives a probability of "transported"; the blend averages them (50/50) and predicts True above 0.5. Hard voting would count yes/no votes instead. |
| Why didn't the LightGBM + RF blend help? | The two models agree on 93% of passengers (probability correlation 0.96), so blending only changes 7%. On those, LightGBM is right 55.8% vs the blend 48.8%: averaging with the weaker forest costs correct answers. Result: 80.93% vs 81.42%, within noise. |
| Why not tune the blend weights? | That would be another best-of-many pick on the same folds, making the score more optimistic. Fixed 50/50 adds no selection; issue #19 asked to keep the same tuning budget. |
| When would a blend help? | When the models are about equally strong and make different mistakes (low correlation). Here one model is stronger and they mostly make the same mistakes. |
| Why are the spending features so important? | Spending means the passenger was awake and active; EDA showed sleepers spend nothing and are transported far more often. |

## 9. Open items / handover
- [x] `05_lightgbm.py`: default + tuned LightGBM, overfitting check, importance chart
- [x] `results/best_params_lightgbm.json` saved; `lightgbm_default` and `lightgbm` rows in `results.csv`
- [x] Leakage review passed (no leakage)
- [x] Compared with Random Forest (re-run on up-to-date main after Role 3's merge; section 4)
- [x] Report complete
- [x] Issue #19 (Role 4 part): 50/50 soft-voting blend in `05c_blend.py`, `lgbm_rf_blend` row and `best_params_lgbm_rf_blend.json`; leakage review done. Does not beat LightGBM → no change to 07.
- For Role 5/1 (`06_compare.py`): add `"lgbm_rf_blend": "LightGBM + RF blend (50/50)"` to `LABELS`, otherwise the chart shows the raw name. 06's conclusion now names the blend (not RF) as LightGBM's closest rival from another algorithm; the slide text may need updating.
- For Role 3 (issue #19): CatBoost/XGBoost part is still open.
- For Role 1 (07): `lightgbm` row and `best_params_lightgbm.json` are in place. LightGBM vs RF is within noise (+0.82 vs ~1.31 points spread); see section 4 for the tie-breakers.
- For Role 1: the Thai console needs `PYTHONUTF8=1` to print "±" and "≈"; `evaluate()` rewriting the CSV trims the last float digits of other rows (cosmetic diff only).
- Role 3 flagged that `05_lightgbm.py` prints "≈", which crashes on Windows consoles that can't encode it when output is redirected. With `PYTHONUTF8=1` the run completed (run log contains the line). Not changed in code yet.
- For `results.csv` merge conflicts: re-run the scripts; `evaluate()` replaces only its own rows.
