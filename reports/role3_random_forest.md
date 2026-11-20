# Role 3 – Model B: Random Forest (Pat)
Last updated: 2026-11-20 · Status: Done

## 1. Slide summary
- Tuned Random Forest: **80.60% ± 1.31%** cross-validated accuracy (ROC-AUC 0.895), against 77.41% for Logistic Regression and 49.63% for the Dummy floor.
- Tuning helped only a little: accuracy went from 80.05% to 80.60% (+0.55 points), which is within the 1.31-point fold-to-fold noise.
- Tuning roughly halved overfitting: the gap between training and validation accuracy fell from **19.9 to 9.2 points**.
- Spending drives the forest: FoodCourt, Spa and VRDeck spending are the top 3 features, followed by CryoSleep.
- Tuned LightGBM is 0.82 points ahead (81.42%). That is within the fold-to-fold spread, so the two models are effectively tied.
- **More models (issue #19, with Role 2):** tuned CatBoost scores **81.77% ± 1.03%** and tuned XGBoost **81.15% ± 1.01%**. CatBoost is 0.35 points above LightGBM, which is within fold noise, so the final model stays LightGBM.

## 2. What I did
1. Built the model as `Pipeline([("prep", build_preprocessor()), ("model", RandomForestClassifier(random_state=SEED))])`, with the same `X = add_features(df).drop(columns=[TARGET])` as every other model (`04_random_forest.py`: `load_data()`, `make_pipeline()`).
2. Scored the default forest with `evaluate()` and logged it as `random_forest_default`.
3. Tuned it with `RandomizedSearchCV`: 30 random settings × 5 `get_cv()` folds = 150 fits, using `groups` and accuracy scoring. This took about 1 minute (`tune()`).
4. Saved the best settings to `results/best_params_random_forest.json` and scored them with `evaluate()` as `random_forest`, which is the name `07_final_model.py` looks for.
5. Overfitting check: compared the mean training-fold accuracy with the mean validation-fold accuracy, for both the default and the tuned model (`overfit_report()`).
6. Permutation importance, fold by fold on the validation folds only, 5 repeats each (`permutation_importance_cv()`). The chart is in `plots/` (`plot_importance()`).
7. Compared the forest with the baselines and LightGBM from `results.csv` (`compare_with_others()`).
8. Issue #19, with Role 2 (Champ): added XGBoost and CatBoost in the shared Pipeline (`04b_xgboost_catboost.py`). Each gets a default run, the same 30-setting random search on the shared folds, a saved `best_params_<name>.json`, a tuned row from `evaluate()` and an overfitting check. The script then compares both with tuned LightGBM using the one-std rule. It runs in about 3 minutes.

## 3. Key decisions and why
| Decision | Why | Evidence (number, chart or source) |
|---|---|---|
| Randomized search with 30 settings, not a full grid | The full grid has 4×5×5×4 = 400 settings (2,000 fits). 30 random ones find a good region quickly, and efficiency is graded | Search took 59 s. The top 5 settings are all within 0.06 points of each other (0.8054–0.8060) |
| Search `max_depth` and `min_samples_leaf` | A default forest grows trees until every leaf is pure, so it memorises its training rows | Default training accuracy 0.9995 vs validation 0.8005 |
| Search `max_features` (`sqrt`, `log2`, 0.3, 0.5) | Fewer features per split → more varied trees → a less correlated vote | Best: `sqrt` |
| `n_estimators` capped at 500 | Extra trees stop helping after a few hundred and only cost time | Best: 200 trees. The 500-tree settings didn't do better (0.8055, 0.8054) |
| `refit=False` in the search | `evaluate()` refits the best settings fold by fold, and the fit on all of train.csv is 07's job | Matches `05_lightgbm.py` |
| Permutation importance on the validation folds, not `feature_importances_` | The built-in importance is measured on training rows and favours columns with many distinct values (e.g. CabinNum). Permutation measures accuracy on unseen passengers | `plots/10_random_forest_permutation_importance.png` |
| #19: same Pipeline, folds and 30-setting budget for XGBoost and CatBoost | A difference in score must come from the model, not from better inputs or a bigger search | Issue #19 rules |
| #19: CatBoost gets the shared one-hot columns, not raw categories | CatBoost can read categories directly, but rule 4 says one preparation step for every model | `build_preprocessor()` |
| #19: 1000 iterations is in CatBoost's search space | It is CatBoost's default, so leaving it out would let tuning only do worse than the default run | Best: 1000 iterations |
| #19: XGBoost wrapped so it predicts True/False | XGBoost predicts 1/0 and the shared `evaluate()` expects True/False. The wrapper converts the answer only | `BoolXGBClassifier` in the script |
| #19: final model stays LightGBM | The issue's rule is to switch only for a win larger than one std. CatBoost is +0.35 points with std 1.12 | Script output, section 4 below |
| Parameters stored as plain lists, not scipy distributions | The JSON must match the logged params exactly. 07 checks this | `best_params_random_forest.json` = `params` in the `random_forest` row |

## 4. Results
From `results/results.csv` (mean over 5 GroupKFold folds, ± std where logged):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Fit time/fold |
|---|---|---|---|---|---|---|
| random_forest_default | 0.8005 ± 0.0128 | 0.8270 | 0.7638 | 0.7940 | 0.8852 | 1.25 s |
| **random_forest** (tuned) | **0.8060 ± 0.0131** | 0.8129 | 0.7991 | 0.8057 | 0.8950 | 2.05 s |

**Best parameters** (`results/best_params_random_forest.json`): `n_estimators=200`, `max_depth=20`, `min_samples_leaf=4`, `max_features="sqrt"`.

**Search ranges:** `n_estimators` [100, 200, 300, 500] · `max_depth` [None, 8, 12, 16, 20] · `min_samples_leaf` [1, 2, 4, 8, 16] · `max_features` ["sqrt", "log2", 0.3, 0.5].

**Overfitting check** (training-fold vs validation-fold accuracy):
| Model | Train | Validation | Gap |
|---|---|---|---|
| default | 0.9995 | 0.8005 | 19.9 points |
| tuned | 0.8978 | 0.8060 | 9.2 points |

A 9-point gap is normal for a forest, because every tree has seen most of its training rows. What matters is that validation accuracy went up while the gap shrank. The tuned forest relies less on memorising rows.

**Out-of-fold confusion matrix (tuned):** rows = true False/True, columns = predicted False/True:
```
[[3510  805]
 [ 881 3497]]
```
Errors are balanced: 805 false "transported" vs 881 missed transported passengers. Tuning raised recall from 0.764 to 0.799.

**Top features** (accuracy lost when the feature is shuffled, percentage points, mean ± std over the folds): FoodCourt_log 3.92 ± 0.41 · Spa_log 2.39 ± 0.61 · VRDeck_log 2.23 ± 0.72 · CryoSleep 1.62 ± 0.39 · ShoppingMall_log 1.61 ± 0.32 · TotalSpend_log 1.40 ± 0.25 · CabinNum 1.20 ± 0.26 · RoomService_log 1.19 ± 0.36 · Side 1.04 ± 0.40 · Deck 1.02 ± 0.47.

**Comparison:**
- vs Logistic Regression: **+3.20 points**, larger than the fold-to-fold spread (std ~ 1.31 points).
- vs tuned LightGBM: **−0.82 points**, within the fold-to-fold spread (std ~ 1.31 points). The forest is also about 11× slower per fold (2.05 s vs 0.18 s).

**XGBoost and CatBoost (issue #19, `04b_xgboost_catboost.py`)**

From `results/results.csv`, same folds and same 30-setting budget as the other tuned models:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Fit time/fold |
|---|---|---|---|---|---|---|
| xgboost_default | 0.8028 ± 0.0142 | 0.8083 | 0.7975 | 0.8028 | 0.8966 | 0.21 s |
| xgboost (tuned) | 0.8115 ± 0.0101 | 0.8147 | 0.8098 | 0.8121 | 0.9028 | 0.63 s |
| catboost_default | 0.8174 ± 0.0106 | 0.8117 | 0.8304 | 0.8208 | 0.9084 | 3.94 s |
| **catboost** (tuned) | **0.8177 ± 0.0103** | 0.8141 | 0.8269 | 0.8203 | 0.9085 | 2.55 s |
| lightgbm (tuned, Role 4, for reference) | 0.8142 ± 0.0112 | 0.8088 | 0.8267 | 0.8176 | 0.9025 | 0.17 s |

- **Best parameters, XGBoost:** `n_estimators=500`, `learning_rate=0.03`, `max_depth=8`, `min_child_weight=10`, `subsample=1.0`, `colsample_bytree=0.8`.
- **Best parameters, CatBoost:** `iterations=1000`, `learning_rate=0.05`, `depth=4`, `l2_leaf_reg=3`, `subsample=1.0`, `rsm=1.0`.
- **Search time:** XGBoost 22 s, CatBoost 88 s (30 settings × 5 folds each).
- **Overfitting (train vs validation):** XGBoost default 0.9435 vs 0.8028 (gap 14.1 points), tuned 0.8858 vs 0.8115 (7.4). CatBoost default 0.8853 vs 0.8174 (6.8), tuned 0.8817 vs 0.8177 (6.4).
- **Confusion matrices (tuned, out-of-fold, rows = true False/True):** XGBoost `[[3509, 806], [833, 3545]]`, CatBoost `[[3488, 827], [758, 3620]]`.
- Script output: "xgboost vs lightgbm: -0.28 points (within the fold-to-fold spread, std ~1.12 points)" and "catboost vs lightgbm: +0.35 points (within the fold-to-fold spread, std ~1.12 points)".
- **Reading:** the three boosting libraries land within 0.62 points of each other, well inside the 1.12-point fold noise. The features, not the library, set the ceiling at about 81 to 82%.
- CatBoost is the least overfit and barely needs tuning (+0.03 points over its default). It is also the slowest: about 15× LightGBM's fit time per fold.
- **Decision:** neither model beats LightGBM by more than one std, so `07_final_model.py` and `submission.csv` stay unchanged (issue #19 rule).

## 5. Charts for the slides
| File in plots/ | One-line caption (the finding) | Suggested slide |
|---|---|---|
| `plots/10_random_forest_permutation_importance.png` | Random Forest relies most on FoodCourt and Spa spending. VIP and GroupSize add nothing | Best model: feature importance / Random Forest results |

## 6. Bias and leakage checks
- Same folds as every other model: `get_cv()` (GroupKFold by travel group), with `groups` passed to both `RandomizedSearchCV.fit()` and `evaluate()`. No travel group is ever in both training and validation.
- All preprocessing (imputer, one-hot, scaler) is inside the Pipeline, so it is refitted on the training part of each fold.
- Tuning used cross-validation on train.csv only. `test.csv` is never read in `04_random_forest.py`.
- Permutation importance is measured on validation folds that each fold's model never saw.
- `random_state=SEED` for the forest, the search and the permutations. Two full re-runs gave identical scores.
- `n_jobs=-1` only parallelises the work; it doesn't change any score.
- One logged row per model name (`evaluate()` replaces old rows), so `results.csv` can't hold a best-of-many result.
- Known small optimism: the best of 30 settings is chosen on the same folds that score it, so the tuned score (and the +0.55-point gain over default) is a slight upper bound. LightGBM is tuned the same way, so RF vs LightGBM stays fair. Nested CV would remove this bias, at about 5× the cost.
- Issue #19: XGBoost and CatBoost use the same `get_cv()` folds, the same preprocessing inside the Pipeline, `random_state=SEED` and the same 30-setting budget. test.csv is never read. Each tuned score from the search equals the score from `evaluate()` (0.8115 and 0.8177), so the thread settings used to speed up the search do not change any result.
- Issue #19 caveat: we have now compared five tuned candidates on the same folds. Picking the top one by 0.35 points would be choosing on noise, which is why the one-std rule keeps LightGBM.
- Leakage review (leakage-reviewer agent, 2026-10-30): no leakage found. It reproduced the tuned score exactly, and 07 can rebuild the model from the saved params.

## 7. Problems and how I fixed them
- No pandas in the system Python → made a project `.venv` (gitignored) with Python 3.14 and `pip install -r requirements.txt`.
- The script crashed at the very end on Windows: cp1252 consoles can't print `≈` (`UnicodeEncodeError`) → replaced it with `~`. `05_lightgbm.py` has the same line and crashes when its output is piped or redirected on Windows (told Role 4).
- Default forest training accuracy was 99.95% → added `max_depth` and `min_samples_leaf` to the search. The gap fell from 19.9 to 9.2 points.
- Issue #19: the first run crashed in `evaluate()` because XGBoost predicts 1/0, not True/False → a small wrapper in the script converts its predictions. `src/evaluation.py` was not changed.
- Issue #19: CatBoost writes a `catboost_info/` folder by default → `allow_writing_files=False`.

## 8. Likely Q&A questions
| Question | Short answer |
|---|---|
| How does a Random Forest work? | Many decision trees, each trained on a bootstrap sample of passengers and a random subset of features at every split, then a majority vote. Averaging different trees cancels out most of each tree's noise. |
| Why is training accuracy 99.95% for the default forest? Isn't that overfitting? | Default trees grow until every leaf is pure, so each one memorises its rows. Validation accuracy (80.05%) is the honest number. Tuning `min_samples_leaf=4` and `max_depth=20` cut the gap from 19.9 to 9.2 points. |
| Why RandomizedSearchCV instead of GridSearchCV? | The full grid has 400 settings (2,000 fits). 30 random settings took about 1 minute, and the top 5 differ by only 0.06 points, so a bigger search wouldn't change the result. |
| Did tuning really help? | Only +0.55 points of accuracy, which is within the 1.31-point fold noise. The clearer gains are less overfitting (gap 19.9 → 9.2) and higher ROC-AUC (0.885 → 0.895). |
| Why permutation importance and not `feature_importances_`? | The built-in importance is computed on training data and inflates columns with many distinct values. Permutation importance measures the accuracy lost on unseen validation passengers. |
| Why is spending more important than CryoSleep here, when CryoSleep was the strongest single feature in the EDA? | Passengers in cryosleep spend nothing, so the spending columns already carry most of CryoSleep's information. Shuffling CryoSleep alone loses little because the forest can still read it from spending. Correlated features share credit. |
| Why GroupSize ≈ 0 importance when the EDA showed a difference? | Its bar on the chart is slightly below zero and its error bar crosses zero, so there's no measurable effect once the other features are known. Most likely the other columns already carry the same information. Permutation importance measures what a feature adds on top of the rest, not its effect on its own. |
| Isn't the tuned score biased if you tune and score on the same folds? | Slightly, yes: picking the best of 30 on the same folds flatters it a little. That's why we call the +0.55-point gain "within noise". Both tuned models are handled the same way, so the comparison is fair. Nested CV would fix it, at about 5× the cost. |
| Random Forest or LightGBM? | LightGBM is 0.82 points ahead, which is within fold noise, so they are effectively tied. LightGBM is also about 11× faster per fold. |
| Why try XGBoost and CatBoost? | To check that LightGBM was not just a lucky choice of library. All three are gradient boosting and all three land between 81.15% and 81.77%, inside the fold noise. |
| CatBoost has the highest score. Why is it not the final model? | It is 0.35 points ahead and the fold-to-fold std is 1.12 points, so the lead could be chance. Our rule, set in the issue before the run, was to switch only for a win larger than one std. LightGBM is also about 15× faster. |
| How do the three boosting libraries differ? | LightGBM grows the most useful leaf first. XGBoost grows level by level to a fixed depth. CatBoost grows symmetric trees and uses ordered boosting, which makes it harder to overfit. |
| Why does CatBoost not use its own category handling? | Every model must use the same preparation step (rule 4). With different inputs we could not say whether the model or the encoding made the difference. |
| How did you avoid leakage? | Same GroupKFold folds as everyone, preprocessing inside the Pipeline, tuning only by CV on train.csv, test.csv never read. |

## 9. Open items / handover
- [x] `04_random_forest.py`: `Pipeline(build_preprocessor(), RandomForestClassifier(random_state=SEED))`
- [x] Tuning: `RandomizedSearchCV` with `cv=get_cv()` and `groups`, searching `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`
- [x] Best parameters saved to `results/best_params_random_forest.json`
- [x] Tuned result logged with `evaluate()` (`random_forest` row in `results.csv`)
- [x] Overfitting check: training vs validation accuracy, gap explained
- [x] Permutation importance chart → `plots/10_random_forest_permutation_importance.png`
- [x] Report complete
- **For Role 1 (07):** `random_forest` row and `best_params_random_forest.json` are in place. LightGBM currently has the higher accuracy.
- **For Role 4:** `05_lightgbm.py` prints `≈`, which crashes on Windows when output is redirected. Its RF comparison also needs a re-run now that the `random_forest` row exists.
- [x] Issue #19 (Roles 3 and 2 part): `04b_xgboost_catboost.py`, rows `xgboost_default`, `xgboost`, `catboost_default`, `catboost`, and `best_params_xgboost.json` / `best_params_catboost.json`.
- **For Role 1:** `requirements.txt` now pins `xgboost==3.4.1` and `catboost==1.2.10`, please review. `07_final_model.py` is unchanged. `06_compare.py` was re-run and now names CatBoost as most accurate (within one std) while 07 ships LightGBM. Please confirm that is the story for the slides.
- **For Role 5/1 (`06_compare.py`):** add labels for `xgboost`, `xgboost_default`, `catboost` and `catboost_default` to `LABELS`, otherwise charts 12 and 13 show the raw names.
- **For Role 5:** the chart and numbers above are slide-ready. The Q&A table can go into `notes/qa_prep.md`.
