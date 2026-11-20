# Role 5 – Analysis & presentation (Bard)
Last updated: 2026-11-20 · Status: In progress

## 1. Slide summary
- Three boosting libraries land within 0.62 points: CatBoost (tuned) 0.8177 ± 0.0103, LightGBM (tuned) 0.8142 ± 0.0112, XGBoost (tuned) 0.8115 ± 0.0101 (5 GroupKFold folds).
- CatBoost is slightly ahead of LightGBM (+0.35 pts, better on 4/5 folds), but the fold-by-fold check needs > 0.36 pts, so it is not a clear win; LightGBM fits in 0.17 s per fold against 2.55 s.
- LightGBM clearly beats tuned Random Forest (0.8060 ± 0.0131): +0.82 pts and better on all 5 folds (threshold 0.39 pts).
- Tuning matters little: CatBoost default (0.8174 ± 0.0106) is almost the same as tuned; every tree model beats Logistic Regression (0.7741 ± 0.0071).
- Every model beats the Dummy floor of 0.4963, so they all learned real signal.

## 2. What I did
Numbered steps in the order done, each with the script/function that does it.

1. Created the five role reports in `reports/` from the CLAUDE.md template, each with its role's deliverables as a checklist (`/create-reports`).
2. Wrote `06_compare.py` with Role 1 (Pun), test first: `python 06_compare.py --self-test` checks `latest_per_model()`, `comparison_table()` and `conclusion()` on a small made-up table (including a model logged twice), then the implementation was written until it passed.
3. `06_compare.py` reads `results/results.csv` only (no retraining, no recomputed scores) and keeps the latest row per model name (`latest_per_model()`).
4. `comparison_table()` writes `results/model_comparison.csv`, sorted by mean CV accuracy; `conclusion()` prints the winner, its gap to the best model of a *different* algorithm (so tuned vs untuned CatBoost is not mistaken for the model choice), the tuning gain over its own default, and whether each gap is within one std. A nested-CV line is printed only when the winner has a nested row; the Dummy claim only when it holds.
5. `plot_accuracy()` → `plots/12_model_accuracy.png`; `plot_prf()` → `plots/13_precision_recall_f1.png` (top 3 models, the F1 leader highlighted).
6. Added display labels (`LABELS` in `06_compare.py`) for the new rows `xgboost`, `xgboost_default`, `catboost`, `catboost_default` (PR #28, Roles 3 and 2) and `lgbm_rf_blend` (Role 4), and re-ran the script.
7. Used Role 1's `06b_paired_check.py` to compare the challengers with LightGBM fold by fold on the shared folds, because the ±1 std overlap in `06_compare.py` cannot tell whether a small lead is real (results in section 4).

## 3. Key decisions and why
| Decision | Why | Evidence (number, chart or source) |
|---|---|---|
| Comparison reads `results.csv` only, never retrains | Rule 3: every slide number must trace to `evaluate()`; retraining could give numbers different from the role reports | `06_compare.py` reads only `results/results.csv`; its table matches `results.csv` row for row |
| Keep the latest row per model name | A re-run appends a new row; the newest run is the one the owner's report describes | `latest_per_model()`; covered by the self-test (duplicate `rf` row) |
| Dot + ±1 std plot, not bars | A bar must start at 0, which would squash the 0.77–0.82 differences into nearly identical bars; a dot plot can zoom the axis honestly | `plots/12_model_accuracy.png` |
| Neutral `#8a8984` + dark ink for the winner, not blue/orange | The team reserves `#2a78d6` / `#eb6834` for transported / not transported; reusing them for models would confuse the audience | CLAUDE.md shared rule 8 |
| Small multiples (one panel per metric) for precision/recall/F1 instead of a 3-colour grouped bar | Same colour reason, and each metric gets its own readable panel | `plots/13_precision_recall_f1.png` |
| Dummy shown as a dashed floor line, not a dot | It is a reference point (0.4963), not a competitor | `plots/12_model_accuracy.png` |
| Judge "is model A better than B?" fold by fold, not by std overlap | All models use the same folds, so fold difficulty cancels; a clear win needs mean per-fold gain > 2 × standard error | `06b_paired_check.py`: LightGBM vs RF 0.82 pts (needs > 0.39, clear); CatBoost vs LightGBM +0.35 pts (needs > 0.36, not clear) |
| Ship LightGBM, not CatBoost | CatBoost's lead is not a clear win fold by fold; LightGBM was the pre-set choice and fits about 15× faster | `06b_paired_check.py` output; `results.csv` fit times 2.5505 s (CatBoost) vs 0.1735 s (LightGBM) per fold |
| No separate task board or meeting notes file | No regular meetings are planned, and a Markdown board would duplicate the `CLAUDE.md` Status checklist, each report's section 9 and the GitHub PRs | Teamwork evidence = `CLAUDE.md` Status checklist + reviewed, merged PRs (#1–#4 so far) |

## 4. Results
Copied from the printed output of `python 06_compare.py` (run 2026-11-20); also saved as `results/model_comparison.csv`. CV = 5-fold GroupKFold by travel group.

| Model | Owner | Accuracy (mean ± std) | Precision | Recall | F1 | ROC-AUC | Fit time (s) |
|---|---|---|---|---|---|---|---|
| catboost (tuned) | Roles 3 and 2 | 0.8177 ± 0.0103 | 0.8141 | 0.8269 | 0.8203 | 0.9085 | 2.5505 |
| catboost_default | Roles 3 and 2 | 0.8174 ± 0.0106 | 0.8117 | 0.8304 | 0.8208 | 0.9084 | 3.9357 |
| lightgbm (tuned) | Role 4 | 0.8142 ± 0.0112 | 0.8088 | 0.8267 | 0.8176 | 0.9025 | 0.1735 |
| xgboost (tuned) | Roles 3 and 2 | 0.8115 ± 0.0101 | 0.8147 | 0.8098 | 0.8121 | 0.9028 | 0.6325 |
| lightgbm_nested | Role 1 | 0.8109 ± 0.0099 | 0.8056 | 0.8234 | 0.8142 | 0.9042 | 35.7492* |
| lightgbm_default | Role 4 | 0.8106 ± 0.0121 | 0.8080 | 0.8188 | 0.8132 | 0.9030 | 0.1002 |
| lgbm_rf_blend | Role 4 | 0.8093 ± 0.0129 | 0.8078 | 0.8154 | 0.8114 | 0.9022 | 1.3082 |
| random_forest (tuned) | Role 3 | 0.8060 ± 0.0131 | 0.8129 | 0.7991 | 0.8057 | 0.8950 | 2.0473 |
| xgboost_default | Roles 3 and 2 | 0.8028 ± 0.0142 | 0.8083 | 0.7975 | 0.8028 | 0.8966 | 0.2056 |
| random_forest_nested | Role 1 | 0.8025 ± 0.0133 | 0.8078 | 0.7979 | 0.8026 | 0.8944 | 53.2444* |
| random_forest_default | Role 3 | 0.8005 ± 0.0128 | 0.8270 | 0.7638 | 0.7940 | 0.8852 | 1.2509 |
| logistic_regression | Role 2 | 0.7741 ± 0.0071 | 0.7668 | 0.7922 | 0.7792 | 0.8525 | 0.0448 |
| dummy_most_frequent | Role 2 | 0.4963 ± 0.0141 | 0.3999 | 0.8000 | 0.5332 | 0.5000 | 0.0025 |

\* Nested rows come from `05b_nested_cv.py` (Role 1, issue #16): an honest re-score of the tuned model, not a separate model. Their fit time includes the whole inner search, so it is not comparable.

Printed conclusion (verbatim): "CatBoost (tuned) has the highest mean CV accuracy, 0.8177 ± 0.0103, 0.3 points ahead of LightGBM (tuned) (0.8142 ± 0.0112). The gap is within one standard deviation, so the difference may not be meaningful. It fits in 2.55 s per fold against 0.17 s. Tuning adds 0.0 points over CatBoost (default) (0.8174 ± 0.0106), also within one standard deviation. Every model beats the Dummy floor of 0.4963, so they all learned real signal."

The gaps in the conclusion are printed by the script. Extra gaps for Q&A, by plain subtraction of the printed means: tuning added 0.0055 accuracy to Random Forest (0.8060 vs 0.8005), 0.0036 to LightGBM (0.8142 vs 0.8106) and 0.0087 to XGBoost (0.8115 vs 0.8028); the three tuned boosting models span 0.0062 (0.8115 to 0.8177).

**Fold-by-fold check:** copied from the printed output of `python 06b_paired_check.py` (Role 1, run 2026-11-20). Each model is refitted with its saved `best_params_<name>.json` on the shared `get_cv()` folds (no re-tuning); the means equal the `results.csv` rows. Rule: a challenger is a clear win only if its mean per-fold gain is more than 2 × the standard error of the per-fold gains.

| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|---|---|---|---|---|---|---|
| lightgbm | 0.8281 | 0.8217 | 0.8189 | 0.7986 | 0.8038 | 0.8142 |
| random_forest | 0.8194 | 0.8212 | 0.8074 | 0.7894 | 0.7929 | 0.8060 |
| catboost | 0.8321 | 0.8269 | 0.8154 | 0.8055 | 0.8084 | 0.8177 |

Verdicts (verbatim):
- "random_forest vs lightgbm: gain -0.82 pts (per fold -0.86, -0.06, -1.15, -0.92, -1.09; better on 0/5); needs > 2 x SE = 0.39 pts -> not a clear win: lightgbm stays the final model."
- "catboost vs lightgbm: gain +0.35 pts (per fold +0.40, +0.52, -0.35, +0.69, +0.46; better on 4/5); needs > 2 x SE = 0.36 pts -> not a clear win: lightgbm stays the final model."

Read the first verdict the other way round: LightGBM beats tuned Random Forest on all 5 folds, by 0.82 pts on average, more than the 0.39-pt threshold, so that is a clear win for LightGBM. XGBoost is not checked by the script because it scores below LightGBM on average (0.8115 vs 0.8142).

## 5. Charts for the slides
| File in plots/ | One-line caption (the finding) | Suggested slide |
|---|---|---|
| `plots/12_model_accuracy.png` | CatBoost (tuned) is most accurate (0.8177), 0.3 points ahead of LightGBM (tuned) (0.8142), a gap within one std; all tree and boosting models sit between 0.8005 and 0.8177 with overlapping ±1 std bars, far above the Dummy floor (0.4963) | Slide 10 – Model comparison |
| `plots/13_precision_recall_f1.png` | Among the top 3 by accuracy, CatBoost (default) leads on F1 (0.8208) and recall (0.8304), CatBoost (tuned) has the best precision (0.8141), and LightGBM (tuned) is close behind (F1 0.8176) | Slide 10 (second half) or Slide 11 – Best model |

## 6. Bias and leakage checks
- **Numbers only from `evaluate()`:** `06_compare.py` reads `results/results.csv` and nothing else; it does not retrain or recompute any score (shared rule 3).
- **Same folds for every model:** every row in `results.csv` comes from `evaluate()` with `get_cv()` (GroupKFold, 5 folds by travel group, `SEED = 42`), so all models are compared on identical splits.
- **No test.csv:** the comparison and the model choice use CV scores on train.csv only.
- **Uncertainty shown, not hidden:** the chart and the conclusion report ±1 std and say when a gap is within it, so a small lead is not over-sold.
- **Paired check before switching models:** `06b_paired_check.py` refits the saved best parameters (no re-tuning) on the same folds; a challenger replaces LightGBM only with a clear fold-by-fold win, so we do not chase a lead that could be fold luck.
- **Honest axis:** dot plot with a labelled Dummy floor instead of a truncated bar chart, so the visual gap matches the numeric gap.

## 7. Problems and how I fixed them
Short bullets: issue → fix. These make good "lessons learned" slides.
- A re-run appends a second row for the same model to `results.csv` → `latest_per_model()` keeps the newest row per model and prints a note; checked by the self-test.
- Bars must start at 0, so 0.77 vs 0.82 would look identical → dot plot with ±1 std error bars.
- Blue/orange are reserved for the target classes → team neutral plus dark ink for the winner, and small multiples instead of a 3-colour grouped bar.
- "Within one std" suggested the LightGBM vs Random Forest gap might not be real, but each model's std also contains how hard each fold is → the fold-by-fold check (`06b_paired_check.py`) cancels fold difficulty and shows LightGBM wins on all 5 folds.

## 8. Likely Q&A questions
| Question | Short answer |
|---|---|
| Is LightGBM really better than Random Forest? | Yes. On the same 5 folds LightGBM beats tuned RF on every fold, by 0.82 pts on average; the clear-win threshold (2 × standard error) is 0.39 pts. The means (0.8142 ± 0.0112 vs 0.8060 ± 0.0131) overlap within one std only because each std includes fold difficulty, which cancels when comparing fold by fold. |
| CatBoost scores highest, why not ship it? | Its lead over LightGBM is +0.35 pts and it is better on 4/5 folds, but a clear win needs > 0.36 pts, so it is not a clear win. LightGBM fits about 15× faster (0.17 s vs 2.55 s per fold) and was the pre-set choice, so it stays. |
| Why compare fold by fold? | Every model uses the same `get_cv()` folds, so how hard each fold is cancels out when we subtract scores fold by fold. The spread of the per-fold gain is then the right noise measure, not the ±1 std of each model's mean. |
| Why does tuning add so little? | The tree models are already strong with default settings: tuning added 0.0036 (LightGBM), 0.0055 (RF) and 0.0087 (XGBoost) accuracy, and CatBoost tuned (0.8177) is almost the same as default (0.8174), all within one std. Most of the gain comes from the features and the model family, not the hyperparameters. |
| Where do the numbers on the comparison slide come from? | Only from `results/results.csv`, written by `evaluate()` in each model script. `06_compare.py` copies them; it never retrains. |
| Were all models compared fairly? | Yes: the same `get_cv()` GroupKFold folds (by travel group), the same `build_preprocessor()` inside a Pipeline, the same seed, and tuning only through CV on train.csv. |
| Why a dot plot and not a bar chart? | Bars must start at 0, which hides 0.77–0.82 differences; a dot with ±1 std shows both the gap and the uncertainty honestly. |
| Why is the Dummy's recall 0.8000? | It always predicts the majority class of the training folds; recall is 1 on folds where that class is "True" and 0 otherwise, so the 5-fold mean is 0.8000. Its accuracy (0.4963) is the floor every model must beat. |

## 9. Open items / handover
What's not done yet and who needs what from me.

- [x] Create the report template files for all 5 roles in `reports/`
- [ ] Start the slides: problem → data → EDA findings → pipeline diagram → how we avoided bias → model comparison → best model (confusion matrix + feature importance) → limitations & next steps
- [ ] Build slides from `reports/`: read reports regularly, ask owners when a section is empty or a number doesn't match `results.csv`
- [ ] Keep this report up to date: slide plan with source reports, missing content, rehearsal notes (timing, who presents what)
- [ ] `notes/qa_prep.md`: collect section 8 from every report + likely questions ("Why GroupKFold?", "Why accuracy?", "Why drop Name?", "How did you prevent leakage?", "Why this model?"); make sure every member rehearses them
- [x] `06_compare.py` (with Role 1): comparison table and chart from `results.csv` (mean ± std across folds) – script, `results/model_comparison.csv` and plots 12–13 done; merged in PR #13. Nested-CV rows added by Role 1 (issue #16): they show in the table and chart 12 but never as the runner-up or in chart 13. Labels for XGBoost, CatBoost and the LightGBM + RF blend added 2026-11-20
- [ ] Put the team's agreed wording on slide 10: "Three boosting libraries land within 0.62 points (81.15%–81.77%). CatBoost is slightly ahead (+0.35 pts) but not a clear win fold by fold, so we ship LightGBM: about 15× faster to fit. LightGBM does clearly beat Random Forest (better on all 5 folds)." Checked against `results.csv` and `06b_paired_check.py`: xgboost 0.8115, catboost 0.8177 (gap 0.0062); fit times 2.5505 s vs 0.1735 s per fold; CatBoost +0.35 pts (needs > 0.36); LightGBM better than RF on 5/5 folds.
- [ ] Re-run `python 06_compare.py` whenever a model script is re-run, so the table and plots match `results.csv`
- [x] Teamwork evidence: decided to use the `CLAUDE.md` Status checklist + GitHub PRs instead of a task board and `notes/meetings.md` (see section 3). `CLAUDE.md` updated to match (Pun to review the PR).
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
| 10 | Model comparison (mean ± std) | `role5_presentation.md` (`06_compare.py`, `results.csv`) | Bard + Pun | Content ready (plots 12–13, table in section 4) |
| 11 | Best model: confusion matrix + feature importance | `role1_lead.md`, `role3_random_forest.md` / `role4_lightgbm.md` | Pun | Waiting on content |
| 12 | Final model and submission | `role1_lead.md` | Pun | Waiting on content |
| 13 | Limitations and next steps | all reports (sections 7 and 9) | Bard | Not started |
