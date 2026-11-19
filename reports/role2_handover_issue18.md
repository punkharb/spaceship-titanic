# Issue #18 handover: two new features (Role 2, Champ)
Date: 2026-11-19 · Deadline: 22 November 2026

## What changed
`src/features.py` now creates two more features. No function name or signature changed.

| Feature | Definition | Type |
|---|---|---|
| `LuxuryShare` | (RoomService + Spa + VRDeck) / TotalSpend. 0 when nothing was spent. | numeric |
| `CabinRegion` | CabinNum in fixed blocks of 300: R0 to R6. | categorical |

Model inputs go from 29 to 37. Both features are row-wise and use no target.

## Why
- Logistic Regression: 0.7741 → **0.7951** (+2.10 pts, better on 5 of 5 folds).
- LightGBM, re-tuned: 0.8142 → **0.8167** (+0.25 pts, better on 4 of 5 folds).
- The LightGBM gain passes our keep rule narrowly (0.253 vs 0.248 needed). It held on 9 of 10 reshuffled splits.
- With the old saved LightGBM parameters the new features score 0.8132. Re-tuning is required.

Evidence: `python 02b_feature_experiments.py` and `python 02b_feature_experiments.py --retune`.

## What you need to do
Your scripts need **no code changes**. Work on the branch `role2/adopt-luxury-region` and push to it. The PR is merged only after steps 2 to 4 are in, so `main` never mixes old and new features. Run in this order, one role at a time, because each run rewrites `results/results.csv`.

| Order | Role | Run | Then |
|---|---|---|---|
| 1 | Role 2 (Champ) | 01, 02, 03 | Done. |
| 2 | Role 3 (Pat) | `python 04_random_forest.py` | Commit the new `best_params_random_forest.json`, `results.csv` and chart 10. Update your report. |
| 3 | Role 4 (Get) | `python 05_lightgbm.py` | Expect about 0.8167. Commit the new `best_params_lightgbm.json`, `results.csv` and chart 11. Update your report. |
| 4 | Role 1 (Pun) | `python 05b_nested_cv.py`, `06_compare.py`, `07_final_model.py` | Commit the new `submission.csv`. Update your report and the feature list in `CLAUDE.md`. |
| 5 | Role 5 (Bard) | Nothing to run | Update the slides from the reports once steps 2 to 4 are in. |

## Numbers that will change
- Every Random Forest and LightGBM score, their best parameters and their importance charts.
- The nested CV scores, the comparison table, charts 12 and 13, and `submission.csv`.
- Baseline slide: Logistic Regression is now 0.7951 ± 0.0105, ROC-AUC 0.8739.
- Chart 09 (coefficients) is already regenerated.

## For issue #19
Start new models and blends only after step 3, so they use the final features. Nothing in this change touches models or ensembling.

## Be honest about it in Q&A
- The gain for the final model is small, about 22 of 8,693 passengers, and within fold noise.
- It was found after about 11 tries on the same folds.
- It passes our keep rule only after re-tuning. With the saved parameters it fails.
- Nested CV nests the tuning but not the feature choice. The only untouched check is the Kaggle test score.
