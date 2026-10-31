# Role 5 – Analysis & presentation (Bard)
Last updated: 2026-10-31 · Status: In progress

## 1. Slide summary
- Tuned LightGBM has the highest mean CV accuracy: **0.8142 ± 0.0112** (5 GroupKFold folds).
- Its lead is small: 0.8 points ahead of the best other algorithm, Random Forest tuned (0.8060 ± 0.0131), a gap within one standard deviation, so it may not be meaningful; it fits in 0.17 s per fold against 2.05 s.
- Tuning added only 0.4 points over LightGBM default (0.8106 ± 0.0121), also within one standard deviation.
- The four tree models are close together (0.8005–0.8142); all beat Logistic Regression (0.7741 ± 0.0071).
- Every model beats the Dummy floor of 0.4963, so they all learned real signal.
- Tuned LightGBM also has the best recall (0.8267) and F1 (0.8176); Random Forest tuned has slightly higher precision (0.8129).

## 2. What I did
Numbered steps in the order done, each with the script/function that does it.

1. Created the five role reports in `reports/` from the CLAUDE.md template, each with its role's deliverables as a checklist (`/create-reports`).
2. Wrote `06_compare.py` with Role 1 (Pun), test first: `python 06_compare.py --self-test` checks `latest_per_model()`, `comparison_table()` and `conclusion()` on a small made-up table (including a model logged twice), then the implementation was written until it passed.
3. `06_compare.py` reads `results/results.csv` only (no retraining, no recomputed scores) and keeps the latest row per model name (`latest_per_model()`).
4. `comparison_table()` writes `results/model_comparison.csv`, sorted by mean CV accuracy; `conclusion()` prints the winner, its gap to the best model of a *different* algorithm (so tuned vs untuned LightGBM is not mistaken for the model choice), the tuning gain over its own default, and whether each gap is within one std. The Dummy claim is only printed when it holds.
5. `plot_accuracy()` → `plots/12_model_accuracy.png`; `plot_prf()` → `plots/13_precision_recall_f1.png` (top 3 models).

## 3. Key decisions and why
| Decision | Why | Evidence (number, chart or source) |
|---|---|---|
| Comparison reads `results.csv` only, never retrains | Rule 3: every slide number must trace to `evaluate()`; retraining could give numbers different from the role reports | `06_compare.py` reads only `results/results.csv`; its table matches `results.csv` row for row |
| Keep the latest row per model name | A re-run appends a new row; the newest run is the one the owner's report describes | `latest_per_model()`; covered by the self-test (duplicate `rf` row) |
| Dot + ±1 std plot, not bars | A bar must start at 0, which would squash the 0.77–0.81 differences into nearly identical bars; a dot plot can zoom the axis honestly | `plots/12_model_accuracy.png` |
| Neutral `#8a8984` + dark ink for the winner, not blue/orange | The team reserves `#2a78d6` / `#eb6834` for transported / not transported; reusing them for models would confuse the audience | CLAUDE.md shared rule 8 |
| Small multiples (one panel per metric) for precision/recall/F1 instead of a 3-colour grouped bar | Same colour reason, and each metric gets its own readable panel | `plots/13_precision_recall_f1.png` |
| Dummy shown as a dashed floor line, not a dot | It is a reference point (0.4963), not a competitor | `plots/12_model_accuracy.png` |
| No separate task board or meeting notes file | No regular meetings are planned, and a Markdown board would duplicate the `CLAUDE.md` Status checklist, each report's section 9 and the GitHub PRs | Teamwork evidence = `CLAUDE.md` Status checklist + reviewed, merged PRs (#1–#4 so far) |

## 4. Results
Copied from the printed output of `python 06_compare.py` (run 2026-10-31); also saved as `results/model_comparison.csv`. CV = 5-fold GroupKFold by travel group.

| Model | Owner | Accuracy (mean ± std) | Precision | Recall | F1 | ROC-AUC | Fit time (s) |
|---|---|---|---|---|---|---|---|
| lightgbm (tuned) | Role 4 | 0.8142 ± 0.0112 | 0.8088 | 0.8267 | 0.8176 | 0.9025 | 0.1735 |
| lightgbm_default | Role 4 | 0.8106 ± 0.0121 | 0.8080 | 0.8188 | 0.8132 | 0.9030 | 0.1002 |
| random_forest (tuned) | Role 3 | 0.8060 ± 0.0131 | 0.8129 | 0.7991 | 0.8057 | 0.8950 | 2.0473 |
| random_forest_default | Role 3 | 0.8005 ± 0.0128 | 0.8270 | 0.7638 | 0.7940 | 0.8852 | 1.2509 |
| logistic_regression | Role 2 | 0.7741 ± 0.0071 | 0.7668 | 0.7922 | 0.7792 | 0.8525 | 0.0448 |
| dummy_most_frequent | Role 2 | 0.4963 ± 0.0141 | 0.3999 | 0.8000 | 0.5332 | 0.5000 | 0.0025 |

Printed conclusion (verbatim): "LightGBM (tuned) has the highest mean CV accuracy, 0.8142 ± 0.0112, 0.8 points ahead of Random Forest (tuned) (0.8060 ± 0.0131). The gap is within one standard deviation, so the difference may not be meaningful. It fits in 0.17 s per fold against 2.05 s. Tuning adds 0.4 points over LightGBM (default) (0.8106 ± 0.0121), also within one standard deviation. Every model beats the Dummy floor of 0.4963, so they all learned real signal."

The 0.8- and 0.4-point gaps above are printed by the script. One extra gap for Q&A, by plain subtraction of the printed means: tuning added 0.0055 accuracy to Random Forest (0.8060 vs 0.8005).

## 5. Charts for the slides
| File in plots/ | One-line caption (the finding) | Suggested slide |
|---|---|---|
| `plots/12_model_accuracy.png` | Tuned LightGBM is most accurate (0.8142), but the four tree models' ±1 std bars overlap; every model is far above the Dummy floor (0.4963) | Slide 10 – Model comparison |
| `plots/13_precision_recall_f1.png` | Tuned LightGBM leads on F1 (0.8176) and recall (0.8267); Random Forest tuned trades recall (0.7991) for slightly higher precision (0.8129) | Slide 10 (second half) or Slide 11 – Best model |

## 6. Bias and leakage checks
- **Numbers only from `evaluate()`:** `06_compare.py` reads `results/results.csv` and nothing else; it does not retrain or recompute any score (shared rule 3).
- **Same folds for every model:** every row in `results.csv` comes from `evaluate()` with `get_cv()` (GroupKFold, 5 folds by travel group, `SEED = 42`), so all models are compared on identical splits.
- **No test.csv:** the comparison and the model choice use CV scores on train.csv only.
- **Uncertainty shown, not hidden:** the chart and the conclusion report ±1 std and say when a gap is within it, so a small lead is not over-sold.
- **Honest axis:** dot plot with a labelled Dummy floor instead of a truncated bar chart, so the visual gap matches the numeric gap.

## 7. Problems and how I fixed them
Short bullets: issue → fix. These make good "lessons learned" slides.
- A re-run appends a second row for the same model to `results.csv` → `latest_per_model()` keeps the newest row per model and prints a note; checked by the self-test.
- Bars must start at 0, so 0.77 vs 0.81 would look identical → dot plot with ±1 std error bars.
- Blue/orange are reserved for the target classes → team neutral plus dark ink for the winner, and small multiples instead of a 3-colour grouped bar.

## 8. Likely Q&A questions
| Question | Short answer |
|---|---|
| Is LightGBM really better than Random Forest? | Not clearly. Tuned LightGBM 0.8142 ± 0.0112 vs tuned RF 0.8060 ± 0.0131: a 0.0082 gap, smaller than either std. LightGBM is best on average, has the best F1 (0.8176) and recall (0.8267), and fits much faster (0.1735 s vs 2.0473 s), so it is a reasonable pick, not a proven winner. |
| Why does tuning add so little? | Both tree models are already strong with default settings; tuning added 0.0036 (LightGBM) and 0.0055 (RF) accuracy, both within one std. Most of the gain comes from the features and the model family, not the hyperparameters. |
| Where do the numbers on the comparison slide come from? | Only from `results/results.csv`, written by `evaluate()` in each model script. `06_compare.py` copies them; it never retrains. |
| Were all models compared fairly? | Yes: the same `get_cv()` GroupKFold folds (by travel group), the same `build_preprocessor()` inside a Pipeline, the same seed, and tuning only through CV on train.csv. |
| Why a dot plot and not a bar chart? | Bars must start at 0, which hides 0.77–0.81 differences; a dot with ±1 std shows both the gap and the uncertainty honestly. |
| Why is the Dummy's recall 0.8000? | It always predicts the majority class of the training folds; recall is 1 on folds where that class is "True" and 0 otherwise, so the 5-fold mean is 0.8000. Its accuracy (0.4963) is the floor every model must beat. |

## 9. Open items / handover
What's not done yet and who needs what from me.

- [x] Create the report template files for all 5 roles in `reports/`
- [ ] Start the slides: problem → data → EDA findings → pipeline diagram → how we avoided bias → model comparison → best model (confusion matrix + feature importance) → limitations & next steps
- [ ] Build slides from `reports/`: read reports regularly, ask owners when a section is empty or a number doesn't match `results.csv`
- [ ] Keep this report up to date: slide plan with source reports, missing content, rehearsal notes (timing, who presents what)
- [ ] `notes/qa_prep.md`: collect section 8 from every report + likely questions ("Why GroupKFold?", "Why accuracy?", "Why drop Name?", "How did you prevent leakage?", "Why this model?"); make sure every member rehearses them
- [x] `06_compare.py` (with Role 1): comparison table and chart from `results.csv` (mean ± std across folds) – script, `results/model_comparison.csv` and plots 12–13 done; not yet committed or merged (Pun to review)
- [ ] State on slide 10 that tuned LightGBM's lead over LightGBM default (0.4 pts: 0.8142 vs 0.8106) and over Random Forest tuned (0.8142 vs 0.8060) is within one std, so the choice rests on the best mean plus speed, not on a proven difference
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
