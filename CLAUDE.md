# Spaceship Titanic – Classification Project (5-person team)

Data Science assignment: train a classification model in Python that predicts
whether a passenger was **`Transported`** (True/False).

- **Data:** `train.csv` (8,693 labelled rows), `test.csv` (4,277 unlabelled rows), Kaggle "Spaceship Titanic".
- **Columns:** `PassengerId, HomePlanet, CryoSleep, Cabin, Destination, Age, VIP, RoomService, FoodCourt, ShoppingMall, Spa, VRDeck, Name` (+ `Transported` in train only).
- **Final outputs:** a reproducible pipeline of numbered `.py` scripts, a model comparison, `submission.csv` for test.csv, and a presentation.

## How this project is graded

| Weight | Criterion | What it means for us |
|---|---|---|
| 50% | DS/DM pipeline: completeness, clarity, efficiency, correctness, **minimum bias** | Every stage is present, readable and justified, with **no data leakage** and fair validation |
| 20% | ML model performance | A strong, honestly validated model. Don't over-tune. |
| 20% | Presentation | A clear story with charts |
| 5% | Q&A | **Everyone** can explain the **whole** pipeline, not just their own part |
| 5% | Teamwork | Up-to-date Status checklist (below), reviewed and merged PRs on GitHub |

Pipeline quality is worth more than accuracy. When in doubt, choose the cleaner, better-justified option.

---

## Project structure and file ownership

Run every script **from the project folder** (`python 01_explore.py`) so that `from src... import ...` works.
Only edit the files your role owns. Changes to `src/` go through **Role 1**.

```
DS Model Training/
├── CLAUDE.md                 # this file (Role 1 keeps it up to date)
├── requirements.txt          # Role 1
├── train.csv, test.csv       # raw data – NEVER edited
├── src/
│   ├── __init__.py
│   ├── config.py             # Role 1 – paths, TARGET, SEED, N_SPLITS
│   ├── evaluation.py         # Role 1 – get_groups(), get_cv(), evaluate()
│   └── features.py           # Role 2 – add_features(), build_preprocessor(), feature lists
├── 01_explore.py             # Role 2 – EDA
├── 02_prepare.py             # Role 2 – demonstrates and checks the feature and preprocessing steps
├── 03_baseline.py            # Role 2 – Dummy + Logistic Regression
├── 04_random_forest.py       # Role 3 – Random Forest + tuning
├── 05_lightgbm.py            # Role 4 – LightGBM + tuning
├── 06_compare.py             # Role 5 (with Role 1) – comparison table and charts from results.csv
├── 07_final_model.py         # Role 1 – retrain the best model on all of train.csv, write submission.csv
├── plots/                    # all charts, numbered NN_name.png (used in slides)
├── results/                  # results.csv, best_params_<model>.json, eda_findings.txt
├── reports/                  # one report per role – the source for the slides (see below)
│   ├── role1_lead.md
│   ├── role2_data_baseline.md
│   ├── role3_random_forest.md
│   ├── role4_lightgbm.md
│   └── role5_presentation.md
├── slides/                   # Role 5
└── notes/                    # Role 5 – qa_prep.md
```

`requirements.txt`: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `lightgbm`. Pin the versions the team actually uses.

---

## Shared rules (everyone must follow)

These keep the model comparison **fair** (minimum bias) and the code **mergeable**.

1. **One configuration.** Import paths, `TARGET = "Transported"`, `SEED = 42` and `N_SPLITS = 5` from `src/config.py`. Never hard-code them.
2. **One validation split.** Use `get_cv()`, a `GroupKFold(n_splits=5)` grouped by travel group (the `gggg` part of `PassengerId` = `gggg_pp`). Members of a group tend to share circumstances, so a random split would put the same group in both training and validation and inflate scores. Role 2's EDA must confirm the reason for this: no travel group should appear in both train.csv and test.csv.
3. **One scoring function.** Every model is scored with `evaluate()`, which appends a row to `results/results.csv`. No hand-computed scores in slides.
4. **One preparation step.** Every model uses `Pipeline([("prep", build_preprocessor()), ("model", ...)])`.
5. **No data leakage:**
   - Anything that **learns from data** (imputer medians/modes, scaler, one-hot categories) lives **inside** the Pipeline, so it is fitted on the training folds only.
   - `add_features()` may only do **row-wise rules** that use no target and no statistics from other rows, plus group counts within the same file.
   - `Transported` is never a feature. `test.csv` is never used for tuning or model choice.
   - Hyperparameter tuning happens **only** through cross-validation on train.csv.
6. **Reproducible.** Set `random_state=SEED` everywhere, and re-run every script from a clean state before merging.
7. **Clear code.** Short functions, a docstring at the top of each script explaining its purpose, comments on *why* rather than just *what*, and printed conclusions in plain English.
8. **Charts:** saved to `plots/` with numbered names and a title that states the finding. Use the same colours across the team: transported `#2a78d6`, not transported `#eb6834`, neutral `#8a8984`.

### Shared interfaces (contracts – don't change without telling the team)

```python
# src/config.py            (Role 1)
DATA_DIR, TRAIN_PATH, TEST_PATH, PLOT_DIR, RESULTS_DIR
TARGET = "Transported"; ID_COL = "PassengerId"; SEED = 42; N_SPLITS = 5

# src/features.py          (Role 2)
add_features(df) -> pd.DataFrame     # row-wise feature creation, safe before the split
build_preprocessor() -> ColumnTransformer   # imputation + encoding + scaling (fitted inside CV)
NUMERIC_FEATURES, CATEGORICAL_FEATURES, DROP_COLUMNS

# src/evaluation.py        (Role 1)
get_groups(df) -> pd.Series          # travel group from PassengerId
get_cv() -> GroupKFold
evaluate(pipeline, X, y, groups, name, owner, params=None) -> dict
    # cross-validated accuracy, precision, recall, F1, ROC-AUC (mean ± std) + fit time
    # also out-of-fold confusion matrix; appends one row to results/results.csv
```

`results/results.csv` columns: `model, owner, accuracy_mean, accuracy_std, precision_mean, recall_mean, f1_mean, roc_auc_mean, fit_time_s, params, date`.

**For `07_final_model.py` to find the tuned models:**
- Log the tuned model with `evaluate(..., name="random_forest", params=best_params)` / `name="lightgbm"`, and save the same dict to `results/best_params_<name>.json`. Use a different name for untuned runs (e.g. `"random_forest_default"`).
- Every model script builds `X = add_features(df).drop(columns=[TARGET])`; `build_preprocessor()` selects its own columns. 07 does the same, so the final fit matches CV.

---

## Role reports (`reports/`) – what each role has done

Every role keeps **one Markdown report** describing what they have done, using the **same template**.
Role 5 builds the slides from these reports instead of chasing people or reading code.

**Rules**
- **Write as you go, not at the end:** update your report every time you finish a task or make a decision, and commit it together with your code.
- **Use real numbers only:** copy scores from `results/results.csv` and counts from your script's printed output. Never estimate or round differently from the source.
- **Point to charts, don't paste them:** give the path in `plots/` plus a one-line caption that states the finding.
- **Keep it slide-ready:** short bullets, plain English, no code dumps. A code snippet is fine only when the slide needs it.
- Update the **Last updated** line each time.

**Template** (copy into your report file)

```markdown
# Role N – <role name> (<your name>)
Last updated: YYYY-MM-DD · Status: In progress / Done

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
```

**Which report feeds which slides**

| Report | Slide sections |
|---|---|
| `role1_lead.md` | Pipeline overview diagram, validation method (GroupKFold), final model, submission |
| `role2_data_baseline.md` | Data overview, EDA findings, data cleaning and features, baseline results |
| `role3_random_forest.md` | Random Forest: tuning, results, feature importance |
| `role4_lightgbm.md` | LightGBM: tuning, results, comparison with Random Forest |
| `role5_presentation.md` | Slide plan, which report each slide uses, missing content, rehearsal notes |

---

## Roles and responsibilities

### Role 1 – Lead / integrator ([Pun])
*Best fit: the strongest coder.*

**Does**
- Sets up the shared repo or folder, `requirements.txt`, the `src/` package and `config.py`. Keeps this CLAUDE.md up to date.
- Writes `src/evaluation.py`: the fixed `GroupKFold` split, `evaluate()` and logging to `results.csv`.
- Reviews every merge for leakage (rule 5), reproducibility and shared-rule compliance, then merges.
- Writes `07_final_model.py`: retrains the chosen model on **all** of train.csv with its best parameters, predicts test.csv, and writes `submission.csv` (`PassengerId, Transported` as True/False).
- Runs the whole pipeline from 01 to 07 from a clean state before the deadline.
- Pairs with Role 5 to make sure the technical slides are correct.
- Keeps **`reports/role1_lead.md`** up to date: project setup, how the validation split and `evaluate()` work and why, the final model choice, the submission, and merge or review notes.

**Done when:** every script runs start to finish without errors, `results.csv` has one row per model, `submission.csv` has 4,277 rows in the right format, and the report is complete.

### Role 2 – Data + baseline model ([Champ])
*Best fit: a strong coder.*

**Does**
- **`01_explore.py` (EDA – not done yet).** Each section prints a plain-English conclusion and saves a chart:
  1. Overview: rows, columns, types.
  2. Target balance → which metric is fair.
  3. Missing values per column in train and test; whether rows with gaps differ in their transported rate (is missingness biased?).
  4. Data quality: duplicates, negative spending, age range, the logical link between CryoSleep and spending.
  5. Train vs test: do category shares and numeric distributions match? (Will the model generalise?)
  6. Feature vs target: transported rate per category, including Cabin split into Deck/Num/Side; spending and age patterns; ignore tiny groups when ranking.
  7. ID-like columns: uniqueness of `Name` and `Cabin`; travel groups from `PassengerId` (group size vs target, overlap between train and test → confirms the GroupKFold rule).
  8. Summary of decisions → saved to `results/eda_findings.txt`.
- **`src/features.py` + `02_prepare.py`:** create `Deck`, `CabinNum`, `Side`, `GroupSize`, `TotalSpend` and log-spending; apply rule-based filling (e.g. CryoSleep ↔ zero spending) if the EDA supports it; drop the ID-like columns; build `build_preprocessor()`.
- **`03_baseline.py`:** a `DummyClassifier` (the floor to beat), then **Logistic** Regression. Not Linear Regression: that predicts numbers and is unsuitable for a True/False target.
- Keeps **`reports/role2_data_baseline.md`** up to date: EDA findings with their charts, every cleaning and feature decision with its evidence, the columns dropped and why, and baseline scores.

**Done when:** EDA findings and charts are saved, `features.py` is merged and reused by Roles 3 and 4, both baselines are in `results.csv`, and the report is complete.

### Role 3 – Model B: Random Forest ([Pat])
*Best fit: medium coding level.*

**Does**
- **`04_random_forest.py`:** `Pipeline(build_preprocessor(), RandomForestClassifier(random_state=SEED))`.
- **Tuning:** `RandomizedSearchCV` with `cv=get_cv()` and `groups`, searching `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`. Saves the best parameters to `results/best_params_random_forest.json` and logs the result with `evaluate()`.
- **Overfitting check:** compare training and validation accuracy, and explain any gap.
- **Feature importance:** permutation importance chart → `plots/`, for the presentation.
- Keeps **`reports/role3_random_forest.md`** up to date: the search ranges and best parameters, scores before and after tuning, the overfitting check, and the top features with their chart.

**Done when:** the tuned RF is in `results.csv`, the parameters are saved, the importance chart is in `plots/`, and the report is complete.

### Role 4 – Model C: LightGBM ([Get])
*Best fit: medium to strong coder.*

**Does**
- **`05_lightgbm.py`:** `Pipeline(build_preprocessor(), LGBMClassifier(random_state=SEED, verbose=-1))`.
- **Tuning** on the same `get_cv()` folds: `learning_rate`, `num_leaves`/`max_depth`, `n_estimators`, `min_child_samples`, `subsample`, `colsample_bytree`. Prefer a modest search with sensible ranges over an exhaustive one (efficiency is graded).
- **Overfitting check:** training vs validation score. Saves the best parameters to `results/best_params_lightgbm.json` and logs the result with `evaluate()`.
- **Explanation:** why LightGBM beats or loses to Random Forest, written in the report.
- Keeps **`reports/role4_lightgbm.md`** up to date: the search ranges and best parameters, scores before and after tuning, the overfitting check, and the comparison with Random Forest.

**Done when:** tuned LightGBM is in `results.csv`, its parameters are saved, and the report (including the comparison) is complete.

### Role 5 – Analysis & presentation ([Bard])
*Best fit: a good communicator, or a lighter coder paired with Role 1.*

**Does**
- **Starts the slides on day 1:** problem → data → EDA findings → pipeline diagram → how we avoided bias → model comparison → best model (confusion matrix + feature importance) → limitations & next steps.
- **Builds slides from `reports/`:** creates the report template files for everyone on day 1, reads the reports regularly, and asks the owner directly when a section is empty or a number doesn't match `results.csv`.
- Keeps **`reports/role5_presentation.md`** up to date: the slide plan with the report each slide draws from, what content is still missing, and rehearsal notes (timing, who presents what).
- **Builds `notes/qa_prep.md`** by collecting section 8 (Likely Q&A questions) from every report.
- **`06_compare.py` (with Role 1):** reads `results.csv` and produces the comparison table and chart (mean ± std across folds).
- **`notes/qa_prep.md`:** likely questions with answers, e.g. "Why GroupKFold?", "Why accuracy?", "Why drop Name?", "How did you prevent leakage?", "Why this model?". Makes sure every member rehearses them.
- **Teamwork evidence:** no regular meetings are planned, so the evidence is the **Status** checklist below plus reviewed, merged PRs. Role 5 checks that every finished deliverable is ticked and went through a reviewed PR.
- **Rehearsals:** runs timed rehearsals with the whole team.

**Done when:** the slides are final, every slide's content is traceable to a report, Q&A prep is shared, and at least one full timed rehearsal is done.

---

## Order of work

1. **Setup (Role 1):** repo, `config.py`, `evaluation.py`. In parallel, Role 2 starts the EDA, and Role 5 starts the slide skeleton and the five report files from the template.
2. **Data (Role 2):** finish the EDA → `features.py` → baselines. *Roles 3 and 4 can draft their scripts but must use the merged `features.py` for final results.*
3. **Models (Roles 3 and 4, in parallel):** tuning on the shared folds.
4. **Integration (Roles 1 and 5):** comparison, choosing the final model, `submission.csv`, slides.
5. **Rehearsal (all):** everyone can explain every stage. All reports are marked *Done* before the final slides are locked.

## Status

- [x] `reports/` created with the template for all 5 roles (Role 5)
- [x] Repo, `requirements.txt`, `src/config.py` (Role 1)
- [x] `src/evaluation.py` (Role 1)
- [x] `01_explore.py` + `results/eda_findings.txt` (Role 2)
- [x] `src/features.py` + `02_prepare.py` (Role 2)
- [x] `03_baseline.py` (Role 2)
- [x] `04_random_forest.py` (Role 3)
- [x] `05_lightgbm.py` (Role 4)
- [ ] `06_compare.py` + comparison chart (Roles 5 and 1)
- [x] `07_final_model.py` + `submission.csv` (Role 1)
- [ ] Slides, `qa_prep.md` (Role 5)
- [ ] All 5 reports marked Done (each role)
- [ ] Full clean run + rehearsal (all)

---

## Notes for Claude

- This is a **learning assignment graded partly on Q&A**: write code together with the student, explain each part, and add comments that explain *why*.
- Ask which role the person has if it isn't clear, and stay within that role's files. Suggest changes to shared files (`src/`, contracts) rather than making them silently, and flag that Role 1 must review.
- Always follow the shared rules: `config.py` values, `get_cv()` folds, `build_preprocessor()` inside a Pipeline, `evaluate()` for scores.
- Flag anything that could leak information or bias the comparison, such as fitting on the full data, tuning on test.csv, or a different split per model.
- When unsure about a requirement or a decision, **ask instead of assuming**.
- After finishing a task with someone, **update their role's report in `reports/`**, following the template: what was done, decisions and why, real numbers from the script output, chart paths with captions, and likely Q&A questions. Never invent numbers; run the script and copy them.
- Update the **Status** checklist when a deliverable is finished.
