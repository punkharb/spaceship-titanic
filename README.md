# Spaceship Titanic

A five-person data science project to predict whether a passenger was **Transported** to another dimension during the Spaceship Titanic incident.

Our goal is a reproducible classification pipeline with fair model comparisons and clear explanations. We prioritize sound reasoning, pipeline quality, and honest evaluation over leaderboard tuning.

**Status:** Planning and Claude Code team configuration are complete. Python scripts, experiments, results, and slides have not been added yet.

## Problem and dataset

This is a binary classification problem with `Transported` (`True` / `False`) as the target. Features describe passengers, their cabins, destinations, cryosleep status, and onboard spending.

| File | Purpose | Rows |
|---|---|---:|
| `train.csv` | Labelled data for training and validation | 8,693 |
| `test.csv` | Unlabelled data for final predictions | 4,277 |
| `sample_submission.csv` | Expected submission format | 4,277 |

Source: [Kaggle — Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic).

Raw data is not included in this repository. Preserve original files and save derived outputs separately.

## Planned workflow

```text
Problem understanding → Data inspection and EDA → Feature engineering
→ Preprocessing inside CV → Baselines → Model training and tuning
→ Evaluation and comparison → Final predictions → Insights and presentation
```

1. **Explore:** check types, missing values, duplicates, invalid values, target balance, feature relationships, and differences between train and test.
2. **Prepare:** investigate cabin deck/number/side, travel-group size, total spending, and log-spending. Justify cleaning rules with EDA evidence.
3. **Establish baselines:** most-frequent `DummyClassifier`, followed by Logistic Regression.
4. **Train:** Random Forest and LightGBM, with modest hyperparameter searches.
5. **Compare:** validation scores, training–validation gaps, feature importance, runtime, and limitations.
6. **Predict:** fit the selected pipeline on all labelled data and verify the submission.

### Evaluation and leakage prevention

- **Primary metric:** accuracy, supported by precision, recall, F1, ROC-AUC, and fit time. EDA will check class balance and justify the metric.
- **Planned validation:** five-fold `GroupKFold`, grouped by the travel-group prefix of `PassengerId`. EDA must inspect train/test group overlap to assess the validation assumptions; overlap has not been checked yet.
- **Fair comparison:** shared folds, seed (`42`), preprocessing contract, and scoring function across all models.
- **Fold-local preprocessing:** imputation, encoding, and scaling are fitted inside a scikit-learn `Pipeline` on each training fold only.
- **Feature safety:** no target-derived features; follow the row-wise rules and within-file group-count contract in `CLAUDE.md`.
- **Test isolation:** never use `test.csv` for tuning or model selection.

Scores from folds also used for tuning may be optimistic. Report this limitation; fold standard deviations alone do not establish statistical significance.

## Team responsibilities

| Role | Focus | Main deliverables |
|---|---|---|
| 1 — Lead / integrator | Setup, shared evaluation, reviews, integration | Configuration, evaluation utilities, final model, submission |
| 2 — Data and baselines | EDA, cleaning, features, preprocessing | EDA findings, shared features, Dummy and Logistic Regression baselines |
| 3 — Random Forest | Training, tuning, overfitting checks | Model results, best parameters, feature importance |
| 4 — LightGBM | Training, tuning, overfitting checks | Model results, best parameters, comparison with Random Forest |
| 5 — Analysis and presentation | Comparison, reports, slides, teamwork records | Comparison charts, Q&A preparation, meeting and rehearsal notes |

Each member maintains a report containing decisions, evidence, results, charts, and likely Q&A. Everyone should be able to explain the entire workflow.

[CLAUDE.md](CLAUDE.md) contains the detailed plan, file ownership, shared interfaces, report template, and implementation checklist.

## Repository structure

Currently available:

```text
spaceship-titanic/
├── README.md                 # Project overview
├── CLAUDE.md                 # Team plan and shared contracts
└── .claude/                  # Claude Code commands, agents, and hooks
```

Planned implementation:

```text
├── requirements.txt          # Pinned dependencies
├── train.csv / test.csv      # Local raw data
├── src/
│   ├── config.py             # Paths and shared constants
│   ├── evaluation.py         # Grouping, CV, scoring, and logging
│   └── features.py           # Feature creation and preprocessing
├── 01_explore.py
├── 02_prepare.py
├── 03_baseline.py
├── 04_random_forest.py
├── 05_lightgbm.py
├── 06_compare.py
├── 07_final_model.py
├── results/                  # Scores, best parameters, EDA findings
├── plots/                    # Charts for reports and slides
├── reports/                  # One report per role
├── slides/                   # Presentation
├── notes/                    # Meetings, task board, Q&A
└── submission.csv            # Final predictions
```

## Getting started

```bash
git clone https://github.com/punkharb/spaceship-titanic.git
cd spaceship-titanic
```

1. Read [CLAUDE.md](CLAUDE.md), identify your role, and check the shared interfaces.
2. Download the competition data and place local copies at the repository root as `train.csv`, `test.csv`, and `sample_submission.csv`. For files named `Space ship ...`, rename only your project copies. Keep raw CSV files out of commits.
3. Once `requirements.txt` is added, create a virtual environment and install dependencies:

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate
python -m pip install -r requirements.txt
```

Planned libraries: pandas, NumPy, scikit-learn, Matplotlib, and LightGBM. **The requirements file and numbered scripts are not available yet.** Once added, run scripts from the repository root in order, starting with `python 01_explore.py`.

### Claude Code support

See the [.claude guide](.claude/README.md) for the included commands, agents, and hooks. Examples include `/start-role 1`, `/eda`, `/baseline`, `/check-leakage`, and `/slide-plan`.

These helpers support collaboration. Team members must understand AI-assisted code and verify every reported result. The presentation should disclose how AI was used.

## Results and deliverables

**No model scores are available yet.** Experiments will be logged in `results/results.csv`; comparison tables and charts will use those recorded results.

- [x] Public repository and project documentation
- [x] Claude Code team workflow configuration
- [ ] Shared Python setup and evaluation utilities
- [ ] EDA, feature engineering, and preprocessing
- [ ] Dummy and Logistic Regression baselines
- [ ] Random Forest and LightGBM experiments
- [ ] Model comparison, explanations, and limitations
- [ ] Final submission: 4,277 rows, original test IDs and order, columns `PassengerId,Transported`, boolean predictions
- [ ] Five role reports and presentation
- [ ] AI-use declaration, improvement plan, and team Q&A rehearsal

Track detailed progress in [CLAUDE.md](CLAUDE.md). Every final claim and presentation figure must be traceable to actual outputs.