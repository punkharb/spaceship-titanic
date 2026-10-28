---
description: (Role 1) Create the shared project foundation - requirements, src/config.py, src/evaluation.py
---
I'm Role 1. Help me set up the shared foundation described in @CLAUDE.md, writing the code together with me and explaining each part.

Create these, but ask before overwriting anything that exists:
1. `requirements.txt`: pandas, numpy, scikit-learn, matplotlib, lightgbm. Pin the versions installed here (`pip show`).
2. Folders `src/`, `plots/`, `results/`, `reports/`, `slides/`, `notes/`, plus `src/__init__.py`.
3. `src/config.py`: paths built from the project folder, plus `TARGET`, `ID_COL`, `SEED = 42`, `N_SPLITS = 5`.
4. `src/evaluation.py`, exactly matching the contract in CLAUDE.md:
   - `get_groups(df)`: travel group from `PassengerId` (`gggg` of `gggg_pp`)
   - `get_cv()`: `GroupKFold(n_splits=N_SPLITS)`
   - `evaluate(pipeline, X, y, groups, name, owner, params=None)`: `cross_validate` for accuracy, precision, recall, F1, ROC-AUC (mean ± std) and fit time; an out-of-fold confusion matrix via `cross_val_predict`; appends one row to `results/results.csv` with the columns listed in CLAUDE.md; returns a dict.
5. A short self-test (`python -m src.evaluation`) that runs `evaluate()` on a `DummyClassifier` to prove it works.
6. A `.gitignore` covering `__pycache__/`, `.claude/settings.local.json` and virtual environments.

Then update `reports/role1_lead.md` and tick the matching Status items in CLAUDE.md.
