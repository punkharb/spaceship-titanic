---
description: (Roles 3 & 4) Build, tune and evaluate a model - random_forest or lightgbm
argument-hint: <random_forest | lightgbm>
---
Build the model script for **$ARGUMENTS** together with me, following @CLAUDE.md:
- `random_forest` → `04_random_forest.py` (Role 3)
- `lightgbm` → `05_lightgbm.py` (Role 4)

If the argument is missing or different, ask. If `src/features.py` or `src/evaluation.py` don't exist yet, say who I'm waiting on and offer to draft the script around their documented contracts.

Steps:
1. Load data → `add_features()` → X, y, `groups = get_groups(...)`.
2. `Pipeline([("prep", build_preprocessor()), ("model", <model>(random_state=SEED))])`. For LightGBM, add `verbose=-1`.
3. Score the default settings first with `evaluate(..., name="<model> (default)")`.
4. Tune with `RandomizedSearchCV(cv=get_cv(), scoring="accuracy", n_iter≈30, random_state=SEED)` and `.fit(X, y, groups=groups)`. Use modest ranges from my role section in CLAUDE.md, and explain each parameter in one line.
5. Save the best parameters to `results/best_params_$ARGUMENTS.json`, then score the tuned model with `evaluate(..., name="<model> (tuned)")`.
6. Overfitting check: training accuracy vs. cross-validated accuracy, explained in plain English.
7. A permutation importance chart (top 15) → `plots/`, with a title stating the finding.
8. For LightGBM, compare with Random Forest from `results.csv` and explain the difference.

Never touch test.csv here. Explain each step, run the script, then run the **leakage-reviewer** on it and use /update-report.
