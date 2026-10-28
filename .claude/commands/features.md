---
description: (Role 2) Build src/features.py (add_features, build_preprocessor) and 02_prepare.py
---
I'm Role 2. Build `src/features.py` and `02_prepare.py` together with me, matching the contract in @CLAUDE.md exactly. Roles 3 and 4 depend on it.

Base every choice on `results/eda_findings.txt`. If the EDA isn't done yet, say so and ask whether to continue.

1. `add_features(df)`, **row-wise only** (safe before the split):
   - Split `Cabin` into `Deck`, `CabinNum`, `Side`
   - `GroupSize` from the travel group (counted within the same file)
   - `TotalSpend`; log1p of the spending columns
   - Rule-based filling supported by the EDA, e.g. CryoSleep=True → spending 0; spending > 0 → CryoSleep=False
   - Must not use `Transported` or any statistic learned across rows
2. `NUMERIC_FEATURES`, `CATEGORICAL_FEATURES`, `DROP_COLUMNS` (`Name`, `Cabin`, `PassengerId`, …), each with a comment saying why.
3. `build_preprocessor()`: an **unfitted** `ColumnTransformer` with numeric median imputation + `StandardScaler`, and categorical most-frequent imputation + `OneHotEncoder(handle_unknown="ignore")`.
4. `02_prepare.py`: demonstrates and checks the steps. It shows before/after columns, confirms no missing values remain after the preprocessor is fitted on a training fold, and prints the final feature count.

Then run the **leakage-reviewer** on `src/features.py`, tell me that Role 1 must review this shared file, and use /update-report 2.
