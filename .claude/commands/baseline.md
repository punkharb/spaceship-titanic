---
description: (Role 2) Build 03_baseline.py - DummyClassifier and Logistic Regression baselines
---
I'm Role 2. Build `03_baseline.py` together with me.

1. Load train.csv, apply `add_features()`, split into X / y, and get `groups` with `get_groups()`.
2. Model 1: `DummyClassifier(strategy="most_frequent")`, the floor every model must beat.
3. Model 2: `Pipeline(build_preprocessor(), LogisticRegression(max_iter=1000, random_state=SEED))`. Explain why this is **Logistic**, not Linear, Regression.
4. Score both with `evaluate(..., owner="Role 2")` so they go into `results/results.csv`.
5. Print a small comparison and a one-line conclusion.
6. Optional: a chart of the largest Logistic Regression coefficients, which shows which features push toward "transported".

Explain each step, run the script, then use /update-report 2.
