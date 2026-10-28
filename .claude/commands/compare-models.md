---
description: (Role 5 with Role 1) Build 06_compare.py - the model comparison table and charts
---
Build `06_compare.py` together with me (Role 5, paired with Role 1).

1. Read `results/results.csv`. If a model was logged several times, keep the latest row for each model name, and say so.
2. Save a comparison table, sorted by mean accuracy, to `results/model_comparison.csv`. Columns: model, owner, accuracy mean ± std, precision, recall, F1, ROC-AUC, fit time.
3. Chart 1: accuracy per model with ± std error bars, the Dummy baseline as a reference line, and a title stating the winner and margin → `plots/`.
4. Chart 2: a grouped chart of precision / recall / F1 for the top models → `plots/`.
5. Print a one-paragraph plain-English conclusion. If the top two overlap within one std, say the difference may not be meaningful.
6. Use only the numbers in results.csv. Don't recompute scores.

Then use /update-report 5.
