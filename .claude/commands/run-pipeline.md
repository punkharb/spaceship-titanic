---
description: (Role 1) Run the whole pipeline 01->07 from a clean state and check the outputs
---
Run the full pipeline as Role 1, the integrator.

1. From the project folder, run each existing numbered script in order (`01_explore.py` … `07_final_model.py`) with `python`. Skip missing scripts but list them.
2. Stop at the first failure. Show the error, explain the likely cause, and name the role that owns the file. Don't fix another role's file without asking.
3. After a successful run, check that:
   - `results/results.csv` has one row per model, with no duplicates from earlier runs. If there are duplicates, explain and ask before cleaning.
   - Every chart referenced in `reports/` exists in `plots/`.
   - `submission.csv`, if created, has 4,277 rows, columns `PassengerId,Transported`, and True/False values.
   - The numbers stay the same when re-run, which shows `SEED` is respected.
4. Report a table: Script | Owner | Result (ok / failed / missing) | Run time.
