---
description: (Role 1) Train the chosen model on all of train.csv and write submission.csv
argument-hint: [model name from results.csv - default the best accuracy]
---
I'm Role 1. Build `07_final_model.py` together with me.

1. Read `results/results.csv`. Use the model "$ARGUMENTS", or if none was given, the best mean accuracy. Show me the top 3 and confirm the choice before continuing. If two scores are within one std of each other, mention that the simpler model is a defensible choice.
2. Load its best parameters from `results/best_params_<model>.json`.
3. Build the same `Pipeline(build_preprocessor(), model)`, apply `add_features()` to train and test, and fit on **all** of train.csv. No re-tuning, and never tune on test.csv.
4. Predict test.csv and write `submission.csv` with columns `PassengerId,Transported`, values True/False, 4,277 rows. Verify all three.
5. Print the predicted transported share, which should be roughly balanced like the training data.

Then update `reports/role1_lead.md` (final model, why it was chosen, submission check) and tick the Status item.
