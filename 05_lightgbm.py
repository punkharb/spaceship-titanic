"""LightGBM: default score, tuning, overfitting check, feature importance (Role 4).

LightGBM is a gradient-boosted tree model: it builds small trees one after
another, each one correcting the mistakes of the trees before it. Random
Forest (Role 3) instead builds many deep trees independently and averages
them. Comparing the two tells us whether boosting helps on this data.

Steps:
1. Score LightGBM with its default settings -> "lightgbm_default" row.
2. Tune it with RandomizedSearchCV on the shared GroupKFold folds.
3. Save the best parameters and score the tuned model -> "lightgbm" row
   (07_final_model.py looks for exactly this name and its json file).
4. Overfitting check: training-fold vs validation-fold accuracy.
5. Permutation importance on the validation folds -> plots/.
6. Compare with Random Forest and the baselines from results.csv.

test.csv is never read here: all tuning and model choice use train.csv only.

Run from the project folder:  python 05_lightgbm.py
"""
import json
import time

import matplotlib
matplotlib.use("Agg")  # write PNG files without needing a screen
import matplotlib.pyplot as plt
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from src.config import PLOT_DIR, RESULTS_DIR, SEED, TARGET, TRAIN_PATH
from src.evaluation import RESULTS_PATH, evaluate, get_cv, get_groups
from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, add_features, build_preprocessor

OWNER = "Role 4"
NEUTRAL = "#8a8984"  # team colour for charts that aren't about one class
N_ITER = 30          # 30 random settings x 5 folds = 150 fits: enough to find a
                     # good region, small enough to stay fast (efficiency is graded)

# Plain Python lists, not scipy distributions: the chosen values must survive
# json.dumps() unchanged, because 07_final_model.py checks that
# best_params_lightgbm.json matches the params logged in results.csv.
PARAM_SPACE = {
    # Step size: how much each new tree corrects. Smaller = slower but steadier.
    "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
    # Number of trees. Works together with learning_rate (small rate needs more trees).
    "model__n_estimators": [100, 200, 300, 500],
    # Max leaves per tree = how complex one tree can be. Default 31; fewer = simpler.
    "model__num_leaves": [7, 15, 31],
    # Hard cap on tree depth (-1 = no cap). A second brake on complexity.
    "model__max_depth": [-1, 4, 6, 8],
    # Min passengers in a leaf. Higher = a leaf can't be built on a handful of rows.
    "model__min_child_samples": [10, 20, 40, 80],
    # Share of rows each tree sees. < 1 adds randomness, which reduces overfitting.
    "model__subsample": [0.7, 0.85, 1.0],
    # LightGBM ignores subsample unless bagging is switched on (subsample_freq > 0).
    # Kept in the search space so it is saved with the best params and 07 uses it too.
    "model__subsample_freq": [1],
    # Share of features each tree sees. < 1 stops a few strong features dominating.
    "model__colsample_bytree": [0.6, 0.8, 1.0],
}


def load_data():
    df = pd.read_csv(TRAIN_PATH)
    X = add_features(df).drop(columns=[TARGET])  # same X as every model script
    return X, df[TARGET], get_groups(df)


def make_pipeline():
    # verbose=-1 silences LightGBM's per-fit log lines.
    return Pipeline([("prep", build_preprocessor()),
                     ("model", LGBMClassifier(random_state=SEED, verbose=-1))])


def tune(X, y, groups):
    """Random search over PARAM_SPACE, scored on the shared GroupKFold folds."""
    search = RandomizedSearchCV(
        make_pipeline(), PARAM_SPACE, n_iter=N_ITER, scoring="accuracy",
        cv=get_cv(), random_state=SEED,
        # No refit: evaluate() refits the best settings fold by fold anyway,
        # and the final fit on all of train.csv is 07_final_model.py's job.
        refit=False)
    start = time.time()
    search.fit(X, y, groups=groups)  # groups -> GroupKFold keeps travel groups together
    print(f"   searched {N_ITER} settings x {get_cv().get_n_splits()} folds "
          f"in {time.time() - start:.0f} s")

    table = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    print("   top 5 settings (mean ± std validation accuracy):")
    for _, row in table.head(5).iterrows():
        short = {k.replace("model__", ""): v for k, v in row["params"].items()
                 if k != "model__subsample_freq"}
        print(f"     {row['mean_test_score']:.4f} ± {row['std_test_score']:.4f}  {short}")
    return search.best_params_, table


def permutation_importance_cv(pipe, X, y, groups, n_repeats=5):
    """Accuracy lost when one input column is shuffled, on validation folds only.

    Each fold's model is fitted on its training part and judged on rows it
    never saw, so a feature only counts as important if it helps on new
    passengers. Columns are shuffled *before* the preprocessor, so one-hot
    columns are judged together as one feature (e.g. "Deck", not "Deck_F").
    Caveat: correlated features (each bill and TotalSpend_log) share credit,
    so each of them can look less important than the group really is.
    """
    cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES  # the columns the model uses
    X = X[cols]
    per_fold = []
    for train_idx, val_idx in get_cv().split(X, y, groups):
        fitted = clone(pipe).fit(X.iloc[train_idx], y.iloc[train_idx])
        r = permutation_importance(fitted, X.iloc[val_idx], y.iloc[val_idx],
                                   scoring="accuracy", n_repeats=n_repeats,
                                   random_state=SEED)
        per_fold.append(pd.Series(r.importances_mean, index=cols))
    folds = pd.concat(per_fold, axis=1)
    return pd.DataFrame({"mean": folds.mean(axis=1), "std": folds.std(axis=1)}) \
             .sort_values("mean", ascending=False)


def plot_importance(imp, top=15):
    shown = imp.head(top).iloc[::-1]  # largest at the top of the chart
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(shown.index, shown["mean"] * 100, xerr=shown["std"] * 100,
            color=NEUTRAL, ecolor="black", capsize=3)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Accuracy lost when the feature is shuffled (percentage points, "
                  "mean ± std over 5 folds)")
    first, second = imp.index[0], imp.index[1]
    ax.set_title(f"LightGBM relies most on {first} and {second}")
    fig.tight_layout()
    path = PLOT_DIR / "11_lightgbm_permutation_importance.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"   chart: plots/{path.name}")


def overfit_report(label, res):
    gap = res["train_accuracy_mean"] - res["accuracy_mean"]
    print(f"   {label}: train {res['train_accuracy_mean']:.4f} vs validation "
          f"{res['accuracy_mean']:.4f} -> gap {gap * 100:.1f} points")
    return gap


def compare_with_others(res_tuned):
    """Put the tuned LightGBM next to Random Forest and the baselines."""
    results = pd.read_csv(RESULTS_PATH).set_index("model")
    cols = ["accuracy_mean", "accuracy_std", "f1_mean", "roc_auc_mean", "fit_time_s"]
    print(results[cols].round(4).to_string())

    for other in ["logistic_regression", "random_forest"]:
        if other not in results.index:
            print(f"   {other} is not in results.csv yet -> re-run this comparison later.")
            continue
        diff = res_tuned["accuracy_mean"] - results.loc[other, "accuracy_mean"]
        # The fold-to-fold std is the noise level: a difference smaller than it
        # could come from which passengers happened to land in which fold.
        # A rule of thumb, not a statistical test, so the wording stays modest.
        noise = max(res_tuned["accuracy_std"], results.loc[other, "accuracy_std"])
        verdict = ("larger than the fold-to-fold spread" if abs(diff) > noise
                   else "within the fold-to-fold spread")
        print(f"   LightGBM vs {other}: {diff * 100:+.2f} points ({verdict}, "
              f"std ≈ {noise * 100:.2f} points)")


def main():
    PLOT_DIR.mkdir(exist_ok=True)
    X, y, groups = load_data()

    print("1. LightGBM with default settings")
    res_default = evaluate(make_pipeline(), X, y, groups, name="lightgbm_default",
                           owner=OWNER, params={})

    print(f"\n2. Random search ({N_ITER} settings) on the shared GroupKFold folds")
    best_params, _ = tune(X, y, groups)

    print("\n3. Best settings: save, then score with evaluate()")
    params_path = RESULTS_DIR / "best_params_lightgbm.json"
    params_path.write_text(json.dumps(best_params, indent=2))
    print(f"   saved {params_path.relative_to(RESULTS_DIR.parent)}: {best_params}")
    tuned = make_pipeline().set_params(**best_params)
    res_tuned = evaluate(tuned, X, y, groups, name="lightgbm", owner=OWNER,
                         params=best_params)
    cm = res_tuned["confusion_matrix"]
    print(f"   out-of-fold confusion matrix (rows = true False/True, "
          f"cols = predicted False/True):\n{cm}")

    print("\n4. Overfitting check (mean over the 5 folds)")
    gap_default = overfit_report("default", res_default)
    gap_tuned = overfit_report("tuned  ", res_tuned)

    print("\n5. Permutation importance (validation folds)")
    imp = permutation_importance_cv(tuned, X, y, groups)
    print((imp.head(10) * 100).round(2).rename(columns=lambda c: f"{c} (pts)").to_string())
    plot_importance(imp)

    print("\n6. Comparison with the other models in results.csv")
    compare_with_others(res_tuned)

    gain = res_tuned["accuracy_mean"] - res_default["accuracy_mean"]
    # Same noise rule as compare_with_others(): smaller than the fold std = not clearly real.
    noise = max(res_tuned["accuracy_std"], res_default["accuracy_std"])
    print(f"\n-> Tuning moved LightGBM from {res_default['accuracy_mean']:.2%} to "
          f"{res_tuned['accuracy_mean']:.2%} accuracy ({gain * 100:+.2f} points, "
          + ("more than" if abs(gain) > noise else "within")
          + f" the {noise * 100:.2f}-point fold-to-fold noise). "
          f"The train-validation gap went from {gap_default * 100:.1f} to "
          f"{gap_tuned * 100:.1f} points: "
          + ("the tuned model memorises the training folds less."
             if gap_tuned < gap_default else "tuning did not reduce overfitting.")
          + f" Most useful features: {', '.join(imp.index[:3])}.")


if __name__ == "__main__":
    main()
