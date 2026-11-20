"""XGBoost and CatBoost in the shared Pipeline (Roles 3 and 2, issue #19).

Question: is LightGBM the best we can do, or does another gradient-boosting
library score higher on the same features and the same folds?

All three build trees one after another, each correcting the last. They
differ in how a tree is grown:
- LightGBM grows leaf by leaf: it always splits the leaf that helps most.
- XGBoost grows level by level, to a fixed depth.
- CatBoost grows "symmetric" trees (the same question for every node of a
  level) and uses ordered boosting, both of which make it harder to overfit.

Kept the same as every other model, so the comparison is fair:
- the shared Pipeline([("prep", build_preprocessor()), ("model", ...)]),
- the shared get_cv() folds and evaluate(),
- the same tuning budget: 30 random settings (issue #19 rule).
CatBoost can read category columns directly, but here it gets the same
one-hot columns as everyone else. That keeps rule 4 (one preparation step)
and means a difference in score comes from the model, not from the inputs.

Steps, for each of the two models:
1. Score it with default settings      -> "<name>_default" row.
2. Tune it with RandomizedSearchCV on the shared GroupKFold folds.
3. Save the best parameters and score  -> "<name>" row + best_params_<name>.json.
4. Overfitting check: training-fold vs validation-fold accuracy.
Then:
5. Compare both with tuned LightGBM, using the team's one-std rule.

test.csv is never read here: all tuning and model choice use train.csv only.
07_final_model.py is not changed by this script (Role 1 owns it).

Run from the project folder:  python 04b_xgboost_catboost.py   (about 3 minutes)
"""
import json
import time

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import RESULTS_DIR, SEED, TARGET, TRAIN_PATH
from src.evaluation import RESULTS_PATH, evaluate, get_cv, get_groups
from src.features import add_features, build_preprocessor

OWNER = "Roles 3 and 2"
CURRENT_BEST = "lightgbm"  # the model 07_final_model.py ships today
N_ITER = 30  # same budget as 04 and 05: 30 random settings x 5 folds = 150 fits

# Plain Python lists, not scipy distributions: the chosen values must survive
# json.dumps() unchanged, so best_params_<name>.json matches results.csv.
XGBOOST_SPACE = {
    # Step size: how much each new tree corrects. Smaller = slower but steadier.
    "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
    # Number of trees. Works together with learning_rate (small rate needs more trees).
    "model__n_estimators": [100, 200, 300, 500],
    # Depth of every tree. XGBoost grows level by level, so this is its main
    # complexity control (LightGBM's is num_leaves).
    "model__max_depth": [3, 4, 6, 8],
    # Min "weight" of passengers in a leaf. Higher = no leaf built on a handful of rows.
    "model__min_child_weight": [1, 3, 5, 10],
    # Share of rows each tree sees. < 1 adds randomness, which reduces overfitting.
    "model__subsample": [0.7, 0.85, 1.0],
    # Share of features each tree sees. < 1 stops a few strong features dominating.
    "model__colsample_bytree": [0.6, 0.8, 1.0],
}
CATBOOST_SPACE = {
    "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
    # Number of trees. 1000 is CatBoost's default tree count, so it has to be
    # in the space, or tuning could hardly match the default run. Caveat:
    # LightGBM and XGBoost stop at 500 trees, so CatBoost has a small edge here.
    "model__iterations": [200, 300, 500, 1000],
    # Depth of the symmetric trees (default 6).
    "model__depth": [4, 6, 8],
    # L2 penalty on the leaf values. Higher = smaller, more cautious corrections.
    "model__l2_leaf_reg": [1, 3, 5, 10],
    # Share of rows each tree sees.
    "model__subsample": [0.7, 0.85, 1.0],
    # Share of features tried at each split (CatBoost calls it rsm).
    "model__rsm": [0.6, 0.8, 1.0],
}


class BoolXGBClassifier(XGBClassifier):
    """XGBoost that answers True/False, like every other model here.

    XGBoost turns the True/False target into 1/0 and predicts 1/0. The shared
    evaluate() builds its confusion matrix from True/False predictions, so
    the answer is converted back. The model and its scores are unchanged.
    """

    def predict(self, X, **kwargs):
        return super().predict(X, **kwargs).astype(bool)


def make_xgboost():
    return BoolXGBClassifier(random_state=SEED)


def make_catboost():
    # verbose=0 silences the per-tree log. allow_writing_files=False stops
    # CatBoost creating a catboost_info/ folder in the project.
    return CatBoostClassifier(random_seed=SEED, verbose=0, allow_writing_files=False)


# name -> (model factory, search space, the parameter that sets its thread count)
MODELS = {
    "xgboost": (make_xgboost, XGBOOST_SPACE, "model__n_jobs"),
    "catboost": (make_catboost, CATBOOST_SPACE, "model__thread_count"),
}


def load_data():
    df = pd.read_csv(TRAIN_PATH)
    X = add_features(df).drop(columns=[TARGET])  # same X as every model script
    return X, df[TARGET], get_groups(df)


def make_pipeline(name):
    return Pipeline([("prep", build_preprocessor()), ("model", MODELS[name][0]())])


def tune(name, X, y, groups):
    """Random search over the model's space, scored on the shared GroupKFold folds."""
    _, space, threads = MODELS[name]
    # The 150 fits run side by side (n_jobs=-1), so each single model uses one
    # thread here. That changes the run time, not the scores: the best
    # setting's search score equals its evaluate() score below, which runs
    # with the default thread count.
    pipe = make_pipeline(name).set_params(**{threads: 1})
    search = RandomizedSearchCV(
        pipe, space, n_iter=N_ITER, scoring="accuracy", cv=get_cv(),
        random_state=SEED, n_jobs=-1,
        # No refit: evaluate() refits the best settings fold by fold anyway.
        refit=False)
    start = time.time()
    search.fit(X, y, groups=groups)  # groups -> GroupKFold keeps travel groups together
    print(f"   searched {N_ITER} settings x {get_cv().get_n_splits()} folds "
          f"in {time.time() - start:.0f} s")

    table = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    print("   top 3 settings (mean ± std validation accuracy):")
    for _, row in table.head(3).iterrows():
        short = {k.replace("model__", ""): v for k, v in row["params"].items()}
        print(f"     {row['mean_test_score']:.4f} ± {row['std_test_score']:.4f}  {short}")
    return search.best_params_


def overfit_report(label, res):
    gap = res["train_accuracy_mean"] - res["accuracy_mean"]
    print(f"   {label}: train {res['train_accuracy_mean']:.4f} vs validation "
          f"{res['accuracy_mean']:.4f} -> gap {gap * 100:.1f} points")


def run_model(name, X, y, groups):
    """Steps 1-4 for one model; returns its tuned result."""
    print(f"\n=== {name} ===")
    print("1. Default settings")
    res_default = evaluate(make_pipeline(name), X, y, groups, name=f"{name}_default",
                           owner=OWNER, params={})

    print(f"2. Random search ({N_ITER} settings) on the shared GroupKFold folds")
    best_params = tune(name, X, y, groups)

    print("3. Best settings: save, then score with evaluate()")
    params_path = RESULTS_DIR / f"best_params_{name}.json"
    params_path.write_text(json.dumps(best_params, indent=2))
    print(f"   saved {params_path.relative_to(RESULTS_DIR.parent)}: {best_params}")
    res_tuned = evaluate(make_pipeline(name).set_params(**best_params), X, y, groups,
                         name=name, owner=OWNER, params=best_params)
    print(f"   out-of-fold confusion matrix (rows = true False/True, "
          f"cols = predicted False/True):\n{res_tuned['confusion_matrix']}")

    print("4. Overfitting check (mean over the 5 folds)")
    overfit_report("default", res_default)
    overfit_report("tuned  ", res_tuned)
    return res_tuned


def compare_with_current_best(tuned):
    """Each new model vs tuned LightGBM, with the issue #19 one-std rule."""
    results = pd.read_csv(RESULTS_PATH).set_index("model")
    cols = ["accuracy_mean", "accuracy_std", "f1_mean", "roc_auc_mean", "fit_time_s"]
    shown = [m for m in [CURRENT_BEST, *[n for name in MODELS for n in (f"{name}_default", name)]]
             if m in results.index]
    print(results.loc[shown, cols].round(4).to_string())

    best = results.loc[CURRENT_BEST]
    winners = []
    for name, res in tuned.items():
        diff = res["accuracy_mean"] - best["accuracy_mean"]
        # A rule of thumb, not a statistical test: a gap smaller than the
        # fold-to-fold std could come from which passengers landed in which fold.
        noise = max(res["accuracy_std"], best["accuracy_std"])
        print(f"   {name} vs {CURRENT_BEST}: {diff * 100:+.2f} points ("
              + ("larger than" if abs(diff) > noise else "within")
              + f" the fold-to-fold spread, std ~{noise * 100:.2f} points); "
              f"fit time {res['fit_time_s']:.2f} s vs {best['fit_time_s']:.2f} s per fold")
        if diff > noise:
            winners.append(name)
    return winners


def main():
    X, y, groups = load_data()
    tuned = {name: run_model(name, X, y, groups) for name in MODELS}

    print(f"\n5. Comparison with the current final model ({CURRENT_BEST})")
    winners = compare_with_current_best(tuned)

    scores = ", ".join(f"{name} {res['accuracy_mean']:.2%}" for name, res in tuned.items())
    print(f"\n-> Tuned on the same folds with the same budget: {scores}. "
          + (f"{' and '.join(winners)} beat {CURRENT_BEST} by more than one std: Role 1 "
             "should consider adding it to 07_final_model.py (issue #19)."
             if winners else
             f"Neither beats {CURRENT_BEST} by more than one std, so 07_final_model.py "
             "stays unchanged (issue #19 rule)."))


if __name__ == "__main__":
    main()
