"""Nested cross-validation: an unbiased score for the tuned models (Role 1, issue #16).

Why: 04 and 05 pick the best of 30 settings on the 5 shared folds and then
score that pick on the *same* folds. The winner is partly lucky on those
folds, so its score is slightly optimistic (a best-of-30 pick).

Nested CV removes that luck. For each of the 5 outer folds:
  1. run the *same* random search (same space, same 30 settings, same seed)
     on the other 4 folds only, with inner folds grouped by travel group;
  2. refit the best setting on those 4 folds;
  3. score it on the outer fold, which the search never saw.
The mean over the outer folds is an honest estimate of "tune, then predict".

Logged with the shared evaluate() as random_forest_nested / lightgbm_nested,
on the same outer get_cv() folds as every other model. The search spaces are
imported unchanged from 04_random_forest.py and 05_lightgbm.py.

Cost: 5 outer x 30 settings x 5 inner folds = 750 fits per model, about 5x
the original tuning, so this script is slow (several minutes).

Run from the project folder:  python 05b_nested_cv.py
Self-test:                    python 05b_nested_cv.py --self-test
"""
import importlib
import sys

import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import RandomizedSearchCV

from src.config import SEED, TARGET, TRAIN_PATH
from src.evaluation import RESULTS_PATH, evaluate, get_cv, get_groups
from src.features import add_features

OWNER = "Role 1"
# Model scripts whose search is repeated inside each outer fold.
MODELS = {"random_forest": "04_random_forest", "lightgbm": "05_lightgbm"}


class GroupedSearch(ClassifierMixin, BaseEstimator):
    """A RandomizedSearchCV that evaluate() can treat like any pipeline.

    cross_validate() only passes X and y to fit(), not the groups, so the
    travel groups are read back from PassengerId in the rows it is given.
    That keeps the inner folds grouped too: no travel group is split
    between inner training and inner validation.
    """

    def __init__(self, pipeline, param_space, n_iter, n_jobs=None):
        self.pipeline = pipeline
        self.param_space = param_space
        self.n_iter = n_iter
        self.n_jobs = n_jobs

    def fit(self, X, y):
        self.search_ = RandomizedSearchCV(
            clone(self.pipeline), self.param_space, n_iter=self.n_iter,
            scoring="accuracy", cv=get_cv(), random_state=SEED, n_jobs=self.n_jobs,
            refit=True)  # refit the best setting on this outer training part only
        self.search_.fit(X, y, groups=get_groups(X))
        self.classes_ = self.search_.classes_
        return self

    def predict(self, X):
        return self.search_.predict(X)

    def predict_proba(self, X):  # needed for ROC-AUC
        return self.search_.predict_proba(X)


def main():
    df = pd.read_csv(TRAIN_PATH)
    X, y, groups = add_features(df).drop(columns=[TARGET]), df[TARGET], get_groups(df)
    logged = pd.read_csv(RESULTS_PATH).set_index("model")
    # Check before the slow part, not after many minutes of fitting.
    missing = [n for n in MODELS if n not in logged.index]
    if missing:
        sys.exit(f"No tuned row for {missing} in results.csv: run 04/05 first.")

    for name, script in MODELS.items():
        m = importlib.import_module(script)  # reuse the owner's exact search
        # RF already searches on all cores in 04; keep that, LightGBM as in 05.
        nested = GroupedSearch(m.make_pipeline(), m.PARAM_SPACE, m.N_ITER,
                               n_jobs=-1 if name == "random_forest" else None)
        print(f"\n{name}: nested CV, {m.N_ITER} settings per outer fold ...")
        res = evaluate(nested, X, y, groups, name=f"{name}_nested", owner=OWNER,
                       params={"n_iter": m.N_ITER, "outer": "get_cv()", "inner": "get_cv() on outer-train"})
        drop = logged.loc[name, "accuracy_mean"] - res["accuracy_mean"]
        print(f"   tuned (same folds) {logged.loc[name, 'accuracy_mean']:.4f} vs nested "
              f"{res['accuracy_mean']:.4f} ± {res['accuracy_std']:.4f}: "
              + (f"the best-of-{m.N_ITER} pick was {drop * 100:.1f} points optimistic."
                 if drop > 0 else f"no optimism from the best-of-{m.N_ITER} pick."))


def _self_test():
    """Nested evaluate() runs end to end on tiny data with grouped inner folds."""
    import tempfile
    from pathlib import Path

    import pandas as pd
    from sklearn.base import clone
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    import src.evaluation as ev

    # 40 travel groups x 3 people; x separates the classes, so a tuned
    # model should beat chance on every outer fold.
    ids = [f"{g:04d}_{p:02d}" for g in range(1, 41) for p in range(1, 4)]
    y = pd.Series([g % 2 == 0 for g in range(1, 41) for _ in range(3)])
    X = pd.DataFrame({"PassengerId": ids, "x": y.astype(float) + [0.1 * (i % 5) for i in range(120)]})
    pipe = Pipeline([("prep", ColumnTransformer([("x", "passthrough", ["x"])])),
                     ("model", LogisticRegression())])
    nested = GroupedSearch(pipe, {"model__C": [0.1, 1.0]}, n_iter=2)

    # cross_validate() clones the estimator for every outer fold.
    assert clone(nested).get_params()["n_iter"] == 2

    with tempfile.TemporaryDirectory() as tmp:
        ev.RESULTS_PATH = Path(tmp) / "results.csv"
        res = ev.evaluate(nested, X, y, ev.get_groups(X), name="selftest_nested", owner="selftest")
        logged = pd.read_csv(ev.RESULTS_PATH)
    assert logged["model"].tolist() == ["selftest_nested"]
    assert res["accuracy_mean"] > 0.9

    # The inner search reads travel groups from PassengerId; without it,
    # it must fail instead of silently falling back to ungrouped folds.
    try:
        clone(nested).fit(X.drop(columns=["PassengerId"]), y)
        raise AssertionError("fit without PassengerId was accepted")
    except KeyError:
        pass
    print("Self-test passed: GroupedSearch works inside evaluate().")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
    else:
        main()
