"""Train the chosen model on all of train.csv and write submission.csv (Role 1).

Usage:
    python 07_final_model.py              # best tuned model by CV accuracy
    python 07_final_model.py lightgbm     # or name the model explicitly
    python 07_final_model.py --self-test  # check the selection/format logic

The model is refitted once on all 8,693 training rows with the parameters
found by cross-validation (no re-tuning). test.csv is only ever predicted,
never used to choose or tune anything.
"""
import json
import sys

import pandas as pd

from src.config import DATA_DIR, ID_COL, RESULTS_DIR, SEED, TARGET, TEST_PATH, TRAIN_PATH

# Tuned models that can be final. Each has best_params_<name>.json.
MODELS = ("random_forest", "lightgbm")
SUBMISSION_PATH = DATA_DIR / "submission.csv"


def choose_model(results, results_dir, requested=None):
    """Pick the final model: the requested one, or the best tuned CV accuracy.

    Only models with a saved best_params file are candidates, so baselines
    (Dummy, Logistic Regression) are compared against but never shipped.
    """
    tuned = [m for m in MODELS if (results_dir / f"best_params_{m}.json").exists()]
    # Ties go to the faster model, so the choice never depends on row order.
    ranked = (results[results["model"].isin(tuned)]
              .sort_values(["accuracy_mean", "fit_time_s"],
                           ascending=[False, True], kind="stable"))
    if ranked.empty:
        raise ValueError(f"No tuned model in results.csv with a best_params file ({MODELS})")

    print("Top tuned models by CV accuracy:")
    print(ranked[["model", "accuracy_mean", "accuracy_std", "fit_time_s"]]
          .head(3).to_string(index=False))
    if len(ranked) > 1:
        best, second = ranked.iloc[0], ranked.iloc[1]
        if best["accuracy_mean"] - second["accuracy_mean"] <= best["accuracy_std"]:
            print(f"Note: {best['model']} and {second['model']} are within one std; "
                  "the difference is not clearly real, so the simpler/faster "
                  "model would be a defensible choice too.")

    if requested is None:
        return ranked.iloc[0]["model"]
    if requested not in ranked["model"].values:
        raise ValueError(f"{requested!r} is not a tuned model in results.csv: {list(ranked['model'])}")
    return requested


def read_results(path):
    """results.csv as a DataFrame, or a clear message if no model is logged yet."""
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"{path} is missing or empty: run 03-05 first.")
    return pd.read_csv(path)


def load_params(name, results, results_dir):
    """The saved best parameters, checked against the row that earned the score.

    If the json were rewritten after evaluate() ran, we would ship a model
    different from the one whose CV score justified choosing it.
    """
    params = json.loads((results_dir / f"best_params_{name}.json").read_text())
    logged = json.loads(results.loc[results["model"] == name, "params"].iloc[0] or "{}")
    if logged and logged != params:
        raise ValueError(f"best_params_{name}.json {params} differs from the params "
                         f"logged in results.csv {logged}; re-run that model's script.")
    return params


def pipeline_params(params):
    """Saved params as Pipeline keys; the shared SEED always wins."""
    # A search over the Pipeline saves keys with a step prefix ('model__x',
    # 'prep__...'); bare keys belong to the model.
    out = {k if "__" in k else f"model__{k}": v for k, v in params.items()}
    out["model__random_state"] = SEED
    return out


def make_model(name):
    """The untuned estimator; its tuned values are applied with set_params()."""
    if name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(random_state=SEED)
    from lightgbm import LGBMClassifier
    return LGBMClassifier(random_state=SEED, verbose=-1)


def check_submission(sub, test):
    """Raise if the submission isn't exactly Kaggle's expected format."""
    if list(sub.columns) != [ID_COL, TARGET]:
        raise ValueError(f"columns must be {[ID_COL, TARGET]}, got {list(sub.columns)}")
    if len(sub) != len(test):
        raise ValueError(f"expected {len(test)} rows, got {len(sub)}")
    if sub[TARGET].dtype != bool:
        raise ValueError(f"{TARGET} must be True/False, got dtype {sub[TARGET].dtype}")
    if not sub[ID_COL].reset_index(drop=True).equals(test[ID_COL].reset_index(drop=True)):
        raise ValueError(f"{ID_COL} does not match test.csv row for row")


def main(requested=None):
    # Imported here so --self-test runs even before Role 2's features.py exists.
    from sklearn.pipeline import Pipeline
    from src.evaluation import RESULTS_PATH
    from src.features import add_features, build_preprocessor

    results = read_results(RESULTS_PATH)
    name = choose_model(results, RESULTS_DIR, requested)
    params = pipeline_params(load_params(name, results, RESULTS_DIR))
    print(f"\nFinal model: {name} with {params}")

    train_raw, test_raw = pd.read_csv(TRAIN_PATH), pd.read_csv(TEST_PATH)
    # add_features() is row-wise (plus group counts within each file), so
    # applying it to each file separately leaks nothing between them.
    train, test = add_features(train_raw), add_features(test_raw)
    X, y = train.drop(columns=[TARGET]), train[TARGET]

    # Same pipeline as in cross-validation; the preprocessor is now fitted on
    # all of train.csv and only applied (never fitted) to test.csv.
    pipe = Pipeline([("prep", build_preprocessor()), ("model", make_model(name))])
    pipe.set_params(**params)
    pipe.fit(X, y)

    sub = pd.DataFrame({ID_COL: test_raw[ID_COL],
                        TARGET: pipe.predict(test).astype(bool)})
    check_submission(sub, test_raw)
    sub.to_csv(SUBMISSION_PATH, index=False)

    print(f"\nWrote {SUBMISSION_PATH.name}: {len(sub)} rows, columns {list(sub.columns)}.")
    print(f"Predicted transported share: {sub[TARGET].mean():.1%} "
          f"(training data: {y.mean():.1%}). A similar share means the model "
          "isn't biased towards one class on the test set.")


def _self_test():
    """Check model choice and submission checks on tiny made-up inputs."""
    import json
    import tempfile
    from pathlib import Path

    import pandas as pd

    results = pd.DataFrame({
        "model": ["dummy", "random_forest", "lightgbm"],
        "accuracy_mean": [0.50, 0.80, 0.81],
        "accuracy_std": [0.01, 0.02, 0.02],
        "fit_time_s": [0.01, 2.0, 1.0],
    })
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # Only tuned models (with a best_params file) are candidates,
        # so the dummy baseline can never be chosen.
        for m in ["random_forest", "lightgbm"]:
            (tmp / f"best_params_{m}.json").write_text(json.dumps({}))
        assert choose_model(results, tmp) == "lightgbm"
        # Equal accuracy -> the faster model wins (deterministic tie-break).
        tie = results.assign(accuracy_mean=[0.5, 0.81, 0.81], fit_time_s=[0, 0.5, 1.0])
        assert choose_model(tie, tmp) == "random_forest"

        # Saved params must match the ones logged with the CV score.
        logged = results.assign(params=["{}", "{}", json.dumps({"model__n_estimators": 50})])
        (tmp / "best_params_lightgbm.json").write_text(json.dumps({"model__n_estimators": 50}))
        assert load_params("lightgbm", logged, tmp) == {"model__n_estimators": 50}
        (tmp / "best_params_lightgbm.json").write_text(json.dumps({"model__n_estimators": 99}))
        try:
            load_params("lightgbm", logged, tmp)
            raise AssertionError("params that differ from results.csv were accepted")
        except ValueError:
            pass

        # Missing or empty results.csv -> clear message, not a pandas error.
        (tmp / "empty.csv").write_text("")
        for path in [tmp / "missing.csv", tmp / "empty.csv"]:
            try:
                read_results(path)
                raise AssertionError(f"{path.name} was accepted")
            except FileNotFoundError:
                pass

    # Keys get the pipeline prefix; the shared SEED always wins.
    assert pipeline_params({"n_estimators": 5, "model__max_depth": 3,
                            "prep__num__strategy": "mean", "random_state": 0}) == {
        "model__n_estimators": 5, "model__max_depth": 3,
        "prep__num__strategy": "mean", "model__random_state": SEED}

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for m in ["random_forest", "lightgbm"]:
            (tmp / f"best_params_{m}.json").write_text(json.dumps({}))
        assert choose_model(results, tmp, requested="random_forest") == "random_forest"
        for bad in ["dummy", "svm"]:
            try:
                choose_model(results, tmp, requested=bad)
                raise AssertionError(f"{bad} was accepted")
            except ValueError:
                pass

    test = pd.DataFrame({"PassengerId": ["0001_01", "0002_01"]})
    good = pd.DataFrame({"PassengerId": ["0001_01", "0002_01"],
                         "Transported": [True, False]})
    check_submission(good, test)
    bad_subs = [
        good.iloc[:1],                                     # missing a row
        good.assign(Transported=[1, 0]),                   # not True/False
        good.rename(columns={"Transported": "target"}),    # wrong column
        good.assign(PassengerId=["0001_01", "0001_01"]),   # wrong ids
    ]
    for sub in bad_subs:
        try:
            check_submission(sub, test)
            raise AssertionError(f"bad submission accepted:\n{sub}")
        except ValueError:
            pass
    print("Self-test passed: choose_model() and check_submission() work.")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else None)
