"""Shared validation and scoring for every model (Role 1).

Every model is scored the same way here: the same GroupKFold folds, the same
metrics, one row per model in results/results.csv. That makes the comparison
fair and keeps hand-computed numbers out of the slides.

Self-test: python -m src.evaluation
"""
import json
from datetime import date

import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold, cross_validate

from src.config import ID_COL, N_SPLITS, RESULTS_DIR

RESULTS_PATH = RESULTS_DIR / "results.csv"
RESULTS_COLUMNS = ["model", "owner", "accuracy_mean", "accuracy_std",
                   "precision_mean", "recall_mean", "f1_mean", "roc_auc_mean",
                   "fit_time_s", "params", "date"]
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def get_groups(df):
    """Travel group of each passenger: the 'gggg' part of PassengerId 'gggg_pp'.

    People in one group often share a cabin, a planet and spending habits.
    Grouping the folds by it stops the model being validated on the relatives
    of people it was trained on, which would inflate the score.
    """
    return df[ID_COL].str.split("_").str[0]


def get_cv():
    """The one validation split every model uses: GroupKFold by travel group.

    No shuffling, so the folds depend only on the groups and are identical
    for every role and every run.
    """
    return GroupKFold(n_splits=N_SPLITS)


def evaluate(pipeline, X, y, groups, name, owner, params=None):
    """Cross-validate a pipeline, log one row to results.csv, return the scores.

    The pipeline is refitted inside each fold, so any preprocessing it holds
    (imputer, scaler, encoder) only ever sees that fold's training rows.
    """
    # Precision/recall/F1 treat True as the positive class; a string target
    # would make them fail. Stop early with a clear message instead.
    if y.dtype != bool:
        raise ValueError(f"y must be True/False (bool), got dtype {y.dtype}")

    # error_score="raise": a fold that fails stops the run instead of being
    # logged as NaN and silently dragging the mean around.
    scores = cross_validate(pipeline, X, y, groups=groups, cv=get_cv(),
                            scoring=METRICS, return_train_score=True,
                            return_estimator=True, return_indices=True,
                            error_score="raise")
    # Out-of-fold predictions from the same 5 fitted models: each row is
    # predicted by a model that never saw it, so this confusion matrix is an
    # honest picture of the errors, at no extra training cost.
    oof_pred = pd.Series(index=y.index, dtype=bool)
    for model, test_idx in zip(scores["estimator"], scores["indices"]["test"]):
        oof_pred.iloc[test_idx] = model.predict(X.iloc[test_idx])

    result = {"model": name, "owner": owner}
    for m in METRICS:
        result[f"{m}_mean"] = scores[f"test_{m}"].mean()
        result[f"{m}_std"] = scores[f"test_{m}"].std()
    # Average fit time per fold, in seconds (efficiency is graded too).
    result["fit_time_s"] = scores["fit_time"].mean()
    result["params"] = json.dumps(params or {}, default=str)
    result["date"] = date.today().isoformat()
    # Not logged, but returned for the overfitting check (Roles 3 and 4).
    result["train_accuracy_mean"] = scores["train_accuracy"].mean()
    result["confusion_matrix"] = confusion_matrix(y, oof_pred)

    # One row per model: re-running a script replaces its old row, so the
    # comparison can never pick a stale or best-of-many result.
    new_row = pd.DataFrame([result])[RESULTS_COLUMNS]
    if RESULTS_PATH.exists() and RESULTS_PATH.stat().st_size > 0:
        old = pd.read_csv(RESULTS_PATH)
        new_row = pd.concat([old[old["model"] != name], new_row], ignore_index=True)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_row.to_csv(RESULTS_PATH, index=False)

    print(f"{name}: accuracy {result['accuracy_mean']:.4f} "
          f"± {result['accuracy_std']:.4f}, ROC-AUC {result['roc_auc_mean']:.4f} "
          f"(train accuracy {result['train_accuracy_mean']:.4f})")
    return result


def _self_test():
    """Check get_groups() and evaluate() on tiny data whose answers we know."""
    import tempfile
    from pathlib import Path

    import pandas as pd
    from sklearn.dummy import DummyClassifier

    import src.evaluation as ev

    # 20 travel groups x 4 people; each group has 3 transported, 1 not.
    # Every fold therefore has the same 75% majority, so a "most frequent"
    # dummy always predicts True and its scores are known exactly.
    ids = [f"{g:04d}_{p:02d}" for g in range(1, 21) for p in range(1, 5)]
    df = pd.DataFrame({"PassengerId": ids, "x": range(80)})
    y = pd.Series([p != 4 for g in range(1, 21) for p in range(1, 5)])

    groups = ev.get_groups(df)
    assert groups.iloc[0] == "0001" and groups.iloc[-1] == "0020"
    assert groups.nunique() == 20
    assert ev.get_cv().get_n_splits() == 5

    with tempfile.TemporaryDirectory() as tmp:
        # Write to a temp folder so the real results/results.csv stays clean.
        ev.RESULTS_PATH = Path(tmp) / "results.csv"
        dummy = DummyClassifier(strategy="most_frequent")
        res = ev.evaluate(dummy, df[["x"]], y, groups, name="dummy", owner="selftest")
        # Re-running a script must replace its row, not add a duplicate.
        ev.evaluate(dummy, df[["x"]], y, groups, name="dummy", owner="selftest")
        logged = pd.read_csv(ev.RESULTS_PATH)
        ev.evaluate(dummy, df[["x"]], y, groups, name="other", owner="selftest")
        assert len(pd.read_csv(ev.RESULTS_PATH)) == 2

        # A target that isn't True/False must fail loudly, not log NaN scores.
        try:
            ev.evaluate(dummy, df[["x"]], y.astype(str), groups, "bad", "selftest")
            raise AssertionError("string target was accepted")
        except ValueError:
            pass

    assert abs(res["accuracy_mean"] - 0.75) < 1e-9
    assert res["accuracy_std"] < 1e-9
    assert abs(res["precision_mean"] - 0.75) < 1e-9
    assert abs(res["recall_mean"] - 1.0) < 1e-9
    assert abs(res["f1_mean"] - 6 / 7) < 1e-9
    assert abs(res["roc_auc_mean"] - 0.5) < 1e-9
    # Rows = true class (False, True); columns = predicted class.
    assert res["confusion_matrix"].tolist() == [[0, 20], [0, 60]]

    assert list(logged.columns) == [
        "model", "owner", "accuracy_mean", "accuracy_std", "precision_mean",
        "recall_mean", "f1_mean", "roc_auc_mean", "fit_time_s", "params", "date"]
    assert len(logged) == 1 and logged.loc[0, "model"] == "dummy"
    print("Self-test passed: get_groups(), get_cv() and evaluate() work.")


if __name__ == "__main__":
    _self_test()
