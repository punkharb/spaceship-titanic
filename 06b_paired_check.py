"""Is a challenger clearly better than the final model? Fold-by-fold check (Role 1).

06_compare.py ranks models by mean accuracy, and every mean has a fold-to-fold
std of about 1 point. But all models use the same 5 get_cv() folds, so they
can be compared fold by fold: fold difficulty then cancels out, and the noise
that matters is how much the per-fold *gain* varies.

Rule (same as 02b_feature_experiments.py): a challenger is a clear win only if
its mean gain is more than 2 standard errors of the per-fold gains (a rough
95% check). Otherwise the final model stays, because a smaller lead could be
which passengers landed in which fold.

Each model is refitted with its saved best_params_<name>.json on the shared
folds (no re-tuning); the means equal the rows in results.csv. Nothing is
written. xgboost is not checked: it scores below the final model on average.

Run from the project folder:  python 06b_paired_check.py   (about 1 minute)
Self-test:                    python 06b_paired_check.py --self-test
"""
import importlib
import json
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate

from src.config import RESULTS_DIR, SEED, TARGET, TRAIN_PATH
from src.evaluation import get_cv, get_groups
from src.features import add_features

FINAL = "lightgbm"  # the model 07_final_model.py ships


def _catboost_pipeline():
    # Built exactly as in 04b_xgboost_catboost.py (imported there next to xgboost,
    # so it is rebuilt here to keep this check runnable without xgboost).
    from catboost import CatBoostClassifier
    from sklearn.pipeline import Pipeline
    from src.features import build_preprocessor
    return Pipeline([("prep", build_preprocessor()),
                     ("model", CatBoostClassifier(random_seed=SEED, verbose=0,
                                                  allow_writing_files=False))])


# name -> untuned pipeline, built by the owner's script
PIPELINES = {
    "lightgbm": lambda: importlib.import_module("05_lightgbm").make_pipeline(),
    "random_forest": lambda: importlib.import_module("04_random_forest").make_pipeline(),
    "catboost": _catboost_pipeline,
}


def paired_verdict(base, challenger):
    """Mean per-fold gain of challenger over base, and whether it is clear."""
    gain = pd.Series(challenger) - pd.Series(base)
    threshold = 2 * gain.std(ddof=1) / np.sqrt(len(gain))
    return {"gain": gain.mean(), "threshold": threshold, "per_fold": gain,
            "better_folds": int((gain > 0).sum()), "clear_win": gain.mean() > threshold}


def fold_scores(name, X, y, groups):
    """Accuracy on each shared fold with the model's saved best parameters."""
    params = json.loads((RESULTS_DIR / f"best_params_{name}.json").read_text())
    pipe = PIPELINES[name]().set_params(**params)
    return pd.Series(cross_validate(pipe, X, y, groups=groups, cv=get_cv(),
                                    scoring="accuracy", error_score="raise")["test_score"])


def main():
    raw = pd.read_csv(TRAIN_PATH)
    X, y, groups = add_features(raw).drop(columns=[TARGET]), raw[TARGET], get_groups(raw)
    scores = {name: fold_scores(name, X, y, groups) for name in PIPELINES}

    print("Accuracy per shared fold (saved best params):")
    for name, s in scores.items():
        print(f"   {name:14} {', '.join(f'{a:.4f}' for a in s)}  mean {s.mean():.4f}")
    for name in PIPELINES:
        if name == FINAL:
            continue
        v = paired_verdict(scores[FINAL], scores[name])
        print(f"\n{name} vs {FINAL}: gain {v['gain'] * 100:+.2f} pts (per fold "
              f"{', '.join(f'{g * 100:+.2f}' for g in v['per_fold'])}; better on "
              f"{v['better_folds']}/5); needs > 2 x SE = {v['threshold'] * 100:.2f} pts -> "
              + (f"clear win: the team should consider switching to {name}."
                 if v["clear_win"] else f"not a clear win: {FINAL} stays the final model."))


def _self_test():
    """The verdict follows the 2 x standard-error rule on the per-fold gains."""
    import pandas as pd

    base = pd.Series([0.80, 0.81, 0.82, 0.79, 0.80])
    # Ahead on every fold by about 1 point: a clear, consistent win.
    clear = paired_verdict(base, base + [0.010, 0.011, 0.009, 0.010, 0.012])
    assert clear["clear_win"] and clear["better_folds"] == 5
    # Same mean gain but it swings across folds: not clearly real.
    noisy = paired_verdict(base, base + [0.04, -0.02, 0.03, -0.01, 0.01])
    assert not noisy["clear_win"]
    assert abs(noisy["gain"] - 0.01) < 1e-12
    print("Self-test passed: paired_verdict() applies the 2 x SE rule.")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
    else:
        main()
