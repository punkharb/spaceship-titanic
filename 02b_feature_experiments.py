"""Feature experiments: does a candidate change actually help? (Role 2 with Role 1)

Each variant is scored with the final model (tuned LightGBM, its saved
best_params_lightgbm.json, no re-tuning) on the same 5 shared get_cv()
folds as the current features, and compared fold by fold. Same folds on
both sides make the comparison paired: fold difficulty cancels out, so the
noise that matters is the spread of the per-fold *gains*, not the spread of
the scores. Default LightGBM and Logistic Regression are run too, as a check
that an effect is not just LightGBM having been tuned on the current features.

Decision rule (issues #17 and #18): keep a change only if its mean gain is
more than 2 standard errors of the per-fold gains (a rough 95% check).

Experiments are not logged to results.csv: they are decisions about the
features, not models. Kept changes go into src/features.py, and 03-07 are
re-run so every logged score uses them.

Run from the project folder:  python 02b_feature_experiments.py
Self-test:                    python 02b_feature_experiments.py --self-test
"""
import json
import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

from src.config import ID_COL, RESULTS_DIR, SEED, TARGET, TEST_PATH, TRAIN_PATH
from src.evaluation import get_cv, get_groups
from src.features import add_features, build_preprocessor


class GroupModeImputer(TransformerMixin, BaseEstimator):
    """#17: fill a gap with the most common value among the travel group.

    Learns nothing in fit(): each batch is filled from its own rows. Safe
    because GroupKFold keeps a whole travel group in one fold and no group
    spans train.csv and test.csv, so group mates are always in the same
    batch, and the target is never read. It must sit inside the Pipeline
    (rule 5), because it uses other rows, which add_features() may not.
    """

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        group = get_groups(X)
        for c in self.columns:
            X[c] = X[c].mask(X[c].isna(), group.map(_group_mode(group, X[c])))
        return X


def _group_mode(group, values):
    """Most common known value per travel group (ties: alphabetical), vectorised."""
    counts = pd.DataFrame({"g": group, "v": values}).dropna().value_counts().reset_index()
    best = counts.sort_values(["g", "count", "v"], ascending=[True, False, True])
    return best.drop_duplicates("g").set_index("g")["v"]


def describe_fill(df, columns):
    """How often group mates agree, and how many gaps a group fill could close."""
    group = get_groups(df)
    multi = group.map(group.value_counts()) > 1
    for c in columns:
        known = df[multi].groupby(group[multi])[c].agg(["count", "nunique"])
        known = known[known["count"] >= 2]  # groups with at least 2 known values
        fillable = df[c].isna() & group.map(df.groupby(group)[c].count() > 0)
        print(f"   {c:12} groups agree {(known['nunique'] == 1).mean():6.1%} "
              f"| missing {df[c].isna().sum():3} | fillable from group mates {fillable.sum():3}")


def with_group_fill(columns):
    """Preprocessor variant: group fill first, then the shared preprocessor."""
    return lambda: Pipeline([("group_fill", GroupModeImputer(columns)),
                             ("prep", build_preprocessor())])


# name -> (add_features function, preprocessor factory)
VARIANTS = {
    "current features": (add_features, build_preprocessor),
    "#17 group fill: HomePlanet, Side": (add_features, with_group_fill(["HomePlanet", "Side"])),
    "#17 group fill: + Deck": (add_features, with_group_fill(["HomePlanet", "Side", "Deck"])),
}


def models(params):
    """The final model decides; two untuned models check the effect isn't a
    side effect of LightGBM having been tuned on the current features."""
    return {
        "tuned LightGBM (decides)": lambda: LGBMClassifier(random_state=SEED, verbose=-1).set_params(
            **{k.removeprefix("model__"): v for k, v in params.items()}),
        "default LightGBM (check)": lambda: LGBMClassifier(random_state=SEED, verbose=-1),
        "Logistic Regression (check)": lambda: LogisticRegression(max_iter=1000, random_state=SEED),
    }


def fold_scores(make_features, make_prep, make_model, raw):
    """Accuracy on each of the 5 shared folds."""
    X = make_features(raw).drop(columns=[TARGET])
    pipe = Pipeline([("prep", make_prep()), ("model", make_model())])
    cv = cross_validate(pipe, X, raw[TARGET], groups=get_groups(raw), cv=get_cv(),
                        scoring="accuracy", error_score="raise")
    return pd.Series(cv["test_score"])


def main():
    raw = pd.read_csv(TRAIN_PATH)
    print("Group agreement (multi-person groups with >= 2 known values):")
    for name, df in [("train", add_features(raw)), ("test", add_features(pd.read_csv(TEST_PATH)))]:
        print(f" {name}")
        describe_fill(df, ["HomePlanet", "Side", "Deck", "Destination", "VIP", "CryoSleep", "Age"])
    params = json.loads((RESULTS_DIR / "best_params_lightgbm.json").read_text())
    for model_name, make_model in models(params).items():
        scores = {name: fold_scores(f, p, make_model, raw) for name, (f, p) in VARIANTS.items()}
        base = scores["current features"]
        decides = "decides" in model_name
        print(f"\n{model_name}: current features {base.mean():.4f} ± {base.std(ddof=0):.4f}")
        for name, s in scores.items():
            if name == "current features":
                continue
            gain = s - base
            line = (f"  {name}: {s.mean():.4f} | gain {gain.mean() * 100:+.2f} pts (per fold "
                    f"{', '.join(f'{g * 100:+.2f}' for g in gain)}; better on {(gain > 0).sum()}/5)")
            if decides:
                se = gain.std(ddof=1) / np.sqrt(len(gain))
                keep = gain.mean() > 2 * se
                line += (f" -> {'KEEP' if keep else 'drop'}: needs > 2 x SE = {2 * se * 100:.2f} pts")
            print(line)


def _self_test():
    """GroupModeImputer fills only from group mates in the same batch."""
    df = pd.DataFrame({
        ID_COL: ["0001_01", "0001_02", "0002_01", "0003_01", "0003_02"],
        "HomePlanet": ["Earth", None, None, "Mars", "Mars"],
        "Side": [None, "S", "P", None, None],
    })
    out = GroupModeImputer(["HomePlanet", "Side"]).fit(df).transform(df).fillna("?")
    assert out["HomePlanet"].tolist() == ["Earth", "Earth", "?", "Mars", "Mars"]
    # A solo traveller (0002) or a group with no known value (0003) stays
    # missing; the normal imputer in the Pipeline handles those later.
    assert out["Side"].tolist() == ["S", "S", "P", "?", "?"]
    assert df["HomePlanet"].isna().sum() == 2, "input must not be changed in place"
    # Nothing is learned in fit(): another batch only uses its own group mates.
    other = pd.DataFrame({ID_COL: ["0001_03"], "HomePlanet": [None]})
    assert GroupModeImputer(["HomePlanet"]).fit(df).transform(other)["HomePlanet"].isna().all()
    print("Self-test passed: GroupModeImputer fills only from group mates in the same batch.")


if __name__ == "__main__":
    _self_test() if "--self-test" in sys.argv else main()
