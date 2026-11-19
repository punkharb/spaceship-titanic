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

Candidates:
- #17 group-aware imputation (GroupModeImputer).
- #18 richer features, one at a time: family size from surnames, cabin-number
  regions, spending shares. They live here, not in src/features.py, until
  one of them earns its place.

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
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer

from src.config import ID_COL, RESULTS_DIR, SEED, TARGET, TEST_PATH, TRAIN_PATH
from src.evaluation import get_cv, get_groups
from src.features import SPEND_COLUMNS, add_features, build_preprocessor

REGION_WIDTH = 300  # cabin numbers per region; fixed before looking at any score


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


# ---------------------------------------------------------------------------
# Issue #18: richer features. Each function = the current add_features() plus
# ONE new idea, so every variant is scored on its own.
# ---------------------------------------------------------------------------

def _surname(df):
    """Last word of Name ('Maham Ofracculy' -> 'Ofracculy'); NaN if Name is missing."""
    return df["Name"].str.split().str[-1]


def add_family_size(df):
    """FamilySize: passengers in the same file with the same surname.

    This is the feature as the issue words it. Its weakness: the count
    depends on how many rows the file has. test.csv is half the size of
    train.csv, so the same family gets a smaller number there (see
    describe_richer()). It also counts unrelated people who share a name.
    """
    df = add_features(df)
    surname = _surname(df)
    df["FamilySize"] = surname.map(surname.value_counts())  # missing Name stays NaN
    return df


def add_family_in_group(df):
    """FamilyInGroup: passengers with the same surname in the same travel group.

    A stricter family: relatives who travel together. Counted inside the
    travel group, so it does not depend on the file size, and GroupKFold
    keeps a whole group in one fold, so the count never crosses folds.
    """
    df = add_features(df)
    key = get_groups(df) + "_" + _surname(df)  # NaN when Name is missing
    df["FamilyInGroup"] = key.map(key.value_counts())
    return df


def add_cabin_region(df):
    """CabinRegion: CabinNum in fixed blocks of REGION_WIDTH (0-299 -> R0, ...).

    A fixed rule on the row's own value, so it is allowed in add_features().
    """
    df = add_features(df)
    region = "R" + (df["CabinNum"] // REGION_WIDTH).astype("Int64").astype(str)
    df["CabinRegion"] = region.where(df["CabinNum"].notna())  # keep gaps as NaN
    return df


def add_spend_shares(df):
    """<column>_share: each bill as a fraction of the passenger's TotalSpend.

    Says *what* the money went on, separately from how much. Row-wise only.
    """
    df = add_features(df)
    total = df["TotalSpend"]
    for c in SPEND_COLUMNS:
        share = df[c] / total
        share[total == 0] = 0.0  # 0/0: spent nothing, so every share is 0
        df[f"{c}_share"] = share
    return df


def add_luxury_share(df):
    """LuxuryShare: RoomService + Spa + VRDeck as a fraction of TotalSpend.

    One column instead of five: the EDA and the Logistic Regression both
    show these three bills push towards 'not transported', while FoodCourt
    and ShoppingMall push the other way.
    """
    df = add_features(df)
    total = df["TotalSpend"]
    share = df[["RoomService", "Spa", "VRDeck"]].sum(axis=1, min_count=1) / total
    share[total == 0] = 0.0
    df["LuxuryShare"] = share
    return df


def with_columns(numeric=(), categorical=(), extra=()):
    """Preprocessor variant: the shared one plus the new columns.

    New numeric columns get the same median + scaling, new categorical ones
    the same mode + one-hot, as every existing feature. `extra` adds whole
    transformers (used for bin edges that are learned inside the fold).
    """
    more = {"num": list(numeric), "cat": list(categorical)}

    def make():
        prep = build_preprocessor()
        prep.transformers = [(name, steps, columns + more[name])
                             for name, steps, columns in prep.transformers] + list(extra)
        return prep
    return make


def learned_regions(n_bins=7):
    """Cabin regions whose edges are learned: 7 equal-count bins of CabinNum.

    The edges are quantiles, i.e. statistics of other rows, so by rule 5
    this must sit inside the Pipeline and is fitted on the training folds only.
    """
    return ("region", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("bins", KBinsDiscretizer(n_bins=n_bins, encode="onehot-dense", strategy="quantile",
                                  quantile_method="averaged_inverted_cdf")),
    ]), ["CabinNum"])


def describe_richer(train, test):
    """Plain-number checks on the #18 features, before any model is scored."""
    print("\nIssue #18 feature checks:")
    fam = {n: add_family_size(d)["FamilySize"] for n, d in [("train", train), ("test", test)]}
    print(f"   FamilySize (surname count in file): mean {fam['train'].mean():.2f} in train vs "
          f"{fam['test'].mean():.2f} in test; max {fam['train'].max():.0f} vs {fam['test'].max():.0f} "
          f"-> the scale depends on the file size")
    grp = {n: add_family_in_group(d) for n, d in [("train", train), ("test", test)]}
    print(f"   FamilyInGroup (surname count in travel group): mean "
          f"{grp['train']['FamilyInGroup'].mean():.2f} in train vs {grp['test']['FamilyInGroup'].mean():.2f} "
          f"in test; max {grp['train']['FamilyInGroup'].max():.0f} vs {grp['test']['FamilyInGroup'].max():.0f}")
    t = grp["train"].dropna(subset=["FamilyInGroup"])
    same = (t["FamilyInGroup"] == t["GroupSize"]).mean()
    print(f"   FamilyInGroup equals GroupSize for {same:.1%} of train passengers with a name "
          f"-> mostly a copy of a feature we already have")
    multi = t[t["GroupSize"] > 1]
    one_name = multi.groupby(get_groups(multi))["Name"].agg(lambda n: _surname(n.to_frame()).nunique() == 1)
    print(f"   {one_name.mean():.1%} of multi-person train groups share one surname")
    reg = add_cabin_region(train)
    rate = reg.groupby("CabinRegion")[TARGET].agg(["size", "mean"])
    print("   CabinRegion (train) size and transported rate: "
          + ", ".join(f"{r} n={int(v['size'])} {v['mean']:.0%}" for r, v in rate.iterrows()))
    sh = add_spend_shares(train)
    spenders = sh[sh["TotalSpend"] > 0]
    print("   Mean spending share (train, spenders only), transported vs not: "
          + ", ".join(f"{c} {spenders.loc[spenders[TARGET], c + '_share'].mean():.0%} vs "
                      f"{spenders.loc[~spenders[TARGET], c + '_share'].mean():.0%}" for c in SPEND_COLUMNS))


# name -> (add_features function, preprocessor factory)
VARIANTS = {
    "current features": (add_features, build_preprocessor),
    "#17 group fill: HomePlanet, Side": (add_features, with_group_fill(["HomePlanet", "Side"])),
    "#17 group fill: + Deck": (add_features, with_group_fill(["HomePlanet", "Side", "Deck"])),
    "#18 family: FamilySize (surname in file)": (add_family_size, with_columns(numeric=["FamilySize"])),
    "#18 family: FamilyInGroup (surname in travel group)": (
        add_family_in_group, with_columns(numeric=["FamilyInGroup"])),
    f"#18 cabin: fixed regions of {REGION_WIDTH}": (add_cabin_region, with_columns(categorical=["CabinRegion"])),
    "#18 cabin: learned regions (7 quantile bins, in Pipeline)": (
        add_features, with_columns(extra=[learned_regions()])),
    "#18 spending: 5 shares of TotalSpend": (
        add_spend_shares, with_columns(numeric=[f"{c}_share" for c in SPEND_COLUMNS])),
    "#18 spending: LuxuryShare": (add_luxury_share, with_columns(numeric=["LuxuryShare"])),
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
    describe_richer(raw, pd.read_csv(TEST_PATH))
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
    _self_test_richer()


def _self_test_richer():
    """#18 features on six passengers whose answers can be checked by hand."""
    df = pd.DataFrame({
        ID_COL: ["0001_01", "0001_02", "0001_03", "0002_01", "0003_01", "0004_01"],
        "Name": ["Ann Lee", "Bob Lee", "Cy Fox", "Di Lee", None, "Ed Roy"],
        "Cabin": ["F/0/S", "F/299/S", "F/300/P", "G/1500/P", None, "A/5/S"],
        "CryoSleep": [False, False, False, False, True, False],
        "Age": [30, 30, 30, 30, 30, 30],
        "RoomService": [10.0, 0.0, 50.0, 0.0, None, None],
        "FoodCourt": [30.0, 0.0, 50.0, 0.0, None, 20.0],
        "ShoppingMall": [0.0, 0.0, 0.0, 0.0, None, 0.0],
        "Spa": [0.0, 0.0, 0.0, 0.0, None, 20.0],
        "VRDeck": [0.0, 0.0, 0.0, 0.0, None, 0.0],
    })
    # Three people are called Lee, but only two of them travel together.
    assert add_family_size(df)["FamilySize"].fillna(-1).tolist() == [3, 3, 1, 3, -1, 1]
    assert add_family_in_group(df)["FamilyInGroup"].fillna(-1).tolist() == [2, 2, 1, 1, -1, 1]
    assert add_cabin_region(df)["CabinRegion"].fillna("?").tolist() == ["R0", "R0", "R1", "R5", "?", "R0"]
    shares = add_spend_shares(df)
    assert shares["RoomService_share"].iloc[:5].tolist() == [0.25, 0.0, 0.5, 0.0, 0.0]
    # A missing bill stays missing (the imputer fills it later); it is not guessed here.
    assert pd.isna(shares.loc[5, "RoomService_share"]) and shares.loc[5, "Spa_share"] == 0.5
    assert add_luxury_share(df)["LuxuryShare"].tolist() == [0.25, 0.0, 0.5, 0.0, 0.0, 0.5]
    # The variant preprocessor adds columns without touching the shared one.
    n_before = len(build_preprocessor().transformers[0][2])
    assert len(with_columns(numeric=["FamilySize"])().transformers[0][2]) == n_before + 1
    assert len(build_preprocessor().transformers[0][2]) == n_before
    print("Self-test passed: #18 features match the hand-checked answers.")


if __name__ == "__main__":
    _self_test() if "--self-test" in sys.argv else main()
