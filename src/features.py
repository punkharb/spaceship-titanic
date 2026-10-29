"""Feature creation and preprocessing shared by every model (Role 2).

Two steps, kept apart on purpose:
- add_features(df): row-wise rules only (plus group counts within the same
  file). It learns nothing from other rows or the target, so it is safe to
  apply to train.csv and test.csv before any split.
- build_preprocessor(): everything that *learns* from data (imputer medians
  and modes, scaler means, one-hot categories). It is returned unfitted and
  used inside a Pipeline, so it is fitted on the training folds only.

Every choice here is backed by results/eda_findings.txt (01_explore.py).
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import ID_COL

SPEND_COLUMNS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
CHILD_AGE = 13  # EDA: 0 of 806 under-13s in train spent anything; the youngest spender is 13

NUMERIC_FEATURES = [
    "Age",
    "CabinNum",    # position along the deck; from Cabin
    "GroupSize",   # EDA: solo travellers 45% transported vs 57% in groups
    # log1p of spending: EDA skewness 6-13. Trees don't care, but it stops a
    # few huge bills from dominating the scaled Logistic Regression.
    *[f"{c}_log" for c in SPEND_COLUMNS],
    "TotalSpend_log",
]
CATEGORICAL_FEATURES = [
    "HomePlanet",
    "CryoSleep",   # EDA: strongest single feature (82% vs 33% transported)
    "Destination",
    "VIP",
    "Deck",        # EDA: 38 pp spread in transported rate between decks
    "Side",
]
DROP_COLUMNS = [
    ID_COL,        # an ID; its travel group is used via GroupSize and the CV folds
    "Name",        # 99.6% unique: a model could only memorise it
    "Cabin",       # 77.5% unique; its useful parts are Deck, CabinNum, Side
    *SPEND_COLUMNS, "TotalSpend",  # replaced by their _log versions
]


def _split_cabin(df):
    """Cabin 'Deck/Num/Side' (e.g. 'F/123/S') -> Deck, CabinNum, Side."""
    # reindex: always three columns, even for a batch where every Cabin is missing.
    parts = df["Cabin"].str.split("/", expand=True).reindex(columns=[0, 1, 2])
    df["Deck"] = parts[0]
    df["CabinNum"] = pd.to_numeric(parts[1])
    df["Side"] = parts[2]
    return df


def _fill_by_rules(df):
    """Fill gaps that follow logically from the same row (no statistics).

    EDA (train): 0 of 3,037 CryoSleep passengers and 0 of 806 children under 13
    spent anything. So spending > 0 means awake, and a sleeper's or a
    child's missing bill must be 0. The reverse (no spending -> asleep) is
    not used: 9% of awake passengers also spent nothing.
    """
    spent = df[SPEND_COLUMNS].sum(axis=1) > 0      # NaN counts as 0 here
    df.loc[df["CryoSleep"].isna() & spent, "CryoSleep"] = False

    no_spend = (df["CryoSleep"] == True) | (df["Age"] < CHILD_AGE)  # noqa: E712 (column holds NaN too)
    for c in SPEND_COLUMNS:
        df.loc[no_spend & df[c].isna(), c] = 0.0
    return df


def add_features(df):
    """Row-wise feature creation; safe to run before the train/validation split.

    Keeps every original column (PassengerId is still needed for the CV
    groups); build_preprocessor() selects the model's columns itself.
    """
    df = _split_cabin(df.copy())
    df = _fill_by_rules(df)

    # Counted within the same file only. GroupKFold keeps a whole group in
    # one fold, so this never counts passengers from another fold, and it
    # never looks at the target.
    group = df[ID_COL].str.split("_").str[0]
    df["GroupSize"] = group.map(group.value_counts())

    # Sum of the known bills; NaN only if all five are missing (then imputed).
    df["TotalSpend"] = df[SPEND_COLUMNS].sum(axis=1, min_count=1)
    for c in SPEND_COLUMNS + ["TotalSpend"]:
        df[f"{c}_log"] = np.log1p(df[c])
    return df


def build_preprocessor():
    """Unfitted imputation + scaling + one-hot encoding, for use inside a Pipeline.

    Medians, modes, scaling and categories are learned in .fit(), i.e. from
    the training folds only. handle_unknown="ignore" means a category unseen
    in training (e.g. the 5-passenger Deck T) becomes all zeros instead of
    crashing.
    """
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),  # robust to skewed columns
        ("scale", StandardScaler()),                   # needed by Logistic Regression
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ], remainder="drop")  # anything not listed (DROP_COLUMNS, the target) is never used
