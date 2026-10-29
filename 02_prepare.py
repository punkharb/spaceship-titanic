"""Demonstrate and check the feature and preprocessing steps (Role 2).

Shows what src/features.py does to the data and checks the things the
models rely on:
1. add_features(): the new columns and how many gaps the logical rules fill.
2. Every column is either a feature or deliberately dropped; the target is never a feature.
3. add_features() is safe before the split: on any training fold alone it
   gives exactly the same values as on the whole file.
4. build_preprocessor(), fitted on one training fold, leaves no missing
   values in the validation fold or in test.csv; prints the final feature count.

Run from the project folder:  python 02_prepare.py
"""
import numpy as np
import pandas as pd

from src.config import TARGET, TEST_PATH, TRAIN_PATH
from src.evaluation import get_cv, get_groups
from src.features import (CATEGORICAL_FEATURES, DROP_COLUMNS, NUMERIC_FEATURES,
                          SPEND_COLUMNS, add_features, build_preprocessor)


def show_new_columns(raw, feat):
    print("1. add_features()")
    new = [c for c in feat.columns if c not in raw.columns]
    print(f"   columns: {raw.shape[1]} raw -> {feat.shape[1]} after add_features()")
    print(f"   new: {new}")
    example = feat.loc[0, ["Cabin", "Deck", "CabinNum", "Side", "GroupSize",
                           "TotalSpend", "TotalSpend_log"]]
    print(f"   example (first passenger): {example.to_dict()}")


def show_rule_fills(name, raw, feat):
    """Missing values before vs after the row-wise rules (CryoSleep, spending)."""
    cols = ["CryoSleep"] + SPEND_COLUMNS
    before, after = raw[cols].isna().sum(), feat[cols].isna().sum()
    print(f"   {name}: rules filled {before.sum() - after.sum()} of {before.sum()} gaps in "
          f"{cols} ({(before - after).to_dict()})")


def check_columns(feat):
    print("\n2. Column accounting")
    listed = NUMERIC_FEATURES + CATEGORICAL_FEATURES + DROP_COLUMNS
    assert len(listed) == len(set(listed)), "a column is listed twice"
    unaccounted = set(feat.columns) - set(listed) - {TARGET}
    assert not unaccounted, f"columns neither used nor dropped: {unaccounted}"
    assert TARGET not in NUMERIC_FEATURES + CATEGORICAL_FEATURES, "target used as a feature!"
    print(f"   {len(NUMERIC_FEATURES)} numeric + {len(CATEGORICAL_FEATURES)} categorical "
          f"features, {len(DROP_COLUMNS)} dropped: {DROP_COLUMNS}")
    print("-> Every column is either a feature or deliberately dropped, and Transported is not a feature.")


def check_rowwise(raw):
    """Features computed on a training fold alone == features from the whole file."""
    print("\n3. Is add_features() safe before the split?")
    full = add_features(raw)
    for k, (train_idx, _) in enumerate(get_cv().split(raw, groups=get_groups(raw)), 1):
        fold = add_features(raw.iloc[train_idx])
        pd.testing.assert_frame_equal(fold, full.iloc[train_idx])
    print(f"   all {k} training folds: identical values, including GroupSize")
    print("-> add_features() uses nothing from other folds (GroupKFold keeps each travel group "
          "whole, so even GroupSize is unchanged). It is safe to run before the split.")


def check_preprocessor(train, test):
    print("\n4. build_preprocessor() fitted on one training fold")
    X = train.drop(columns=[TARGET])
    train_idx, val_idx = next(get_cv().split(X, groups=get_groups(X)))
    prep = build_preprocessor().fit(X.iloc[train_idx])
    X_val, X_test = prep.transform(X.iloc[val_idx]), prep.transform(test)

    names = prep.get_feature_names_out()
    print(f"   fitted on {len(train_idx):,} rows; validation {len(val_idx):,} rows; test {len(test):,} rows")
    print(f"   missing values left: validation {np.isnan(X_val).sum()}, test {np.isnan(X_test).sum()}")
    print(f"   final feature count: {len(names)}")
    print(f"   features: {list(names)}")
    assert np.isnan(X_val).sum() == 0 and np.isnan(X_test).sum() == 0
    assert X_test.shape[1] == len(names)
    print(f"-> After the preprocessor every row has {len(names)} model inputs and no gaps, "
          f"for validation rows and test.csv alike. Its medians, modes and categories came "
          f"from the training fold only.")


def main():
    raw_train, raw_test = pd.read_csv(TRAIN_PATH), pd.read_csv(TEST_PATH)
    train, test = add_features(raw_train), add_features(raw_test)

    show_new_columns(raw_train, train)
    show_rule_fills("train", raw_train, train)
    show_rule_fills("test", raw_test, test)
    print("-> The logical rules (CryoSleep/children never spend; spenders are awake) fill "
          "these gaps without any learned statistics; the rest is imputed inside the Pipeline.")
    check_columns(train)
    check_rowwise(raw_train)
    check_preprocessor(train, test)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
