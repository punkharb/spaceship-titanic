"""Shared settings for every script (Role 1).

Import from here instead of hard-coding paths or numbers, so all five roles
use the same data, the same seed and the same number of folds. That is what
keeps the model comparison fair.
"""
from pathlib import Path

# Paths are built from this file's location, not the current working folder,
# so they are correct on every team member's computer.
DATA_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
PLOT_DIR = DATA_DIR / "plots"
RESULTS_DIR = DATA_DIR / "results"

TARGET = "Transported"   # the column we predict (True/False)
ID_COL = "PassengerId"   # format gggg_pp: travel group + person number
SEED = 42                # one seed everywhere -> reproducible results
N_SPLITS = 5             # number of GroupKFold folds
