"""Baseline models: the scores every tuned model must beat (Role 2).

1. DummyClassifier: always predicts the most common class. This is the floor;
   a model that can't beat it has learned nothing.
2. Logistic Regression: a simple, explainable linear model on the shared
   features. This is a classifier: it outputs a probability between 0 and 1
   for "transported". Linear Regression would predict an unbounded number,
   which doesn't suit a True/False target.

Both are scored with the shared evaluate() (same GroupKFold folds as every
model) and logged to results/results.csv.

Run from the project folder:  python 03_baseline.py
"""
import warnings

import matplotlib
matplotlib.use("Agg")  # write PNG files without needing a screen
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import N_SPLITS, PLOT_DIR, SEED, TARGET, TRAIN_PATH
from src.evaluation import evaluate, get_groups
from src.features import add_features, build_preprocessor

BLUE, ORANGE = "#2a78d6", "#eb6834"  # pushes towards transported / not transported
OWNER = "Role 2"


def load_data():
    df = pd.read_csv(TRAIN_PATH)
    X = add_features(df).drop(columns=[TARGET])  # same X as every model script
    return X, df[TARGET], get_groups(df)


def plot_coefficients(pipe, X, y, top=12):
    """Largest Logistic Regression coefficients: which features push which way.

    Fitted once on all of train.csv for interpretation only; the scores
    above come from cross-validation, not from this fit.
    """
    pipe.fit(X, y)
    coefs = pd.Series(pipe["model"].coef_[0],
                      index=pipe["prep"].get_feature_names_out())
    coefs.index = coefs.index.str.replace(r"^(num|cat)__", "", regex=True)
    shown = coefs.loc[coefs.abs().sort_values().index[-top:]]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(shown.index, shown, color=[BLUE if v > 0 else ORANGE for v in shown])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Coefficient (scaled features; > 0 = more likely transported)")
    ax.set_title("Deck C/B and Europa push towards transported; Spa and VRDeck spending away")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "09_logreg_coefficients.png", dpi=120)
    plt.close(fig)
    print("   chart: plots/09_logreg_coefficients.png")
    return coefs.sort_values()


def main():
    PLOT_DIR.mkdir(exist_ok=True)
    X, y, groups = load_data()

    print("1. DummyClassifier (most frequent class)")
    # No preprocessor: the dummy ignores X, so the shared Pipeline would change nothing.
    dummy = DummyClassifier(strategy="most_frequent")
    # In a fold where it predicts "False" for everyone, precision is 0/0;
    # scikit-learn warns and scores it 0. That is expected for this model only.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        res_dummy = evaluate(dummy, X, y, groups, name="dummy_most_frequent", owner=OWNER,
                             params={"strategy": "most_frequent"})
    print(f"   recall {res_dummy['recall_mean']:.2f}: it predicted 'transported' for everyone "
          f"in {res_dummy['recall_mean'] * N_SPLITS:.0f} of {N_SPLITS} folds. The majority flips between folds "
          f"(50.4% overall is nearly a tie), so its accuracy is just below 50%.")

    print("\n2. Logistic Regression")
    # max_iter=1000: the default 100 iterations may stop before converging.
    logreg_params = {"max_iter": 1000}
    logreg = Pipeline([("prep", build_preprocessor()),
                       ("model", LogisticRegression(random_state=SEED, **logreg_params))])
    res_lr = evaluate(logreg, X, y, groups, name="logistic_regression", owner=OWNER,
                      params=logreg_params)

    print(f"\nComparison ({N_SPLITS}-fold GroupKFold, mean ± std):")
    table = pd.DataFrame([res_dummy, res_lr]).set_index("model")
    table["accuracy"] = (table["accuracy_mean"].map("{:.4f}".format) + " ± "
                         + table["accuracy_std"].map("{:.4f}".format))
    print(table[["accuracy", "f1_mean", "roc_auc_mean", "train_accuracy_mean",
                 "fit_time_s"]].round(4).to_string())

    cm = res_lr["confusion_matrix"]
    print(f"\nLogistic Regression out-of-fold confusion matrix "
          f"(rows = true False/True, cols = predicted False/True):\n{cm}")

    print("\n3. Coefficients")
    coefs = plot_coefficients(logreg, X, y)
    # Caveat: features overlap (each bill and TotalSpend, one-hot pairs), so a
    # single coefficient's size is a rough guide, not a precise effect.
    print(f"   most towards transported: {coefs.tail(3).round(2).to_dict()}")
    print(f"   most towards not transported: {coefs.head(3).round(2).to_dict()}")

    gain = res_lr["accuracy_mean"] - res_dummy["accuracy_mean"]
    gap = res_lr["train_accuracy_mean"] - res_lr["accuracy_mean"]
    print(f"\n-> Logistic Regression reaches {res_lr['accuracy_mean']:.1%} accuracy, "
          f"{gain * 100:.1f} points above the {res_dummy['accuracy_mean']:.1%} dummy floor, "
          f"with a train-validation gap of only {gap * 100:.1f} points (no overfitting). "
          f"The tuned Random Forest and LightGBM must beat {res_lr['accuracy_mean']:.1%}.")


if __name__ == "__main__":
    main()
