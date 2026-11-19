"""Soft-voting blend of the tuned LightGBM and tuned Random Forest (Role 4, issue #19).

Idea: LightGBM (boosting) and Random Forest (bagging) build trees in different
ways, so they may make *different* mistakes. Soft voting averages the two
models' predicted probabilities and predicts "transported" when the average
is above 0.5. If the models disagree often and each is right in different
places, the average can beat both; if they agree almost everywhere, the blend
can only match them.

Design choices (both keep the comparison fair):
- Each model is rebuilt from its saved results/best_params_<name>.json, checked
  against the params logged in results.csv, so we blend exactly the models
  whose CV scores are already on the slides.
- Fixed 50/50 weights, no tuning: tuning the weights on the same folds would
  add another lucky pick (more optimism). With fixed weights the blend adds
  no selection of its own.

Steps:
1. Load both tuned parameter sets (checked against results.csv).
2. Score the blend with evaluate() on the shared GroupKFold folds
   -> "lgbm_rf_blend" row + results/best_params_lgbm_rf_blend.json.
   Nothing new is tuned: that json only records the configuration (the two
   components' params, as Pipeline keys) under the team's file-name convention.
   07_final_model.py ignores it unless Role 1 adds the blend to its MODELS.
3. Overfitting check: training-fold vs validation-fold accuracy.
4. Why the blend helps or not: how similar the two models' out-of-fold
   probabilities are, and who is right when they disagree.
5. Compare with LightGBM and Random Forest using the fold-to-fold std rule.

test.csv is never read here.

Run from the project folder:  python 05c_blend.py
"""
import importlib
import json

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline

from src.config import RESULTS_DIR, SEED, TARGET, TRAIN_PATH
from src.evaluation import RESULTS_PATH, evaluate, get_cv, get_groups
from src.features import add_features, build_preprocessor

OWNER = "Role 4"
NAME = "lgbm_rf_blend"
COMPONENTS = ("lightgbm", "random_forest")

# Reuse 07's check that a best_params json still matches the params logged
# with its CV score (the module name starts with a digit, hence importlib).
load_params = importlib.import_module("07_final_model").load_params


def load_data():
    df = pd.read_csv(TRAIN_PATH)
    X = add_features(df).drop(columns=[TARGET])  # same X as every model script
    return X, df[TARGET], get_groups(df)


def load_component_params():
    """Tuned params of each component, as plain model keywords (no 'model__')."""
    results = pd.read_csv(RESULTS_PATH)
    params = {}
    for name in COMPONENTS:
        saved = load_params(name, results, RESULTS_DIR)  # raises if json != results.csv
        params[name] = {k.removeprefix("model__"): v for k, v in saved.items()}
    return params


def make_component(name, params):
    """One tuned model, exactly as 04/05 validated it (same SEED)."""
    if name == "lightgbm":
        return LGBMClassifier(random_state=SEED, verbose=-1, **params)
    return RandomForestClassifier(random_state=SEED, **params)


def make_blend(params):
    """One shared preprocessor, then a 50/50 average of the two models' probabilities.

    The preprocessor is the same build_preprocessor() both models use alone,
    fitted inside each fold like every other pipeline (rule 4). Equal weights
    are VotingClassifier's default, so no weights are tuned.
    """
    voter = VotingClassifier(
        [(name, make_component(name, params[name])) for name in COMPONENTS],
        voting="soft")  # soft = average probabilities, not count votes
    return Pipeline([("prep", build_preprocessor()), ("model", voter)])


def flat_params(params):
    """Params as Pipeline keys (model__lightgbm__learning_rate, ...).

    pipe.set_params(**flat_params(params)) rebuilds exactly this blend, so the
    json and the results.csv row describe the model that was scored.
    """
    return {f"model__{name}__{k}": v
            for name in COMPONENTS for k, v in params[name].items()}


def oof_probabilities(params, X, y, groups):
    """Out-of-fold P(transported) of each model alone, on the shared folds."""
    probs = {}
    for name in COMPONENTS:
        pipe = Pipeline([("prep", build_preprocessor()),
                         ("model", make_component(name, params[name]))])
        probs[name] = cross_val_predict(pipe, X, y, groups=groups, cv=get_cv(),
                                        method="predict_proba")[:, 1]
    return pd.DataFrame(probs, index=y.index)


def agreement_report(probs, y):
    """How alike the two models are; returns the numbers for the conclusion."""
    corr = probs["lightgbm"].corr(probs["random_forest"])
    pred = probs > 0.5
    disagree = pred["lightgbm"] != pred["random_forest"]
    print(f"   correlation of the two models' probabilities: {corr:.4f}")
    print(f"   they predict different classes for {disagree.sum()} of {len(y)} "
          f"passengers ({disagree.mean():.1%})")
    if disagree.any():
        # Only these passengers can be changed by blending: where both models
        # agree, the average lands on the same side of 0.5.
        lgbm_right = (pred.loc[disagree, "lightgbm"] == y[disagree]).mean()
        blend_right = ((probs.loc[disagree].mean(axis=1) > 0.5) == y[disagree]).mean()
        print(f"   on those passengers: LightGBM right {lgbm_right:.1%}, "
              f"Random Forest right {1 - lgbm_right:.1%}, blend right {blend_right:.1%}")
        return corr, disagree.mean(), lgbm_right, blend_right
    return corr, 0.0, float("nan"), float("nan")


def compare_with_components(res):
    """Blend vs each tuned model, with the team's fold-to-fold std rule."""
    results = pd.read_csv(RESULTS_PATH).set_index("model")
    cols = ["accuracy_mean", "accuracy_std", "f1_mean", "roc_auc_mean", "fit_time_s"]
    print(results.loc[[*COMPONENTS, NAME], cols].round(4).to_string())
    verdicts = {}
    for other in COMPONENTS:
        diff = res["accuracy_mean"] - results.loc[other, "accuracy_mean"]
        # A rule of thumb, not a statistical test: a gap smaller than the
        # fold-to-fold std could come from which passengers landed in which fold.
        noise = max(res["accuracy_std"], results.loc[other, "accuracy_std"])
        clear = diff > noise
        print(f"   blend vs {other}: {diff * 100:+.2f} points ("
              + ("larger than" if abs(diff) > noise else "within")
              + f" the fold-to-fold spread, std ~{noise * 100:.2f} points)")
        verdicts[other] = (diff, clear)
    return verdicts


def main():
    X, y, groups = load_data()

    print("1. Tuned parameters (checked against results.csv)")
    params = load_component_params()
    for name in COMPONENTS:
        print(f"   {name}: {params[name]}")

    print("\n2. Soft-voting blend (50/50) on the shared GroupKFold folds")
    logged = flat_params(params)
    params_path = RESULTS_DIR / f"best_params_{NAME}.json"
    params_path.write_text(json.dumps(logged, indent=2))
    print(f"   saved {params_path.relative_to(RESULTS_DIR.parent)}")
    res = evaluate(make_blend(params), X, y, groups, name=NAME, owner=OWNER,
                   params=logged)
    print(f"   out-of-fold confusion matrix (rows = true False/True, "
          f"cols = predicted False/True):\n{res['confusion_matrix']}")

    print("\n3. Overfitting check (mean over the 5 folds)")
    gap = res["train_accuracy_mean"] - res["accuracy_mean"]
    print(f"   blend: train {res['train_accuracy_mean']:.4f} vs validation "
          f"{res['accuracy_mean']:.4f} -> gap {gap * 100:.1f} points")

    print("\n4. Do the two models make different mistakes? (out-of-fold)")
    probs = oof_probabilities(params, X, y, groups)
    corr, disagree_share, lgbm_right, blend_right = agreement_report(probs, y)

    print("\n5. Comparison with the two tuned models in results.csv")
    verdicts = compare_with_components(res)

    # Compare with whichever component scores best in results.csv, not an
    # assumed winner, so the conclusion stays right if the tuned rows change.
    accs = pd.read_csv(RESULTS_PATH).set_index("model")["accuracy_mean"]
    best = accs[list(COMPONENTS)].idxmax()
    diff_best, beats_best = verdicts[best]
    # Blending can only change passengers where the two models disagree, so
    # who is right there decides whether the average helps.
    why = (f"Blending only changes passengers where the models disagree "
           f"({disagree_share:.1%}; probability correlation {corr:.2f}). There, "
           f"LightGBM alone is right {lgbm_right:.1%} of the time and the blend "
           f"{blend_right:.1%}, so "
           + ("averaging in the weaker forest costs LightGBM some correct answers. "
              if blend_right < lgbm_right else
              "the forest corrects some of LightGBM's mistakes. "))
    print(f"\n-> The 50/50 blend scores {res['accuracy_mean']:.2%} accuracy, "
          f"{diff_best * 100:+.2f} points vs the best single model ({best}). " + why
          + ("It beats the current best by more than one std: Role 1 should "
             "consider it for 07_final_model.py (issue #19)."
             if beats_best else
             "It does not beat the current best by more than one std, so "
             "07_final_model.py stays unchanged (issue #19 rule)."))


if __name__ == "__main__":
    main()
