"""Compare every model logged in results/results.csv (Role 5 with Role 1).

Reads only the numbers evaluate() logged, so every score on the slides is
traceable to results.csv (rule 3). Nothing is retrained or recomputed here.

Outputs:
    results/model_comparison.csv        table sorted by mean CV accuracy
    plots/12_model_accuracy.png         accuracy ± std per model vs the Dummy floor
    plots/13_precision_recall_f1.png    precision / recall / F1 for the top 3

Self-test: python 06_compare.py --self-test
"""
import sys

import matplotlib.pyplot as plt
import pandas as pd

from src.config import PLOT_DIR, RESULTS_DIR

NEUTRAL, INK = "#8a8984", "#3d3d3a"  # team neutral; darker ink marks the winner
LABELS = {"dummy_most_frequent": "Dummy (most frequent)",
          "logistic_regression": "Logistic Regression",
          "random_forest_default": "Random Forest (default)",
          "random_forest": "Random Forest (tuned)",
          "lightgbm_default": "LightGBM (default)",
          "lightgbm": "LightGBM (tuned)"}
METRICS = {"precision_mean": "Precision", "recall_mean": "Recall", "f1_mean": "F1"}


def latest_per_model(results):
    """One row per model name: the last one logged (the most recent run)."""
    dupes = results["model"].duplicated(keep="last")
    if dupes.any():
        print(f"Note: {sorted(set(results.loc[dupes, 'model']))} logged more than once; "
              "keeping the latest row for each.")
    return results[~dupes].reset_index(drop=True)


def comparison_table(results):
    """Slide-ready table, best mean accuracy first; numbers copied, not recomputed."""
    t = results.sort_values("accuracy_mean", ascending=False, kind="stable").reset_index(drop=True)
    t["accuracy"] = [f"{m:.4f} ± {s:.4f}" for m, s in zip(t["accuracy_mean"], t["accuracy_std"])]
    return t[["model", "owner", "accuracy", "accuracy_mean", "accuracy_std",
              "precision_mean", "recall_mean", "f1_mean", "roc_auc_mean", "fit_time_s"]]


def _family(model):
    """'lightgbm_default' and 'lightgbm' are the same algorithm, tuned or not."""
    return model.removesuffix("_default")


def _within_std(a, b):
    """True when the gap is smaller than the fold-to-fold spread (could flip)."""
    return a["accuracy_mean"] - b["accuracy_mean"] <= max(a["accuracy_std"], b["accuracy_std"])


def _verdict(table):
    """The winner and its closest rival from a *different* algorithm.

    Comparing against its own untuned version would be the wrong question:
    the choice the slides defend is which algorithm, then whether tuning helped.
    """
    models = table[~table["model"].str.startswith("dummy")]
    best = models.iloc[0]
    rival = models[models["model"].map(_family) != _family(best["model"])].iloc[0]
    return best, rival, best["accuracy_mean"] - rival["accuracy_mean"], _within_std(best, rival)


def conclusion(table):
    """Plain-English verdict: winner vs other algorithm, tuning gain, Dummy floor."""
    best, rival, gap, overlap = _verdict(table)
    text = (f"{_label(best['model'])} has the highest mean CV accuracy, "
            f"{best['accuracy_mean']:.4f} ± {best['accuracy_std']:.4f}, "
            f"{gap * 100:.1f} points ahead of {_label(rival['model'])} "
            f"({rival['accuracy_mean']:.4f} ± {rival['accuracy_std']:.4f}). ")
    text += ("The gap is within one standard deviation, so the difference may not be "
             "meaningful. " if overlap else
             "The gap is larger than one standard deviation, so the lead is clear. ")
    text += (f"It fits in {best['fit_time_s']:.2f} s per fold against "
             f"{rival['fit_time_s']:.2f} s. ")

    default = table[table["model"] == f"{best['model']}_default"]
    if not default.empty:
        d = default.iloc[0]
        text += (f"Tuning adds {(best['accuracy_mean'] - d['accuracy_mean']) * 100:.1f} points "
                 f"over {_label(d['model'])} ({d['accuracy_mean']:.4f} ± {d['accuracy_std']:.4f}), "
                 + ("also within one standard deviation. " if _within_std(best, d)
                    else "more than one standard deviation. "))

    floor = table[table["model"].str.startswith("dummy")]
    others = table[~table["model"].str.startswith("dummy")]
    if not floor.empty and (others["accuracy_mean"] > floor.iloc[0]["accuracy_mean"]).all():
        text += (f"Every model beats the Dummy floor of {floor.iloc[0]['accuracy_mean']:.4f}, "
                 "so they all learned real signal.")
    return text.strip()


def _label(model):
    return LABELS.get(model, model)


def plot_accuracy(table, path):
    """Dot + ± std bar per model, Dummy as a dashed floor, winner in dark ink."""
    models = table[~table["model"].str.startswith("dummy")].iloc[::-1]  # best on top
    floor = table[table["model"].str.startswith("dummy")]
    colors = [INK if m == table.iloc[0]["model"] else NEUTRAL for m in models["model"]]
    y = range(len(models))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for yi, m, s, c in zip(y, models["accuracy_mean"], models["accuracy_std"], colors):
        ax.errorbar(m, yi, xerr=s, fmt="none", ecolor=c, elinewidth=2, capsize=4)
    ax.scatter(models["accuracy_mean"], y, s=60, color=colors, zorder=3)
    for yi, (m, s) in zip(y, zip(models["accuracy_mean"], models["accuracy_std"])):
        ax.annotate(f"{m:.4f}", (m + s, yi), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK)
    if not floor.empty:
        d = floor.iloc[0]["accuracy_mean"]
        ax.axvline(d, color=NEUTRAL, linestyle="--", linewidth=1.5)
        ax.annotate(f"Dummy floor\n{d:.4f}", (d, 0), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK)

    best, rival, gap, overlap = _verdict(table)
    ax.set_title(f"{_label(best['model'])} is most accurate: {gap * 100:.1f} points ahead of "
                 f"{_label(rival['model'])}\n"
                 + ("a gap within one std, so not clearly real" if overlap else "a clear lead"),
                 fontsize=11)
    _style(ax, [_label(m) for m in models["model"]], "Mean CV accuracy (± 1 std over 5 GroupKFold folds)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_prf(table, path, top=3):
    """Three small panels (precision | recall | F1) for the top models."""
    models = table[~table["model"].str.startswith("dummy")].head(top).iloc[::-1]
    colors = [INK if m == table.iloc[0]["model"] else NEUTRAL for m in models["model"]]
    lo = models[list(METRICS)].min().min() - 0.02
    hi = models[list(METRICS)].max().max() + 0.02

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), sharey=True)
    for ax, (col, name) in zip(axes, METRICS.items()):
        ax.scatter(models[col], range(len(models)), s=60, color=colors, zorder=3)
        for yi, v in enumerate(models[col]):
            ax.annotate(f"{v:.4f}", (v, yi), xytext=(0, -14), textcoords="offset points",
                        ha="center", fontsize=9, color=INK)
        ax.set_ylim(-0.6, len(models) - 0.5)
        ax.set_xlim(lo, hi)
        ax.set_title(name, fontsize=10)
        _style(ax, [_label(m) for m in models["model"]], "")
    # Recall vs precision tells the slide which way each model errs.
    best = models.loc[models["f1_mean"].idxmax()]  # computed, not assumed
    leans = "finding transported passengers (recall)" if best["recall_mean"] > best["precision_mean"] \
        else "being right when it says transported (precision)"
    fig.suptitle(f"{_label(best['model'])} leads on F1 and is slightly better at {leans}",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _style(ax, labels, xlabel):
    """Recessive axes: light x-grid only, no box."""
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel(xlabel, fontsize=9, color=INK)
    ax.grid(axis="x", color="#e5e4df", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK, labelsize=9, length=0)


def main():
    table = comparison_table(latest_per_model(pd.read_csv(RESULTS_DIR / "results.csv")))
    out = table.drop(columns=["accuracy_mean", "accuracy_std"]).round(4)
    out.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    print(out.to_string(index=False))

    PLOT_DIR.mkdir(exist_ok=True)
    plot_accuracy(table, PLOT_DIR / "12_model_accuracy.png")
    plot_prf(table, PLOT_DIR / "13_precision_recall_f1.png")
    print(f"\nSaved results/model_comparison.csv, plots/12_model_accuracy.png "
          f"and plots/13_precision_recall_f1.png.\n\nConclusion: {conclusion(table)}")


def _self_test():
    """Check de-duplication, sorting and the conclusion on made-up rows."""
    import pandas as pd

    rows = pd.DataFrame({
        "model": ["dummy", "rf", "lgbm", "rf"],
        "owner": ["R2", "R3", "R4", "R3"],
        "accuracy_mean": [0.50, 0.70, 0.81, 0.80],
        "accuracy_std": [0.01, 0.02, 0.02, 0.02],
        "precision_mean": [0.4, 0.8, 0.8, 0.8], "recall_mean": [0.8, 0.8, 0.8, 0.8],
        "f1_mean": [0.5, 0.8, 0.8, 0.8], "roc_auc_mean": [0.5, 0.9, 0.9, 0.9],
        "fit_time_s": [0.0, 2.0, 0.2, 2.0], "params": ["{}"] * 4,
        "date": ["2026-10-30"] * 4,
    })
    latest = latest_per_model(rows)
    # rf was logged twice: the later row (0.80) replaces the earlier (0.70).
    assert len(latest) == 3
    assert latest.loc[latest["model"] == "rf", "accuracy_mean"].item() == 0.80

    table = comparison_table(latest)
    assert list(table["model"]) == ["lgbm", "rf", "dummy"]
    assert table.loc[0, "accuracy"] == "0.8100 ± 0.0200"

    # 0.81 vs 0.80 with std 0.02 -> overlap; 0.81 vs 0.70 -> clear win.
    assert "may not be meaningful" in conclusion(table)
    clear = comparison_table(latest_per_model(rows.iloc[:3]))
    assert "may not be meaningful" not in conclusion(clear)

    # The winner is compared with the best *other* family, not its own
    # untuned version; the tuning gain is reported separately.
    fam = comparison_table(pd.concat([rows, rows.iloc[[2]].assign(
        model="lgbm_default", accuracy_mean=0.805)], ignore_index=True).pipe(latest_per_model))
    text = conclusion(fam)
    assert "ahead of rf " in text and "over lgbm_default" in text

    # "Every model beats the Dummy" is only claimed when it is true.
    assert "Every model beats" in conclusion(table)
    worse = comparison_table(latest_per_model(rows.assign(
        accuracy_mean=[0.50, 0.70, 0.81, 0.45])))
    assert "Every model beats" not in conclusion(worse)
    print("Self-test passed: latest_per_model(), comparison_table() and conclusion() work.")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
    else:
        main()
