"""Exploratory data analysis of Spaceship Titanic (Role 2).

Eight sections, each printing a plain-English conclusion ("-> ...") and
saving a chart to plots/. The conclusions are also written to
results/eda_findings.txt, which is the evidence behind every cleaning and
feature decision in src/features.py.

Run from the project folder:  python 01_explore.py
Read-only: train.csv and test.csv are never modified.
"""
import matplotlib
matplotlib.use("Agg")  # write PNG files without needing a screen
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import ID_COL, PLOT_DIR, RESULTS_DIR, SEED, TARGET, TEST_PATH, TRAIN_PATH

# Team colours (CLAUDE.md rule 8), used the same way in every chart.
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8984"  # transported / not / neutral
SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
CATEGORY_COLS = ["HomePlanet", "CryoSleep", "Destination", "VIP"]
MIN_GROUP = 30  # categories smaller than this are too noisy to rank

findings = []  # every conclusion, in order, for eda_findings.txt


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")
    findings.append(f"\n## {title}")


def conclude(text):
    """Print a conclusion and keep it for the findings file."""
    print(f"-> {text}")
    findings.append(f"-> {text}")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(PLOT_DIR / name, dpi=120)
    plt.close(fig)
    print(f"   chart: plots/{name}")


def split_cabin(df):
    """Cabin 'Deck/Num/Side' -> three columns (EDA copy; features.py does the same)."""
    parts = df["Cabin"].str.split("/", expand=True)
    return df.assign(Deck=parts[0], CabinNum=pd.to_numeric(parts[1]), Side=parts[2])


def rate_table(df, col):
    """Transported rate and size per category of one column (missing excluded)."""
    t = df.groupby(col, observed=True)[TARGET].agg(rate="mean", n="size")
    return t.sort_values("rate", ascending=False)


# ---------------------------------------------------------------------------
def overview(train, test):
    section("1. Overview")
    print(f"train: {train.shape[0]:,} rows x {train.shape[1]} columns")
    print(f"test:  {test.shape[0]:,} rows x {test.shape[1]} columns")
    print(train.dtypes.to_string())
    num = train.select_dtypes("number").columns.tolist()
    conclude(f"train has {len(train):,} rows and test {len(test):,}; the only extra column "
             f"in train is {sorted(set(train) - set(test))}. {len(num)} columns are numeric "
             f"({', '.join(num)}); the rest are text, True/False or IDs.")


def target_balance(train):
    section("2. Target balance")
    share = train[TARGET].mean()
    counts = train[TARGET].value_counts()
    print(counts.to_string())
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(["Transported", "Not transported"], [counts[True], counts[False]],
           color=[BLUE, ORANGE])
    ax.set_ylabel("Passengers")
    ax.set_title(f"Classes are balanced: {share:.1%} transported")
    save(fig, "01_target_balance.png")
    conclude(f"{share:.1%} of passengers were transported ({counts[True]:,} vs "
             f"{counts[False]:,}). The classes are balanced, so accuracy is a fair main "
             f"metric; a model that always guesses the majority scores only {max(share, 1 - share):.1%}.")


def missing_values(train, test):
    section("3. Missing values")
    cols = [c for c in test.columns if c != ID_COL]
    miss = pd.DataFrame({"train_%": train[cols].isna().mean() * 100,
                         "test_%": test[cols].isna().mean() * 100})
    # Is missingness informative? Compare the transported rate with/without a gap.
    gap = {c: train.loc[train[c].isna(), TARGET].mean() - train.loc[train[c].notna(), TARGET].mean()
           for c in cols if train[c].isna().any()}
    miss["rate_gap_pp"] = pd.Series(gap) * 100
    print(miss.round(2).to_string())

    any_gap = train[cols].isna().any(axis=1).mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(cols))
    ax.bar(x - 0.2, miss["train_%"], 0.4, color=GREY, label="train")
    ax.bar(x + 0.2, miss["test_%"], 0.4, color=BLUE, label="test")
    ax.set_xticks(x, cols, rotation=45, ha="right")
    ax.set_ylabel("% missing")
    ax.set_ylim(0, miss[["train_%", "test_%"]].max().max() * 1.3)  # room for the legend
    ax.legend(ncol=2)
    ax.set_title("Every column is ~2% missing, the same in train and test")
    save(fig, "02_missing_values.png")

    worst = miss["rate_gap_pp"].abs().idxmax()
    conclude(f"Each feature column is {miss['train_%'].min():.1f}-{miss['train_%'].max():.1f}% "
             f"missing in train and {miss['test_%'].min():.1f}-{miss['test_%'].max():.1f}% in test; "
             f"{any_gap:.1%} of train rows have at least one gap, so dropping rows would lose "
             f"too much data. We impute instead.")
    conclude(f"Missingness is only weakly related to the target: the largest gap is {worst} "
             f"({miss.loc[worst, 'rate_gap_pp']:+.1f} percentage points). Simple "
             f"median/most-frequent imputation inside the Pipeline is therefore reasonable.")


def data_quality(train, test):
    section("4. Data quality")
    both = pd.concat([train.drop(columns=TARGET), test])
    dup_rows = train.drop(columns=ID_COL).duplicated().sum()
    dup_ids = both[ID_COL].duplicated().sum()
    neg = (both[SPEND_COLS] < 0).sum().sum()
    print(f"duplicate rows (ignoring ID) in train: {dup_rows}; duplicate IDs across files: {dup_ids}")
    print(f"negative spending values: {neg}")
    print(f"age range: {both['Age'].min():.0f}-{both['Age'].max():.0f}, "
          f"age 0: {(both['Age'] == 0).sum()} passengers")

    spend = both[SPEND_COLS].sum(axis=1, min_count=1)
    cryo = both["CryoSleep"]
    cryo_spend = ((cryo == True) & (spend > 0)).sum()
    cryo_n = (cryo == True).sum()
    awake_zero = (spend.loc[cryo == False] == 0).mean()
    kids = both["Age"] < 13
    kid_spend = (both.loc[kids, SPEND_COLS] > 0).any(axis=1).sum()
    # Rows whose gaps the logical rules below could fill.
    cryo_fill = (cryo.isna() & (spend > 0)).sum()
    spend_fill = ((cryo == True) | kids).loc[both[SPEND_COLS].isna().any(axis=1)].sum()
    print(f"CryoSleep=True passengers: {cryo_n:,}, of whom spent anything: {cryo_spend}")
    print(f"awake passengers who spent nothing: {awake_zero:.1%}")
    print(f"children under 13: {kids.sum():,}, of whom spent anything: {kid_spend}")
    print(f"missing CryoSleep with spending > 0: {cryo_fill}; "
          f"rows with a missing spend column that are asleep or under 13: {spend_fill}")

    fig, ax = plt.subplots(figsize=(6, 3.5))
    groups = {"CryoSleep": both.loc[cryo == True], "Awake": both.loc[cryo == False]}
    shares = [(spend.loc[g.index] > 0).mean() * 100 for g in groups.values()]
    ax.bar(list(groups), shares, color=[GREY, BLUE])
    ax.set_ylabel("% who spent anything")
    ax.set_title(f"Nobody in CryoSleep spends money ({cryo_spend} of {cryo_n:,})")
    save(fig, "03_cryosleep_spending.png")

    conclude(f"No duplicate rows ({dup_rows}) or duplicate IDs ({dup_ids}), and no negative "
             f"spending ({neg}). Ages run {both['Age'].min():.0f}-{both['Age'].max():.0f}; the "
             f"{(both['Age'] == 0).sum()} age-0 passengers are plausible babies, so we keep them.")
    conclude(f"CryoSleep passengers never spend ({cryo_spend} of {cryo_n:,}) and children "
             f"under 13 never spend ({kid_spend} of {kids.sum():,}). These are reliable "
             f"row-wise rules: missing spending for them is 0, and anyone who spent > 0 was not "
             f"in CryoSleep. Across both files this fills {cryo_fill} missing CryoSleep values "
             f"and the gaps in {spend_fill} rows with missing spending. "
             f"The reverse does not hold: {awake_zero:.0%} of awake passengers also spent nothing.")


def train_vs_test(train, test):
    section("5. Train vs test distributions")
    rows = []
    for c in CATEGORY_COLS + ["Deck", "Side"]:
        a = train[c].value_counts(normalize=True)
        b = test[c].value_counts(normalize=True)
        rows.append((c, (a - b).abs().max() * 100))
    cat_diff = pd.DataFrame(rows, columns=["column", "max_share_diff_pp"]).set_index("column")
    print(cat_diff.round(2).to_string())

    # Standardised mean difference: < 0.1 is the usual "negligible" threshold.
    num = ["Age", "CabinNum"] + SPEND_COLS
    smd = ((train[num].mean() - test[num].mean()) / train[num].std()).abs()
    print(smd.round(3).rename("SMD").to_string())

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    bins = np.arange(0, 81, 5)
    axes[0].hist(train["Age"].dropna(), bins, density=True, alpha=0.6, color=GREY, label="train")
    axes[0].hist(test["Age"].dropna(), bins, density=True, alpha=0.6, color=BLUE, label="test")
    axes[0].set_xlabel("Age")
    axes[0].legend()
    shares = pd.DataFrame({"train": train["HomePlanet"].value_counts(normalize=True),
                           "test": test["HomePlanet"].value_counts(normalize=True)}) * 100
    shares.plot.bar(ax=axes[1], color=[GREY, BLUE], rot=0)
    axes[1].set_ylabel("% of passengers")
    fig.suptitle("Train and test look alike (age, home planet), so the model should generalise")
    save(fig, "04_train_vs_test.png")

    conclude(f"Category shares differ by at most {cat_diff['max_share_diff_pp'].max():.1f} "
             f"percentage points between train and test, and every numeric column has a "
             f"standardised mean difference below {smd.max():.2f} (< 0.1 = negligible). "
             f"Test looks like train, so cross-validation scores should carry over.")


def feature_vs_target(train):
    section("6. Features vs target")
    cols = CATEGORY_COLS + ["Deck", "Side"]
    tables = {c: rate_table(train, c) for c in cols}
    for c, t in tables.items():
        print(f"\n{c}:\n{t.round(3).to_string()}")

    # Spread = best minus worst category rate, ignoring tiny categories.
    big = {c: t.loc[t["n"] >= MIN_GROUP, "rate"] for c, t in tables.items()}
    spread = pd.Series({c: (r.max() - r.min()) * 100
                        for c, r in big.items()}).sort_values(ascending=False)
    small = {c: t.index[t["n"] < MIN_GROUP].tolist() for c, t in tables.items()}
    small = {c: v for c, v in small.items() if v}
    print(f"\nrate spread (pp), categories with n >= {MIN_GROUP}:\n{spread.round(1).to_string()}")
    print(f"ignored as too small: {small}")

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharey=True)
    for ax, c in zip(axes.flat, cols):
        t = tables[c][tables[c]["n"] >= MIN_GROUP]
        ax.bar(t.index.astype(str), t["rate"] * 100,
               color=[BLUE if r > 0.5 else ORANGE for r in t["rate"]])
        ax.axhline(train[TARGET].mean() * 100, color=GREY, ls="--", lw=1)
        ax.set_title(c)
        ax.tick_params(axis="x", rotation=30)
    for ax in axes[:, 0]:
        ax.set_ylabel("% transported")
    fig.suptitle("CryoSleep, home planet and deck change the transported rate the most "
                 "(dashed = overall)")
    save(fig, "05_rate_by_category.png")

    # Spending: compare spenders and non-spenders per spend column.
    spend_rate = pd.DataFrame({
        "spent_nothing": [train.loc[train[c] == 0, TARGET].mean() for c in SPEND_COLS],
        "spent_something": [train.loc[train[c] > 0, TARGET].mean() for c in SPEND_COLS],
    }, index=SPEND_COLS) * 100
    total = train[SPEND_COLS].sum(axis=1, min_count=1)
    skew = train[SPEND_COLS].skew()
    print(f"\n% transported by spending:\n{spend_rate.round(1).to_string()}")
    print(f"skewness of spending columns:\n{skew.round(1).to_string()}")

    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(len(SPEND_COLS))
    ax.bar(x - 0.2, spend_rate["spent_nothing"], 0.4, color=GREY, label="spent nothing")
    ax.bar(x + 0.2, spend_rate["spent_something"], 0.4, color=ORANGE, label="spent something")
    ax.set_xticks(x, SPEND_COLS)
    ax.set_ylabel("% transported")
    ax.legend()
    ax.set_title("Passengers who spend are transported far less often")
    save(fig, "06_spending_vs_target.png")

    # Age: rate per 5-year band.
    age_band = pd.cut(train["Age"], bins=np.arange(0, 85, 5), right=False)
    age_rate = train.groupby(age_band, observed=True)[TARGET].agg(rate="mean", n="size")
    age_rate = age_rate[age_rate["n"] >= MIN_GROUP]
    print(f"\n% transported by age band:\n{(age_rate.assign(rate=age_rate.rate * 100)).round(1).to_string()}")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar([f"{b.left}" for b in age_rate.index], age_rate["rate"] * 100,
           color=[BLUE if r > 0.5 else ORANGE for r in age_rate["rate"]])
    ax.axhline(train[TARGET].mean() * 100, color=GREY, ls="--", lw=1)
    ax.set_xlabel("Age band (start)")
    ax.set_ylabel("% transported")
    ax.set_title("Children are transported more often than adults")
    save(fig, "07_age_vs_target.png")

    cryo = tables["CryoSleep"]["rate"]
    zero_rate = train.loc[total == 0, TARGET].mean()
    conclude(f"Strongest categories (best-worst rate spread, ignoring groups < {MIN_GROUP}): "
             + ", ".join(f"{c} {v:.0f} pp" for c, v in spread.items()) + ". "
             f"CryoSleep is the single strongest: {cryo[True]:.0%} transported vs "
             f"{cryo[False]:.0%} awake. Deck and Side are useful, so Cabin should be split.")
    if small:
        conclude(f"Ignored as too small to rank: {small}.")
    conclude(f"Spending is a strong signal: {zero_rate:.0%} of passengers with zero total "
             f"spending were transported. Spending is heavily right-skewed (skewness "
             f"{skew.min():.0f}-{skew.max():.0f}), so log1p-transforming it makes sense, and "
             f"TotalSpend summarises it in one feature.")
    conclude(f"Age matters at the young end: under-5s are transported "
             f"{age_rate['rate'].iloc[0]:.0%} of the time vs {train[TARGET].mean():.0%} overall.")


def id_columns(train, test):
    section("7. ID-like columns and travel groups")
    both = pd.concat([train.drop(columns=TARGET), test])
    name_unique = both["Name"].nunique() / both["Name"].notna().sum()
    cabin_unique = both["Cabin"].nunique() / both["Cabin"].notna().sum()
    print(f"Name: {both['Name'].nunique():,} unique of {both['Name'].notna().sum():,} non-missing")
    print(f"Cabin: {both['Cabin'].nunique():,} unique of {both['Cabin'].notna().sum():,} non-missing")

    group = lambda df: df[ID_COL].str.split("_").str[0]
    g_train, g_test = set(group(train)), set(group(test))
    overlap = g_train & g_test
    print(f"travel groups: train {len(g_train):,}, test {len(g_test):,}, in both: {len(overlap)}")

    size = group(train).map(group(train).value_counts())
    size_rate = train.groupby(size)[TARGET].agg(rate="mean", n="size")
    print(f"\n% transported by group size:\n{size_rate.assign(rate=size_rate.rate * 100).round(1).to_string()}")
    solo = size_rate.loc[1, "rate"]
    big = train.loc[size > 1, TARGET].mean()

    # Do group members share their fate? Agreement within multi-person groups.
    multi = train[size > 1].assign(g=group(train)[size > 1])
    same_fate = multi.groupby("g")[TARGET].nunique().eq(1).mean()
    # Baseline: the same agreement if outcomes were shuffled at random between
    # passengers (same group sizes), so "44%" can be judged against chance.
    shuffled = multi.assign(**{TARGET: multi[TARGET].sample(frac=1, random_state=SEED).values})
    same_fate_chance = shuffled.groupby("g")[TARGET].nunique().eq(1).mean()
    same_planet = multi.groupby("g")["HomePlanet"].nunique().le(1).mean()
    print(f"multi-person groups where everyone has the same outcome: {same_fate:.1%} "
          f"(by chance, with shuffled outcomes: {same_fate_chance:.1%})")
    print(f"multi-person groups with one home planet: {same_planet:.1%}")

    fig, ax = plt.subplots(figsize=(7, 3.5))
    shown = size_rate[size_rate["n"] >= MIN_GROUP]
    ax.bar(shown.index.astype(str), shown["rate"] * 100,
           color=[BLUE if r > 0.5 else ORANGE for r in shown["rate"]])
    ax.axhline(train[TARGET].mean() * 100, color=GREY, ls="--", lw=1)
    ax.set_xlabel("Travel group size")
    ax.set_ylabel("% transported")
    ax.set_title("Solo travellers are transported less often than groups")
    save(fig, "08_group_size_vs_target.png")

    conclude(f"Name is {name_unique:.1%} unique and Cabin {cabin_unique:.1%} unique, so as "
             f"raw values they are IDs that a model could only memorise. Drop Name and the "
             f"raw Cabin (after splitting it into Deck/CabinNum/Side) and PassengerId.")
    conclude(f"Group size matters: solo travellers {solo:.0%} transported vs {big:.0%} in "
             f"groups, so GroupSize is a useful feature.")
    conclude(f"Group members share their features ({same_planet:.0%} of multi-person groups "
             f"have one home planet) but their outcomes only slightly more than chance "
             f"({same_fate:.0%} of groups all agree vs {same_fate_chance:.0%} with shuffled "
             f"outcomes). So the risk of a random split is mild but real: the model would be "
             f"validated on the companions of people it trained on, which never happens for test.")
    conclude(f"{len(overlap)} travel groups appear in both train.csv and test.csv "
             f"({len(g_train):,} vs {len(g_test):,} groups). The train/test split keeps whole groups "
             f"together, so our validation must too: this confirms the GroupKFold-by-group rule.")


def summary():
    section("8. Decisions for src/features.py")
    for d in [
        "Metric: accuracy as the main score (balanced classes), with F1 and ROC-AUC alongside.",
        "Validation: GroupKFold by travel group (no group is in both train and test).",
        "Split Cabin into Deck, CabinNum, Side; then drop Cabin, Name and PassengerId (ID-like).",
        "Add GroupSize (count of the travel group within the same file) and TotalSpend.",
        "log1p-transform the five spending columns and TotalSpend (heavy right skew).",
        "Rule-based filling: CryoSleep=True or Age<13 -> missing spending = 0; "
        "any spending > 0 -> CryoSleep = False. These are logical rules, not learned statistics.",
        "Remaining gaps (~2% per column) are imputed inside the Pipeline (median / most frequent), "
        "so imputation is learned from training folds only.",
        "Do not fill values from other group members (e.g. HomePlanet), even though they "
        "always agree: that uses other rows' data, which shared rule 5 does not allow.",
    ]:
        conclude(d)


def main():
    PLOT_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    train = split_cabin(pd.read_csv(TRAIN_PATH))
    test = split_cabin(pd.read_csv(TEST_PATH))
    # Section 1 describes the raw files, so hide the EDA-only Cabin columns.
    raw = ["Deck", "CabinNum", "Side"]
    overview(train.drop(columns=raw), test.drop(columns=raw))
    target_balance(train)
    missing_values(train.drop(columns=raw), test.drop(columns=raw))
    data_quality(train, test)
    train_vs_test(train, test)
    feature_vs_target(train)
    id_columns(train, test)
    summary()

    out = RESULTS_DIR / "eda_findings.txt"
    out.write_text("EDA findings (generated by 01_explore.py)\n" + "\n".join(findings) + "\n",
                   encoding="utf-8")
    print(f"\nSaved {len(findings)} lines of findings to results/{out.name}")


if __name__ == "__main__":
    main()
