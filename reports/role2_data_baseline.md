# Role 2 – Data + baseline model (Champ)
Last updated: 2026-11-19 · Status: Done

## 1. Slide summary
- Balanced target: **50.4%** transported (4,378 vs 4,315), so accuracy is a fair main metric.
- **CryoSleep** is the strongest single signal: **82%** of sleepers were transported vs **33%** of awake passengers. Deck (38 pp spread) and HomePlanet (23 pp) come next.
- Two exact rules found in train: **nobody in CryoSleep spends (0 of 3,037)** and **no child under 13 spends (0 of 806)**. Test confirms both (0 of 1,544; 0 of 351). They fill 525 of 1,160 gaps in train with no learned statistics.
- **0 travel groups** appear in both train.csv and test.csv, so we validate with GroupKFold by travel group, the same way the test set is split.
- Baseline to beat: Logistic Regression **0.7951 ± 0.0105** accuracy (ROC-AUC 0.8739), vs a **0.4963** dummy floor.
- Next phase: of nine feature ideas tested, two were adopted (**LuxuryShare** and **CabinRegion**). They lift the baseline from 0.7741 to 0.7951 and re-tuned LightGBM by about +0.25 pts.

## 2. What I did
1. EDA in 8 sections, each printing a conclusion and saving a chart (`01_explore.py`). The conclusions are saved in `results/eda_findings.txt`.
2. Feature creation in `add_features()` (`src/features.py`): split Cabin into Deck/CabinNum/Side, add GroupSize, apply the rule-based fills, add TotalSpend, and log1p the spending columns. Since issue #18 it also adds CabinRegion and LuxuryShare.
3. Preprocessing in `build_preprocessor()` (`src/features.py`): median imputation + StandardScaler for numeric columns, most-frequent imputation + OneHotEncoder(handle_unknown="ignore") for categorical columns. Returned unfitted.
4. Checks in `02_prepare.py`: before/after columns, the gaps each rule fills, every column accounted for, add_features() identical per fold vs the whole file, and no missing values after the preprocessor.
5. Baselines in `03_baseline.py`: DummyClassifier(most_frequent) and Logistic Regression, both logged with `evaluate()`, plus a coefficient chart.
6. **Next phase, issue #17 (group-aware imputation, with Role 1):** `02b_feature_experiments.py` prints how often group mates agree, then scores each candidate change with the final model on the same 5 shared folds, fold by fold, plus two untuned models as a check (runs in ~17 s). Result: dropped; `src/features.py` unchanged.
7. **Next phase, issue #18 (richer features):** added six candidates to `02b_feature_experiments.py`, two per idea: family size from surnames (counted in the file, and counted in the travel group), cabin-number regions (fixed blocks of 300, and 7 learned quantile bins inside the Pipeline), and spending ratios (5 shares of TotalSpend, and one LuxuryShare). Each is scored on its own against the current features on the same 5 folds. Result: none passes alone; `src/features.py` unchanged, so no logged score changes.
8. **Issue #18 follow-up:** tested the ideas together (3 combinations), then repeated Role 4's random search for the two most promising feature sets (`python 02b_feature_experiments.py --retune`), and re-checked them on 10 reshuffled group splits. Result: LuxuryShare + CabinRegion gives a small gain of about +0.25 pts that passes the rule narrowly.
9. **Adoption (decided 2026-11-19):** added `LuxuryShare` and `CabinRegion` to `add_features()` and the feature lists in `src/features.py`, added checks for both to `02_prepare.py`, and re-ran `03_baseline.py`. `02b_feature_experiments.py` now compares against "features before #18", so it still reproduces the evidence. Handover for the other roles: `reports/role2_handover_issue18.md`.

## 3. Key decisions and why
| Decision | Why | Evidence (number, chart or source) |
|---|---|---|
| Accuracy as the main metric (F1 and ROC-AUC alongside) | Classes are balanced | 50.4% transported; `plots/01_target_balance.png` |
| GroupKFold by travel group | Test groups are never seen in train; group members share features | 0 groups in both files; 100% of multi-person groups share one HomePlanet; `plots/08_group_size_vs_target.png` |
| Impute rather than drop rows | Only ~2% per column is missing, but across 24.0% of rows | 2.1–2.5% missing per column in train, 1.9–2.5% in test; `plots/02_missing_values.png` |
| Simple imputation inside the Pipeline | Missingness is only weakly tied to the target | Largest gap: RoomService, −4.6 pp |
| Rule: CryoSleep=True or Age<13 → missing spending = 0 | Logically and empirically exact | Train: 0 of 3,037 sleepers and 0 of 806 under-13s spent anything, and the youngest spender is 13. Test confirms (0 of 1,544; 0 of 351). `plots/03_cryosleep_spending.png` (both files: 0 of 4,581) |
| Rule: spending > 0 → CryoSleep = False | Same fact, the other way round | Fills 119 CryoSleep gaps in train and 55 in test |
| Not used: zero spending → CryoSleep=True | The reverse rule isn't true | 9% of awake passengers also spent nothing |
| Split Cabin into Deck, CabinNum, Side | Deck and Side matter, but the raw Cabin is ID-like | Deck spread 38 pp, Side 10 pp; Cabin is 77.5% unique |
| Add GroupSize | Solo travellers differ | Solo 45% transported vs 57% in groups |
| log1p of spending + TotalSpend | Heavy right skew would let a few huge bills dominate the linear model | Skewness 6–13; spenders transported 26–35% vs 59–64% for non-spenders; `plots/06_spending_vs_target.png` |
| Drop PassengerId, Name, Cabin | ID-like: a model could only memorise them | Name is 99.6% unique |
| No filling from other group members in `add_features()` | Uses other rows' data, which rule 5 doesn't allow there; tested inside the Pipeline in #17 and not adopted | Kept add_features() strictly row-wise |
| Logistic, not Linear, Regression | The target is True/False; Logistic outputs a probability, Linear an unbounded number | — |
| Group fill (#17) not adopted | It lowered the final model by 0.55 pts (worse on 4/5 folds) and moved untuned models by between −0.17 and +0.14 pts; the rule needs a gain above 2 standard errors of the per-fold gains | `python 02b_feature_experiments.py` |
| Paired decision rule: mean gain > 2 × standard error of the per-fold gains | Both sides use the same folds, so fold difficulty cancels; the fold-to-fold std of the scores (1.12 pts) would overstate the noise | Docstring of `02b_feature_experiments.py` |
| Only HomePlanet and Side were candidates for group fill (Deck tested as a variant) | Among multi-person groups with ≥ 2 known values, they agree in 100% (train and test); Deck 69.2% / 64.2%, Destination 47.7% / 50.0%, CryoSleep 40.5% / 41.3%, Age 1.5% / 1.6% (train / test) | Printed by `python 02b_feature_experiments.py` |
| Richer features (#18): none of the six adopted on its own | With the final model, five lower accuracy (−0.13 to −0.35 pts) and the best, LuxuryShare, gains +0.14 pts where the rule needs > 0.33 pts | Table in section 4 |
| Two versions of each #18 idea | Each idea has an obvious weakness that a second version removes: the surname count depends on file size (mean 5.43 in train vs 3.56 in test), fixed bin edges are arbitrary, and five shares add five correlated columns | Printed by `python 02b_feature_experiments.py` |
| LuxuryShare not adopted although it adds +2.17 pts to Logistic Regression | The rule was fixed before the test: the final model decides. Shares are a division of two columns, which a linear model cannot build itself but trees can approximate with splits. Adding it would change the features of every model for a gain the final model does not show | Table in section 4 |
| Adopt LuxuryShare + CabinRegion (decided by Champ, Role 2, on 2026-11-19; Role 1 to review) | Passes the keep rule only after re-tuning (+0.253 pts vs 0.248 needed) and adds +2.10 pts to Logistic Regression. Under the rule as first written (saved parameters) it is a drop (−0.10 pts). The gain for the final model is small (about 22 of 8,693 passengers) and within fold noise, which we say openly | Follow-up tables in section 4 |
| LuxuryShare = (RoomService + Spa + VRDeck) / TotalSpend | These three bills are higher for passengers who were not transported, the other two lower. One ratio column carries that | Mean share, transported vs not: RoomService 12% vs 27%, Spa 9% vs 24%, VRDeck 10% vs 23%, FoodCourt 41% vs 15%, ShoppingMall 30% vs 14% |
| CabinRegion = CabinNum in fixed blocks of 300 | A fixed width needs no statistics, so it is allowed in `add_features()`. 300 is a round width picked before any score was seen, and no other width was tried. Learned bins were tested too and were no better. CabinRegion is the weaker of the two: alone it lowers LightGBM (−0.30 pts) and on top of LuxuryShare it adds +0.11 pts after re-tuning | Transported rate by block: 41% to 61% |
| Second look on 10 reshuffled group splits | Nine variants and two re-tunings were tried on the same 5 folds, so a narrow pass could be luck with those folds | `second_look()` in `02b_feature_experiments.py` |
| Candidates stayed in `02b_feature_experiments.py` until one passed | An unproven feature should not touch the shared contract | — |

## 4. Results
From `results/results.csv` (5-fold GroupKFold, `evaluate()`):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Train acc. | Fit time (s) |
|---|---|---|---|---|---|---|---|
| dummy_most_frequent | 0.4963 ± 0.0141 | 0.3999 | 0.8000 | 0.5332 | 0.5000 | 0.5037 | 0.0020 |
| logistic_regression | 0.7951 ± 0.0105 | 0.7727 | 0.8407 | 0.8051 | 0.8739 | 0.7967 | 0.0524 |

Fit time changes slightly with each run; the values above are the ones in the committed `results.csv`. All the scores are identical on every run.

- The dummy scores just below 50%. The training majority flips between folds, so it predicted "transported" for everyone in 4 of 5 folds and "not transported" in the fifth.
- Logistic Regression out-of-fold confusion matrix (rows = true False/True, columns = predicted False/True): `[[3232, 1083], [698, 3680]]`.
- Before issue #18 the Logistic Regression scored 0.7741 ± 0.0071 (ROC-AUC 0.8525). The two new features add **+2.10 points**. Recall rose from 0.7922 to 0.8407.
- Largest coefficients now: CabinRegion_R3 +0.81, Deck_C +0.80, HomePlanet_Europa +0.59 towards transported. LuxuryShare −1.32, Deck_G −0.85, CabinRegion_R1 −0.54 away.
- Train vs validation gap for Logistic Regression is **0.2 points** (0.7967 vs 0.7951), so it does not overfit.
- After preprocessing there are **37 model inputs** (10 numeric + 27 one-hot) and **0** missing values in validation and test.
- Train vs test: category shares differ by at most **1.7 pp**, and every numeric column has a standardised mean difference below **0.02**.

**Issue #17: group-aware imputation** (`python 02b_feature_experiments.py`, accuracy on the 5 shared folds):

| Model | Current features | + group fill HomePlanet, Side | + group fill HomePlanet, Side, Deck |
|---|---|---|---|
| tuned LightGBM (decides) | 0.8142 ± 0.0112 | 0.8087 (−0.55 pts, better on 1/5 folds) | 0.8123 (−0.20 pts, 1/5) |
| default LightGBM (check) | 0.8106 ± 0.0121 | 0.8108 (+0.01 pts, 2/5) | 0.8089 (−0.17 pts, 1/5) |
| Logistic Regression (check) | 0.7741 ± 0.0071 | 0.7755 (+0.14 pts, 4/5) | 0.7738 (−0.02 pts, 2/5) |

- Fillable gaps: in train, 90 HomePlanet and 100 Side values (and 100 Deck) could be filled from group mates, out of 201 / 199 / 199 missing.
- Scale of the change: 190 values in train (90 HomePlanet + 100 Side) out of 8,693 passengers, so it can only move a small number of predictions.

In the experiment tables below, "Current features" means the features **before issue #18**.

**Issue #18: richer features** (`python 02b_feature_experiments.py`; mean accuracy on the 5 shared folds, gain vs current features, folds where it was better):

| Candidate (added alone) | Tuned LightGBM (decides) | Needs > 2 × SE | Verdict | Default LightGBM (check) | Logistic Regression (check) |
|---|---|---|---|---|---|
| Current features | 0.8142 ± 0.0112 | – | – | 0.8106 ± 0.0121 | 0.7741 ± 0.0071 |
| FamilySize (surname count in file) | 0.8115 (−0.28 pts, 0/5) | 0.14 pts | drop | 0.8086 (−0.21, 2/5) | 0.7729 (−0.12, 0/5) |
| FamilyInGroup (surname count in travel group) | 0.8108 (−0.35 pts, 0/5) | 0.30 pts | drop | 0.8108 (+0.01, 2/5) | 0.7738 (−0.02, 1/5) |
| CabinRegion, fixed blocks of 300 | 0.8112 (−0.30 pts, 0/5) | 0.27 pts | drop | 0.8095 (−0.12, 1/5) | 0.7761 (+0.21, 3/5) |
| CabinRegion, 7 learned quantile bins (in Pipeline) | 0.8109 (−0.33 pts, 0/5) | 0.21 pts | drop | 0.8097 (−0.09, 1/5) | 0.7768 (+0.28, 4/5) |
| 5 spending shares of TotalSpend | 0.8130 (−0.13 pts, 2/5) | 0.34 pts | drop | 0.8109 (+0.02, 2/5) | 0.7937 (+1.97, 5/5) |
| LuxuryShare (RoomService + Spa + VRDeck) / TotalSpend | 0.8156 (+0.14 pts, 4/5) | 0.33 pts | drop | 0.8136 (+0.30, 4/5) | 0.7958 (+2.17, 5/5) |

- Family: FamilyInGroup equals GroupSize for **87.3%** of train passengers with a name, and 82.4% of multi-person groups share one surname, so it mostly repeats a feature we already have.
- Family: the in-file surname count is not comparable between files: mean **5.43** in train vs **3.56** in test (max 18 vs 14), because test.csv has half the rows.
- Cabin regions (train): transported rate by block of 300 is R0 54%, R1 41%, R2 54%, R3 61%, R4 42%, R5 41%, R6 44% (n = 3,560 / 1,488 / 976 / 992 / 938 / 460 / 80). The pattern is real, but trees already split on CabinNum directly.
- Spending shares (train, spenders only), transported vs not: FoodCourt 41% vs 15%, ShoppingMall 30% vs 14%, RoomService 12% vs 27%, Spa 9% vs 24%, VRDeck 10% vs 23%.

**Issue #18 follow-up 1: features together** (same run, saved LightGBM parameters):

| Combination | Tuned LightGBM (decides) | Needs > 2 × SE | Verdict | Default LightGBM | Logistic Regression |
|---|---|---|---|---|---|
| LuxuryShare + CabinRegion (fixed 300) | 0.8132 (−0.10 pts, 1/5) | 0.26 pts | drop | +0.25 (4/5) | +2.10 (5/5) |
| LuxuryShare + FamilyInGroup | 0.8143 (+0.01 pts, 2/5) | 0.20 pts | drop | +0.20 (4/5) | +2.22 (5/5) |
| LuxuryShare + CabinRegion + FamilyInGroup | 0.8126 (−0.16 pts, 2/5) | 0.31 pts | drop | +0.20 (3/5) | +2.12 (5/5) |

**Issue #18 follow-up 2: LightGBM re-tuned per feature set** (`python 02b_feature_experiments.py --retune`; Role 4's search space, 30 settings, same seed, shared folds):

| Feature set | Best of 30 | Gain vs current (per fold) | Needs > 2 × SE | Verdict | 10 reshuffled splits |
|---|---|---|---|---|---|
| Current features | 0.8142 ± 0.0112 (same parameters as saved) | – | – | – | 0.8112 on average |
| + LuxuryShare | 0.8156 ± 0.0102 (same parameters as saved) | +0.14 pts (+0.17, −0.35, +0.12, +0.06, +0.69) | 0.33 pts | drop | +0.22 pts, better on 9/10 |
| + LuxuryShare + CabinRegion | 0.8167 ± 0.0102 | +0.25 pts (+0.12, +0.46, −0.17, +0.40, +0.46) | 0.25 pts | **keep (narrowly: 0.253 vs 0.248)** | +0.26 pts, better on 9/10 |

- Best parameters for LuxuryShare + CabinRegion: learning_rate 0.05, n_estimators 500, num_leaves 15, max_depth −1, min_child_samples 10, subsample 1.0, colsample_bytree 0.6.
- Reading: the gain is small but consistent, about **+0.25 pts** (roughly 22 of 8,693 passengers). On the reshuffled splits it ranged from −0.16 to +0.44 pts.
- Caveat: the reshuffled splits reuse the same passengers, so the 10 gains are not independent. They are exploration only; every logged score still uses `get_cv()`.

## 5. Charts for the slides
| File in plots/ | One-line caption (the finding) | Suggested slide |
|---|---|---|
| `01_target_balance.png` | Classes are balanced: 50.4% transported | Data overview |
| `02_missing_values.png` | Every column is ~2% missing, the same in train and test | Data cleaning |
| `03_cryosleep_spending.png` | Nobody in CryoSleep spends money: 0.0% of sleepers vs 90.8% of awake passengers (0 of 4,581) | Data cleaning (rule-based fill) |
| `04_train_vs_test.png` | Train and test look alike, so the model should generalise | EDA / generalisation |
| `05_rate_by_category.png` | CryoSleep, deck and home planet change the transported rate the most | EDA findings |
| `06_spending_vs_target.png` | Passengers who spend are transported far less often | EDA findings |
| `07_age_vs_target.png` | Children are transported more often (under-5s: 77%) | EDA findings |
| `08_group_size_vs_target.png` | Solo travellers are transported less often than groups | EDA findings / GroupKFold |
| `09_logreg_coefficients.png` | Deck C/B and Europa push towards transported; Spa and VRDeck spending away | Baseline results |

## 6. Bias and leakage checks
- `add_features()` is row-wise, plus GroupSize counted within the same file. `02_prepare.py` checks that running it on each training fold alone gives **identical** values to running it on the whole file, for all 5 folds. GroupKFold keeps each group whole, so even GroupSize doesn't change.
- Everything that learns (imputer medians and modes, scaler, one-hot categories) is in `build_preprocessor()`, used inside `Pipeline([("prep", ...), ("model", ...)])`, so it is fitted on the training folds only.
- `02_prepare.py` asserts that `Transported` is not a feature and that every column is either used or deliberately dropped.
- Both baselines are scored with the shared `evaluate()`: the same `get_cv()` folds, `SEED = 42`, and one row each in `results.csv`.
- test.csv is only used in the EDA to compare distributions and group overlap, and in `02_prepare.py` to check that it transforms without gaps. It is never used for training or any choice of parameters.
- The coefficient chart comes from one fit on all of train.csv, for interpretation only. No score comes from it.
- Honest caveat: the feature choices were made from EDA on all of train.csv, target included. This is standard practice and the same for every model, so the comparison stays fair. The absolute CV scores may be very slightly optimistic.
- The independent leakage review (leakage-reviewer agent) found no leakage and no blockers. I fixed its should-fix (a hard-coded fold count, now `N_SPLITS`) and its minor notes.
- Issue #18: no candidate reads the target. Shares and fixed regions are row-wise; FamilyInGroup is counted inside the travel group, which GroupKFold keeps in one fold; the learned bin edges sit inside the Pipeline. FamilySize is the exception: a surname count over the whole file also counts passengers in the validation fold. It uses no target, but it is one more reason not to adopt it.
- Issue #18: the keep rule and the deciding model were fixed before the six single candidates were scored, and all results are reported, not only the best one. **The rule was relaxed afterwards for the adopted pair:** it fails with the saved parameters and passes only in the re-tuning check, which I added after seeing the first results. Testing six candidates makes a lucky pass more likely, so a marginal pass needs a second look. One did pass narrowly in the follow-up (LuxuryShare + CabinRegion after re-tuning), so I repeated the comparison on 10 reshuffled group splits before believing it: +0.26 pts, better on 9/10.
- The re-tuning check gives every feature set its own best-of-30 from the same search, so the comparison is like for like. Best-of-30 on the same folds is slightly optimistic for both sides (nested CV: 0.8109 vs 0.8142), which is why I report the difference, not the new absolute score.
- Honest caveat on the adoption: the two features were chosen after nine variants and two re-tunings (about 11 looks) on the same folds. The gain is 0.253 pts against 0.248 needed, which is about 2.0 standard errors. With only 5 folds a strict 95% test needs about 2.8, so the gain is **not** statistically significant and some of it may be selection luck.
- The 10 reshuffled splits reuse the same 8,693 passengers. They show the result is not luck with one set of folds. They are not an independent replication.
- Nested CV (`05b_nested_cv.py`) nests the tuning but not the feature choice, so it is also slightly optimistic here. The only fully untouched check is the Kaggle test score, which we do not use to choose.
- LuxuryShare treats a missing bill as 0 (518 train and 267 test passengers still have at least one after the rule fills). It uses no other row, so this is a small inaccuracy, not leakage.
- test.csv was read for label-free checks only (for example the FamilySize scale in train vs test). No score or choice of parameters used it.
- Both adopted features are row-wise and use no target. `02_prepare.py` still confirms that `add_features()` gives identical values on each training fold and on the whole file.

## 7. Problems and how I fixed them
- The dummy scored 49.6%, not the expected 50.4%. The majority class flips in one fold. I explained this in the output and silenced the expected precision 0/0 warning for the dummy only.
- My first claim that "group members share their outcome" was weak: 44% of groups all agree vs 41% by chance. I reworded it: the case for GroupKFold rests on shared features and on test groups being unseen, not on shared outcomes.
- The coefficient chart's first title named CryoSleep, which isn't in the top 3. I retitled it from the actual coefficients.
- Review notes → fixes: the EDA now derives the fill rules from train alone and uses test only as confirmation. It imports `SPEND_COLUMNS`/`CHILD_AGE` from `features.py` so the evidence and the features can't drift apart. Cabin splitting no longer crashes on a batch with no Cabin values. Charts comparing train vs test now use neutral colours, keeping blue/orange for the target.
- Role 5 found that chart 03 showed about 23% of sleepers spending, contradicting its title. The cause was `pd.concat` of train and test without `ignore_index=True`: both files number their rows from 0, so looking up the sleepers' rows by label also picked up the other file's rows. The fix was `ignore_index=True` in both concats in `01_explore.py`. The chart now shows 0.0% vs 90.8%. Every number in `eda_findings.txt` was already correct and is unchanged. Lesson: open every chart, not just a sample, and check it against the printed numbers.
- Issue #18: the feature as first worded ("surname count within the same file") gives different numbers for the same family in train and test (mean 5.43 vs 3.56). I added a second version counted inside the travel group, which is stable (1.79 vs 1.75), and tested both.
- After adoption the coefficient chart's fixed title no longer matched its bars. The title is now built from the coefficients, so it cannot go stale again.
- The packages weren't installed on my machine. I created `.venv` and installed `requirements.txt` (Python 3.14).

## 8. Likely Q&A questions
| Question | Short answer |
|---|---|
| Why GroupKFold and not a random split? | 0 travel groups are in both train and test, so the test set is "unseen groups". Group members share features (100% share a home planet), so a random split would score the model on companions of people it trained on. |
| Why accuracy? | The classes are balanced (50.4%), so accuracy isn't misleading. We report F1 and ROC-AUC too. |
| Why is the dummy below 50%? | It predicts the training folds' majority. That flips in one fold because 50.4% is nearly a tie. |
| Isn't filling CryoSleep/spending before the split leakage? | No. These are logical rules within one row (in train, sleepers and children never spend: 0 of 3,037, 0 of 806). Nothing is learned from other rows or the target. |
| Why the age cutoff 13? | It's where spending starts in train: nobody under 13 spends, and the youngest spender is 13. Test confirms it (0 of 351 under-13s spend). |
| Did the EDA bias the model choice? | The features were chosen on all of train.csv, which is standard and identical for every model, so the comparison is fair. Absolute scores may be very slightly optimistic. |
| Why not fill HomePlanet from group members? | It would use other rows' values, which our leakage rule forbids. We kept add_features() strictly row-wise. |
| Why drop Name and Cabin? | Name is 99.6% unique and Cabin 77.5%. A model could only memorise them. Cabin's useful parts (Deck, Num, Side) are kept. |
| Why log-transform spending? | Skewness is 6–13. log1p stops a few huge bills dominating the linear model. Trees are unaffected. |
| Why Logistic and not Linear Regression? | The target is True/False. Logistic gives a probability between 0 and 1, while Linear predicts any number. |
| Is GroupSize leakage? | No. It's counted within the same file, and since GroupKFold keeps groups whole it's identical per fold (checked in 02_prepare.py). |
| How much better are the tree models than your baseline? | Before issue #18: tuned LightGBM 0.8142 and tuned Random Forest 0.8060, vs 0.7741 for Logistic Regression (+4.0 and +3.2 points). With the new features the baseline is 0.7951, so the gap is smaller. Final numbers come after Roles 3 and 4 re-run. Trees can combine features (interactions) and bend at thresholds, which a linear model can't. |
| Why did two features help Logistic Regression by 2 points but LightGBM by only 0.25? | A linear model cannot divide one column by another or bend at a cabin number. Trees can approximate both with splits, so the new columns tell them little they could not already find. |
| Is LuxuryShare leakage? | No. It divides a passenger's own bills by their own total. No other row and no target is used. |
| Why is CryoSleep not the biggest coefficient? | Sleepers never spend, so the spending columns already carry most of CryoSleep's information and the model splits the credit. Its transported rate (82% vs 33%) is the clearer measure of its strength. |
| Does the model generalise to test? | Train and test distributions match: at most 1.7 pp difference in category shares, and SMD < 0.02 for numeric columns. |
| Did you try filling gaps from travel-group members? | Yes (issue #17). HomePlanet and Side agree in 100% of groups, so the fill is very likely correct, but it did not improve CV: −0.55 pts for the final model (worse on 4/5 folds) and between −0.17 and +0.14 pts for untuned models. We kept the simpler pipeline. |
| How did you test it without leakage? | The fill sat inside the Pipeline and only used group mates in the same batch; GroupKFold keeps a group in one fold, so nothing crossed folds and the target was never read. |
| You planned richer features. What happened? | We tested six (issue #18): two family-size counts from surnames, two kinds of cabin region, and two kinds of spending ratio. With the final model five lowered accuracy by 0.13–0.35 pts and the best, LuxuryShare, gained +0.14 pts where our rule needed more than 0.33. Tested together and with LightGBM re-tuned, LuxuryShare + cabin regions gained about +0.25 pts, which passes narrowly and held on 9 of 10 reshuffled splits. |
| Why did they not help? | The information was already there. Family-in-group equals GroupSize for 87.3% of passengers; trees already split on CabinNum; and trees can approximate a ratio from the spending columns. A new column that repeats old information mostly adds noise. |
| Why adopt a gain as small as 0.25 points? | Both features are simple row-wise rules, they help the baseline by 2 points, and the LightGBM gain held on 9 of 10 reshuffled splits. We say openly that it passed only after re-tuning, that it is within fold noise, and that we tried about 11 variants. |
| Why 2 standard errors and not just "higher is better"? | With 5 folds a gain of a few tenths of a point appears by chance. The same folds are used on both sides, so we look at the per-fold gains and ask whether their mean is clearly above zero. |
| Is a near-zero result a wasted phase? | No. It shows little accuracy is left on the table by the features: the best we found is about +0.25 pts, roughly 22 passengers in 8,693. |
| Why re-tune before judging a feature? | The saved parameters were chosen with the current features, so they can undersell a new one. We gave each feature set the same 30-setting search and compared best with best. |
| Why reshuffled splits if everyone must use the same folds? | The shared folds are for every logged score. The reshuffles are only a check that a narrow pass was not luck with those five folds; groups are still never split. |

## 9. Open items / handover
- Issue #17 closed as **not adopted** (evidence above). 
- Issue #18 (richer features): six candidates and three combinations tested (section 4). **LuxuryShare + CabinRegion adopted** into `src/features.py`.
- **Action for Roles 1, 3, 4 and 5:** `src/features.py` changed, so 04 to 07 must be re-run. Steps per role are in `reports/role2_handover_issue18.md`. Until then the Random Forest and LightGBM rows in `results.csv` and `submission.csv` still come from the old features.
- For Role 5: the #18 table is ready for the "what we tried in the next phase" slide.
- `src/features.py` was merged in PR #7 after Role 1 approved it. Roles 3 and 4 reuse it without editing it: `04_random_forest.py` and `05_lightgbm.py` both build `X = add_features(df).drop(columns=[TARGET])` and `Pipeline([("prep", build_preprocessor()), ("model", ...)])`.
- **How the tuned models compare with my baseline** (from `results/results.csv`, same folds):

| Model | Accuracy | vs Logistic Regression |
|---|---|---|
| logistic_regression (baseline, new features) | 0.7951 ± 0.0105 | — |
| random_forest (tuned, Role 3, old features) | 0.8060 ± 0.0131 | to be updated after the re-run |
| lightgbm (tuned, Role 4, old features) | 0.8142 ± 0.0112 | to be updated after the re-run |

- Before issue #18 both tuned models beat the baseline (0.7741) by well over one std. The full comparison chart is Role 5/Role 1's `06_compare.py`.
- **Role 5:** charts `plots/01`–`09` and the Q&A table above are ready for the slides.
