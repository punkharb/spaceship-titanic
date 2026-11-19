# Role 2 – Data + baseline model (Champ)
Last updated: 2026-11-19 · Status: Done

## 1. Slide summary
- Balanced target: **50.4%** transported (4,378 vs 4,315), so accuracy is a fair main metric.
- **CryoSleep** is the strongest single signal: **82%** of sleepers were transported vs **33%** of awake passengers. Deck (38 pp spread) and HomePlanet (23 pp) come next.
- Two exact rules found in train: **nobody in CryoSleep spends (0 of 3,037)** and **no child under 13 spends (0 of 806)**. Test confirms both (0 of 1,544; 0 of 351). They fill 525 of 1,160 gaps in train with no learned statistics.
- **0 travel groups** appear in both train.csv and test.csv, so we validate with GroupKFold by travel group, the same way the test set is split.
- Baseline to beat: Logistic Regression **0.7741 ± 0.0071** accuracy (ROC-AUC 0.8525), vs a **0.4963** dummy floor.

## 2. What I did
1. EDA in 8 sections, each printing a conclusion and saving a chart (`01_explore.py`). The conclusions are saved in `results/eda_findings.txt`.
2. Feature creation in `add_features()` (`src/features.py`): split Cabin into Deck/CabinNum/Side, add GroupSize, apply the rule-based fills, add TotalSpend, and log1p the spending columns.
3. Preprocessing in `build_preprocessor()` (`src/features.py`): median imputation + StandardScaler for numeric columns, most-frequent imputation + OneHotEncoder(handle_unknown="ignore") for categorical columns. Returned unfitted.
4. Checks in `02_prepare.py`: before/after columns, the gaps each rule fills, every column accounted for, add_features() identical per fold vs the whole file, and no missing values after the preprocessor.
5. Baselines in `03_baseline.py`: DummyClassifier(most_frequent) and Logistic Regression, both logged with `evaluate()`, plus a coefficient chart.
6. **Next phase, issue #17 (group-aware imputation, with Role 1):** `02b_feature_experiments.py` prints how often group mates agree, then scores each candidate change with the final model on the same 5 shared folds, fold by fold, plus two untuned models as a check (runs in ~17 s). Result: dropped; `src/features.py` unchanged.
7. **Next phase, issue #18 (richer features):** added six candidates to `02b_feature_experiments.py`, two per idea: family size from surnames (counted in the file, and counted in the travel group), cabin-number regions (fixed blocks of 300, and 7 learned quantile bins inside the Pipeline), and spending ratios (5 shares of TotalSpend, and one LuxuryShare). Each is scored on its own against the current features on the same 5 folds. Result: **none kept**; `src/features.py` unchanged, so no logged score changes.

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
| Richer features (#18) not adopted: none of the six | With the final model, five lower accuracy (−0.13 to −0.35 pts) and the best, LuxuryShare, gains +0.14 pts where the rule needs > 0.33 pts | Table in section 4 |
| Two versions of each #18 idea | Each idea has an obvious weakness that a second version removes: the surname count depends on file size (mean 5.43 in train vs 3.56 in test), fixed bin edges are arbitrary, and five shares add five correlated columns | Printed by `python 02b_feature_experiments.py` |
| LuxuryShare not adopted although it adds +2.17 pts to Logistic Regression | The rule was fixed before the test: the final model decides. Shares are a division of two columns, which a linear model cannot build itself but trees can approximate with splits. Adding it would change the features of every model for a gain the final model does not show | Table in section 4 |
| #18 candidates live in `02b_feature_experiments.py`, not `src/features.py` | An unproven feature should not touch the shared contract; nothing in 03–07 had to be re-run | — |

## 4. Results
From `results/results.csv` (5-fold GroupKFold, `evaluate()`):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Train acc. | Fit time (s) |
|---|---|---|---|---|---|---|---|
| dummy_most_frequent | 0.4963 ± 0.0141 | 0.3999 | 0.8000 | 0.5332 | 0.5000 | 0.5037 | 0.0025 |
| logistic_regression | 0.7741 ± 0.0071 | 0.7668 | 0.7922 | 0.7792 | 0.8525 | 0.7758 | 0.0448 |

Fit time changes slightly with each run; the values above are the ones in the committed `results.csv`. All the scores are identical on every run.

- The dummy scores just below 50%. The training majority flips between folds, so it predicted "transported" for everyone in 4 of 5 folds and "not transported" in the fifth.
- Logistic Regression out-of-fold confusion matrix (rows = true False/True, columns = predicted False/True): `[[3261, 1054], [910, 3468]]`.
- Train vs validation gap for Logistic Regression is **0.2 points** (0.7758 vs 0.7741), so it does not overfit.
- After preprocessing there are **29 model inputs** (9 numeric + 20 one-hot) and **0** missing values in validation and test.
- Train vs test: category shares differ by at most **1.7 pp**, and every numeric column has a standardised mean difference below **0.02**.

**Issue #17: group-aware imputation** (`python 02b_feature_experiments.py`, accuracy on the 5 shared folds):

| Model | Current features | + group fill HomePlanet, Side | + group fill HomePlanet, Side, Deck |
|---|---|---|---|
| tuned LightGBM (decides) | 0.8142 ± 0.0112 | 0.8087 (−0.55 pts, better on 1/5 folds) | 0.8123 (−0.20 pts, 1/5) |
| default LightGBM (check) | 0.8106 ± 0.0121 | 0.8108 (+0.01 pts, 2/5) | 0.8089 (−0.17 pts, 1/5) |
| Logistic Regression (check) | 0.7741 ± 0.0071 | 0.7755 (+0.14 pts, 4/5) | 0.7738 (−0.02 pts, 2/5) |

- Fillable gaps: in train, 90 HomePlanet and 100 Side values (and 100 Deck) could be filled from group mates, out of 201 / 199 / 199 missing.
- Scale of the change: 190 values in train (90 HomePlanet + 100 Side) out of 8,693 passengers, so it can only move a small number of predictions.

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
- Issue #18: the keep rule and the deciding model were fixed before any candidate was scored, and all six results are reported, not only the best one. Testing six candidates makes a lucky pass more likely, so a marginal pass would have needed a second look; none passed.

## 7. Problems and how I fixed them
- The dummy scored 49.6%, not the expected 50.4%. The majority class flips in one fold. I explained this in the output and silenced the expected precision 0/0 warning for the dummy only.
- My first claim that "group members share their outcome" was weak: 44% of groups all agree vs 41% by chance. I reworded it: the case for GroupKFold rests on shared features and on test groups being unseen, not on shared outcomes.
- The coefficient chart's first title named CryoSleep, which isn't in the top 3. I retitled it from the actual coefficients.
- Review notes → fixes: the EDA now derives the fill rules from train alone and uses test only as confirmation. It imports `SPEND_COLUMNS`/`CHILD_AGE` from `features.py` so the evidence and the features can't drift apart. Cabin splitting no longer crashes on a batch with no Cabin values. Charts comparing train vs test now use neutral colours, keeping blue/orange for the target.
- Role 5 found that chart 03 showed about 23% of sleepers spending, contradicting its title. The cause was `pd.concat` of train and test without `ignore_index=True`: both files number their rows from 0, so looking up the sleepers' rows by label also picked up the other file's rows. The fix was `ignore_index=True` in both concats in `01_explore.py`. The chart now shows 0.0% vs 90.8%. Every number in `eda_findings.txt` was already correct and is unchanged. Lesson: open every chart, not just a sample, and check it against the printed numbers.
- Issue #18: the feature as first worded ("surname count within the same file") gives different numbers for the same family in train and test (mean 5.43 vs 3.56). I added a second version counted inside the travel group, which is stable (1.79 vs 1.75), and tested both.
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
| How much better are the tree models than your baseline? | Tuned LightGBM is 0.8142 and tuned Random Forest 0.8060, vs 0.7741 for Logistic Regression: +4.0 and +3.2 points, more than the fold-to-fold std. Trees can combine features (interactions) and bend at thresholds, which a linear model can't. |
| Why is CryoSleep not the biggest coefficient? | Sleepers never spend, so the spending columns already carry most of CryoSleep's information and the model splits the credit. Its transported rate (82% vs 33%) is the clearer measure of its strength. |
| Does the model generalise to test? | Train and test distributions match: at most 1.7 pp difference in category shares, and SMD < 0.02 for numeric columns. |
| Did you try filling gaps from travel-group members? | Yes (issue #17). HomePlanet and Side agree in 100% of groups, so the fill is very likely correct, but it did not improve CV: −0.55 pts for the final model (worse on 4/5 folds) and between −0.17 and +0.14 pts for untuned models. We kept the simpler pipeline. |
| How did you test it without leakage? | The fill sat inside the Pipeline and only used group mates in the same batch; GroupKFold keeps a group in one fold, so nothing crossed folds and the target was never read. |
| You planned richer features. What happened? | We tested six (issue #18): two family-size counts from surnames, two kinds of cabin region, and two kinds of spending ratio. With the final model five lowered accuracy by 0.13–0.35 pts and the best, LuxuryShare, gained +0.14 pts where our rule needed more than 0.33. None was kept. |
| Why did they not help? | The information was already there. Family-in-group equals GroupSize for 87.3% of passengers; trees already split on CabinNum; and trees can approximate a ratio from the spending columns. A new column that repeats old information mostly adds noise. |
| Spending shares gave Logistic Regression +2 points. Why not keep them? | The rule, set before testing, was that the final model decides, and LightGBM gained only +0.14 pts (within noise). A linear model cannot divide two columns itself, so it benefits; trees already can. Logistic Regression would still be about 2 pts behind LightGBM (0.7958 vs 0.8142). |
| Why 2 standard errors and not just "higher is better"? | With 5 folds a gain of a few tenths of a point appears by chance. The same folds are used on both sides, so we look at the per-fold gains and ask whether their mean is clearly above zero. |
| Is a negative result a wasted phase? | No. It shows the 0.8142 is not left on the table by weak features, and the pipeline stays simpler (29 model inputs). |

## 9. Open items / handover
- Issue #17 closed as **not adopted** (evidence above). 
- Issue #18 (richer features) done: six candidates tested, **none adopted** (table in section 4). `src/features.py`, `results/results.csv` and `submission.csv` are unchanged, so nobody has to re-run anything.
- For Role 5: the #18 table is ready for the "what we tried in the next phase" slide.
- `src/features.py` was merged in PR #7 after Role 1 approved it. Roles 3 and 4 reuse it unchanged: `04_random_forest.py` and `05_lightgbm.py` both build `X = add_features(df).drop(columns=[TARGET])` and `Pipeline([("prep", build_preprocessor()), ("model", ...)])`.
- **How the tuned models compare with my baseline** (from `results/results.csv`, same folds):

| Model | Accuracy | vs Logistic Regression |
|---|---|---|
| logistic_regression (baseline) | 0.7741 ± 0.0071 | — |
| random_forest (tuned, Role 3) | 0.8060 ± 0.0131 | +3.2 points |
| lightgbm (tuned, Role 4) | 0.8142 ± 0.0112 | +4.0 points |

- Both tuned models beat the baseline by well over one std, so the gain is real and not fold-to-fold noise. The full comparison chart is Role 5/Role 1's `06_compare.py`.
- **Role 5:** charts `plots/01`–`09` and the Q&A table above are ready for the slides.
