# Role 2 – Data + baseline model (Champ)
Last updated: 2026-10-28 · Status: In progress

## 1. Slide summary
3–5 bullets a slide could show as is, each one finding or result.

## 2. What I did
Numbered steps in the order done, each with the script/function that does it.

## 3. Key decisions and why
| Decision | Why | Evidence (number, chart or source) |
|---|---|---|

## 4. Results
Numbers copied from the script output or results/results.csv (mean ± std where available).

## 5. Charts for the slides
| File in plots/ | One-line caption (the finding) | Suggested slide |
|---|---|---|

## 6. Bias and leakage checks
What I did to keep the result fair (e.g. GroupKFold, fitting inside Pipeline, same seed).

## 7. Problems and how I fixed them
Short bullets: issue → fix. These make good "lessons learned" slides.

## 8. Likely Q&A questions
| Question | Short answer |
|---|---|

## 9. Open items / handover
What's not done yet and who needs what from me.

- [ ] `01_explore.py` – each section prints a plain-English conclusion and saves a chart:
  - [ ] 1. Overview: rows, columns, types
  - [ ] 2. Target balance → which metric is fair
  - [ ] 3. Missing values per column in train and test; is missingness biased?
  - [ ] 4. Data quality: duplicates, negative spending, age range, CryoSleep ↔ spending
  - [ ] 5. Train vs test: category shares and numeric distributions
  - [ ] 6. Feature vs target: rate per category (incl. Deck/Num/Side), spending and age patterns
  - [ ] 7. ID-like columns: `Name`, `Cabin` uniqueness; travel groups (size vs target, train/test overlap → confirms GroupKFold)
  - [ ] 8. Summary of decisions → `results/eda_findings.txt`
- [ ] `src/features.py`: `Deck`, `CabinNum`, `Side`, `GroupSize`, `TotalSpend`, log-spending; rule-based filling if the EDA supports it; drop ID-like columns; `build_preprocessor()`
- [ ] `02_prepare.py`: demonstrate and check the feature and preprocessing steps
- [ ] `03_baseline.py`: `DummyClassifier` + Logistic Regression, both logged in `results.csv`
- [ ] `features.py` merged and reused by Roles 3 and 4
- [ ] Keep this report up to date: EDA findings with charts, every cleaning/feature decision with evidence, dropped columns and why, baseline scores
