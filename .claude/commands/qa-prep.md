---
description: (Role 5) Build notes/qa_prep.md from every report's Q&A section plus likely examiner questions
---
Build `notes/qa_prep.md` for the team.

1. Collect section 8 (Likely Q&A questions) from every file in `reports/`.
2. Add the cross-cutting questions examiners usually ask, each with a short answer drawn from our reports and code. For example:
   - Why GroupKFold instead of a random split?
   - How did you prevent data leakage?
   - Why accuracy as the main metric?
   - Why drop Name, Cabin and PassengerId?
   - Why this final model?
   - What would you improve with more time?
3. Group the questions by pipeline stage (Data → EDA → Features → Validation → Models → Results). Next to each, note who built that part, but **everyone** should be able to answer.
4. Mark any question we can't answer yet with ⚠ and the role who should supply the answer.
5. Keep every answer to 2–3 sentences, in plain English.
