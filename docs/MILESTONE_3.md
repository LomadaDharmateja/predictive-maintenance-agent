# MILESTONE 3 — Baselines, models, honest evaluation

Read `docs/DATA.md`, `docs/FEATURES.md` and `docs/MILESTONE_2.md` first.

This milestone produces the first performance numbers in the project. Every one of
them must be defensible in a technical interview, which means the constraints below
matter more than the scores.

**Scope: no agent, no API, no UI.** Models and evaluation only.

---

## 0. The rule that governs everything

> **The test split is evaluated exactly once, at the very end, after every modelling
> decision is final.**

Model selection, hyperparameter tuning, threshold selection, and feature decisions
all happen on the training split and the validation split. Not the test split.

Enforce it structurally:

- `src/eval/test_evaluation.py` is the only module permitted to load
  `features_test.parquet`. A test asserts by source inspection that no other module
  in `src/` reads it.
- That script writes its output to `docs/EVALUATION.md` and records a run count in
  `data/generated/build_manifest.json`. If it has run more than once with different
  model artefacts, say so in the report rather than hiding it.

If you find yourself wanting to adjust something after seeing test numbers, the
honest move is to record what you wanted to change and why, and leave the number
alone.

---

## 1. Baselines first

No model is reported before these three. A gradient boosting score means nothing
without the floor it clears.

1. **Majority class.** Always predict negative. Report its precision, recall and
   PR-AUC per component. Recall will be 0 and precision undefined — report that
   plainly, and report the accuracy it achieves (99.9%+) once, in a sentence
   explaining why accuracy is never mentioned again.
2. **Single-rule baseline.** Predict positive if the machine had any error in the
   preceding 24 hours. This is what a maintenance team could do with a spreadsheet.
3. **Logistic regression** on the full feature set, with standardisation.

Any model that does not clearly beat baseline 2 on PR-AUC has not earned its
complexity.

---

## 2. Models

Four independent binary classifiers, one per component, as `docs/DATA.md` specifies.

- Logistic regression (baseline 3 above)
- Gradient boosting — LightGBM or XGBoost, your choice, stated with a reason

Requirements:

- Class imbalance handled explicitly. State the approach (class weights, scale_pos_
  weight, or none) and why. Do not resample without justifying it; resampling
  distorts calibration, which section 4 depends on.
- Every hyperparameter search uses **rolling-origin cross-validation** on the
  training split — expanding window, never shuffled, never k-fold. Report the fold
  boundaries.
- All runs tracked in MLflow: parameters, metrics, the feature list, the data
  content hash from `build_manifest.json`, and the git commit. A run that cannot be
  traced back to an exact dataset state is not a result.

---

## 3. Metrics

Per component, on validation, and once at the end on test:

- Precision, recall, F1 at the chosen threshold
- **PR-AUC** as the primary comparison metric, with the positive rate marked on
  every PR curve as the no-skill line
- Full confusion matrix in counts, not percentages
- The majority baseline alongside every table

**Accuracy is not reported.** Neither is ROC-AUC as a headline: with positive rates
between 0.36% and 0.71%, ROC-AUC is dominated by true negatives and will look
excellent regardless of whether the model is useful. It may appear in an appendix
with that caveat attached.

### Uncertainty

The test period contains roughly 127 failure events, spread across four components —
so per-component recall rests on tens of positives, not hundreds.

Report **bootstrap 95% confidence intervals** on precision and recall, resampling at
the level of failure events rather than rows. A point estimate of "recall 0.62" from
32 positives is not a result; "0.62 (95% CI 0.45–0.78)" is.

---

## 4. Calibration

The agent in a later milestone will act on these probabilities, so they must mean
what they say.

- Reliability curve per component, on validation
- Brier score, against the base-rate reference
- If poorly calibrated, apply isotonic or Platt scaling **fitted on validation only**
  and report before and after

---

## 5. Thresholds from the cost assumption

`docs/DATA.md` records a stated cost ratio for a missed failure versus a false alarm.

- Build the expected-cost curve across thresholds **on validation** for each
  component.
- Select the cost-minimising threshold per component. They will differ; that is
  expected and worth a sentence.
- Report a sensitivity table: the chosen threshold and resulting recall at cost
  ratios of 3:1, 10:1 and 30:1. This shows the conclusion's dependence on an
  assumption you cannot measure.
- Record each chosen threshold as a named constant with the ratio it came from.

> v1 of this project used hardcoded 0.5 and 0.8 bands with no stated rationale. The
> point of this section is that every threshold traces to an assumption written down
> somewhere.

---

## 6. Interpretability

- **Permutation importance on validation**, not split-gain importance. Gain
  importance is biased toward high-cardinality features and is not a claim about
  predictive contribution.
- Report the top ten features per component with confidence intervals from repeated
  permutation.
- Sanity check and state the outcome: do the maintenance-recency features dominate?
  If they do, verify against the Milestone 2 leakage tests and say explicitly that
  the guards still hold.

---

## 7. The leakage case study

Write `docs/leakage-case-study.md` now — you finally have the honest pipeline to
contrast against.

Using the archived v1 code in `archive/v1-app/`:

- What `db_setup.py` did: duplicated 10,000 AI4I rows five times
- What a naive random split then reports: F1 0.964
- What honest 5-fold CV on the 10,000 source rows reports: F1 0.736, recall 0.646
- The 23-point gap, and the mechanism that produced it
- The majority baseline on AI4I: 96.61% accuracy, and why that made the inflated
  number look plausible
- How the current pipeline makes this structurally impossible: natural-key primary
  keys that abort on duplication, temporal splits, the embargo, and the leakage
  test suite

Reproduce the numbers from the archived code rather than quoting them from an
earlier document. If they differ, the earlier document was wrong and this one is
right.

---

## 8. Outputs

- `docs/EVALUATION.md` — every table above, plus a short "what this model cannot do"
  section covering the simulated-data caveat, the 18 anomalous training failures,
  and the confidence intervals
- `docs/leakage-case-study.md`
- Model artefacts and MLflow runs, with the data content hash recorded
- PR curves and reliability curves as committed images

---

## Acceptance

- `make train` and `make evaluate` run deterministically from a clean state
- `pytest` passes, including the test asserting single-consumer access to the test
  split
- Every table in `docs/EVALUATION.md` shows the majority baseline alongside
- No accuracy figure appears outside the one explanatory sentence in section 1
- Every threshold traces to a stated cost ratio
- The test split was evaluated once

Then stop and summarise. Do not build the agent.
