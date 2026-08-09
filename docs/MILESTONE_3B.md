# MILESTONE 3B — Deriving the prediction horizon from the decision it serves

Read `docs/EVALUATION.md` first, then this.

Milestone 3 produced test PR-AUC of 1.000 on comp2 and comp3 with LightGBM. The
controls in that milestone establish this is not leakage and not memorisation: the
simulator injects a matched error-code and sensor-channel signature that fires before
100% of failures, and the pair is near-deterministic 24 hours out.

That is a correct result. It is also a useless one, and this milestone explains why
and fixes it.

---

## 1. The problem with the 24-hour horizon

The system exists so that an agent can decide whether to reserve or order a
replacement part.

`data/generated/parts_inventory.csv` records supplier lead times of **10 to 34 days**,
median 23.

A 24-hour failure warning cannot inform a decision whose action takes three weeks to
execute. The horizon was inherited from the standard framing of this dataset, not
derived from the decision the model serves. Everything downstream — calibration,
thresholds, the cost curve — was rendered vacuous by a horizon chosen before anyone
asked what the prediction was for.

Fixing this is the substance of the milestone. Do not skip to the modelling.

---

## 2. Characterise the fault signature first

Before changing any horizon, measure the thing that made 24 hours easy.

For each component, using training data only:

- For every failure, find the first occurrence of the matched error code in the
  preceding 30 days. Report the distribution of lead time between that first
  occurrence and the failure: median, 10th and 90th percentile.
- Do the same for the matched sensor channel, defining "onset" as the first hour the
  channel's 24h rolling mean crosses two standard deviations from that machine's
  own baseline.
- Report the false-positive rate of each signal at each horizon: how often does the
  code fire with no failure following within the window.

Write this to `docs/SIGNAL_ANALYSIS.md` as a table. It answers the question the
1.000 raised, and it tells you where the horizon stops being trivial.

---

## 3. Horizon sweep

Rebuild labels and re-evaluate at horizons of **24h, 72h, 7d, 14d, and 30d**.

For each horizon, per component, report on **validation only**:

- Positive rate (it will rise sharply; 7d is roughly 14% of hours, not 2%)
- PR-AUC for: majority baseline, matched-code baseline, logistic regression, LightGBM
- The three-way ablation from Milestone 3 — telemetry-only, errors-only, combined —
  since the collapse of that interaction is the mechanism you are looking for

Produce one plot: PR-AUC against horizon, per component, with the no-skill line.

The expected shape is a cliff. Find where it is and say so.

---

## 4. Choose the operational horizon

Select a single horizon and justify it in writing against three constraints:

1. **Lead time.** It must exceed the median supplier lead time of 23 days, or you
   must state explicitly which parts it can and cannot support and why that is
   acceptable.
2. **Predictability.** The model must still beat the matched-code baseline by a
   margin larger than the bootstrap confidence interval.
3. **Actionability.** State what an operator does with a prediction at this horizon
   that they could not do at 24 hours.

If constraints 1 and 2 cannot both be satisfied, **say so plainly.** A documented
finding that "this data cannot support a prediction horizon long enough to order
parts, therefore the agent's parts-ordering tool must work from stock on hand rather
than from predictions" is a legitimate and valuable engineering result. Do not
manufacture a horizon that satisfies neither.

Record the choice and its rationale in `docs/DATA.md`.

---

## 5. Redo the sections the 1.000 emptied out

At the chosen horizon, now with a model that is not deterministic:

- **Calibration.** Reliability curve and Brier score per component, on validation.
  Apply isotonic or Platt scaling fitted on validation if needed; report before and
  after.
- **Cost-based thresholds.** Expected-cost curve across thresholds using the ratio
  in `docs/DATA.md`. Per-component threshold selection. Sensitivity table at 3:1,
  10:1 and 30:1.
- **Permutation importance** on validation, with confidence intervals.

Then evaluate the test split **once** at the chosen horizon, under the same
single-consumer unlock protocol Milestone 3 established. Bootstrap confidence
intervals on precision and recall, resampled at the level of failure events.

---

## 6. What to keep from Milestone 3

Do not delete the 24-hour results. They become the opening section of
`docs/EVALUATION.md`:

> The standard framing of this dataset scores near-perfectly. Here is the
> measurement, here are the two controls proving it is neither leakage nor
> memorisation, here is the fault signature that explains it, and here is why a
> 24-hour horizon cannot serve a decision with a 23-day lead time.

That section is more valuable than any score in the document. Write it as the
finding it is.

---

## 7. Housekeeping

- The three window-coverage features are constant across validation and test, so
  they carry zero information there. Note this in `docs/FEATURES.md`; consider
  whether they earn their place or should be dropped.
- `test_results.json` currently re-renders by extraction rather than regeneration
  for run 1. Make the next run persist properly and remove the extraction path.

---

## Acceptance

- `docs/SIGNAL_ANALYSIS.md` exists with the lead-time distributions
- The horizon sweep plot is committed
- The chosen horizon is justified against all three constraints in writing, or the
  impossibility is documented
- Calibration, cost thresholds and permutation importance are redone at the chosen
  horizon and are non-trivial
- The test split was evaluated once at the new horizon
- `docs/EVALUATION.md` opens with the 24-hour finding and its explanation

Then stop and summarise. Do not build the agent.
