# MILESTONE 2 — Labels, features, splits, and leakage enforcement

Read `docs/DATA.md` in full first. This milestone implements the constraints it
specifies.

**Scope: no model.** Do not train anything, do not import a classifier, do not
compute a performance metric. This milestone produces a labelled feature matrix and
proves it is free of temporal leakage. Milestone 3 trains on it.

The reason for the split: every metric produced in Milestone 3 is only as
trustworthy as the guarantees established here. Those guarantees must be tested
before any model exists to be tempted by them.

---

## 0. Correction to DATA.md

Section 5.1 currently states that features must use records with `datetime < t`,
"strictly less than, never `<=`". Replace that rule with:

> **No feature may use any record with `datetime > t`.**

Rationale to record in the document: the label for prediction time `t` covers
`(t, t + 24h]`. Every failure in that window has `datetime > t`, so its coincident
`maint` record is excluded by this rule. A `maint` record at exactly `t` is a
replacement that has already occurred and is legitimately available to an operator
at prediction time. Excluding it, and excluding the telemetry reading at `t`,
discards real observed signal for no gain.

---

## 1. Labels

Four independent binary labels, one per component, as specified in `docs/DATA.md`
section 4.

For machine `m` at time `t`, `label_comp{k}` is 1 if a `comp{k}` failure occurs for
`m` in `(t, t + 24h]`, else 0.

Requirements:

- The horizon is a named constant, not a literal scattered through the code.
- Rows whose label window extends past the last observed failure timestamp
  (2015-12-31 06:00) cannot be confirmed negative. Trim them. Assert the trim
  boundary in a test rather than assuming it.
- Report the resulting positive count and rate per component. `docs/DATA.md`
  estimates roughly 2.1% overall and 0.71 / 0.53 / 0.49 / 0.36% for comp2 / comp1 /
  comp4 / comp3. If your measured figures differ, the estimate was wrong —
  correct the document with the measured values and note that they are now
  measured rather than derived.

---

## 2. Features

All features are computed by a single function taking an explicit `as_of` timestamp.
There is exactly one implementation, used identically for training and inference.

> v1 of this project had a defect where `predict_failure` passed features positionally
> and a swapped feature order changed the output from 0.00 to 0.70. The feature
> function must return a named DataFrame, and every consumer must select by column
> name. A test asserts that permuting the input column order does not change output.

### Feature groups

**Telemetry rolling aggregates.** Mean and standard deviation of `volt`, `rotate`,
`pressure`, `vibration` over 3-hour and 24-hour windows ending at `t` inclusive,
per machine. 4 sensors x 2 statistics x 2 windows = 16 features.

**Error counts.** Count of each `errorID` for the machine in the 24 hours and the
7 days ending at `t` inclusive. 5 error types x 2 windows = 10 features.

**Maintenance recency.** Hours since the most recent replacement of each component
for that machine, as of `t`. 4 features. Use the pre-2015 `maint` records so this is
defined from the first telemetry hour — see `docs/DATA.md` section 3.

**Machine attributes.** `age`, and `model` as four indicator columns. 5 features.

Approximately 35 features total. If you add or drop any, say which and why.

### Constraints

- Windows are time-based, not row-count-based.
- Every window boundary is closed at `t` and open at the start.
- No feature may reference a future record. This is the subject of section 4.
- Feature computation must be deterministic and seeded where any ordering is
  ambiguous.

---

## 3. Splits

Temporal, as specified in `docs/DATA.md` section 5.2:

| Split | Period |
|---|---|
| Train | 2015-01-01 to 2015-09-30 |
| Validation | 2015-10-01 to 2015-10-31 |
| Test | 2015-11-01 to 2015-12-31 |

Requirements:

- **Embargo.** Drop rows in the final 24 hours of each split, so no training row's
  label window extends into validation, and no validation row's label window extends
  into test. 24 hours is the label horizon; state the chosen embargo as a constant
  derived from that horizon, not a magic number.
- Feature windows reaching backwards across a split boundary are **permitted and
  correct** — in production, history is available. Do not truncate them. Document
  this asymmetry explicitly, because it looks like leakage and is not.
- No random or shuffled splitting anywhere in the codebase.
- Report the row count and positive count per component per split. Note in the
  output that the test period contains roughly 127 failure events, so recall
  estimated on it carries wide uncertainty.

A secondary machine-holdout split is **out of scope for this milestone**. It is
recorded in `docs/DATA.md` and will be added in Milestone 3 if time allows.

---

## 4. Leakage tests

This is the substance of the milestone. `tests/test_no_future_leakage.py`:

1. **Future-record invariance.** For a sample of `(machine, t)` pairs, compute all
   features. Insert a synthetic record — telemetry, error, maint, and failure, each
   tested separately — at `t + 1h`. Recompute. Assert every feature value is
   unchanged.

2. **Boundary inclusion.** Assert that a record at exactly `t` *does* affect the
   relevant feature, and a record at `t + 1s` does not. This proves the boundary is
   where the document says it is, and stops the test above from passing vacuously
   because features ignore recent data entirely.

3. **Failure-coincident maint exclusion.** Construct a case with a failure and its
   coincident `maint` record inside the label window. Assert no maintenance-recency
   feature reflects it.

4. **Split ordering.** Assert `max(train.datetime) + embargo <= min(val.datetime)`,
   and the same for validation to test.

5. **Label window containment.** Assert no row's label window extends beyond its own
   split.

6. **No shuffled splitting.** Assert by source inspection that no call to a shuffling
   splitter exists in `src/`.

7. **Column-order invariance.** Permute the input DataFrame's columns; assert
   identical feature output.

Each test must be shown to fail when the constraint it guards is deliberately
broken. A test that cannot fail is not a test — the Milestone 1 guard-test pattern
applies here too.

---

## 5. Outputs

- `data/generated/features_{train,val,test}.parquet`
- A row in `data/generated/build_manifest.json` per artefact with a content hash
- A short `docs/FEATURES.md`: every feature, its definition, its window, and the
  source table it derives from. One table, no prose padding.

---

## Acceptance

- `make features` builds all three splits deterministically from a clean state;
  two consecutive runs produce identical content hashes.
- `pytest` passes, including every test in section 4.
- Each section-4 test has been demonstrated to fail when its constraint is broken.
- `docs/DATA.md` section 5.1 carries the corrected `datetime > t` rule and the
  rationale.
- `docs/FEATURES.md` exists.

Then stop and summarise. Do not train a model.
