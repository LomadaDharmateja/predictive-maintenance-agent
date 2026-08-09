# The v1 leakage case study

Every number here was recomputed by `scripts/leakage_case_study.py` from
`archive/v1-data/maintenance.csv`, following what
`archive/v1-app/tools/db_setup.py` and `archive/v1-app/tools/train_model.py`
actually did. Nothing is quoted from an earlier document. Reproduce with
`make case-study`.

---

## What v1 did

`db_setup.py` contained this, under a comment calling it robustness:

```python
large_maint = pd.concat([maint_df] * 5, ignore_index=True)  # Multiply by 5
large_maint['Torque [Nm]'] = large_maint['Torque [Nm]'] + \
    np.random.normal(0, 2, large_maint.shape[0])
```

The AI4I table has 10,000 rows and 10,000 unique product IDs -- one row per product. After inflation:

| | Source | After inflation |
|---|---|---|
| Rows | 10,000 | 50,000 |
| Unique `UDI` | 10,000 | 10,000 |
| Rows sharing a `UDI` with another row | 0 | 40,000 |
| Exact duplicate rows | 0 | 0 |

**The last row is the trap.** Adding Gaussian noise to torque means no two
copies are byte-identical, so `DataFrame.duplicated()` reports zero and a
casual check passes. Every copy still carries the same label and four
identical features out of five.

`train_model.py` then fitted a RandomForest on the inflated table with no
held-out evaluation of any kind -- no split, no cross-validation, no metric.
The model was pickled and shipped. The numbers below are what it *would* have
reported under the evaluation a reader would assume.

---

## The gap

| | Naive random split of the inflated table | Honest 5-fold CV on the source rows |
|---|---|---|
| F1 | **0.958** | **0.736** |
| Precision | 0.973 | 0.855 |
| Recall | 0.944 | 0.646 |
| Evaluated on | 10,000 rows drawn from the inflated table | all 10,000 source rows, out of fold |
| Confusion (TP/FP/FN/TN) | 320/9/19/9652 | 219/37/120/9624 |

**22.2 F1 points of pure illusion.**

### Reconciliation with the figures in `docs/MILESTONE_3.md`

The milestone brief quoted F1 0.964 for the naive split and a 23-point gap.
Recomputed here: **0.9581** and **22.2 points**. The
honest-CV figures match the brief exactly (F1 0.736, recall 0.646), as does
the majority accuracy (0.9661).

The naive figure differs because **`db_setup.py` called `np.random.normal`
with no seed**. The torque noise -- and therefore the inflated table, and
therefore every number derived from it -- was different on every run and could
not be reproduced even by the person who wrote it. This script seeds the noise
so the comparison is stable; the seed changes the third decimal, not the
mechanism. That the original number cannot be recovered is itself part of the
case study.

### The mechanism

Each source row exists five times. A random 80/20 split puts, in expectation,
four copies in train and one in test. The model does not generalise to the
test row -- it *remembers* it, from four near-identical rows it was fitted on,
differing only by a couple of Newton-metres of injected noise. The test set
is not held out in any meaningful sense; it is a subset of the training set
wearing a disguise.

Recall moves most:
0.646 honest against 0.944 naive. Positives
are rare, so a memorised positive is worth far more than a memorised
negative, and duplication rewards exactly that.

---

## Why nobody noticed

The AI4I positive rate is 0.0339, so predicting 'no failure' for every row scores **96.61% accuracy**.

That is the number that made the inflated result look plausible. Against a
96.6% floor, an F1 of 0.96 reads as 'about as good as you would expect'
rather than 'impossibly good'. The two figures are not comparable -- one is
accuracy on a 3.4% positive rate, the other is F1 -- but they are the same
size, and a reader skimming a README will not stop to notice.

v1's README claimed the model provided 'Future-Sight'. It reported no metric
at all, so there was nothing to check.

---

## How the current pipeline makes this impossible

Not 'unlikely'. Each of these fails the build rather than degrading quietly.

| Failure mode from v1 | What now prevents it |
|---|---|
| Silent row duplication | Every table has a `PRIMARY KEY` over its natural key and ingestion uses plain `INSERT`, never `INSERT OR IGNORE`. A duplicated source row aborts `make data` with an `IntegrityError`. `tests/test_ingest.py::test_primary_keys_reject_duplicates` |
| Random split over autocorrelated rows | Splits are temporal. No shuffling splitter exists anywhere in `src/`, asserted by parsing the source, and the scanner is itself tested against a planted `train_test_split` call. `tests/test_no_future_leakage.py::test_no_shuffled_splitting_anywhere_in_src` |
| A training row's label window reaching into the evaluation period | A 24-hour embargo derived from the label horizon, plus four independent assertions on the split boundary, each demonstrated to fail when the embargo is removed |
| Features reading the future | 56 leakage tests: future-record invariance, boundary inclusion at exactly `t`, failure-coincident `maint` exclusion, and column-order invariance. Each is paired with a demonstration that it can fail |
| Tuning against the test set | `features_test.parquet` is behind a runtime lock, and one module holds the key. `tests/test_single_test_split_consumer.py` |
| No evaluation at all | `docs/EVALUATION.md` reports PR-AUC against three baselines with bootstrap intervals, and accuracy appears exactly once, in a sentence explaining why it is never used again |

---

## The honest reading of this comparison

The 5-fold CV column is not a *good* result either. F1
0.736 at recall 0.646 means the model misses
35% of failures. The point of this document
is not that the honest number is impressive. It is that the honest number is
**true**, and that a project reporting the inflated one would have shipped a
model believing it caught almost everything while missing a third of it.

The AI4I data is retained under `archive/v1-data/` for this reason alone. It
is unsuitable for the current project -- 10,000 rows, 10,000 unique product
IDs, no machine entity and no time dimension, so no fleet and no temporal
split could exist on it.
