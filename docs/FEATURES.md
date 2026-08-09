# FEATURES.md

38 features and 4 labels, produced by `src/features/compute.py` and assembled by
`src/features/build.py`. Definitions live in `src/features/config.py`; nothing below is
a literal repeated in code.

Prediction time is written `t`. Every window is **half-open at the start and closed at
the end**: `(t - W, t]`. The governing rule, from `docs/DATA.md` section 5.1:

> No feature may use any record with `datetime > t`.

Enforced by `tests/test_no_future_leakage.py`, not by convention.

---

## Features

| # | Column | Definition | Window | Source table |
|---|---|---|---|---|
| 1-8 | `{sensor}_mean_3h`, `{sensor}_std_3h` | Mean and standard deviation per machine | `(t - 3h, t]`, 3 hourly readings | `telemetry` |
| 9-16 | `{sensor}_mean_24h`, `{sensor}_std_24h` | Mean and standard deviation per machine | `(t - 24h, t]`, 24 hourly readings | `telemetry` |
| 17-19 | `window_coverage_3h`, `window_coverage_24h`, `window_coverage_7d` | Hourly grid points the window actually spanned | the named window | grid position |
| 20-24 | `{errorID}_count_24h` | Count of that error for the machine | `(t - 24h, t]` | `errors` |
| 25-29 | `{errorID}_count_7d` | Count of that error for the machine | `(t - 7d, t]` | `errors` |
| 30-33 | `hours_since_{comp}` | Hours from the most recent replacement of that component to `t` | all history at or before `t` | `maint` |
| 34 | `age` | Machine age in years | static | `machines` |
| 35-38 | `model_{model}` | 1 if the machine is that model, else 0 | static | `machines` |

`{sensor}` ranges over `volt`, `rotate`, `pressure`, `vibration`.
`{errorID}` ranges over `error1`-`error5`. `{comp}` ranges over `comp1`-`comp4`.
`{model}` ranges over `model1`-`model4`.

Count: 4x2x2 = 16 telemetry, 3 coverage, 5x2 = 10 error, 4 maintenance, 5 machine.
**38 total**.

### Why coverage is three features, not four

There are four rolling windows — telemetry 3h and 24h, error 24h and 7d — but only
three distinct widths. Coverage depends solely on the window width and the position in
the series, never on which table is being aggregated, so a telemetry-24h coverage
column and an error-24h coverage column are the same numbers in both. Shipping both
would add a perfectly collinear duplicate. `COVERAGE_WINDOWS` is built by merging the
two window dictionaries, so the deduplication happens in code rather than by hand, and
`tests/test_features.py::test_coverage_windows_are_deduplicated_by_width` pins it.

**What these features are for.** Windows at the start of the series are partial — at
`2015-01-01 06:00` a 24-hour mean averages one reading. Coverage makes that visible to
the model, which is what justifies keeping partial windows instead of trimming a 23-hour
warm-up. Without it, a 1-sample mean and a 24-sample mean are indistinguishable inputs.

**Their honest limitation.** They vary only over the first 168 hours of `train` and are
constant everywhere after that: 3, 24 and 168 for every row of validation and test. A
model cannot use them outside the warm-up region, and their permutation importance on
validation will be exactly zero by construction. That is expected, not a defect —
they exist to mark a specific 0.27% of rows, and
`tests/test_features.py::test_window_coverage_is_constant_after_the_widest_window_fills`
records it so nobody later reads a zero importance as a bug.

All feature columns are `float64`, including the counts and the model indicators. One
dtype across the matrix means a consumer cannot silently get integer division or an
object column.

### Decisions that are not obvious from the table

**Standard deviation uses `ddof=0`, not pandas' default of 1.** At the first grid point
a 3h window holds one reading, and `ddof=1` returns NaN there. A NaN feature propagates
into every downstream consumer without complaint. `ddof=0` returns 0.0, which is defined
and true: one sample has no spread. `tests/test_features.py::test_partial_windows_at_the_start_of_the_series_give_zero_std`
pins it.

**Windows at the start of the series are partial, and rows are not trimmed for it.**
The `window_coverage_*` features above exist so a model can see this rather than
having to infer it. The alternative — discarding the first 23 hours so every window is full — would delete
all 18 of the anomalous `2015-01-02 03:00` failures described in `docs/DATA.md` section
5.1, which is 2.4% of all positives. Partial windows affect 23 hours x 100 machines =
2,300 rows, 0.27% of the modelling set, all inside `train`. Keeping the positives is
worth the mild distribution shift, but it is a choice and it is recorded here rather
than left implicit.

**A `maint` record stamped exactly `t` gives `hours_since = 0`.** The boundary is closed
at `t`: the replacement has already happened, and a real operator would know about it.
This is the corrected rule; an earlier version of `docs/DATA.md` said `datetime < t`,
which would have discarded it. See section 5.1 of that document for why the change
does not reopen the leak it was guarding against.

**`hours_since_{comp}` is never null on this dataset**, because `maint` begins seven
months before `telemetry` and every machine has a record for all four components before
the first telemetry hour. Where no prior record exists the feature is NaN rather than a
sentinel number — a sentinel would be trained on as if it were a duration —
and `build.py` refuses to write a matrix containing one.

**No randomness anywhere.** Feature computation is a pure function of the source tables
and `as_of`. There is no seed because there is nothing to seed. Row order is
`(datetime, machineID)` by construction and re-sorted to make that a guarantee.

---

## Labels

| Column | Definition |
|---|---|
| `label_comp1` .. `label_comp4` | 1 if a failure of that component occurs for that machine in `(t, t + 24h]`, else 0 |

Four independent binary problems, not one multiclass problem: the downstream agent needs
a per-component probability to decide which part to reserve.

The window is open at `t` — a failure happening *now* is not a prediction — and closed at
`t + 24h`. Both edges are pinned by tests.

`int8`, so the label block is 4 bytes per row rather than 32.

---

## Splits

Temporal. `docs/DATA.md` section 5.2. No random or shuffled split exists anywhere in
`src/`; `tests/test_no_future_leakage.py` asserts that by parsing the source, and the
scanner itself is tested against a planted call so it cannot pass by being broken.

| Split | Period | Prediction times | Rows | First | Last |
|---|---|---|---|---|---|
| train | 2015-01-01 to 2015-09-30 | 6,522 | 652,200 | 2015-01-01 06:00 | 2015-09-29 23:00 |
| val | 2015-10-01 to 2015-10-31 | 720 | 72,000 | 2015-10-01 00:00 | 2015-10-30 23:00 |
| test | 2015-11-01 to 2015-12-31 | 1,423 | 142,300 | 2015-11-01 00:00 | 2015-12-30 06:00 |

Two independent trims produce those end points:

- **Embargo**, 24 hours, dropped from the end of every split. It equals the label
  horizon and is derived from it in code, not written as a separate number. Without it
  a training row's label window reaches into validation.
- **Label observability**, a global cutoff at `2015-12-30 06:00` = the last observed
  failure minus the horizon. Rows past it cannot be confirmed negative, only unobserved.
  This is what binds on `test`, and it binds earlier than that split's own embargo.

**Feature windows reaching backwards across a split boundary are permitted and
correct.** A validation row at 2015-10-01 00:00 has a 7-day error count that reads
September data. This looks like leakage and is not: in production, history is available
at prediction time. What must not cross a boundary is a *label* window, and none does.
The asymmetry is deliberate — features look back, labels look forward, and only the
forward direction is constrained by the split.

---

## Positive counts

Measured, not estimated.

| Split | Rows | comp1 | comp2 | comp3 | comp4 | any component |
|---|---|---|---|---|---|---|
| train | 652,200 | 3,645 (0.559%) | 4,479 (0.687%) | 2,457 (0.377%) | 3,303 (0.506%) | 13,074 (2.005%) |
| val | 72,000 | 192 (0.267%) | 624 (0.867%) | 186 (0.258%) | 336 (0.467%) | 1,296 (1.800%) |
| test | 142,300 | 651 (0.457%) | 1,062 (0.746%) | 444 (0.312%) | 583 (0.410%) | 2,590 (1.820%) |
| **all** | **866,500** | **4,488 (0.518%)** | **6,165 (0.711%)** | **3,087 (0.356%)** | **4,222 (0.487%)** | **16,960 (1.957%)** |

The test split's 2,590 positive rows derive from roughly 127 distinct failure events.
Recall estimated on that carries wide uncertainty and must be reported with an interval,
not as a point estimate.

Note that `comp2` in validation runs at 0.867% against 0.687% in train — the rarest
class in one split is not the rarest in another. A threshold tuned on validation is
tuned on a month with a different mix.

---

## Outputs

| Path | Contents |
|---|---|
| `data/generated/features_train.parquet` | `machineID`, `datetime`, 38 features, 4 labels |
| `data/generated/features_val.parquet` | as above |
| `data/generated/features_test.parquet` | as above |
| `data/generated/build_manifest.json` | rows, positive counts, and a SHA-256 content hash per split, under `features` |

Rebuild with `make features`. Two consecutive builds produce identical content hashes.

The hash covers column bytes in declared order, not the parquet file: parquet embeds a
writer version string, so file bytes would change on a pyarrow upgrade even when nothing
about the data had.

`make data` rewrites `build_manifest.json` and drops the `features` key. That is correct
— rebuilding the database invalidates anything derived from it — and `make features`
restores it.
