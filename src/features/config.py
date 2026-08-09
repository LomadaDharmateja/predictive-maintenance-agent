"""Constants for the label, feature and split definitions.

Everything here is a named constant on purpose. A literal `24` scattered across a
label function, an embargo and a rolling window is three unrelated decisions that
look like one, and changing the horizon later silently changes only some of them.
"""

from __future__ import annotations

import pandas as pd

# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------

#: Prediction horizon. A row at time `t` is positive for component `k` if a `k`
#: failure occurs in `(t, t + LABEL_HORIZON]`. docs/DATA.md section 4.
#:
#: **14 days, derived rather than inherited.** Milestone 3 used 24 hours because
#: that is the standard framing of this dataset, and scored PR-AUC 1.000 with
#: sound controls. It was still the wrong horizon: supplier lead times in
#: data/generated/parts_inventory.csv run 10 to 34 days, so a 24-hour warning
#: cannot inform the decision the model exists to serve.
#:
#: 14 days is the longest horizon at which LightGBM still beats the matched-error
#: baseline with non-overlapping bootstrap intervals on all four components. At
#: 30 days -- the shortest horizon that covers the median 23-day lead time -- the
#: intervals overlap for comp1 and comp2, so predictability is no longer
#: established. No horizon satisfies both constraints. docs/SIGNAL_ANALYSIS.md
#: section 4 documents that impossibility and what follows from it.
LABEL_HORIZON = pd.Timedelta(days=14)

#: The horizon Milestone 3 used, kept so the archived results in
#: docs/EVALUATION_24h.md remain reproducible and so the sweep has its anchor.
LEGACY_HORIZON_24H = pd.Timedelta(hours=24)

#: The last failure timestamp present in the source data. Telemetry runs to
#: 2016-01-01 06:00, but a row whose label window extends past this point cannot
#: be confirmed negative -- only unobserved -- so those rows are trimmed.
#: docs/DATA.md section 3.
LAST_OBSERVED_FAILURE = pd.Timestamp("2015-12-31 06:00:00")

#: Latest prediction time whose whole label window is observed.
MAX_PREDICTION_TIME = LAST_OBSERVED_FAILURE - LABEL_HORIZON  # 2015-12-30 06:00

# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------

SENSORS = ["volt", "rotate", "pressure", "vibration"]
COMPONENTS = ["comp1", "comp2", "comp3", "comp4"]
ERROR_IDS = ["error1", "error2", "error3", "error4", "error5"]
MACHINE_MODELS = ["model1", "model2", "model3", "model4"]

#: Rolling windows for telemetry aggregates. Time-based, not row-count-based.
TELEMETRY_WINDOWS = {"3h": pd.Timedelta(hours=3), "24h": pd.Timedelta(hours=24)}

#: Lookback windows for error counts.
ERROR_WINDOWS = {"24h": pd.Timedelta(hours=24), "7d": pd.Timedelta(days=7)}

#: Distinct window widths, for the coverage features.
#:
#: Three, not four. Coverage -- how many hourly grid points a window actually
#: spanned -- depends only on the width and the position in the series, not on
#: which table is being aggregated. The telemetry 24h window and the error 24h
#: window therefore produce the identical column, and shipping both would be a
#: perfectly collinear duplicate. See docs/FEATURES.md.
COVERAGE_WINDOWS = {
    name: width
    for name, width in {**TELEMETRY_WINDOWS, **ERROR_WINDOWS}.items()
}

#: Delta degrees of freedom for the rolling standard deviation.
#:
#: 0, not pandas' default of 1, and deliberately. The first prediction times in
#: the series have partial windows -- at 2015-01-01 06:00 only one reading exists
#: -- and ddof=1 yields NaN there. A NaN feature propagates silently into every
#: downstream consumer. ddof=0 gives 0.0 for a single observation, which is both
#: defined and true: one sample has no spread. See docs/FEATURES.md.
STD_DDOF = 0

# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------

#: Gap dropped from the end of each split. Derived from the horizon, not chosen:
#: a row within LABEL_HORIZON of a boundary has a label window that reaches into
#: the next split.
EMBARGO = LABEL_HORIZON

#: Inclusive period bounds. docs/DATA.md section 5.2.
SPLITS: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {
    "train": (pd.Timestamp("2015-01-01 00:00:00"), pd.Timestamp("2015-09-30 23:59:59")),
    "val": (pd.Timestamp("2015-10-01 00:00:00"), pd.Timestamp("2015-10-31 23:59:59")),
    "test": (pd.Timestamp("2015-11-01 00:00:00"), pd.Timestamp("2015-12-31 23:59:59")),
}

SPLIT_ORDER = ["train", "val", "test"]


def label_column(component: str) -> str:
    return f"label_{component}"


LABEL_COLUMNS = [label_column(c) for c in COMPONENTS]

#: Feature names in a fixed order. Consumers select by name, never by position --
#: v1 of this project had a defect where a swapped feature order changed a
#: prediction from 0.00 to 0.70 with no error raised.
FEATURE_COLUMNS: list[str] = (
    [
        f"{sensor}_{stat}_{window}"
        for sensor in SENSORS
        for window in TELEMETRY_WINDOWS
        for stat in ("mean", "std")
    ]
    # How many hourly grid points each window actually contained. Constant once
    # the series is wide enough; below that it is what lets a model discount a
    # partial window instead of reading a 1-sample mean as if it were a
    # 24-sample one. Partial windows are kept rather than trimmed because a
    # warm-up trim would delete all 18 anomalous failures -- docs/FEATURES.md.
    + [f"window_coverage_{window}" for window in COVERAGE_WINDOWS]
    + [
        f"{error_id}_count_{window}"
        for error_id in ERROR_IDS
        for window in ERROR_WINDOWS
    ]
    + [f"hours_since_{component}" for component in COMPONENTS]
    + ["age"]
    + [f"model_{model}" for model in MACHINE_MODELS]
)

INDEX_COLUMNS = ["machineID", "datetime"]
