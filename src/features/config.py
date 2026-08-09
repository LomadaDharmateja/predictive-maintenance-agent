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
LABEL_HORIZON = pd.Timedelta(hours=24)

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
