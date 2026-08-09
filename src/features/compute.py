"""The single feature implementation.

`compute_features(store, as_of)` is the only place features are defined. Training
and inference both call it; there is no batch variant and no "fast path". The
batch builder in `build.py` calls it once per prediction hour.

It returns a DataFrame with named columns. Consumers must select by name. v1 of
this project passed features positionally into a model, and a swapped order
changed a prediction from 0.00 to 0.70 with nothing raised -- the return type
here exists to make that mistake impossible rather than unlikely.

The rule every function in this file obeys, from docs/DATA.md section 5.1:

    No feature may use any record with datetime > t.

The boundary is closed at `t`. Every window below is half-open at its start and
closed at its end: `(t - W, t]`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.config import (
    COMPONENTS,
    ERROR_IDS,
    ERROR_WINDOWS,
    FEATURE_COLUMNS,
    LABEL_COLUMNS,
    LABEL_HORIZON,
    MACHINE_MODELS,
    SENSORS,
    STD_DDOF,
    TELEMETRY_WINDOWS,
    label_column,
)
from src.features.store import GRID_PAD, NS_PER_HOUR, FeatureStore


def _window_steps(window: pd.Timedelta) -> int:
    """Number of hourly grid points in a window. The grid is hourly by assertion
    in `FeatureStore`, so this is exact rather than approximate."""
    steps, remainder = divmod(window.value, NS_PER_HOUR)
    if remainder or steps < 1:
        raise ValueError(f"window {window} is not a positive whole number of hours")
    return int(steps)


def compute_features(store: FeatureStore, as_of: pd.Timestamp) -> pd.DataFrame:
    """Features for every machine at one prediction time.

    Returns a DataFrame indexed by machineID with exactly `FEATURE_COLUMNS`, in
    that order. `as_of` must be a telemetry grid point.
    """
    as_of = pd.Timestamp(as_of)
    index = store.index_of(as_of)
    columns: dict[str, np.ndarray] = {}

    # -- Telemetry rolling aggregates ---------------------------------------
    # Rows [index - steps + 1, index] are exactly the grid points in
    # (as_of - W, as_of]. Slicing at `index + 1` is what closes the window at
    # `as_of`; anything later is not in the slice and cannot be read.
    for sensor in SENSORS:
        matrix = store.sensors[sensor]
        for name, window in TELEMETRY_WINDOWS.items():
            steps = _window_steps(window)
            start = max(0, index - steps + 1)
            block = matrix[start : index + 1, :]
            columns[f"{sensor}_mean_{name}"] = block.mean(axis=0)
            # ddof=0, so a partial window at the start of the series gives 0.0
            # rather than NaN. See config.STD_DDOF.
            columns[f"{sensor}_std_{name}"] = block.std(axis=0, ddof=STD_DDOF)

    # -- Error counts --------------------------------------------------------
    # An event is counted in the window ending at `index` if its arrival
    # position lies in [index - steps + 1, index]; see FeatureStore's docstring.
    for error_id in ERROR_IDS:
        prefix = store.error_cumsum[error_id]
        for name, window in ERROR_WINDOWS.items():
            steps = _window_steps(window)
            stop = index + GRID_PAD + 1
            start = index - steps + 1 + GRID_PAD
            columns[f"{error_id}_count_{name}"] = (
                prefix[stop, :] - prefix[start, :]
            ).astype("float64")

    # -- Maintenance recency -------------------------------------------------
    # `maint_last_ns[index]` is the latest replacement at or before `as_of`.
    # A replacement stamped exactly `as_of` gives 0.0 -- it has already happened
    # and an operator would know about it.
    for component in COMPONENTS:
        last_ns = store.maint_last_ns[component][index, :]
        hours_since = (as_of.value - last_ns) / NS_PER_HOUR
        # No prior record is NaN, not a sentinel number: a sentinel would be
        # trained on as if it were a duration. build.py asserts none occur.
        hours_since = np.where(
            last_ns == np.iinfo("int64").min, np.nan, hours_since
        )
        columns[f"hours_since_{component}"] = hours_since

    # -- Machine attributes --------------------------------------------------
    columns["age"] = store.age
    for model in MACHINE_MODELS:
        columns[f"model_{model}"] = store.model_indicator[model]

    frame = pd.DataFrame(columns, index=pd.Index(store.machine_ids, name="machineID"))
    # Reindex rather than trust insertion order: this raises if a feature was
    # added to FEATURE_COLUMNS but not computed here, instead of writing a column
    # of NaN into the training set.
    missing = set(FEATURE_COLUMNS) - set(frame.columns)
    unexpected = set(frame.columns) - set(FEATURE_COLUMNS)
    if missing or unexpected:
        raise ValueError(
            f"feature mismatch: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    return frame[FEATURE_COLUMNS]


def compute_labels(store: FeatureStore, as_of: pd.Timestamp) -> pd.DataFrame:
    """Labels for every machine at one prediction time.

    `label_comp{k}` is 1 if a `comp{k}` failure occurs in `(as_of, as_of + H]`.

    This function reads events after `as_of` by definition -- that is what a label
    is. It is kept separate from `compute_features` so the two are never confused,
    and so the leakage tests can assert that features are invariant to exactly the
    records that labels depend on.
    """
    as_of = pd.Timestamp(as_of)
    index = store.index_of(as_of)
    steps = _window_steps(LABEL_HORIZON)

    columns = {}
    for component in COMPONENTS:
        prefix = store.failure_cumsum[component]
        # Arrival positions in [index + 1, index + steps] correspond to failure
        # times in (as_of, as_of + H]. prefix row x counts arrivals strictly
        # before x - GRID_PAD.
        last = prefix.shape[0] - 1
        upper = min(index + steps + 1 + GRID_PAD, last)
        lower = min(index + 1 + GRID_PAD, last)
        count = prefix[upper, :] - prefix[lower, :]
        columns[label_column(component)] = (count > 0).astype("int8")

    return pd.DataFrame(
        columns, index=pd.Index(store.machine_ids, name="machineID")
    )[LABEL_COLUMNS]
