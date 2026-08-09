"""Relabel and re-split the feature matrices at an arbitrary prediction horizon.

`docs/MILESTONE_3B.md` section 3. Features are backward-looking and therefore
horizon-independent -- only the labels and the two trims change -- so the sweep
reuses the matrices `make features` already built rather than rebuilding them
five times.

Both trims tighten as the horizon grows, so every horizon's row set is a subset
of the 24-hour one. That is what makes reusing the parquet files sound, and
`tests/test_horizons.py` asserts it rather than leaving it as an argument.

    embargo            = horizon      (a training row's label must not reach
                                       into the next split)
    max prediction time = last observed failure - horizon
                                      (a row whose window extends past the last
                                       failure cannot be confirmed negative)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.config import COMPONENTS, LAST_OBSERVED_FAILURE, SPLITS

#: The sweep, as specified.
SWEEP_HORIZONS = [
    pd.Timedelta(hours=24),
    pd.Timedelta(hours=72),
    pd.Timedelta(days=7),
    pd.Timedelta(days=14),
    pd.Timedelta(days=30),
]


def horizon_label(horizon: pd.Timedelta) -> str:
    hours = int(horizon.total_seconds() // 3600)
    return f"{hours}h" if hours < 168 else f"{hours // 24}d"


def parse_horizon(label: str) -> pd.Timedelta:
    return pd.Timedelta(days=int(label[:-1])) if label.endswith("d") else pd.Timedelta(
        hours=int(label[:-1])
    )


@dataclass(frozen=True)
class HorizonSplit:
    horizon: pd.Timedelta
    split: str
    frame: pd.DataFrame

    @property
    def label(self) -> str:
        return horizon_label(self.horizon)


def label_at_horizon(
    frame: pd.DataFrame, failures: pd.DataFrame, horizon: pd.Timedelta
) -> pd.DataFrame:
    """Labels for every component at `horizon`, as `label_comp{k}` columns.

    Vectorised per (machine, component) with two searchsorted calls: the count of
    failures in `(t, t + horizon]` is the difference of the two insertion points.
    `side="right"` on both ends makes the interval open at `t` and closed at
    `t + horizon`, matching `src/features/compute.py`.
    """
    when = pd.to_datetime(frame["datetime"]).to_numpy()
    machines = frame["machineID"].to_numpy()
    out = pd.DataFrame(index=frame.index)

    for component in COMPONENTS:
        relevant = failures[failures["failure"] == component]
        events = {
            machine: np.sort(pd.to_datetime(group["datetime"]).to_numpy())
            for machine, group in relevant.groupby("machineID")
        }
        labels = np.zeros(len(frame), dtype="int8")
        for machine in np.unique(machines):
            times = events.get(machine)
            if times is None:
                continue
            rows = np.flatnonzero(machines == machine)
            start = np.searchsorted(times, when[rows], side="right")
            stop = np.searchsorted(
                times, when[rows] + np.timedelta64(horizon), side="right"
            )
            labels[rows] = (stop > start).astype("int8")
        out[f"label_{component}"] = labels

    return out


def trim_for_horizon(
    frame: pd.DataFrame, split: str, horizon: pd.Timedelta
) -> pd.DataFrame:
    """Apply the embargo and the label-observability cutoff at this horizon."""
    start, end = SPLITS[split]
    latest = min(end - horizon, LAST_OBSERVED_FAILURE - horizon)
    when = pd.to_datetime(frame["datetime"])
    return frame[(when >= start) & (when <= latest)]


def rebuild(
    frame: pd.DataFrame,
    failures: pd.DataFrame,
    split: str,
    horizon: pd.Timedelta,
) -> pd.DataFrame:
    """Trim, then relabel. The 24-hour labels already on the frame are dropped.

    Order matters only for cost: trimming first means relabelling fewer rows.
    """
    trimmed = trim_for_horizon(frame, split, horizon)
    features = trimmed.drop(columns=[f"label_{c}" for c in COMPONENTS])
    labels = label_at_horizon(trimmed, failures, horizon)
    return pd.concat([features, labels], axis=1)


def describe(frame: pd.DataFrame, horizon: pd.Timedelta, split: str) -> dict:
    return {
        "horizon": horizon_label(horizon),
        "split": split,
        "rows": int(len(frame)),
        "prediction_times": int(frame["datetime"].nunique()),
        "first": str(frame["datetime"].min()),
        "last": str(frame["datetime"].max()),
        "positive_rate": {
            component: float(frame[f"label_{component}"].mean())
            for component in COMPONENTS
        },
        "positives": {
            component: int(frame[f"label_{component}"].sum())
            for component in COMPONENTS
        },
    }
