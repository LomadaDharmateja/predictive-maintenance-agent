"""Relabelling and re-trimming at an arbitrary horizon.

The horizon sweep reuses the parquet matrices `make features` built rather than
rebuilding them per horizon. That is only sound if two things hold, and both are
asserted here rather than argued for:

- relabelling at the shipped horizon reproduces the shipped labels exactly, and
- every longer horizon's row set is a subset of a shorter one's.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.eval.horizons import (
    SWEEP_HORIZONS,
    horizon_label,
    label_at_horizon,
    parse_horizon,
    rebuild,
    trim_for_horizon,
)
from src.features.config import (
    COMPONENTS,
    LABEL_HORIZON,
    LAST_OBSERVED_FAILURE,
    SPLITS,
)


@pytest.fixture(scope="module")
def failures(built_db) -> pd.DataFrame:
    import sqlite3

    connection = sqlite3.connect(built_db)
    try:
        return pd.read_sql_query(
            "SELECT * FROM failures", connection, parse_dates=["datetime"]
        )
    finally:
        connection.close()


# ----------------------------------------------------------------------
# Labels
# ----------------------------------------------------------------------


def test_relabelling_at_the_shipped_horizon_reproduces_the_shipped_labels(
    failures,
):
    """The load-bearing assertion. If this drifts, every sweep number is wrong."""
    from src.eval.datasets import load_val

    frame = load_val()
    relabelled = label_at_horizon(frame, failures, LABEL_HORIZON)
    for component in COMPONENTS:
        assert np.array_equal(
            frame[f"label_{component}"].to_numpy(),
            relabelled[f"label_{component}"].to_numpy(),
        ), component


def test_labels_are_monotone_in_the_horizon(failures):
    """A longer window can only turn a 0 into a 1, never the reverse."""
    from src.eval.datasets import load_val

    frame = load_val().head(20000)
    short = label_at_horizon(frame, failures, pd.Timedelta(hours=24))
    long = label_at_horizon(frame, failures, pd.Timedelta(days=14))
    for component in COMPONENTS:
        a = short[f"label_{component}"].to_numpy()
        b = long[f"label_{component}"].to_numpy()
        assert np.all(b >= a), component


def test_label_window_is_open_at_t_and_closed_at_t_plus_horizon():
    horizon = pd.Timedelta(days=7)
    when = pd.Timestamp("2015-05-01 00:00:00")
    frame = pd.DataFrame({"machineID": [1, 1, 1], "datetime": [when, when, when]})

    at_t = pd.DataFrame(
        [{"machineID": 1, "datetime": when, "failure": "comp1"}]
    )
    at_edge = pd.DataFrame(
        [{"machineID": 1, "datetime": when + horizon, "failure": "comp1"}]
    )
    past_edge = pd.DataFrame(
        [
            {
                "machineID": 1,
                "datetime": when + horizon + pd.Timedelta(seconds=1),
                "failure": "comp1",
            }
        ]
    )

    assert label_at_horizon(frame, at_t, horizon)["label_comp1"].iloc[0] == 0
    assert label_at_horizon(frame, at_edge, horizon)["label_comp1"].iloc[0] == 1
    assert label_at_horizon(frame, past_edge, horizon)["label_comp1"].iloc[0] == 0


# ----------------------------------------------------------------------
# Trims
# ----------------------------------------------------------------------


@pytest.mark.parametrize("split", ["train", "val"])
def test_longer_horizons_produce_subsets(failures, split):
    """Every sweep horizon's rows are a subset of the previous horizon's, which
    is what makes reusing the shipped parquet files valid."""
    from src.eval.datasets import load_train, load_val

    frame = load_train() if split == "train" else load_val()
    previous = None
    for horizon in SWEEP_HORIZONS:
        trimmed = trim_for_horizon(frame, split, horizon)
        keys = set(zip(trimmed["machineID"], trimmed["datetime"]))
        if previous is not None:
            assert keys <= previous, f"{horizon} is not a subset of the previous"
        previous = keys


@pytest.mark.parametrize("horizon", SWEEP_HORIZONS)
def test_trim_respects_the_embargo_and_the_observability_cutoff(failures, horizon):
    from src.eval.datasets import load_train

    trimmed = trim_for_horizon(load_train(), "train", horizon)
    if trimmed.empty:
        pytest.skip("horizon trims the split away entirely")

    latest = pd.to_datetime(trimmed["datetime"]).max()
    _, end = SPLITS["train"]
    assert latest <= end - horizon
    assert latest <= LAST_OBSERVED_FAILURE - horizon


def test_rebuild_drops_the_old_labels_and_writes_new_ones(failures):
    from src.eval.datasets import load_val

    rebuilt = rebuild(load_val(), failures, "val", pd.Timedelta(days=7))
    assert all(f"label_{c}" in rebuilt.columns for c in COMPONENTS)
    assert sum(c.startswith("label_") for c in rebuilt.columns) == len(COMPONENTS)


def test_positive_rate_rises_with_the_horizon(failures):
    """The sweep's premise. If it did not hold, the horizons would not be
    ordered by difficulty in the way the analysis assumes."""
    from src.eval.datasets import load_train

    frame = load_train()
    rates = []
    for horizon in [pd.Timedelta(hours=24), pd.Timedelta(days=7), pd.Timedelta(days=14)]:
        rebuilt = rebuild(frame, failures, "train", horizon)
        rates.append(float(rebuilt["label_comp1"].mean()))
    assert rates == sorted(rates)


# ----------------------------------------------------------------------
# Labels round-trip through their string form
# ----------------------------------------------------------------------


@pytest.mark.parametrize("horizon", SWEEP_HORIZONS)
def test_horizon_label_round_trips(horizon):
    assert parse_horizon(horizon_label(horizon)) == horizon


def test_horizon_labels_are_the_expected_strings():
    assert [horizon_label(h) for h in SWEEP_HORIZONS] == [
        "24h",
        "72h",
        "7d",
        "14d",
        "30d",
    ]
