"""Feature and label correctness, independent of the leakage guarantees.

Where a property can be checked against a second, slower implementation written
from the definition in docs/DATA.md rather than from `compute.py`, it is. Two
implementations that agree are evidence; one implementation compared against
itself is a tautology.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.features.build import (
    build_split,
    content_hash,
    prediction_times,
    summarise,
)
from src.features.compute import compute_features, compute_labels
from src.features.config import (
    COMPONENTS,
    COVERAGE_WINDOWS,
    ERROR_IDS,
    ERROR_WINDOWS,
    FEATURE_COLUMNS,
    INDEX_COLUMNS,
    LABEL_COLUMNS,
    LABEL_HORIZON,
    LAST_OBSERVED_FAILURE,
    MACHINE_MODELS,
    MAX_PREDICTION_TIME,
    SENSORS,
    SPLIT_ORDER,
    SPLITS,
    STD_DDOF,
    TELEMETRY_WINDOWS,
    label_column,
)
from src.features.store import FeatureStore, StoreError
from tests import synthetic

HOUR = pd.Timedelta(hours=1)


@pytest.fixture(scope="module")
def frames():
    return synthetic.make_frames()


@pytest.fixture(scope="module")
def store(frames):
    return FeatureStore.from_frames(**frames)


@pytest.fixture
def subset_store(real_subset_frames):
    return FeatureStore.from_frames(**real_subset_frames)


# ----------------------------------------------------------------------
# Feature inventory
# ----------------------------------------------------------------------


def test_feature_count_matches_the_specified_groups():
    telemetry = len(SENSORS) * len(TELEMETRY_WINDOWS) * 2  # mean and std
    coverage = len(COVERAGE_WINDOWS)
    errors = len(ERROR_IDS) * len(ERROR_WINDOWS)
    maint = len(COMPONENTS)
    machine = 1 + len(MACHINE_MODELS)  # age, plus one indicator per model

    assert (telemetry, coverage, errors, maint, machine) == (16, 3, 10, 4, 5)
    assert len(FEATURE_COLUMNS) == 38
    assert len(FEATURE_COLUMNS) == telemetry + coverage + errors + maint + machine


def test_coverage_windows_are_deduplicated_by_width():
    """Three coverage features, not four. Coverage depends only on window width
    and position in the series, so the telemetry 24h and error 24h windows would
    emit the same column."""
    assert set(COVERAGE_WINDOWS) == {"3h", "24h", "7d"}
    assert COVERAGE_WINDOWS["24h"] == TELEMETRY_WINDOWS["24h"] == ERROR_WINDOWS["24h"]


def test_feature_names_are_unique():
    assert len(set(FEATURE_COLUMNS)) == len(FEATURE_COLUMNS)


def test_model_indicators_are_one_hot(store):
    features = compute_features(store, synthetic.grid()[100])
    indicator_columns = [f"model_{model}" for model in MACHINE_MODELS]
    totals = features[indicator_columns].sum(axis=1)
    # Every machine in the synthetic fixture has a known model, so each row has
    # exactly one indicator set.
    assert set(totals.unique()) == {1.0}


# ----------------------------------------------------------------------
# Labels, checked against the definition rather than the implementation
# ----------------------------------------------------------------------


def brute_force_label(
    failures: pd.DataFrame, machine: int, as_of: pd.Timestamp, component: str
) -> int:
    """`(as_of, as_of + H]`, written straight from docs/DATA.md section 4."""
    rows = failures[
        (failures["machineID"] == machine) & (failures["failure"] == component)
    ]
    when = pd.to_datetime(rows["datetime"])
    inside = (when > as_of) & (when <= as_of + LABEL_HORIZON)
    return int(inside.any())


def test_labels_match_a_brute_force_recomputation(real_subset_frames, subset_store):
    failures = real_subset_frames["failures"]
    times = subset_store.times
    sampled = times[:: max(1, len(times) // 40)]

    for as_of in sampled:
        computed = compute_labels(subset_store, as_of)
        for machine in subset_store.machine_ids:
            for component in COMPONENTS:
                assert computed.loc[machine, label_column(component)] == (
                    brute_force_label(failures, int(machine), as_of, component)
                ), f"machine {machine} {component} at {as_of}"


def test_a_failure_at_exactly_t_is_not_in_its_own_label_window(frames):
    """The window is open at the start. A failure at `t` is the present, not a
    prediction."""
    when = synthetic.grid()[250]
    perturbed = synthetic.add_failure(frames, 1, when, "comp1")
    labels = compute_labels(FeatureStore.from_frames(**perturbed), when)
    assert labels.loc[1, label_column("comp1")] == 0


def test_a_failure_at_exactly_t_plus_horizon_is_in_the_window(frames):
    """The window is closed at the end."""
    when = synthetic.grid()[250]
    perturbed = synthetic.add_failure(frames, 1, when + LABEL_HORIZON, "comp1")
    labels = compute_labels(FeatureStore.from_frames(**perturbed), when)
    assert labels.loc[1, label_column("comp1")] == 1


def test_a_failure_one_second_past_the_horizon_is_outside(frames):
    when = synthetic.grid()[250]
    perturbed = synthetic.add_failure(
        frames, 1, when + LABEL_HORIZON + pd.Timedelta(seconds=1), "comp1"
    )
    labels = compute_labels(FeatureStore.from_frames(**perturbed), when)
    assert labels.loc[1, label_column("comp1")] == 0


# ----------------------------------------------------------------------
# The trim boundary
# ----------------------------------------------------------------------


def test_last_observed_failure_matches_the_data(real_store, built_db):
    """The constant is not taken on trust from docs/DATA.md."""
    import sqlite3

    connection = sqlite3.connect(built_db)
    try:
        (latest,) = connection.execute("SELECT MAX(datetime) FROM failures").fetchone()
    finally:
        connection.close()
    assert pd.Timestamp(latest) == LAST_OBSERVED_FAILURE


def test_trim_boundary_is_the_horizon_before_the_last_observed_failure():
    assert MAX_PREDICTION_TIME == LAST_OBSERVED_FAILURE - LABEL_HORIZON
    assert MAX_PREDICTION_TIME == pd.Timestamp("2015-12-30 06:00:00")


def test_no_split_extends_past_the_trim_boundary(real_store):
    for split in SPLIT_ORDER:
        assert prediction_times(real_store, split).max() <= MAX_PREDICTION_TIME


def test_the_test_split_is_bound_by_the_trim_not_the_embargo(real_store):
    """Which of the two trims binds is a fact worth pinning: the test split ends
    at the label-observability boundary, not at its own period end."""
    last = prediction_times(real_store, "test").max()
    assert last == MAX_PREDICTION_TIME
    assert last < SPLITS["test"][1] - LABEL_HORIZON


def test_telemetry_extends_past_the_last_usable_prediction_time(real_store):
    """Rows exist after the boundary and are deliberately discarded, rather than
    the boundary being wherever the data happened to stop."""
    assert real_store.times.max() > MAX_PREDICTION_TIME
    discarded = (real_store.times > MAX_PREDICTION_TIME).sum()
    assert discarded > 0


# ----------------------------------------------------------------------
# Features, checked against the definition
# ----------------------------------------------------------------------


def brute_force_rolling(
    telemetry: pd.DataFrame,
    machine: int,
    as_of: pd.Timestamp,
    sensor: str,
    window: pd.Timedelta,
) -> tuple[float, float]:
    rows = telemetry[telemetry["machineID"] == machine]
    when = pd.to_datetime(rows["datetime"])
    inside = rows[(when > as_of - window) & (when <= as_of)]
    values = inside[sensor].to_numpy(dtype="float64")
    return float(values.mean()), float(values.std(ddof=STD_DDOF))


def test_rolling_aggregates_match_a_brute_force_recomputation(
    real_subset_frames, subset_store
):
    telemetry = real_subset_frames["telemetry"]
    times = subset_store.times
    for as_of in times[:: max(1, len(times) // 12)]:
        features = compute_features(subset_store, as_of)
        for machine in subset_store.machine_ids[:3]:
            for sensor in SENSORS:
                for name, window in TELEMETRY_WINDOWS.items():
                    mean, std = brute_force_rolling(
                        telemetry, int(machine), as_of, sensor, window
                    )
                    assert features.loc[machine, f"{sensor}_mean_{name}"] == pytest.approx(mean)
                    assert features.loc[machine, f"{sensor}_std_{name}"] == pytest.approx(std)


def brute_force_error_count(
    errors: pd.DataFrame,
    machine: int,
    as_of: pd.Timestamp,
    error_id: str,
    window: pd.Timedelta,
) -> int:
    rows = errors[
        (errors["machineID"] == machine) & (errors["errorID"] == error_id)
    ]
    when = pd.to_datetime(rows["datetime"])
    return int(((when > as_of - window) & (when <= as_of)).sum())


def test_error_counts_match_a_brute_force_recomputation(
    real_subset_frames, subset_store
):
    errors = real_subset_frames["errors"]
    times = subset_store.times
    for as_of in times[:: max(1, len(times) // 25)]:
        features = compute_features(subset_store, as_of)
        for machine in subset_store.machine_ids:
            for error_id in ERROR_IDS:
                for name, window in ERROR_WINDOWS.items():
                    assert features.loc[machine, f"{error_id}_count_{name}"] == (
                        brute_force_error_count(
                            errors, int(machine), as_of, error_id, window
                        )
                    ), f"{error_id} {name} machine {machine} at {as_of}"


def brute_force_hours_since(
    maint: pd.DataFrame, machine: int, as_of: pd.Timestamp, component: str
) -> float:
    rows = maint[(maint["machineID"] == machine) & (maint["comp"] == component)]
    when = pd.to_datetime(rows["datetime"])
    prior = when[when <= as_of]
    if prior.empty:
        return float("nan")
    return (as_of - prior.max()).total_seconds() / 3600


def test_maintenance_recency_matches_a_brute_force_recomputation(
    real_subset_frames, subset_store
):
    maint = real_subset_frames["maint"]
    times = subset_store.times
    for as_of in times[:: max(1, len(times) // 25)]:
        features = compute_features(subset_store, as_of)
        for machine in subset_store.machine_ids:
            for component in COMPONENTS:
                assert features.loc[machine, f"hours_since_{component}"] == pytest.approx(
                    brute_force_hours_since(maint, int(machine), as_of, component)
                ), f"{component} machine {machine} at {as_of}"


@pytest.mark.parametrize(
    "hours_back, inside", [(0, True), (1, True), (2, True), (3, False), (10, False)]
)
def test_three_hour_telemetry_window_spans_exactly_three_grid_points(
    store, frames, hours_back, inside
):
    """`(t - 3h, t]` is three hourly readings: t, t-1h, t-2h. A reading at
    t - 3h is on the open end of the window and must not move the mean."""
    as_of = synthetic.grid()[250]
    baseline = compute_features(store, as_of)

    perturbed = synthetic.bump_telemetry(frames, 1, as_of - hours_back * HOUR, 30.0)
    features = compute_features(FeatureStore.from_frames(**perturbed), as_of)

    moved = bool(features.loc[1, "volt_mean_3h"] != baseline.loc[1, "volt_mean_3h"])
    assert moved is inside


@pytest.mark.parametrize(
    "hours_back, inside", [(0, True), (23, True), (24, False), (48, False)]
)
def test_twenty_four_hour_error_window_spans_exactly_one_day(
    store, frames, hours_back, inside
):
    as_of = synthetic.grid()[250]
    baseline = compute_features(store, as_of)

    perturbed = synthetic.add_error(frames, 1, as_of - hours_back * HOUR, "error3")
    features = compute_features(FeatureStore.from_frames(**perturbed), as_of)

    delta = (
        features.loc[1, "error3_count_24h"] - baseline.loc[1, "error3_count_24h"]
    )
    assert delta == (1 if inside else 0)


def test_seven_day_window_is_wider_than_the_twenty_four_hour_one(store, frames):
    """The two error windows are genuinely different lookbacks, not the same
    number under two names."""
    as_of = synthetic.grid()[250]
    perturbed = synthetic.add_error(frames, 1, as_of - 72 * HOUR, "error4")
    features = compute_features(FeatureStore.from_frames(**perturbed), as_of)

    assert features.loc[1, "error4_count_24h"] == 0
    assert features.loc[1, "error4_count_7d"] == 1


@pytest.mark.parametrize(
    "index, expected",
    [
        (0, {"3h": 1, "24h": 1, "7d": 1}),
        (2, {"3h": 3, "24h": 3, "7d": 3}),
        (23, {"3h": 3, "24h": 24, "7d": 24}),
        (167, {"3h": 3, "24h": 24, "7d": 168}),
        (300, {"3h": 3, "24h": 24, "7d": 168}),
    ],
)
def test_window_coverage_counts_the_grid_points_actually_spanned(
    store, index, expected
):
    features = compute_features(store, synthetic.grid()[index])
    for name, count in expected.items():
        assert (features[f"window_coverage_{name}"] == count).all()


def test_window_coverage_matches_the_readings_the_mean_averaged(
    real_subset_frames, subset_store
):
    """Cross-check against the brute-force window: coverage must equal the number
    of telemetry rows the rolling mean actually saw, which is the whole point of
    the feature."""
    telemetry = real_subset_frames["telemetry"]
    machine = int(subset_store.machine_ids[0])
    for index in [0, 1, 5, 30, 200, 1000]:
        as_of = subset_store.times[index]
        features = compute_features(subset_store, as_of)
        for name, window in TELEMETRY_WINDOWS.items():
            rows = telemetry[telemetry["machineID"] == machine]
            when = pd.to_datetime(rows["datetime"])
            n_readings = int(((when > as_of - window) & (when <= as_of)).sum())
            assert features.loc[machine, f"window_coverage_{name}"] == n_readings


def test_window_coverage_is_constant_after_the_widest_window_fills(store):
    """Honest limitation, pinned: these features vary only over the first 168
    hours. Across val and test they are constant and carry no signal."""
    for index in [200, 300, 399]:
        features = compute_features(store, synthetic.grid()[index])
        assert features["window_coverage_3h"].unique().tolist() == [3.0]
        assert features["window_coverage_24h"].unique().tolist() == [24.0]
        assert features["window_coverage_7d"].unique().tolist() == [168.0]


def test_partial_windows_at_the_start_of_the_series_give_zero_std(store):
    """ddof=0, deliberately. At the very first grid point a 3h window holds one
    reading; ddof=1 would make every std feature NaN there. See config.STD_DDOF."""
    features = compute_features(store, synthetic.grid()[0])
    std_columns = [c for c in FEATURE_COLUMNS if c.endswith(("_std_3h", "_std_24h"))]
    assert not features[std_columns].isna().any().any()
    assert (features[std_columns] == 0.0).all().all()


# ----------------------------------------------------------------------
# Store preconditions
# ----------------------------------------------------------------------


def test_off_grid_as_of_is_rejected(store):
    with pytest.raises(StoreError, match="not on the telemetry grid"):
        compute_features(store, synthetic.grid()[100] + pd.Timedelta(seconds=1))


def test_unbalanced_panel_is_rejected(frames):
    broken = {name: frame.copy() for name, frame in frames.items()}
    broken["telemetry"] = broken["telemetry"].iloc[:-1]
    with pytest.raises(StoreError, match="balanced panel"):
        FeatureStore.from_frames(**broken)


def test_non_hourly_grid_is_rejected(frames):
    broken = {name: frame.copy() for name, frame in frames.items()}
    telemetry = broken["telemetry"]
    keep = telemetry["datetime"] != synthetic.grid()[10]
    broken["telemetry"] = telemetry[keep]
    with pytest.raises(StoreError, match="hourly grid"):
        FeatureStore.from_frames(**broken)


def test_machine_absent_from_machines_table_is_rejected(frames):
    broken = {name: frame.copy() for name, frame in frames.items()}
    broken["machines"] = broken["machines"].iloc[:-1]
    with pytest.raises(StoreError):
        FeatureStore.from_frames(**broken)


# ----------------------------------------------------------------------
# Build outputs
# ----------------------------------------------------------------------


def test_build_split_is_deterministic(subset_store):
    first = build_split(subset_store, "train")
    second = build_split(subset_store, "train")
    assert content_hash(first) == content_hash(second)


def test_built_frame_has_the_declared_columns(subset_store):
    frame = build_split(subset_store, "train")
    assert list(frame.columns) == INDEX_COLUMNS + FEATURE_COLUMNS + LABEL_COLUMNS


def test_built_frame_has_no_null_features(subset_store):
    frame = build_split(subset_store, "train")
    assert not frame[FEATURE_COLUMNS].isna().any().any()


def test_built_frame_is_sorted_by_time_then_machine(subset_store):
    frame = build_split(subset_store, "train")
    expected = frame.sort_values(["datetime", "machineID"], kind="stable")
    pd.testing.assert_frame_equal(frame, expected.reset_index(drop=True))


def test_built_frame_has_one_row_per_machine_per_hour(subset_store):
    frame = build_split(subset_store, "train")
    counts = frame.groupby("datetime").size().unique()
    assert list(counts) == [subset_store.n_machines]


def test_labels_are_binary(subset_store):
    frame = build_split(subset_store, "train")
    for column in LABEL_COLUMNS:
        assert set(np.unique(frame[column])) <= {0, 1}


def test_summary_reports_positives_per_component(subset_store):
    frame = build_split(subset_store, "train")
    summary = summarise(frame, "train")
    assert set(summary["positives"]) == set(COMPONENTS)
    for component in COMPONENTS:
        assert summary["positives"][component] == int(
            frame[label_column(component)].sum()
        )


def test_content_hash_changes_when_a_value_changes(subset_store):
    """Guards the determinism test: a hash that never changes proves nothing."""
    frame = build_split(subset_store, "train")
    tampered = frame.copy()
    tampered.loc[0, FEATURE_COLUMNS[0]] += 1e-9
    assert content_hash(frame) != content_hash(tampered)


# ----------------------------------------------------------------------
# Shipped artefacts
# ----------------------------------------------------------------------


def test_manifest_records_a_hash_for_every_split():
    from src.features.build import MANIFEST

    if not MANIFEST.exists():
        pytest.skip(f"{MANIFEST} not built; run `make features`")

    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert set(payload.get("features", {})) == set(SPLIT_ORDER)
    for split in SPLIT_ORDER:
        entry = payload["features"][split]
        assert len(entry["sha256"]) == 64
        assert entry["rows"] > 0
        assert set(entry["positives"]) == set(COMPONENTS)
