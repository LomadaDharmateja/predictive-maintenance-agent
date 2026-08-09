"""No feature may use any record with `datetime > t`.

docs/DATA.md section 5.1. This file is the enforcement; everywhere else the rule
is a comment.

Every guard here is paired with a demonstration that it can fail. A test that
cannot fail is not a test, and an invariance assertion is the easiest kind to
write vacuously -- features that ignore recent data entirely are trivially
invariant to a record inserted after `t`. The paired negatives are named
`test_..._detects_...` and each one asserts that the same harness, pointed at a
deliberately broken implementation, raises.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.build import prediction_times
from src.features.compute import compute_features, compute_labels
from src.features.config import (
    COMPONENTS,
    EMBARGO,
    FEATURE_COLUMNS,
    LABEL_HORIZON,
    SPLIT_ORDER,
    SPLITS,
    label_column,
)
from src.features.store import NS_PER_HOUR, FeatureStore
from tests import synthetic

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"

HOUR = pd.Timedelta(hours=1)
ONE_SECOND = pd.Timedelta(seconds=1)

#: A prediction time deep enough into the synthetic grid that every window is
#: full and there is history on both sides.
AS_OF = synthetic.grid()[250]
MACHINE = 1


def store_of(frames: dict[str, pd.DataFrame]) -> FeatureStore:
    return FeatureStore.from_frames(**frames)


def assert_raises_assertion(check) -> None:
    """Run `check` and require that it fails. Used by the paired negatives."""
    with pytest.raises(AssertionError):
        check()


@pytest.fixture(scope="module")
def frames() -> dict[str, pd.DataFrame]:
    return synthetic.make_frames()


@pytest.fixture(scope="module")
def baseline(frames) -> pd.DataFrame:
    return compute_features(store_of(frames), AS_OF)


# ======================================================================
# 1. Future-record invariance
# ======================================================================


def assert_features_unchanged(baseline_features, perturbed_frames, as_of=AS_OF) -> None:
    after = compute_features(store_of(perturbed_frames), as_of)
    pd.testing.assert_frame_equal(baseline_features, after, check_exact=True)


@pytest.mark.parametrize("offset_hours", [1, 2, 24, 100])
def test_future_error_does_not_change_features(frames, baseline, offset_hours):
    perturbed = synthetic.add_error(frames, MACHINE, AS_OF + offset_hours * HOUR)
    assert_features_unchanged(baseline, perturbed)


@pytest.mark.parametrize("offset_hours", [1, 2, 24, 100])
def test_future_maint_does_not_change_features(frames, baseline, offset_hours):
    perturbed = synthetic.add_maint(frames, MACHINE, AS_OF + offset_hours * HOUR)
    assert_features_unchanged(baseline, perturbed)


@pytest.mark.parametrize("offset_hours", [1, 2, 24, 100])
def test_future_failure_does_not_change_features(frames, baseline, offset_hours):
    perturbed = synthetic.add_failure(frames, MACHINE, AS_OF + offset_hours * HOUR)
    assert_features_unchanged(baseline, perturbed)


@pytest.mark.parametrize("offset_hours", [1, 2, 24, 100])
def test_future_telemetry_does_not_change_features(frames, baseline, offset_hours):
    perturbed = synthetic.bump_telemetry(frames, MACHINE, AS_OF + offset_hours * HOUR)
    assert_features_unchanged(baseline, perturbed)


def test_a_future_failure_does_change_the_label(frames):
    """Anti-vacuity for the invariance tests above.

    The same record that must not move a feature must move the label. If this
    fails, the perturbation helpers are not inserting anything and the four tests
    above prove nothing.
    """
    before = compute_labels(store_of(frames), AS_OF)
    perturbed = synthetic.add_failure(frames, MACHINE, AS_OF + 6 * HOUR, "comp1")
    after = compute_labels(store_of(perturbed), AS_OF)

    assert before.loc[MACHINE, label_column("comp1")] == 0
    assert after.loc[MACHINE, label_column("comp1")] == 1


def _leaky_features(store: FeatureStore, as_of: pd.Timestamp) -> pd.DataFrame:
    """Deliberately broken: computes features one hour into the future."""
    index = store.index_of(as_of)
    return compute_features(store, store.times[index + 1])


@pytest.mark.parametrize(
    "perturb",
    [synthetic.add_error, synthetic.add_maint, synthetic.bump_telemetry],
    ids=["error", "maint", "telemetry"],
)
def test_invariance_harness_detects_a_leaky_feature_function(frames, perturb):
    """The harness above, pointed at a function that reads t + 1h, must fail."""
    leaky_before = _leaky_features(store_of(frames), AS_OF)
    perturbed = perturb(frames, MACHINE, AS_OF + HOUR)
    leaky_after = _leaky_features(store_of(perturbed), AS_OF)

    assert_raises_assertion(
        lambda: pd.testing.assert_frame_equal(
            leaky_before, leaky_after, check_exact=True
        )
    )


# ======================================================================
# 2. Boundary inclusion
# ======================================================================
#
# The window is closed at t. A record stamped exactly t is in; t + 1s is out.
# Without these, section 1 would pass for a feature set that ignored recent data
# altogether.


def test_error_at_exactly_t_is_counted(frames, baseline):
    perturbed = synthetic.add_error(frames, MACHINE, AS_OF, "error1")
    after = compute_features(store_of(perturbed), AS_OF)

    assert (
        after.loc[MACHINE, "error1_count_24h"]
        == baseline.loc[MACHINE, "error1_count_24h"] + 1
    )
    assert (
        after.loc[MACHINE, "error1_count_7d"]
        == baseline.loc[MACHINE, "error1_count_7d"] + 1
    )


def test_error_one_second_after_t_is_not_counted(frames, baseline):
    perturbed = synthetic.add_error(frames, MACHINE, AS_OF + ONE_SECOND, "error1")
    assert_features_unchanged(baseline, perturbed)


def test_maint_at_exactly_t_gives_zero_hours_since(frames, baseline):
    perturbed = synthetic.add_maint(frames, MACHINE, AS_OF, "comp1")
    after = compute_features(store_of(perturbed), AS_OF)

    assert baseline.loc[MACHINE, "hours_since_comp1"] > 0
    assert after.loc[MACHINE, "hours_since_comp1"] == 0.0


def test_maint_one_second_after_t_is_not_seen(frames, baseline):
    perturbed = synthetic.add_maint(frames, MACHINE, AS_OF + ONE_SECOND, "comp1")
    assert_features_unchanged(baseline, perturbed)


def test_telemetry_at_exactly_t_moves_the_rolling_mean(frames, baseline):
    """Telemetry has no `t + 1s` case: the panel is a strict hourly grid, so an
    off-grid reading cannot exist. The `t + 1h` half of this boundary is covered
    by `test_future_telemetry_does_not_change_features`."""
    perturbed = synthetic.bump_telemetry(frames, MACHINE, AS_OF, delta=50.0)
    after = compute_features(store_of(perturbed), AS_OF)

    # A 50.0 bump on one of three readings moves the 3h mean by 50/3.
    assert after.loc[MACHINE, "volt_mean_3h"] == pytest.approx(
        baseline.loc[MACHINE, "volt_mean_3h"] + 50.0 / 3
    )
    assert after.loc[MACHINE, "volt_mean_24h"] == pytest.approx(
        baseline.loc[MACHINE, "volt_mean_24h"] + 50.0 / 24
    )


def test_an_error_at_the_far_edge_of_the_window_is_counted(frames, baseline):
    """The window opens at t - 24h exclusive, so t - 23h is the oldest hour in
    it. Pins the far edge as well as the near one."""
    inside = synthetic.add_error(frames, MACHINE, AS_OF - 23 * HOUR, "error1")
    outside = synthetic.add_error(frames, MACHINE, AS_OF - 24 * HOUR, "error1")

    counted = compute_features(store_of(inside), AS_OF)
    not_counted = compute_features(store_of(outside), AS_OF)

    assert (
        counted.loc[MACHINE, "error1_count_24h"]
        == baseline.loc[MACHINE, "error1_count_24h"] + 1
    )
    assert (
        not_counted.loc[MACHINE, "error1_count_24h"]
        == baseline.loc[MACHINE, "error1_count_24h"]
    )


def _recency_under_the_superseded_rule(
    store: FeatureStore, as_of: pd.Timestamp, machine: int, component: str
) -> float:
    """`hours_since` under the rule DATA.md used to state: `datetime < t`.

    Kept only so the boundary test above can be shown to fail against it.
    """
    column = store.machine_ids.tolist().index(machine)
    index = store.index_of(as_of)
    last_ns = store.maint_last_ns[component][index, column]
    if last_ns == as_of.value:  # a record exactly at t, which the old rule dropped
        previous = store.maint_last_ns[component][index - 1, column]
        last_ns = previous
    return (as_of.value - last_ns) / NS_PER_HOUR


def test_boundary_check_detects_the_superseded_strictly_less_than_rule(frames):
    """Under `datetime < t`, a replacement stamped exactly `t` is invisible and
    `hours_since` stays positive. The section-2 assertion must reject that."""
    perturbed = synthetic.add_maint(frames, MACHINE, AS_OF, "comp1")
    store = store_of(perturbed)
    old_rule = _recency_under_the_superseded_rule(store, AS_OF, MACHINE, "comp1")

    assert old_rule > 0
    assert_raises_assertion(lambda: _assert_equal(old_rule, 0.0))


def _assert_equal(actual, expected) -> None:
    assert actual == expected, f"{actual!r} != {expected!r}"


# ======================================================================
# 3. Failure-coincident maint exclusion
# ======================================================================


@pytest.mark.parametrize("offset_hours", [1, 6, 23, 24])
def test_maint_coincident_with_an_in_window_failure_is_excluded(
    frames, baseline, offset_hours
):
    """A failure inside the label window drags a `maint` record with it at the
    same timestamp -- docs/DATA.md section 5.1, 743 of 761 records. That maint
    record must not reach any feature, or the model is handed its own label."""
    when = AS_OF + offset_hours * HOUR
    perturbed = synthetic.add_failure(frames, MACHINE, when, "comp1")
    perturbed = synthetic.add_maint(perturbed, MACHINE, when, "comp1")
    store = store_of(perturbed)

    # The setup must be real: the label has to be positive, otherwise this test
    # is asserting invariance to a record that is not in the window at all.
    labels = compute_labels(store, AS_OF)
    assert labels.loc[MACHINE, label_column("comp1")] == 1

    after = compute_features(store, AS_OF)
    pd.testing.assert_frame_equal(baseline, after, check_exact=True)


def test_coincident_maint_check_detects_a_recency_feature_that_looks_ahead(frames):
    """A `hours_since` built from the nearest replacement in either direction --
    a plausible-looking bug -- does reflect the coincident record. The test above
    must reject it."""
    when = AS_OF + 6 * HOUR
    perturbed = synthetic.add_failure(frames, MACHINE, when, "comp1")
    perturbed = synthetic.add_maint(perturbed, MACHINE, when, "comp1")

    def nearest_either_direction(maint: pd.DataFrame) -> float:
        rows = maint[(maint["machineID"] == MACHINE) & (maint["comp"] == "comp1")]
        deltas = (pd.to_datetime(rows["datetime"]) - AS_OF).abs()
        return deltas.min().total_seconds() / 3600

    before = nearest_either_direction(frames["maint"])
    after = nearest_either_direction(perturbed["maint"])

    assert after < before
    assert_raises_assertion(lambda: _assert_equal(after, before))


# ======================================================================
# 4. Split ordering
# ======================================================================


@pytest.fixture(scope="module")
def split_times(real_store) -> dict[str, pd.DatetimeIndex]:
    return {split: prediction_times(real_store, split) for split in SPLIT_ORDER}


@pytest.mark.parametrize("earlier, later", [("train", "val"), ("val", "test")])
def test_embargo_separates_consecutive_splits(split_times, earlier, later):
    """Note what this alone does *not* catch: with `EMBARGO = 0` the inequality
    is trivially satisfied, because adjacent splits already abut. Setting the
    embargo to zero was tried, and it is `test_embargo_is_derived_from_the_label_horizon`
    and `test_no_label_window_leaves_its_split` that fail. All three are needed.
    """
    assert split_times[earlier].max() + EMBARGO <= split_times[later].min()


def test_splits_do_not_overlap(split_times):
    for earlier, later in [("train", "val"), ("val", "test")]:
        assert split_times[earlier].max() < split_times[later].min()


def test_embargo_is_derived_from_the_label_horizon():
    """Not a magic 24. If the horizon moves, the embargo moves with it."""
    assert EMBARGO == LABEL_HORIZON


def test_split_ordering_check_detects_a_missing_embargo(real_store):
    """Without the embargo, train's last row and val's first row are adjacent
    and the section-4 assertion must fail."""
    train_end = SPLITS["train"][1]
    val_start = SPLITS["val"][0]
    no_embargo_train_max = real_store.times[real_store.times <= train_end].max()
    val_min = real_store.times[real_store.times >= val_start].min()

    assert_raises_assertion(
        lambda: _assert_true(no_embargo_train_max + EMBARGO <= val_min)
    )


def _assert_true(condition) -> None:
    assert condition


# ======================================================================
# 5. Label window containment
# ======================================================================


@pytest.mark.parametrize("split", SPLIT_ORDER)
def test_no_label_window_leaves_its_split(split_times, split):
    _, end = SPLITS[split]
    assert split_times[split].max() + LABEL_HORIZON <= end


@pytest.mark.parametrize("split", SPLIT_ORDER)
def test_split_starts_inside_its_own_period(split_times, split):
    start, _ = SPLITS[split]
    assert split_times[split].min() >= start


def test_containment_check_detects_a_row_at_the_split_edge(real_store):
    train_end = SPLITS["train"][1]
    edge = real_store.times[real_store.times <= train_end].max()

    assert_raises_assertion(lambda: _assert_true(edge + LABEL_HORIZON <= train_end))


# ======================================================================
# 6. No shuffled splitting
# ======================================================================

#: Names that would introduce a random split. Matched against imported and
#: called identifiers in src/, not against raw text, so a mention in a comment
#: or a docstring does not trip it.
SHUFFLING_NAMES = {
    "train_test_split",
    "ShuffleSplit",
    "StratifiedShuffleSplit",
    "KFold",
    "StratifiedKFold",
    "GroupKFold",
    "cross_val_score",
    "cross_val_predict",
    "shuffle",
    "permutation",
}


def shuffling_calls_in(path: Path) -> list[str]:
    """Identifiers from SHUFFLING_NAMES that this module imports or calls."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = getattr(func, "id", None) or getattr(func, "attr", None)
            if name in SHUFFLING_NAMES:
                found.append(name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name.split(".")[-1] in SHUFFLING_NAMES:
                    found.append(alias.name)
    return found


def test_no_shuffled_splitting_anywhere_in_src():
    offenders = {
        path.relative_to(REPO_ROOT).as_posix(): calls
        for path in sorted(SRC.rglob("*.py"))
        if (calls := shuffling_calls_in(path))
    }
    assert offenders == {}


def test_sample_and_random_state_are_absent_from_the_feature_layer():
    """`DataFrame.sample` and a stray `random_state=` are the other two ways a
    shuffle sneaks in. Neither appears; feature computation has no randomness at
    all, which is why nothing here is seeded."""
    pattern = re.compile(r"\.sample\(|random_state\s*=|np\.random")
    offenders = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in sorted((SRC / "features").rglob("*.py"))
        if pattern.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []


def test_shuffle_scanner_detects_a_planted_call(tmp_path):
    planted = tmp_path / "planted.py"
    planted.write_text(
        "from sklearn.model_selection import train_test_split\n"
        "def go(x, y):\n"
        "    return train_test_split(x, y, test_size=0.2)\n",
        encoding="utf-8",
    )
    assert "train_test_split" in shuffling_calls_in(planted)


def test_shuffle_scanner_ignores_a_mention_in_a_comment(tmp_path):
    """Guards the guard: a text search would flag docs/DATA.md's own prose about
    not using random splits, and a test that cries wolf gets deleted."""
    innocent = tmp_path / "innocent.py"
    innocent.write_text(
        '"""No train_test_split is used anywhere in this project."""\n'
        "# train_test_split would be wrong here\n"
        "VALUE = 1\n",
        encoding="utf-8",
    )
    assert shuffling_calls_in(innocent) == []


# ======================================================================
# 7. Column-order invariance
# ======================================================================


def permute_columns(frames: dict[str, pd.DataFrame]):
    """Reverse each frame's column order. Deterministic on purpose -- a random
    permutation would make a failure hard to reproduce."""
    return {name: frame[list(frame.columns)[::-1]] for name, frame in frames.items()}


def test_permuting_input_columns_does_not_change_features(frames, baseline):
    permuted = permute_columns(frames)
    for name, frame in permuted.items():
        assert list(frame.columns) != list(frames[name].columns)

    after = compute_features(store_of(permuted), AS_OF)
    pd.testing.assert_frame_equal(baseline, after, check_exact=True)


def test_permuting_input_columns_does_not_change_labels(frames):
    before = compute_labels(store_of(frames), AS_OF)
    after = compute_labels(store_of(permute_columns(frames)), AS_OF)
    pd.testing.assert_frame_equal(before, after, check_exact=True)


def test_output_column_order_is_fixed(frames, baseline):
    assert list(baseline.columns) == FEATURE_COLUMNS


def test_column_order_check_detects_positional_selection(baseline):
    """v1 of this project read model inputs by position. Under a reordering that
    returns different numbers silently instead of raising -- exactly the defect
    this milestone exists to prevent.

    Selecting the same three features by name survives the reordering; selecting
    the first three columns by position does not.
    """
    reordered = baseline[FEATURE_COLUMNS[::-1]]
    names = FEATURE_COLUMNS[:3]

    by_name_upright = baseline[names].to_numpy()
    by_name_reordered = reordered[names].to_numpy()
    np.testing.assert_array_equal(by_name_upright, by_name_reordered)

    by_position_upright = baseline.iloc[:, :3].to_numpy()
    by_position_reordered = reordered.iloc[:, :3].to_numpy()
    assert_raises_assertion(
        lambda: np.testing.assert_array_equal(
            by_position_upright, by_position_reordered
        )
    )


# ======================================================================
# Real data: the same invariance, on the shipped tables
# ======================================================================


@pytest.mark.parametrize(
    "perturb", [synthetic.add_error, synthetic.add_maint, synthetic.add_failure],
    ids=["error", "maint", "failure"],
)
def test_future_record_invariance_on_real_data(real_subset_frames, perturb):
    """The synthetic checks prove the code is right. This proves the code is
    right on data with the real distribution of gaps, ties and pre-grid maint."""
    latest = pd.Timestamp(real_subset_frames["telemetry"]["datetime"].max())
    as_of = (latest - pd.Timedelta(days=10)).floor("h")
    machine = int(real_subset_frames["machines"]["machineID"].iloc[0])

    before = compute_features(store_of(real_subset_frames), as_of)
    perturbed = perturb(real_subset_frames, machine, as_of + HOUR)
    after = compute_features(store_of(perturbed), as_of)

    pd.testing.assert_frame_equal(before, after, check_exact=True)


def test_every_component_has_prior_maint_on_real_data(real_store):
    """`hours_since_*` is NaN when a machine has no earlier replacement. The
    build rejects NaN features, so this documents why none occur."""
    first = real_store.times[0]
    features = compute_features(real_store, first)
    columns = [f"hours_since_{component}" for component in COMPONENTS]
    assert not features[columns].isna().any().any()
