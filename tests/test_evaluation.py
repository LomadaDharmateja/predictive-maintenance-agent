"""Tests for the evaluation layer.

Most of these run on small hand-built arrays rather than the real splits, so
they execute in CI where `data/raw/` is absent. Where a property can be checked
against a value computed by hand, it is: a metric function tested only against
itself proves nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.eval import calibration as calib
from src.eval.baselines import (
    BASELINE_THRESHOLDS,
    MATCHED_ERROR,
    any_error_24h_scores,
    majority_class_scores,
    matched_error_24h_scores,
)
from src.eval.importance import permutation_importance
from src.eval.metrics import (
    bootstrap_intervals,
    event_clusters,
    point_metrics,
    safe_divide,
)
from src.eval.report import fmt, fmt_ci
from src.eval.thresholds import (
    DEFAULT_COST_RATIO,
    SENSITIVITY_RATIOS,
    cost_curve,
    select_threshold,
    sensitivity_table,
)
from src.features.config import COMPONENTS, ERROR_IDS

#: These tests are about the clustering and metric mechanics, not about the
#: project's operational horizon. Pinned locally so moving LABEL_HORIZON (24h in
#: Milestone 3, 14 days in 3B) cannot silently reshape the fixture.
CLUSTER_HORIZON = pd.Timedelta(hours=24)


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------


def test_undefined_precision_is_nan_not_zero():
    """The majority baseline predicts no positives. Precision has no
    denominator there, and reporting 0.0 would be a claim about performance
    rather than the absence of one."""
    assert np.isnan(safe_divide(0, 0))
    assert safe_divide(1, 4) == 0.25


def test_point_metrics_against_a_hand_computed_confusion_matrix():
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    y_score = np.array([0.9, 0.8, 0.2, 0.7, 0.1, 0.1, 0.1, 0.1])
    metrics = point_metrics(y_true, y_score, threshold=0.5, component="comp1")

    # Predicted positive: 0.9, 0.8, 0.7 -> TP=2, FP=1. Remaining: FN=1, TN=4.
    assert (metrics.true_positive, metrics.false_positive) == (2, 1)
    assert (metrics.false_negative, metrics.true_negative) == (1, 4)
    assert metrics.precision == pytest.approx(2 / 3)
    assert metrics.recall == pytest.approx(2 / 3)
    assert metrics.f1 == pytest.approx(2 / 3)


def test_majority_baseline_has_undefined_precision_and_zero_recall():
    frame = pd.DataFrame({"x": range(100)})
    y = np.zeros(100, dtype=int)
    y[:3] = 1
    metrics = point_metrics(y, majority_class_scores(frame), 0.5, "comp1")

    assert metrics.true_positive == 0
    assert metrics.false_positive == 0
    assert np.isnan(metrics.precision)
    assert metrics.recall == 0.0


def test_pr_auc_of_a_constant_score_is_the_positive_rate():
    """The no-skill floor drawn on every PR curve. If this were 0.5 the curves
    would be mislabelled."""
    y = np.zeros(1000, dtype=int)
    y[:40] = 1
    metrics = point_metrics(y, np.zeros(1000), 0.5, "comp1")
    assert metrics.pr_auc == pytest.approx(0.04, abs=1e-9)


# ----------------------------------------------------------------------
# Bootstrap clustering
# ----------------------------------------------------------------------


@pytest.fixture
def clustering_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Six machines, one failure each, so the event-level bootstrap has six
    clusters to resample. With a single event it would have nothing to vary --
    which is itself correct behaviour, and is why the real test period's
    intervals are as wide as they are."""
    times = pd.date_range("2015-05-01 00:00", periods=240, freq="h")
    machines = [1, 2, 3, 4, 5, 6]
    failure_times = {m: times[40 + 30 * i] for i, m in enumerate(machines)}

    rows = []
    for machine in machines:
        failure_at = failure_times[machine]
        for when in times:
            positive = int(failure_at - CLUSTER_HORIZON <= when < failure_at)
            rows.append(
                {"machineID": machine, "datetime": when, "label_comp1": positive}
            )
    frame = pd.DataFrame(rows).sort_values(
        ["datetime", "machineID"], ignore_index=True
    )
    failures = pd.DataFrame(
        [
            {"machineID": machine, "datetime": when, "failure": "comp1"}
            for machine, when in failure_times.items()
        ]
    )
    return frame, failures


def test_positive_rows_from_one_event_share_a_cluster(clustering_frame):
    frame, failures = clustering_frame
    clusters = event_clusters(frame, "comp1", failures, CLUSTER_HORIZON)
    positive = frame["label_comp1"].to_numpy().astype(bool)

    assert positive.sum() == 6 * 24
    assert len(np.unique(clusters[positive])) == 6, (
        "all 24 rows produced by one failure must resample together, or the "
        "interval is several times too narrow"
    )


def test_negative_rows_cluster_by_machine_day(clustering_frame):
    frame, failures = clustering_frame
    clusters = event_clusters(frame, "comp1", failures, CLUSTER_HORIZON)
    negative = ~frame["label_comp1"].to_numpy().astype(bool)
    machine_days = (
        frame.loc[negative]
        .assign(day=lambda f: f["datetime"].dt.floor("D"))
        .groupby(["machineID", "day"])
        .ngroups
    )
    assert len(np.unique(clusters[negative])) == machine_days


def test_clustered_bootstrap_is_wider_than_a_row_bootstrap(clustering_frame):
    """The reason for clustering at all. If this fails, the intervals in
    EVALUATION.md are overstating precision."""
    frame, failures = clustering_frame
    y = frame["label_comp1"].to_numpy()
    clustered = event_clusters(frame, "comp1", failures, CLUSTER_HORIZON)
    per_row = np.arange(len(frame))

    # Scores are drawn per cluster, not per row. That is the situation the
    # clustering exists for: the ~24 rows an event produces are near-identical,
    # so the model either catches that event or misses all of it. With
    # independent per-row scores there is nothing for clustering to correct, and
    # the row bootstrap would legitimately look wider.
    rng = np.random.default_rng(0)
    cluster_score = rng.uniform(0.3, 0.9, clustered.max() + 1)
    score = np.where(y == 1, cluster_score[clustered], rng.uniform(0.0, 0.45, len(y)))

    wide = bootstrap_intervals(y, score, clustered, 0.5, n_resamples=300)["recall"]
    narrow = bootstrap_intervals(y, score, per_row, 0.5, n_resamples=300)["recall"]
    assert (wide[1] - wide[0]) > 0, "the clustered interval must not be degenerate"
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


# ----------------------------------------------------------------------
# Baselines
# ----------------------------------------------------------------------


def test_matched_error_mapping_covers_every_component():
    assert set(MATCHED_ERROR) == set(COMPONENTS)
    assert set(MATCHED_ERROR.values()) <= set(ERROR_IDS)


def test_matched_error_mapping_is_not_the_obvious_one():
    """comp3 -> error4 and comp4 -> error5, measured rather than assumed. If
    someone 'tidies' this to comp3 -> error3 the baseline silently becomes
    meaningless, so the surprise is pinned."""
    assert MATCHED_ERROR["comp3"] == "error4"
    assert MATCHED_ERROR["comp4"] == "error5"


def test_baseline_scores_are_binary_and_aligned():
    frame = pd.DataFrame(
        {f"{e}_count_24h": [0.0, 1.0, 0.0, 2.0] for e in ERROR_IDS}
    )
    frame["error1_count_24h"] = [0.0, 1.0, 0.0, 0.0]

    any_error = any_error_24h_scores(frame)
    matched = matched_error_24h_scores(frame, "comp1")

    assert set(np.unique(any_error)) <= {0.0, 1.0}
    assert set(np.unique(matched)) <= {0.0, 1.0}
    assert matched.tolist() == [0.0, 1.0, 0.0, 0.0]
    # Row 3 has other error codes but not error1.
    assert any_error[3] == 1.0 and matched[3] == 0.0


def test_every_baseline_has_a_threshold():
    from src.eval.baselines import BASELINES, COMPONENT_BASELINES

    for name in list(BASELINES) + list(COMPONENT_BASELINES):
        assert name in BASELINE_THRESHOLDS


# ----------------------------------------------------------------------
# Thresholds and cost
# ----------------------------------------------------------------------


@pytest.fixture
def separable() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1)
    y = np.zeros(4000, dtype=int)
    y[:120] = 1
    score = np.where(y == 1, rng.uniform(0.4, 1.0, 4000), rng.uniform(0.0, 0.6, 4000))
    return y, score


def test_cost_curve_matches_a_direct_count(separable):
    y, score = separable
    thresholds, costs = cost_curve(y, score, DEFAULT_COST_RATIO)
    for probe in [0.1, 0.3, 0.5, 0.8]:
        index = int(np.argmin(np.abs(thresholds - probe)))
        threshold = thresholds[index]
        predicted = score >= threshold
        expected = (
            np.sum(~predicted & (y == 1)) * DEFAULT_COST_RATIO
            + np.sum(predicted & (y == 0))
        )
        assert costs[index] == pytest.approx(expected)


def test_selected_threshold_minimises_the_curve(separable):
    y, score = separable
    choice = select_threshold(y, score, "comp1", DEFAULT_COST_RATIO)
    _, costs = cost_curve(y, score, DEFAULT_COST_RATIO)
    assert choice.expected_cost == pytest.approx(costs.min())


def test_a_higher_miss_cost_never_raises_the_threshold(separable):
    """More expensive misses can only make the model more willing to alarm."""
    y, score = separable
    choices = sensitivity_table(y, score, "comp1", SENSITIVITY_RATIOS)
    thresholds = [c.threshold for c in choices]
    assert thresholds == sorted(thresholds, reverse=True), thresholds


def test_a_higher_miss_cost_never_lowers_recall(separable):
    y, score = separable
    recalls = [c.recall for c in sensitivity_table(y, score, "comp1", SENSITIVITY_RATIOS)]
    assert recalls == sorted(recalls)


def test_threshold_choice_records_the_ratio_it_came_from():
    """Every threshold in this project has to trace to a stated assumption."""
    y = np.array([1, 0, 1, 0, 0, 0])
    score = np.array([0.9, 0.4, 0.8, 0.2, 0.1, 0.05])
    choice = select_threshold(y, score, "comp2", 30.0)
    assert choice.cost_ratio == 30.0
    assert choice.component == "comp2"


# ----------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------


def test_brier_of_a_perfect_predictor_is_zero():
    y = np.array([1, 0, 1, 0])
    assert calib.brier(y, y.astype(float)) == 0.0


def test_base_rate_reference_is_p_times_one_minus_p():
    y = np.zeros(1000)
    y[:25] = 1
    assert calib.base_rate_brier(y) == pytest.approx(0.025 * 0.975)


def test_isotonic_calibration_does_not_worsen_the_brier_score():
    rng = np.random.default_rng(3)
    y = rng.binomial(1, 0.05, 5000)
    # Deliberately miscalibrated: scores inflated well above the base rate.
    score = np.clip(y * 0.5 + rng.uniform(0, 0.5, 5000), 0, 1)

    before = calib.brier(y, score)
    calibrator = calib.fit_isotonic(y, score)
    after = calib.brier(y, calib.apply_calibrator(calibrator, score))
    assert after <= before


def test_reliability_bins_are_populated():
    rng = np.random.default_rng(4)
    y = rng.binomial(1, 0.1, 2000)
    score = rng.uniform(0, 1, 2000)
    centres, observed, counts = calib.reliability(y, score)
    assert len(centres) == len(observed) == len(counts)
    assert counts.sum() == 2000


def test_reliability_survives_a_constant_score():
    """The majority baseline emits a constant. A quantile binner must not divide
    by zero on it."""
    y = np.zeros(100, dtype=int)
    y[:5] = 1
    centres, observed, counts = calib.reliability(y, np.zeros(100))
    assert len(centres) == 1
    assert counts.sum() == 100


# ----------------------------------------------------------------------
# Permutation importance
# ----------------------------------------------------------------------


class _OneFeatureModel:
    """Predicts from column 0 alone. Importance must find that."""

    def predict_proba(self, X):
        p = np.clip(X[:, 0], 0.001, 0.999)
        return np.column_stack([1 - p, p])


def test_permutation_importance_finds_the_only_feature_used():
    rng = np.random.default_rng(5)
    n = 2000
    signal = rng.uniform(0, 1, n)
    y = rng.binomial(1, signal * 0.2)
    X = pd.DataFrame(
        {"used": signal, "noise_a": rng.uniform(0, 1, n), "noise_b": rng.uniform(0, 1, n)}
    )

    results = permutation_importance(_OneFeatureModel(), X, y, n_repeats=5, seed=0)
    assert results[0].feature == "used"
    assert results[0].mean_drop > 0
    assert results[0].is_significant

    unused = {r.feature: r for r in results if r.feature != "used"}
    for item in unused.values():
        assert item.mean_drop < results[0].mean_drop


# ----------------------------------------------------------------------
# Report formatting
# ----------------------------------------------------------------------


def test_undefined_values_render_as_undefined_not_zero():
    assert fmt(float("nan")) == "undefined"
    assert fmt_ci(float("nan"), (0.1, 0.2)) == "undefined"


def test_interval_rendering():
    assert fmt_ci(0.625, (0.45, 0.78)) == "0.625 (0.450-0.780)"
