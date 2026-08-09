"""Threshold selection from the stated cost assumption.

`docs/MILESTONE_3.md` section 5, and the cost assumption in `docs/DATA.md`
section 4: a missed failure is treated as 10x the cost of a false alarm.

That ratio is an assumption, not a measurement. Nothing in this repository has
ever touched a factory, so nobody here knows what a callout costs. The point of
selecting thresholds this way is not that 10:1 is right -- it is that every
threshold in the project traces to a number written down somewhere and can be
recomputed when that number changes. v1 used hardcoded 0.5 and 0.8 bands with no
stated rationale at all.

Cost is measured in units of one false alarm:

    cost(threshold) = FN(threshold) * ratio + FP(threshold)

True positives and true negatives are free, which is a simplification: acting on
a true positive still costs a part and a technician. It cancels out of the
comparison between thresholds only because the model does not choose how many
failures occur. Stated so the simplification is visible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: The ratio docs/DATA.md commits to.
DEFAULT_COST_RATIO = 10.0

#: Ratios the sensitivity table reports, to show how much of the conclusion is
#: carried by an assumption that cannot be measured here.
SENSITIVITY_RATIOS = (3.0, 10.0, 30.0)

#: Candidate thresholds. Dense at the low end because with positive rates under
#: 1% the useful operating points sit far below 0.5.
CANDIDATE_THRESHOLDS = np.unique(
    np.concatenate(
        [
            np.linspace(0.0005, 0.05, 100),
            np.linspace(0.05, 0.5, 90),
            np.linspace(0.5, 0.99, 50),
        ]
    )
)


@dataclass(frozen=True)
class ThresholdChoice:
    component: str
    cost_ratio: float
    threshold: float
    expected_cost: float
    cost_of_predicting_nothing: float
    precision: float
    recall: float
    true_positive: int
    false_positive: int
    false_negative: int

    @property
    def cost_reduction_vs_nothing(self) -> float:
        """Fraction of the do-nothing cost avoided. Unitless, so it survives the
        fact that the unit is an invented false-alarm cost."""
        if self.cost_of_predicting_nothing == 0:
            return float("nan")
        return 1 - self.expected_cost / self.cost_of_predicting_nothing


def cost_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    ratio: float,
    thresholds: np.ndarray = CANDIDATE_THRESHOLDS,
) -> tuple[np.ndarray, np.ndarray]:
    """Total cost at each candidate threshold. Vectorised over thresholds."""
    y_true = np.asarray(y_true).astype(bool)
    y_score = np.asarray(y_score, dtype=float)

    order = np.argsort(-y_score, kind="stable")
    sorted_scores = y_score[order]
    sorted_truth = y_true[order]

    # Cumulative counts of positives and negatives among the top-k scores.
    cumulative_tp = np.cumsum(sorted_truth)
    cumulative_fp = np.cumsum(~sorted_truth)
    total_positive = int(sorted_truth.sum())

    # For each threshold, how many rows score >= it.
    k = np.searchsorted(-sorted_scores, -np.asarray(thresholds), side="right")
    tp = np.where(k > 0, cumulative_tp[np.clip(k - 1, 0, None)], 0)
    fp = np.where(k > 0, cumulative_fp[np.clip(k - 1, 0, None)], 0)
    fn = total_positive - tp

    return np.asarray(thresholds), fn * ratio + fp


def select_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    component: str,
    ratio: float = DEFAULT_COST_RATIO,
    thresholds: np.ndarray = CANDIDATE_THRESHOLDS,
) -> ThresholdChoice:
    """The cost-minimising threshold. Ties break toward the higher threshold,
    which is the more conservative choice: fewer callouts for the same cost."""
    grid, costs = cost_curve(y_true, y_score, ratio, thresholds)
    best = int(np.flatnonzero(costs == costs.min())[-1])
    threshold = float(grid[best])

    y_true = np.asarray(y_true).astype(bool)
    predicted = np.asarray(y_score, dtype=float) >= threshold
    tp = int(np.sum(predicted & y_true))
    fp = int(np.sum(predicted & ~y_true))
    fn = int(np.sum(~predicted & y_true))

    return ThresholdChoice(
        component=component,
        cost_ratio=ratio,
        threshold=threshold,
        expected_cost=float(costs[best]),
        cost_of_predicting_nothing=float(int(y_true.sum()) * ratio),
        precision=float("nan") if tp + fp == 0 else tp / (tp + fp),
        recall=float("nan") if tp + fn == 0 else tp / (tp + fn),
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
    )


def sensitivity_table(
    y_true: np.ndarray,
    y_score: np.ndarray,
    component: str,
    ratios: tuple[float, ...] = SENSITIVITY_RATIOS,
) -> list[ThresholdChoice]:
    return [select_threshold(y_true, y_score, component, ratio) for ratio in ratios]
