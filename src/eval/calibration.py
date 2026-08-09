"""Calibration: do the probabilities mean what they say?

`docs/MILESTONE_3.md` section 4. A later milestone has an agent acting on these
numbers, so "0.3" needs to mean roughly three failures in ten, not merely a
higher rank than "0.2".

An honesty note that the numbers below depend on. The calibrator is fitted on
validation, as specified, and the before/after comparison is therefore reported
on the same data the calibrator saw. Those after-figures are in-sample for the
calibrator and optimistic. The out-of-sample check is the test evaluation, which
applies this validation-fitted calibrator to data it has never seen; that is the
number to trust, and it is reported once at the end.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

#: Bins for the reliability curve. Quantile bins, not uniform: with almost all
#: predicted probabilities below 0.05, uniform bins put every row in the first
#: one and the curve says nothing.
N_BINS = 10


@dataclass
class CalibrationReport:
    component: str
    method: str
    brier: float
    brier_base_rate: float
    brier_skill_score: float
    bin_centres: np.ndarray
    bin_observed: np.ndarray
    bin_counts: np.ndarray
    max_deviation: float

    @property
    def beats_base_rate(self) -> bool:
        return self.brier < self.brier_base_rate


def brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_prob - y_true) ** 2))


def base_rate_brier(y_true: np.ndarray) -> float:
    """Brier score of always predicting the base rate.

    This, not zero, is the reference. On a 0.5% positive rate a model that
    outputs 0.005 for every row scores 0.00497 -- a small number that reflects
    the rarity of the event, not skill."""
    rate = float(np.mean(y_true))
    return rate * (1 - rate)


def reliability(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = N_BINS
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Observed frequency against predicted probability, in quantile bins."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    edges = np.unique(np.quantile(y_prob, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:  # degenerate scores, e.g. a constant baseline
        return np.array([float(y_prob.mean())]), np.array([float(y_true.mean())]), np.array([len(y_true)])

    which = np.clip(np.digitize(y_prob, edges[1:-1], right=True), 0, len(edges) - 2)
    centres, observed, counts = [], [], []
    for b in range(len(edges) - 1):
        mask = which == b
        if not mask.any():
            continue
        centres.append(float(y_prob[mask].mean()))
        observed.append(float(y_true[mask].mean()))
        counts.append(int(mask.sum()))
    return np.array(centres), np.array(observed), np.array(counts)


def assess(
    y_true: np.ndarray, y_prob: np.ndarray, component: str, method: str
) -> CalibrationReport:
    centres, observed, counts = reliability(y_true, y_prob)
    score = brier(y_true, y_prob)
    reference = base_rate_brier(y_true)
    return CalibrationReport(
        component=component,
        method=method,
        brier=score,
        brier_base_rate=reference,
        brier_skill_score=float("nan") if reference == 0 else 1 - score / reference,
        bin_centres=centres,
        bin_observed=observed,
        bin_counts=counts,
        max_deviation=float(np.max(np.abs(centres - observed))) if len(centres) else float("nan"),
    )


def fit_isotonic(y_true: np.ndarray, y_prob: np.ndarray) -> IsotonicRegression:
    """Isotonic, not Platt, as the default.

    Platt fits a two-parameter sigmoid, which assumes the miscalibration has a
    particular shape. Isotonic assumes only monotonicity. With 72,000 validation
    rows there is enough data to afford the extra flexibility, and gradient
    boosting miscalibration is not reliably sigmoidal.
    """
    model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    model.fit(np.asarray(y_prob, dtype=float), np.asarray(y_true, dtype=float))
    return model


def fit_platt(y_true: np.ndarray, y_prob: np.ndarray) -> LogisticRegression:
    """Platt scaling, reported alongside isotonic so the choice is visible."""
    model = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
    model.fit(np.asarray(y_prob, dtype=float).reshape(-1, 1), np.asarray(y_true).astype(int))
    return model


def apply_calibrator(calibrator, y_prob: np.ndarray) -> np.ndarray:
    y_prob = np.asarray(y_prob, dtype=float)
    if isinstance(calibrator, IsotonicRegression):
        return np.asarray(calibrator.predict(y_prob), dtype=float)
    return np.asarray(calibrator.predict_proba(y_prob.reshape(-1, 1))[:, 1], dtype=float)
