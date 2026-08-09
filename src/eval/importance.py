"""Permutation importance on validation.

`docs/MILESTONE_3.md` section 6. Not split-gain importance: gain counts how often
a feature was chosen for a split, which is biased toward features with many
distinct values and says nothing about whether the model's predictions would
suffer without it. Permutation importance answers the question actually being
asked -- how much does the score drop when this column is made uninformative.

Scored by average precision, matching the primary metric. Measured on
validation, never on training, because importance on data the model memorised
tells you about memorisation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

#: Repeats per feature. Each is a fresh shuffle; the spread across repeats is
#: what the interval is built from.
N_REPEATS = 10
SEED = 20240605
CONFIDENCE = 0.95


@dataclass
class FeatureImportance:
    feature: str
    mean_drop: float
    std_drop: float
    low: float
    high: float

    @property
    def is_significant(self) -> bool:
        """The interval excludes zero. Anything else is noise dressed as a rank."""
        return self.low > 0


def permutation_importance(
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    n_repeats: int = N_REPEATS,
    seed: int = SEED,
) -> list[FeatureImportance]:
    """Drop in average precision when each column is shuffled, one at a time."""
    y = np.asarray(y).astype(int)
    values = X.to_numpy(dtype=float, copy=True)
    baseline = float(average_precision_score(y, model.predict_proba(values)[:, 1]))

    rng = np.random.default_rng(seed)
    results: list[FeatureImportance] = []

    for position, feature in enumerate(X.columns):
        original = values[:, position].copy()
        drops = np.empty(n_repeats)
        for repeat in range(n_repeats):
            values[:, position] = rng.permutation(original)
            score = float(average_precision_score(y, model.predict_proba(values)[:, 1]))
            drops[repeat] = baseline - score
        values[:, position] = original

        mean, std = float(drops.mean()), float(drops.std(ddof=1))
        # Normal interval on the mean across repeats. This is uncertainty from
        # the shuffling, not from the sample -- a distinction worth keeping
        # straight, and stated in docs/EVALUATION.md.
        half_width = 1.96 * std / np.sqrt(n_repeats)
        results.append(
            FeatureImportance(
                feature=feature,
                mean_drop=mean,
                std_drop=std,
                low=mean - half_width,
                high=mean + half_width,
            )
        )

    results.sort(key=lambda r: r.mean_drop, reverse=True)
    return results


def baseline_score(model, X: pd.DataFrame, y: np.ndarray) -> float:
    return float(average_precision_score(np.asarray(y).astype(int),
                                         model.predict_proba(X.to_numpy(dtype=float))[:, 1]))
