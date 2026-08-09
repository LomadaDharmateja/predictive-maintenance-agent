"""The floors every model has to clear.

`docs/MILESTONE_3.md` section 1. A gradient boosting PR-AUC is a number without
meaning until you know what predicting nothing scores, and what a maintenance
team with a spreadsheet scores.

Both baselines here produce a score array in the same shape a model would, so
every downstream function -- PR-AUC, the bootstrap, the cost curve -- treats them
identically. A baseline evaluated by a different code path is not a comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.config import ERROR_IDS


def majority_class_scores(frame: pd.DataFrame) -> np.ndarray:
    """Baseline 1: always predict negative.

    Constant 0.0. Its PR-AUC is not 0 -- `average_precision_score` on a constant
    score returns the positive rate, which is the correct no-skill floor and the
    line drawn on every PR curve. Precision is undefined (no positives predicted)
    and recall is 0.
    """
    return np.zeros(len(frame), dtype=float)


def any_error_24h_scores(frame: pd.DataFrame) -> np.ndarray:
    """Baseline 2: positive if the machine logged any error in the last 24 hours.

    This is what a maintenance team could run off the error log with a
    spreadsheet and no model at all. It is the honest thing to beat.

    Binary rather than graded, deliberately: making it a count would turn it into
    a weak model rather than the rule it is meant to represent. The consequence
    is that its PR curve has one interior operating point.
    """
    counts = np.zeros(len(frame), dtype=float)
    for error_id in ERROR_IDS:
        counts += frame[f"{error_id}_count_24h"].to_numpy(dtype=float)
    return (counts > 0).astype(float)


def error_count_24h_scores(frame: pd.DataFrame) -> np.ndarray:
    """A graded variant of baseline 2, reported alongside it.

    Not required by the milestone. It is included because the binary rule's
    single operating point makes its PR-AUC hard to read, and the graded version
    shows how much of the rule's value is in "any error" versus "how many".
    """
    counts = np.zeros(len(frame), dtype=float)
    for error_id in ERROR_IDS:
        counts += frame[f"{error_id}_count_24h"].to_numpy(dtype=float)
    return counts


#: Each component's fault signature fires one specific error code. Measured on
#: validation, not assumed: the matched code is present in 100% of positive rows
#: for every component. Note comp3 -> error4 and comp4 -> error5; the numbering
#: does not line up, and assuming it did would have produced a silently wrong
#: baseline. See docs/EVALUATION.md section 0.
MATCHED_ERROR = {
    "comp1": "error1",
    "comp2": "error2",
    "comp3": "error4",
    "comp4": "error5",
}


def matched_error_24h_scores(frame: pd.DataFrame, component: str) -> np.ndarray:
    """Baseline 2b: positive if *that component's* error code fired in 24 hours.

    Not in the milestone; added because baseline 2 as specified fires on any of
    five codes and its precision is correspondingly poor, which makes it easy to
    beat for the wrong reason. This is the sharpest rule available without a
    model, and it is the honest thing for a model to have to beat.
    """
    return (
        frame[f"{MATCHED_ERROR[component]}_count_24h"].to_numpy(dtype=float) > 0
    ).astype(float)


BASELINES = {
    "majority": majority_class_scores,
    "any_error_24h": any_error_24h_scores,
    "error_count_24h": error_count_24h_scores,
}

#: Baselines that need to know which component they are scoring.
COMPONENT_BASELINES = {
    "matched_error_24h": matched_error_24h_scores,
}

#: Threshold at which each baseline's binary prediction is taken. The majority
#: baseline never predicts positive, so any threshold above 0 works; 0.5 is used
#: to keep the table honest about the fact that it emits a constant.
BASELINE_THRESHOLDS = {
    "majority": 0.5,
    "any_error_24h": 0.5,
    "matched_error_24h": 0.5,
    "error_count_24h": 0.5,
}
