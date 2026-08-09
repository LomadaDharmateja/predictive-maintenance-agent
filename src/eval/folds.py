"""Rolling-origin cross-validation on the training split.

Expanding window, never shuffled, never k-fold. Fold `i` trains on everything up
to a cutoff and validates on the month after it, so each fold answers the only
question that matters operationally: given everything known by date X, how well
does the model do on the month that follows.

The same 24-hour embargo that separates train from validation is applied inside
every fold. Without it a fold's last training row has a label window reaching
into that fold's own validation month, and the cross-validation score is
optimistic for exactly the reason `docs/DATA.md` section 5.2 describes. This is
easy to forget precisely because the outer split already handles it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.config import EMBARGO, SPLITS

#: Month boundaries inside the training period. Fold `i` trains on everything
#: before CUTOFFS[i] and validates on [CUTOFFS[i], CUTOFFS[i+1]).
CUTOFFS = [
    pd.Timestamp("2015-06-01"),
    pd.Timestamp("2015-07-01"),
    pd.Timestamp("2015-08-01"),
    pd.Timestamp("2015-09-01"),
    pd.Timestamp("2015-10-01"),  # upper bound of the last fold's validation month
]


@dataclass(frozen=True)
class Fold:
    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp  # inclusive, after the embargo
    val_start: pd.Timestamp
    val_end: pd.Timestamp  # exclusive
    train_rows: np.ndarray
    val_rows: np.ndarray

    def describe(self) -> str:
        return (
            f"fold {self.index}: train {self.train_start.date()} -> "
            f"{self.train_end.date()} ({len(self.train_rows):,} rows), "
            f"val {self.val_start.date()} -> {self.val_end.date()} "
            f"({len(self.val_rows):,} rows)"
        )


def rolling_origin_folds(datetimes: pd.Series) -> list[Fold]:
    """Expanding-window folds over the training split.

    `datetimes` is the `datetime` column of the training matrix, in row order.
    Returns positional index arrays, so the caller can slice any aligned array.
    """
    when = pd.to_datetime(datetimes).to_numpy()
    split_start = SPLITS["train"][0]

    folds: list[Fold] = []
    for index, (val_start, val_end) in enumerate(
        zip(CUTOFFS[:-1], CUTOFFS[1:]), start=1
    ):
        # Everything strictly before the validation month, minus the embargo.
        train_end = val_start - EMBARGO
        train_mask = when < np.datetime64(train_end)
        val_mask = (when >= np.datetime64(val_start)) & (when < np.datetime64(val_end))

        train_rows = np.flatnonzero(train_mask)
        val_rows = np.flatnonzero(val_mask)
        if len(train_rows) == 0 or len(val_rows) == 0:
            continue

        folds.append(
            Fold(
                index=index,
                train_start=split_start,
                train_end=pd.Timestamp(when[train_rows].max()),
                val_start=val_start,
                val_end=val_end,
                train_rows=train_rows,
                val_rows=val_rows,
            )
        )
    return folds


def assert_folds_are_clean(folds: list[Fold]) -> None:
    """Every fold's training data must end at least one horizon before its own
    validation month, and folds must expand rather than slide."""
    for fold in folds:
        gap = fold.val_start - fold.train_end
        if gap < EMBARGO:
            raise AssertionError(
                f"fold {fold.index}: gap {gap} between train end {fold.train_end} "
                f"and val start {fold.val_start} is under the embargo {EMBARGO}"
            )
        if set(fold.train_rows) & set(fold.val_rows):
            raise AssertionError(f"fold {fold.index}: train and val rows overlap")

    for earlier, later in zip(folds[:-1], folds[1:]):
        if len(later.train_rows) <= len(earlier.train_rows):
            raise AssertionError(
                f"fold {later.index} does not expand on fold {earlier.index}"
            )
