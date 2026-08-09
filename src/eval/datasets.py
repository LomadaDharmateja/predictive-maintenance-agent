"""Loading the feature matrices, and the lock on the test split.

`docs/MILESTONE_3.md` section 0:

> The test split is evaluated exactly once, at the very end, after every
> modelling decision is final.

That is enforced two ways rather than one, because a rule that lives only in a
test can be satisfied by editing the test:

1. **At runtime.** `load_features("test")` raises unless the caller passes
   `TEST_SPLIT_UNLOCK`. Nothing forces a caller to be honest, but nothing reads
   the test split by accident either -- and an accident is the realistic failure
   mode, not sabotage.
2. **By source inspection.** `tests/test_single_test_split_consumer.py` parses
   every module under `src/` and asserts that exactly one of them mentions
   `TEST_SPLIT_UNLOCK`: `src/eval/test_evaluation.py`.

No module builds the locked split's filename as a literal -- it is assembled from
the split name -- so a module reading that parquet directly with pandas would
have to spell the name out, which is itself checked for. The source test keys on
the unlock token, which cannot be produced by accident.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features.config import FEATURE_COLUMNS, INDEX_COLUMNS, LABEL_COLUMNS

FEATURES_DIR = Path("data/generated")

#: The only key that opens the test split. Importing this name is the signal the
#: source-inspection test looks for, so it is deliberately unmistakable and
#: deliberately not a bare boolean.
TEST_SPLIT_UNLOCK = "milestone-3: the test split is read once, by src/eval/test_evaluation.py"

TRAIN = "train"
VAL = "val"
LOCKED_SPLIT = "test"


class SplitAccessError(RuntimeError):
    """Raised on an attempt to read the locked split without the unlock token."""


def features_path(split: str, directory: Path = FEATURES_DIR) -> Path:
    return directory / f"features_{split}.parquet"


def load_features(
    split: str,
    directory: Path = FEATURES_DIR,
    unlock: str | None = None,
) -> pd.DataFrame:
    """Read one feature matrix.

    Reading `LOCKED_SPLIT` requires `unlock=TEST_SPLIT_UNLOCK`.
    """
    if split == LOCKED_SPLIT and unlock != TEST_SPLIT_UNLOCK:
        raise SplitAccessError(
            f"the {LOCKED_SPLIT!r} split is locked. It is evaluated exactly once, "
            "at the end, by src/eval/test_evaluation.py -- see docs/MILESTONE_3.md "
            "section 0. Model selection, tuning, thresholds and feature decisions "
            "all happen on train and val."
        )

    path = features_path(split, directory)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `make features` (which needs `make data` first)."
        )

    frame = pd.read_parquet(path)
    expected = INDEX_COLUMNS + FEATURE_COLUMNS + LABEL_COLUMNS
    if list(frame.columns) != expected:
        raise ValueError(
            f"{path} has {len(frame.columns)} columns, expected {len(expected)}. "
            "The parquet files are stale -- rerun `make features`."
        )
    return frame


def load_train(directory: Path = FEATURES_DIR) -> pd.DataFrame:
    return load_features(TRAIN, directory)


def load_val(directory: Path = FEATURES_DIR) -> pd.DataFrame:
    return load_features(VAL, directory)


def xy(frame: pd.DataFrame, component: str) -> tuple[pd.DataFrame, pd.Series]:
    """Features and one component's labels, selected by name.

    Never positional. v1 of this project fed a model a bare list and a swapped
    order changed a prediction from 0.00 to 0.70 with nothing raised.
    """
    return frame[FEATURE_COLUMNS], frame[f"label_{component}"]
