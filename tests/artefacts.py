"""Which built artefacts are present, for tests that need more than raw data.

`tests/conftest.py` already skips everything that needs `data/raw/`, because
that download is licensed and CI has no credential for it. Two other kinds of
artefact are just as absent from a fresh checkout, and until Milestone 9
nothing guarded them:

- **Built feature matrices** (`data/generated/features_*.parquet`), produced by
  `make features`. Gitignored, so a clean clone has none.
- **Recorded evaluation runs** (`evals/results/*.json`), produced by
  `make eval`. Also gitignored.

Four tests read these directly rather than through a fixture, so the
fixture-name sweep in `conftest.pytest_collection_modifyitems` could not catch
them, and they failed rather than skipped on every run that did not happen to
have the artefacts on disk. A test that cannot run without an artefact CI does
not have must say so; failing instead means the signal for "this is broken" and
the signal for "this was never built here" are the same colour.

The floor in the CI workflow (`MIN_EXECUTED`) is what stops this from becoming
the other failure mode, where everything skips and the job reports green.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

FEATURES_DIR = REPO / "data" / "generated"
FEATURE_SPLITS = ("train", "val", "test")
EVAL_RESULTS = REPO / "evals" / "results"


def missing_feature_splits() -> list[str]:
    return [
        split
        for split in FEATURE_SPLITS
        if not (FEATURES_DIR / f"features_{split}.parquet").exists()
    ]


def has_recorded_run() -> bool:
    """A scored run, not a sidecar: `*.traces.json` alone is not one."""
    if not EVAL_RESULTS.exists():
        return False
    return any(p for p in EVAL_RESULTS.glob("*.json") if "." not in p.stem)


requires_features = pytest.mark.skipif(
    bool(missing_feature_splits()),
    reason=(
        f"needs built feature matrices in {FEATURES_DIR} "
        f"(missing: {', '.join(missing_feature_splits()) or 'none'}). "
        "Run `make features`, which needs `make data` and so `make fetch-data`."
    ),
)

requires_recorded_run = pytest.mark.skipif(
    not has_recorded_run(),
    reason=(
        f"needs a scored evaluation run in {EVAL_RESULTS}. "
        "Run `python -m evals.runner`; the outputs are gitignored."
    ),
)
