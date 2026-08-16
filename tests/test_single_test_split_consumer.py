"""Exactly one module may read the test split.

`docs/MILESTONE_3.md` section 0. The rule is enforced twice, and this file checks
both halves:

- **At runtime**, `load_features("test")` raises without the unlock token.
- **By source inspection**, only `src/eval/test_evaluation.py` mentions that
  token anywhere under `src/`.

The source check exists because the runtime guard is a keystroke away from being
bypassed by anyone who wants to peek. It cannot stop a determined person -- and
does not try -- but it turns "I accidentally tuned on test" into something that
shows up in a diff.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import pytest

from tests.artefacts import requires_features

from src.eval.datasets import (
    LOCKED_SPLIT,
    TEST_SPLIT_UNLOCK,
    SplitAccessError,
    load_features,
    load_train,
    load_val,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"

#: The one module allowed to hold the key.
AUTHORISED = "src/eval/test_evaluation.py"

#: Where the token itself is defined. Not a consumer.
DEFINITION = "src/eval/datasets.py"


def modules() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


def relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


# ----------------------------------------------------------------------
# Source inspection
# ----------------------------------------------------------------------


def test_only_one_module_holds_the_unlock_token():
    holders = [
        relative(path)
        for path in modules()
        if "TEST_SPLIT_UNLOCK" in path.read_text(encoding="utf-8")
    ]
    assert sorted(holders) == sorted([DEFINITION, AUTHORISED]), (
        "the test split unlock token appears in an unexpected module: "
        f"{sorted(set(holders) - {DEFINITION, AUTHORISED})}"
    )


def test_no_module_names_the_test_parquet_directly():
    """The filename is built from the split name, so nothing should contain it
    as a literal. A module that hardcoded it would sidestep `load_features`
    entirely and read the file with pandas."""
    offenders = [
        relative(path)
        for path in modules()
        if "features_test" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


def _calls_load_features_with_test(path: Path) -> bool:
    """Does this module ask a loader for the locked split?

    Matches both the literal `"test"` and the `LOCKED_SPLIT` constant, since the
    authorised module uses the constant and a module trying to sneak past would
    use the literal. Anything referring to the split by a computed name defeats
    this check -- which is why the unlock token exists as well.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if name not in {"load_features", "read_parquet"}:
            continue
        for argument in list(node.args) + [kw.value for kw in node.keywords]:
            if isinstance(argument, ast.Constant) and argument.value == LOCKED_SPLIT:
                return True
            if isinstance(argument, ast.Name) and argument.id == "LOCKED_SPLIT":
                return True
    return False


def test_only_the_authorised_module_asks_for_the_locked_split():
    callers = [relative(p) for p in modules() if _calls_load_features_with_test(p)]
    assert callers == [AUTHORISED]


def test_source_scanner_detects_a_planted_reader(tmp_path):
    """Guards the guard. A scanner that never fires proves nothing."""
    planted = tmp_path / "sneaky.py"
    planted.write_text(
        "from src.eval.datasets import load_features\n"
        "def peek():\n"
        "    return load_features('test')\n",
        encoding="utf-8",
    )
    assert _calls_load_features_with_test(planted)


def test_source_scanner_ignores_an_innocent_module(tmp_path):
    innocent = tmp_path / "fine.py"
    innocent.write_text(
        "from src.eval.datasets import load_features\n"
        "def go():\n"
        "    return load_features('val')\n",
        encoding="utf-8",
    )
    assert not _calls_load_features_with_test(innocent)


# ----------------------------------------------------------------------
# Runtime guard
# ----------------------------------------------------------------------


def test_loading_the_test_split_without_the_token_raises():
    with pytest.raises(SplitAccessError, match="locked"):
        load_features(LOCKED_SPLIT)


def test_the_wrong_token_does_not_open_it():
    with pytest.raises(SplitAccessError):
        load_features(LOCKED_SPLIT, unlock="please")


@requires_features
def test_train_and_val_need_no_token(tmp_path):
    """The lock must not be so blunt that ordinary work needs the key."""
    for loader in (load_train, load_val):
        frame = loader()
        assert isinstance(frame, pd.DataFrame)
        assert len(frame) > 0


@requires_features
def test_the_token_does_open_it():
    """Anti-vacuity: if the token did not work, the guard tests above would pass
    for the wrong reason and the authorised module would be broken.

    This does read the file. It asserts on shape and column names only -- it
    never looks at a label, computes a metric, or feeds a model -- so it does not
    consume the one-shot evaluation. The rule in `docs/MILESTONE_3.md` section 0
    is about making modelling decisions on the test split, not about the bytes
    being opened.
    """
    frame = load_features(LOCKED_SPLIT, unlock=TEST_SPLIT_UNLOCK)
    assert len(frame) > 0
    assert "label_comp1" in frame.columns
