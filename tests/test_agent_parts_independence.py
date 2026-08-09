"""`get_parts_position` has no dependency on any prediction.

`docs/MILESTONE_4.md` acceptance: "verified by import graph". Reading the module
and seeing no model import is not verification -- a transitive import three
levels down would be invisible. This walks the closure.

The constraint exists because of the Milestone 3B finding: the model's effective
warning is shorter than the supplier lead time for 8 of 9 parts, so parts
reasoning cannot come from predictions. Enforcing that in the import graph makes
it structural rather than a convention a later change can quietly break.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Anything that is, loads, or computes a prediction.
FORBIDDEN_MODULES = {
    "joblib",
    "sklearn",
    "lightgbm",
    "xgboost",
    "mlflow",
    "src.features",
    "src.features.compute",
    "src.features.store",
    "src.models",
    "src.models.train",
    "src.eval",
    "src.eval.validate",
    "src.eval.calibration",
    "src.agent.risk",
}

ENTRY_POINT = "src.agent.parts"


def module_path(name: str) -> Path | None:
    parts = name.split(".")
    candidate = REPO_ROOT.joinpath(*parts).with_suffix(".py")
    if candidate.exists():
        return candidate
    package = REPO_ROOT.joinpath(*parts, "__init__.py")
    return package if package.exists() else None


def direct_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module)
            found.update(f"{node.module}.{a.name}" for a in node.names)
    return found


def import_closure(entry: str) -> set[str]:
    """Every module reachable from `entry` within this repository."""
    seen: set[str] = set()
    frontier = [entry]
    while frontier:
        name = frontier.pop()
        if name in seen:
            continue
        seen.add(name)
        path = module_path(name)
        if path is None:
            continue  # third-party or stdlib; recorded but not walked
        for imported in direct_imports(path):
            if imported not in seen:
                frontier.append(imported)
    return seen


def test_parts_module_closure_contains_no_prediction_code():
    closure = import_closure(ENTRY_POINT)
    offenders = sorted(
        name
        for name in closure
        if any(
            name == forbidden or name.startswith(forbidden + ".")
            for forbidden in FORBIDDEN_MODULES
        )
    )
    assert offenders == [], (
        "get_parts_position must not reach any prediction code; found "
        f"{offenders} in the import closure of {ENTRY_POINT}"
    )


def test_the_closure_walker_is_not_trivially_empty():
    """Anti-vacuity: a walker that returned nothing would pass the test above."""
    closure = import_closure(ENTRY_POINT)
    assert ENTRY_POINT in closure
    assert "src.agent.db" in closure
    assert "src.agent.contracts" in closure
    assert len(closure) >= 3


def test_the_closure_walker_detects_a_planted_import(tmp_path, monkeypatch):
    """Guards the guard. Point the walker at a module that does import a model
    and it must fail."""
    package = tmp_path / "src" / "agent"
    package.mkdir(parents=True)
    (tmp_path / "src" / "__init__.py").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "parts.py").write_text(
        "import joblib\nfrom src.agent import db\n", encoding="utf-8"
    )
    (package / "db.py").write_text("import sqlite3\n", encoding="utf-8")

    monkeypatch.setattr(
        "tests.test_agent_parts_independence.REPO_ROOT", tmp_path
    )
    closure = import_closure(ENTRY_POINT)
    assert "joblib" in closure


def test_parts_tool_is_importable_without_the_model_stack(monkeypatch):
    """A stronger claim than the static one: importing the module must not pull
    joblib in at runtime either."""
    import sys

    for name in list(sys.modules):
        if name.startswith("src.agent.parts"):
            del sys.modules[name]

    module = importlib.import_module("src.agent.parts")
    assert hasattr(module, "get_parts_position")
    # The function's own globals must hold no model object.
    globals_ = module.get_parts_position.__globals__
    assert "joblib" not in globals_
    assert "compute_features" not in globals_


def test_parts_input_model_has_no_prediction_field():
    """The type system carries the constraint too: there is no way to pass a
    risk score into this tool."""
    from src.agent.contracts import PartsPositionInput

    fields = set(PartsPositionInput.model_fields)
    assert fields == {"component", "model"}
    forbidden = {"risk", "probability", "score", "prediction", "threshold"}
    assert not any(f in name for name in fields for f in forbidden)


def test_parts_output_model_has_no_prediction_field():
    from src.agent.contracts import PartPosition, PartsPosition

    for model in (PartPosition, PartsPosition):
        for name in model.model_fields:
            assert not any(
                token in name
                for token in ("risk", "probability", "score", "prediction")
            ), f"{model.__name__}.{name} looks like a prediction field"
