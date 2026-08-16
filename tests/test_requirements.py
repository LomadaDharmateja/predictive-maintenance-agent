"""Declared dependencies and imported modules must agree, in both directions.

The defect this exists to catch: `requirements.txt` pinned `pandas==3.0.5`
while mlflow 3.15.1 requires `pandas<3`, so `pip install -r requirements.txt`
failed with ResolutionImpossible. CI never reached pytest and reported only a
missing `report.xml`. It survived locally because this repository's venv was
built incrementally over many milestones and had been running pandas 2.3.3 the
whole time -- the pinned version was never installed anywhere.

Underneath that sat a second, quieter fault. `fastapi`, `pydantic` and `PyYAML`
are imported directly by `src/api/` and `evals/`, and none of them was
declared: they arrived transitively through mlflow, pandera and starlette. The
suite passed on dependencies nothing asked for, so removing mlflow -- an
experiment-tracking library the service does not use -- would have broken the
API tests for no reason a reader could see.

Both directions are checked, because each catches a different mistake:

  imported but not declared -> the build depends on luck
  declared but not imported -> the pin is aspirational, and the file's own
                               header says nothing here is

Only **module-level** imports count as hard dependencies. A lazy import inside
a function is precisely how an optional dependency is meant to be expressed,
and `scripts/capture_demo.py` uses one for playwright so that a browser binary
never becomes a test dependency.
"""

from __future__ import annotations

import ast
import pathlib
import re
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
STDLIB = set(sys.stdlib_module_names)

#: First-party packages, which are never declared in a requirements file.
LOCAL = {"src", "tests", "evals", "scripts", "conftest", "synthetic"}

#: Import name -> distribution name, where they differ.
DISTRIBUTION_FOR = {
    "yaml": "pyyaml",
    "sklearn": "scikit-learn",
    "opentelemetry": "opentelemetry-sdk",
    "dateutil": "python-dateutil",
}

#: Declared, and deliberately not imported by name. Each needs a reason.
NOT_IMPORTED_BY_NAME = {
    "pyarrow": "parquet engine for pandas.to_parquet; pandas declares it only "
               "as an optional extra, so `make features` fails without the pin",
    "scipy": "required by scikit-learn and lightgbm; pinned because the "
             "determinism rule needs a fixed numerical stack",
    "uvicorn": "ASGI server; run as a process (`make serve`, the Dockerfile "
               "CMD), and imported lazily by scripts/capture_demo.py",
}


def declared(path: pathlib.Path) -> set[str]:
    """Distribution names pinned in a requirements file, lowercased."""
    names = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if not line or line.startswith("-"):
            continue
        names.add(re.split(r"[=<>!~\[]", line)[0].strip().lower())
    return names


def _third_party(names: set[str]) -> set[str]:
    return {
        DISTRIBUTION_FOR.get(n, n).lower()
        for n in names
        if n not in STDLIB and n not in LOCAL
    }


def module_level_imports(path: pathlib.Path) -> set[str]:
    """Imports executed when the module is imported -- the hard dependencies.

    Walks only `tree.body`, so an import nested in a function or an
    `except ImportError` fallback is correctly not counted.
    """
    names: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return _third_party(names)


def all_imports(path: pathlib.Path) -> set[str]:
    """Every import anywhere in the file, lazy ones included."""
    names: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return _third_party(names)


def python_files(*roots: str) -> list[pathlib.Path]:
    out: list[pathlib.Path] = []
    for root in roots:
        out.extend(sorted((REPO / root).rglob("*.py")))
    return out


TEST_PATH_ROOTS = ("src", "tests", "evals")
ALL_ROOTS = TEST_PATH_ROOTS + ("scripts",)


# ----------------------------------------------------------------------
# Direction 1: imported -> declared
# ----------------------------------------------------------------------


def test_every_module_level_import_is_declared_in_requirements():
    """What CI installs must cover what importing the test path executes."""
    pinned = declared(REPO / "requirements.txt")
    missing: dict[str, set[str]] = {}
    for path in python_files(*TEST_PATH_ROOTS):
        for name in module_level_imports(path) - pinned:
            missing.setdefault(name, set()).add(
                str(path.relative_to(REPO)).replace("\\", "/")
            )
    assert not missing, "imported at module level but not in requirements.txt:\n" + "\n".join(
        f"  {name} <- {', '.join(sorted(files))}" for name, files in sorted(missing.items())
    )


def test_the_service_modules_declare_what_they_import():
    """`requirements-api.txt` is what the container installs.

    Anything under src/agent, src/api or src/obs must be covered by it, or the
    image starts and fails on the first request that touches the gap.
    """
    pinned = declared(REPO / "requirements-api.txt")
    missing: dict[str, set[str]] = {}
    for path in python_files("src/agent", "src/api", "src/obs"):
        for name in module_level_imports(path) - pinned:
            missing.setdefault(name, set()).add(
                str(path.relative_to(REPO)).replace("\\", "/")
            )
    assert not missing, "imported by the service but not in requirements-api.txt:\n" + "\n".join(
        f"  {name} <- {', '.join(sorted(files))}" for name, files in sorted(missing.items())
    )


@pytest.mark.parametrize(
    "module",
    ["src.api.app", "src.api.demo", "src.obs.accounting", "src.obs.tracing",
     "src.agent.loop", "src.agent.tools", "src.agent.contracts"],
)
def test_the_milestone_9_modules_import_cleanly(module):
    """Named individually so a failure says which module, not 'collection error'."""
    __import__(module)


# ----------------------------------------------------------------------
# Direction 2: declared -> imported
# ----------------------------------------------------------------------


def test_every_declared_dependency_is_actually_used():
    """"Nothing here is aspirational", per the file's own header."""
    pinned = declared(REPO / "requirements.txt")
    used: set[str] = set()
    for path in python_files(*ALL_ROOTS):
        used |= all_imports(path)

    unused = pinned - used - set(NOT_IMPORTED_BY_NAME) - {"pytest"}
    assert not unused, (
        "declared in requirements.txt but imported nowhere:\n"
        + "\n".join(f"  {name}" for name in sorted(unused))
        + "\nEither remove it, or add it to NOT_IMPORTED_BY_NAME with a reason."
    )


def test_every_documented_exception_is_still_declared():
    """An allowlist entry for a package nobody pins any more is dead weight."""
    pinned = declared(REPO / "requirements.txt") | declared(REPO / "requirements-api.txt")
    for name, reason in NOT_IMPORTED_BY_NAME.items():
        assert name in pinned, f"{name} is allowlisted but pinned nowhere ({reason})"
        assert reason.strip(), name


# ----------------------------------------------------------------------
# Resolvability, and the two files agreeing
# ----------------------------------------------------------------------


def test_the_two_requirements_files_do_not_disagree_on_a_version():
    """A package pinned twice at different versions means dev and production
    run different code, and only one of them is the one under test."""
    dev, api = {}, {}
    for target, path in ((dev, "requirements.txt"), (api, "requirements-api.txt")):
        for line in (REPO / path).read_text(encoding="utf-8").splitlines():
            line = line.split("#")[0].strip()
            if "==" in line:
                name, version = line.split("==", 1)
                target[name.strip().lower()] = version.strip()

    conflicts = {
        name: (dev[name], api[name])
        for name in set(dev) & set(api)
        if dev[name] != api[name]
    }
    assert not conflicts, "pinned at two different versions:\n" + "\n".join(
        f"  {n}: requirements.txt {a} vs requirements-api.txt {b}"
        for n, (a, b) in sorted(conflicts.items())
    )


def test_no_pin_contradicts_a_declared_dependency_of_another_pin():
    """The exact failure that broke CI: pandas==3.0.5 against mlflow's pandas<3.

    Checked against installed metadata rather than by resolving over the
    network, so it runs offline. It catches the conflict for any package that
    is both pinned here and required by another pinned package.
    """
    from importlib import metadata
    from packaging.requirements import Requirement
    from packaging.version import Version

    pins = {}
    for line in (REPO / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if "==" in line:
            name, version = line.split("==", 1)
            pins[name.strip().lower()] = version.strip()

    problems = []
    for dist_name in list(pins):
        try:
            dist = metadata.distribution(dist_name)
        except metadata.PackageNotFoundError:
            continue
        for raw in dist.requires or []:
            requirement = Requirement(raw)
            if requirement.marker and not requirement.marker.evaluate():
                continue
            target = requirement.name.lower()
            if target in pins and requirement.specifier:
                if not requirement.specifier.contains(Version(pins[target]), prereleases=True):
                    problems.append(
                        f"{dist_name}=={pins[dist_name]} requires "
                        f"{target}{requirement.specifier}, but {target} is pinned "
                        f"at {pins[target]}"
                    )
    assert not problems, "requirements.txt cannot resolve:\n" + "\n".join(
        f"  {p}" for p in sorted(set(problems))
    )


# ----------------------------------------------------------------------
# playwright must never become a test dependency
# ----------------------------------------------------------------------


def test_playwright_is_never_imported_at_module_level_anywhere():
    """The screenshot script needs a browser binary; the suite must not.

    Checked across the whole repository rather than only the test path,
    because `pytest.ini`'s `norecursedirs` is a collection setting and would
    not save a module that something under `tests/` decided to import.
    """
    offenders = []
    for path in python_files(*ALL_ROOTS):
        if "playwright" in module_level_imports(path):
            offenders.append(str(path.relative_to(REPO)).replace("\\", "/"))
    assert not offenders, (
        "playwright imported at module level in: " + ", ".join(offenders)
        + " -- move it inside the function that needs it"
    )


def test_playwright_is_not_declared_as_a_dependency():
    for path in ("requirements.txt", "requirements-api.txt"):
        assert "playwright" not in declared(REPO / path), (
            f"playwright is pinned in {path}; it is a tool for regenerating "
            "screenshots by hand, not a dependency of the suite or the service"
        )


def test_the_screenshot_script_survives_playwright_being_absent(monkeypatch):
    """Importing it must not require the browser stack."""
    import importlib

    module = importlib.import_module("scripts.capture_demo")
    assert hasattr(module, "main")

    real_import = __import__

    def refuse(name, *args, **kwargs):
        if name.startswith("playwright"):
            raise ImportError("playwright is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", refuse)
    assert module.main() == 2, "main() must report the missing browser, not raise"
