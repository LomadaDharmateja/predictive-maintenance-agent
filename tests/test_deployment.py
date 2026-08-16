"""Milestone 7 items 3-5: the image, the compose file, and the CI gates.

Most of this is static analysis of the deployment files, which needs nothing.
The tests that build and inspect an actual image are marked `docker` and skip
when no daemon is reachable -- they must never make the suite require Docker,
but when Docker is there they check the image rather than the Dockerfile.

`.dockerignore` is checked by inspecting a built image, not by reading the
file. A correct exclusion list and a `COPY` that predates it produce the same
document and different images.
"""

from __future__ import annotations

import ast
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
IMAGE = "industrial-ai-agent:pytest"


def _declared_serving_packages() -> set[str]:
    """Package names from requirements-api.txt, ignoring comments."""
    return {
        line.split("==")[0].split(">=")[0].strip().lower()
        for line in (REPO / "requirements-api.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(
            ["docker", "info"], capture_output=True, timeout=20
        ).returncode == 0
    except Exception:  # noqa: BLE001
        return False


docker_required = pytest.mark.skipif(
    not _docker_available(), reason="no reachable Docker daemon"
)


@pytest.fixture(scope="module")
def image() -> str:
    subprocess.run(
        ["docker", "build", "-t", IMAGE, "."],
        cwd=REPO, check=True, capture_output=True, timeout=900,
    )
    return IMAGE


def _in_image(image: str, script: str) -> str:
    result = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "sh", image, "-c", script],
        capture_output=True, text=True, timeout=180,
    )
    return result.stdout


# ----------------------------------------------------------------------
# The Dockerfile, read statically
# ----------------------------------------------------------------------


def test_the_dockerfile_is_multi_stage_and_runs_as_a_named_non_root_user():
    text = (REPO / "Dockerfile").read_text(encoding="utf-8")
    assert len(re.findall(r"^FROM ", text, re.M)) >= 2, "multi-stage"
    assert re.search(r"^USER appuser", text, re.M), "does not end as root"
    assert "--uid 10001" in text, "a fixed non-zero uid, not whatever adduser picks"
    assert 'ENTRYPOINT ["' in text, "exec form, so the process is PID 1 and gets SIGTERM"


def test_no_run_line_hides_a_command_behind_a_continuation_comment():
    """A `#` on a continued line comments out the rest of the joined command.
    That silently skipped a `rm -rf` of pip on an earlier build here, and the
    image looked fine until it was inspected."""
    lines = (REPO / "Dockerfile").read_text(encoding="utf-8").splitlines()
    continued = False
    for number, line in enumerate(lines, start=1):
        if continued and line.lstrip().startswith("#"):
            pytest.fail(f"Dockerfile line {number} comments out a continued command")
        continued = line.rstrip().endswith("\\")


def test_the_service_installs_only_its_serving_dependencies():
    text = (REPO / "Dockerfile").read_text(encoding="utf-8")
    assert "requirements-api.txt" in text
    assert "-r requirements.txt" not in text, (
        "the development set carries mlflow, matplotlib and kaggle, none of "
        "which the service imports"
    )
    # Declared packages only. The file's prose explains *why* mlflow is absent,
    # so matching the whole text would flag its own explanation.
    declared = _declared_serving_packages()
    for absent in ("mlflow", "matplotlib", "kaggle", "lightgbm", "pytest"):
        assert absent not in declared, f"{absent} is not a serving dependency"


def test_every_module_the_service_imports_is_a_declared_serving_dependency():
    """The split is only safe if it is complete."""
    declared = _declared_serving_packages()
    aliases = {"scikit-learn": "sklearn", "opentelemetry-sdk": "opentelemetry"}
    importable = {aliases.get(name, name) for name in declared}

    # Parsed, not grepped: a regex over the source also matches the word
    # "import" inside a docstring, which is how this test first claimed the
    # service depends on a module called `this`.
    third_party: set[str] = set()
    for module in (REPO / "src" / "agent", REPO / "src" / "api", REPO / "src" / "obs"):
        for path in module.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    third_party.update(a.name.split(".")[0] for a in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                    third_party.add(node.module.split(".")[0])

    stdlib = {
        "__future__", "abc", "argparse", "collections", "contextlib", "contextvars",
        "dataclasses", "datetime", "difflib", "enum", "functools", "hashlib", "html",
        "http", "io", "itertools", "json", "logging", "math", "os", "pathlib",
        "random", "re", "shutil", "sqlite3", "subprocess", "sys", "threading",
        "time", "typing", "uuid", "warnings",
    }
    # First-party. `evals` is imported lazily, inside functions, by the replay
    # and viewer *tools* under src/obs -- never on the serving path, and never
    # copied into the image. `test_the_service_imports_inside_the_image` is what
    # proves the service itself resolves in the built image.
    first_party = {"src", "evals"}
    for name in third_party - stdlib - first_party:
        assert name in importable, (
            f"src imports {name!r} but requirements-api.txt does not declare it; "
            "the image would fail at runtime"
        )


# ----------------------------------------------------------------------
# .dockerignore, checked against a built image rather than read
# ----------------------------------------------------------------------


@docker_required
def test_the_image_contains_no_secret_no_data_and_no_database(image):
    """`.dockerignore` is a document; this is the artefact. A correct exclusion
    list and a `COPY` that predates it produce the same file and different
    images."""
    found = _in_image(
        image,
        'find / \\( -name ".env" -o -name ".env.*" -o -name "*.db" '
        '-o -name "*.parquet" -o -name "kaggle.json" -o -name "id_rsa*" '
        '-o -name "*.joblib" \\) -not -path "/proc/*" 2>/dev/null | head -20',
    )
    assert found.strip() == "", f"the image embeds excluded content:\n{found}"


@docker_required
def test_the_image_omits_the_repository_directories_that_must_not_ship(image):
    listing = _in_image(
        image,
        "for p in /app/.env /app/data /app/models /app/archive /app/.git "
        "/app/tests /app/mlruns /app/evals/scenarios.yaml; do "
        '[ -e "$p" ] && echo "PRESENT $p"; done',
    )
    assert listing.strip() == "", f"unexpected content in the image:\n{listing}"


@docker_required
def test_the_final_image_carries_no_build_tooling(image):
    """No compiler and no package installer. With those present, a code
    execution foothold becomes a comfortable one."""
    found = _in_image(
        image,
        "for b in gcc cc g++ make ld pip pip3 easy_install curl wget git; do "
        'command -v $b >/dev/null 2>&1 && echo "$b"; done; '
        'python -c "import pip" 2>/dev/null && echo pip-module; '
        'python -c "import ensurepip" 2>/dev/null && echo ensurepip',
    )
    assert found.strip() == "", f"build tooling present in the image:\n{found}"


@docker_required
def test_the_image_runs_as_a_non_root_user(image):
    assert "uid=10001(appuser)" in _in_image(image, "id")
    configured = subprocess.run(
        ["docker", "image", "inspect", image, "--format", "{{.Config.User}}"],
        capture_output=True, text=True, timeout=60,
    ).stdout.strip()
    assert configured == "appuser", "USER must be set in the image, not just at run time"


@docker_required
def test_the_service_imports_inside_the_image(image):
    out = _in_image(
        image, 'python -c "from src.api.app import app; print(app.version)"'
    )
    assert out.strip() == "7.0.0"


# ----------------------------------------------------------------------
# Configuration by environment variable only
# ----------------------------------------------------------------------


def test_env_example_documents_every_variable_and_carries_no_values():
    from src.api.config import ENV_VARS

    text = (REPO / ".env.example").read_text(encoding="utf-8")
    for name in ENV_VARS:
        assert f"{name}=" in text, f"{name} is undocumented in .env.example"

    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        name, _, value = stripped.partition("=")
        assert value == "", (
            f".env.example line {number} carries a value for {name}; it must "
            "document names only"
        )


def test_dockerignore_excludes_env_but_keeps_the_example():
    text = (REPO / ".dockerignore").read_text(encoding="utf-8")
    assert re.search(r"^\.env$", text, re.M)
    assert re.search(r"^\.env\.\*$", text, re.M)
    assert re.search(r"^!\.env\.example$", text, re.M), (
        ".env.example is the one .env* path that must survive"
    )
    for pattern in ("data/raw/", "*.db", "models/", ".git", "archive/"):
        assert pattern in text, f".dockerignore does not exclude {pattern}"


def test_compose_passes_configuration_only_through_the_environment():
    text = (REPO / "docker-compose.yml").read_text(encoding="utf-8")
    assert "env_file:" in text and ".env" in text
    assert "read_only: true" in text
    assert "no-new-privileges:true" in text
    assert "cap_drop:" in text
    assert ":ro" in text, "the database and artefacts are mounted read-only"
    # No secret may be inlined in a committed file.
    for leak in ("sk-ant", "ANTHROPIC_API_KEY:", "password", "secret:"):
        assert leak not in text, f"docker-compose.yml appears to inline {leak}"


def test_compose_binds_the_port_to_localhost_only():
    """A service with no authentication must not be reachable off-host by
    default. Publishing 8000:8000 binds every interface."""
    text = (REPO / "docker-compose.yml").read_text(encoding="utf-8")
    assert '"127.0.0.1:8000:8000"' in text


# ----------------------------------------------------------------------
# CI
# ----------------------------------------------------------------------


def _workflow() -> str:
    return (REPO / ".github" / "workflows" / "pipeline.yml").read_text(encoding="utf-8")


def test_ci_asserts_the_suite_actually_executed():
    """A green pytest proves nothing on its own: with data/raw absent, a
    regression in conftest that skipped every test would exit 0."""
    text = _workflow()
    assert "MIN_EXECUTED" in text
    assert "collected no tests at all" in text
    assert "junitxml" in text


def test_ci_builds_the_image_and_gates_on_the_replay():
    text = _workflow()
    assert "docker build" in text, "the image must build in CI"
    assert "src.obs.replay" in text, "the deterministic replay is a gate"


def test_ci_has_no_secrets_in_any_job():
    text = _workflow()
    # `${{ secrets.X }}` is the only way a workflow reads one. The word
    # "secrets" also appears in a comment explaining that this workflow has
    # none, so matching the bare word would flag the reassurance.
    assert "${{ secrets." not in text, "no job may read a secret"
    assert "${{secrets." not in text
    assert "ANTHROPIC_API_KEY" not in text
    for hazard in ("sk-ant", "password", "token:"):
        assert hazard not in text.lower()


def test_ci_grants_only_read_permission():
    assert re.search(r"permissions:\s*\n\s*contents: read", _workflow())
