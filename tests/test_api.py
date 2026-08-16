"""Milestone 7: the HTTP service.

No test here needs a network or a live model. The one endpoint that would call
one is driven with a scripted client injected through `Settings.build_client`,
which is the same seam the eval harness uses.
"""

from __future__ import annotations

import json
import logging
from http import HTTPStatus
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.agent.contracts import ErrorCode
from src.agent.loop import LLMResponse, ToolCall
from src.api import config as api_config
from src.api import errors as api_errors
from src.api import logging as structured
from src.api.app import create_app

REPO = Path(__file__).resolve().parents[1]


class ScriptedClient:
    def __init__(self, script):
        self.script = list(script)
        self.last_exchange = {
            "tokens_in": 900, "tokens_out": 80, "cache_read": 0,
            "cache_write": 0, "stop_reason": "end_turn",
        }
        self.identity = None

    def complete(self, messages, tools):
        return self.script.pop(0) if self.script else LLMResponse(text="done")


@pytest.fixture
def client(built_db, tmp_path, monkeypatch):
    settings = api_config.Settings(
        provider="scripted", database=str(built_db), run_store=str(tmp_path / "runs"),
        demo_mode=False,
    )
    monkeypatch.setattr(
        api_config.Settings, "build_client",
        lambda self: ScriptedClient(
            [
                LLMResponse(tool_calls=(ToolCall("get_machine_profile", {"machine_id": 14}),)),
                LLMResponse(text="Machine 14 is a model3, one year old."),
            ]
        ),
    )
    return TestClient(create_app(settings))


# ----------------------------------------------------------------------
# Contracts: one set of models, not two
# ----------------------------------------------------------------------


def test_the_api_reuses_the_existing_contracts_rather_than_redefining_them():
    """A parallel set of API schemas is how a service starts lying: the
    contract says `calibrated: false` and the API model quietly drops it."""
    from src.api import app as api_app
    from src.obs.accounting import RunAccounting

    assert api_app.AskResponse.model_fields["accounting"].annotation is RunAccounting

    import inspect

    source = inspect.getsource(api_app)
    for contract_field in ("calibrated_probability", "warning_adequacy", "days_of_cover"):
        assert contract_field not in source, (
            f"{contract_field} is redeclared in the API layer; it belongs to "
            "src/agent/contracts.py alone"
        )


def test_every_error_code_has_a_status_and_none_is_a_200():
    """Exhaustive by assertion: adding an ErrorCode without deciding its status
    fails here rather than silently becoming a 500."""
    for code in ErrorCode:
        assert code in api_errors.STATUS_FOR, f"{code} has no HTTP status"
        status = api_errors.status_for(code)
        assert 400 <= status <= 599, f"{code} maps to {status}, which is not an error"


@pytest.mark.parametrize(
    "code, expected",
    [
        (ErrorCode.NOT_FOUND, HTTPStatus.NOT_FOUND),
        (ErrorCode.INVALID_INPUT, HTTPStatus.UNPROCESSABLE_ENTITY),
        (ErrorCode.NO_DATA, HTTPStatus.NOT_FOUND),
        (ErrorCode.MODEL_UNAVAILABLE, HTTPStatus.SERVICE_UNAVAILABLE),
        (ErrorCode.DATABASE_ERROR, HTTPStatus.INTERNAL_SERVER_ERROR),
        (ErrorCode.TIMEOUT, HTTPStatus.GATEWAY_TIMEOUT),
        (ErrorCode.INTERNAL, HTTPStatus.INTERNAL_SERVER_ERROR),
    ],
)
def test_the_status_mapping_is_what_it_claims(code, expected):
    assert api_errors.status_for(code) == expected


# ----------------------------------------------------------------------
# /v1/ask
# ----------------------------------------------------------------------


def test_asking_a_question_returns_the_answer_its_tool_calls_and_accounting(client):
    response = client.post("/v1/ask", json={"question": "How old is machine 14?"})
    assert response.status_code == 200
    body = response.json()

    assert body["answer"].startswith("Machine 14")
    assert [c["tool"] for c in body["tool_calls"]] == ["get_machine_profile"]
    assert body["tool_calls"][0]["status"] == "ok"
    assert body["tool_calls"][0]["result"]["data"]["model"] == "model3"
    assert body["accounting"]["tokens_in"] == 1800, "two model calls at 900 each"
    assert body["accounting"]["tool_calls"] == 1
    assert body["run_id"] == response.headers["X-Run-Id"]


def test_a_malformed_request_is_422_not_200(client):
    assert client.post("/v1/ask", json={"question": "hi"}).status_code == 422
    assert client.post("/v1/ask", json={}).status_code == 422
    assert client.post("/v1/ask", json={"question": "ok?", "extra": 1}).status_code == 422


def test_a_missing_database_is_a_500_with_a_typed_body(built_db, tmp_path):
    settings = api_config.Settings(
        database=str(tmp_path / "absent.db"), run_store=str(tmp_path / "runs"),
        demo_mode=False,
    )
    response = TestClient(create_app(settings)).post(
        "/v1/ask", json={"question": "How old is machine 14?"}
    )
    assert response.status_code == 500
    body = response.json()
    assert body["status"] == "error" and body["code"] == "database_error"
    assert body["run_id"], "an error still carries the run id it happened under"


def test_an_unknown_provider_is_a_503_not_a_traceback(built_db, tmp_path):
    settings = api_config.Settings(
        provider="not-a-provider", database=str(built_db),
        run_store=str(tmp_path / "runs"), demo_mode=False,
    )
    response = TestClient(create_app(settings)).post(
        "/v1/ask", json={"question": "How old is machine 14?"}
    )
    assert response.status_code == 503
    assert response.json()["code"] == "model_unavailable"
    assert response.headers["X-Retryable"] == "true"


def test_a_tool_failure_is_reported_in_the_body_not_hidden(client, monkeypatch):
    """The call failed and the answer says so; the HTTP status is still 200
    because the *request* succeeded. What must never happen is the tool error
    vanishing."""
    from src.agent import tools as tools_module
    from src.agent.contracts import ToolError

    monkeypatch.setattr(
        tools_module, "dispatch",
        lambda name, args, db=None: ToolError(
            code=ErrorCode.TIMEOUT, message="injected", tool=name, retryable=False
        ),
    )
    body = client.post("/v1/ask", json={"question": "How old is machine 14?"}).json()
    assert body["tool_calls"][0]["status"] == "error"
    assert body["tool_calls"][0]["error_code"] == "timeout"


# ----------------------------------------------------------------------
# /v1/runs/{run_id}
# ----------------------------------------------------------------------


def test_an_unreachable_provider_is_503_not_500(built_db, tmp_path, monkeypatch):
    """A dependency being down is not this service being broken. A container
    smoke test with no Ollama reachable returned 500, which reads as our bug."""
    from src.agent.providers import ProviderError

    class Unreachable:
        last_exchange: dict = {}
        identity = None

        def complete(self, messages, tools):
            raise ProviderError("connection refused")

    settings = api_config.Settings(
        provider="ollama", database=str(built_db), run_store=str(tmp_path / "runs"),
        demo_mode=False,
    )
    monkeypatch.setattr(api_config.Settings, "build_client", lambda self: Unreachable())
    response = TestClient(create_app(settings)).post(
        "/v1/ask", json={"question": "How old is machine 14?"}
    )
    assert response.status_code == 503
    assert response.json()["code"] == "model_unavailable"
    assert response.headers["X-Retryable"] == "true"


def test_a_run_can_be_fetched_back_by_its_id(client):
    run_id = client.post("/v1/ask", json={"question": "How old is machine 14?"}).json()["run_id"]
    fetched = client.get(f"/v1/runs/{run_id}")
    assert fetched.status_code == 200
    assert fetched.json()["run_id"] == run_id
    assert fetched.json()["accounting"]["tool_calls"] == 1


def test_an_unknown_run_is_404(client):
    response = client.get("/v1/runs/run_doesnotexist")
    assert response.status_code == 404
    assert response.json()["code"] == "not_found"


@pytest.mark.parametrize(
    "hostile", ["../../etc/passwd", "..%2F..%2Fsecrets", "a/b", "x" * 100]
)
def test_a_run_id_cannot_escape_the_run_store(client, hostile):
    """`run_id` reaches the filesystem, so it is validated rather than trusted."""
    response = client.get(f"/v1/runs/{hostile}")
    assert response.status_code in (404, 422), response.status_code
    assert "passwd" not in response.text


# ----------------------------------------------------------------------
# /health
# ----------------------------------------------------------------------


def test_health_checks_the_database_and_the_model_artefacts(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["checks"] == {"database_readable": True, "model_artefacts_load": True}


def test_health_is_503_when_the_database_is_unreadable(tmp_path):
    settings = api_config.Settings(database=str(tmp_path / "absent.db"))
    response = TestClient(create_app(settings)).get("/health")
    assert response.status_code == 503
    assert response.json()["status"] == "degraded"
    assert response.json()["checks"]["database_readable"] is False


def test_health_names_the_provider_but_never_whether_a_key_is_configured(
    client, monkeypatch
):
    """An unauthenticated health endpoint saying "a key is configured" tells an
    attacker this host is worth more of their time."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-do-not-leak-this")
    text = client.get("/health").text
    assert "scripted" in text, "the configured provider is reported"
    assert "do-not-leak-this" not in text
    for word in ("api_key", "apiKey", "credential", "key_present", "has_key", "token"):
        assert word not in text, f"health must not mention {word}"


# ----------------------------------------------------------------------
# Structured logging
# ----------------------------------------------------------------------


def test_every_log_line_is_json_carrying_the_run_id(built_db, tmp_path, monkeypatch, capsys):
    import io

    stream = io.StringIO()
    structured.configure("INFO", stream=stream)
    structured.bind("run_test123")
    structured.info("hello", event="unit", tool="get_failure_risk")

    line = json.loads(stream.getvalue().strip())
    assert line["run_id"] == "run_test123"
    assert line["event"] == "unit" and line["tool"] == "get_failure_risk"
    assert line["level"] == "info" and "ts" in line


def test_a_field_outside_the_allowlist_is_dropped():
    """Fails closed: a careless caller loses a field rather than leaking one."""
    assert structured.scrub({"tool": "x", "unknown_field": "y"}) == {"tool": "x"}


@pytest.mark.parametrize(
    "field", ["api_key", "authorization", "token", "secret", "password", "arguments"]
)
def test_credential_bearing_fields_can_never_be_logged(field):
    assert structured.scrub({field: "sk-ant-secret"}) == {}


def test_tool_arguments_are_logged_as_names_only():
    """`find_machines` takes a free-text filter and a question is whatever
    somebody typed, so argument values are where an unexpected secret shows up."""
    keys = structured.argument_keys({"machine_id": 30, "error_id": "sk-ant-oops"})
    assert keys == ["error_id", "machine_id"]
    assert "sk-ant-oops" not in json.dumps(keys)


def test_a_real_request_never_logs_the_question_or_the_answer(
    built_db, tmp_path, monkeypatch
):
    import io

    stream = io.StringIO()
    settings = api_config.Settings(
        provider="scripted", database=str(built_db), run_store=str(tmp_path / "runs"),
        demo_mode=False,
    )
    monkeypatch.setattr(
        api_config.Settings, "build_client",
        lambda self: ScriptedClient(
            [
                LLMResponse(tool_calls=(ToolCall("get_machine_profile", {"machine_id": 14}),)),
                LLMResponse(text="SENSITIVE ANSWER TEXT"),
            ]
        ),
    )
    app = create_app(settings)
    structured.configure("INFO", stream=stream)
    TestClient(app).post("/v1/ask", json={"question": "SECRET QUESTION TEXT"})

    logs = stream.getvalue()
    assert logs.strip(), "the request did log something"
    assert "SECRET QUESTION TEXT" not in logs
    assert "SENSITIVE ANSWER TEXT" not in logs
    assert '"event": "ask_done"' in logs, "but it logged the run's shape"


def test_the_logger_writes_to_stdout_not_a_file():
    """A container's log is its stdout; a service that writes its own log file
    is a service whose logs vanish with the container."""
    import sys

    logger = structured.configure("INFO")
    handler = logger.handlers[0]
    assert isinstance(handler, logging.StreamHandler)
    assert handler.stream is sys.stdout
    assert not isinstance(handler, logging.FileHandler)
