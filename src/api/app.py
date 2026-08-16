"""The HTTP service. Milestone 7: a real service, hardened. No new behaviour.

Three endpoints and nothing else:

    POST /v1/ask              ask a question, get an answer and a run id
    GET  /v1/runs/{run_id}    that run's trace and accounting
    GET  /health              is this process able to do its job

**No second set of models.** The response bodies are the Pydantic contracts
that already exist -- `src/agent/contracts.py` for tool results,
`src/obs/accounting.py` for the run totals. A parallel set of API schemas is
the classic way a service starts lying: the contract says `calibrated: false`
and the API model quietly drops the field, and nobody notices until a customer
acts on an uncalibrated probability. The wrappers here add a run id and
nothing else.

**A failed tool call is a status code.** `src/api/errors.py` maps every
`ErrorCode` to one. Nothing in this module returns a 200 with an error inside.

**No new agent behaviour.** This process constructs the same `Agent` the eval
harness does, with the same loop, the same tools and the same read-only
database posture. It adds a transport.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from src.agent.contracts import ErrorCode, ToolError
from src.api import config as api_config
from src.api import errors as api_errors
from src.api import logging as structured
from src.obs import accounting, tracing

# ----------------------------------------------------------------------
# Request and response models -- thin wrappers over the existing contracts
# ----------------------------------------------------------------------


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AskRequest(Strict):
    question: str = Field(min_length=3, max_length=2000)
    #: The prediction time. Required in practice: without it the model has no
    #: way to know what "now" is and will guess -- a pilot run guessed
    #: `2025-01-01`, outside the dataset entirely.
    as_of: str | None = None


class ToolCallView(Strict):
    """One tool call as it happened. Mirrors `ToolCallTrace` deliberately."""

    tool: str
    #: Argument *names* only in logs; the values are returned to the caller
    #: that sent them, which is not a disclosure.
    arguments: dict
    status: Literal["ok", "error"]
    error_code: str | None = None
    truncated: bool = False
    duration_ms: float
    result: Any = None


class AskResponse(Strict):
    run_id: str
    trace_id: str | None = None
    answer: str
    tool_calls: list[ToolCallView]
    accounting: accounting.RunAccounting


class ErrorResponse(Strict):
    """The body of every non-2xx. Shaped like `ToolError` on purpose."""

    status: Literal["error"] = "error"
    code: str
    message: str
    tool: str | None = None
    retryable: bool = False
    run_id: str | None = None


class HealthResponse(Strict):
    status: Literal["ok", "degraded"]
    checks: dict[str, bool]
    provider: str
    model: str
    #: Deliberately absent: whether a credential is configured. See `_health`.
    detail: dict[str, str] = Field(default_factory=dict)


# ----------------------------------------------------------------------
# App
# ----------------------------------------------------------------------


def _error(code: ErrorCode, message: str, tool: str | None = None) -> JSONResponse:
    body = ErrorResponse(
        code=code.value,
        message=message,
        tool=tool,
        retryable=api_errors.is_retryable(code),
        run_id=structured.current_run_id(),
    )
    return JSONResponse(
        status_code=api_errors.status_for(code),
        content=json.loads(body.model_dump_json()),
        headers={"X-Retryable": "true" if body.retryable else "false"},
    )


def create_app(settings: api_config.Settings | None = None) -> FastAPI:
    settings = settings or api_config.Settings.from_env()
    structured.configure(settings.log_level)

    app = FastAPI(
        title="Predictive maintenance planning agent",
        version="7.0.0",
        description=(
            "Flags elevated component risk over a 14-day window so maintenance "
            "attention can be scheduled. Parts are managed from stock and "
            "consumption, never from predictions."
        ),
        docs_url="/docs" if settings.enable_docs else None,
        redoc_url=None,
    )
    app.state.settings = settings

    @app.middleware("http")
    async def bind_run_id(request: Request, call_next):
        run_id = structured.bind()
        started = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            structured.info(
                "request",
                event="http_request",
                method=request.method,
                path=request.url.path,
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
            )
        response.headers["X-Run-Id"] = run_id
        structured.clear()
        return response

    # ---- ask ---------------------------------------------------------
    @app.post(
        "/v1/ask",
        response_model=AskResponse,
        responses={code: {"model": ErrorResponse} for code in (404, 422, 500, 503, 504)},
    )
    def ask(body: AskRequest) -> Any:
        from src.agent.loop import Agent, LoopConfig
        from src.agent.providers import ProviderError

        run_id = structured.current_run_id() or structured.bind()
        settings: api_config.Settings = app.state.settings

        if settings.database is None or not Path(settings.database).exists():
            structured.error("database missing", event="ask", error_code="database_error")
            return _error(ErrorCode.DATABASE_ERROR, "the database is not readable")

        client = settings.build_client()
        if client is None:
            return _error(
                ErrorCode.MODEL_UNAVAILABLE,
                f"no model client for provider {settings.provider!r}",
            )

        exporter = tracing.MemorySpanExporter()
        tracing.configure(exporter, reset=True)
        structured.info(
            "ask", event="ask_start", provider=settings.provider,
            question_chars=len(body.question),
        )
        try:
            agent = Agent(
                client, database=Path(settings.database),
                config=LoopConfig(max_iterations=settings.max_iterations),
            )
            outcome = agent.run(body.question, as_of=body.as_of)
        except ProviderError as exc:
            # The provider is unreachable or refused the request. That is a
            # dependency failure, not a bug in this service, and 503 tells a
            # caller to retry rather than to open a ticket. A container smoke
            # test surfaced this: with no Ollama reachable the endpoint
            # returned 500, which reads as "we are broken".
            structured.error(
                "provider unavailable", event="ask_error",
                error_code=ErrorCode.MODEL_UNAVAILABLE.value,
                detail=type(exc).__name__,
            )
            return _error(ErrorCode.MODEL_UNAVAILABLE, "the model provider is unavailable")
        except TimeoutError as exc:
            structured.error("timed out", event="ask_error", detail=type(exc).__name__)
            return _error(ErrorCode.TIMEOUT, "the run timed out")
        except Exception as exc:  # noqa: BLE001 - mapped, never swallowed
            structured.error("ask failed", event="ask_error", detail=type(exc).__name__)
            return _error(ErrorCode.INTERNAL, f"{type(exc).__name__} during the run")

        calls: list[ToolCallView] = []
        for entry in outcome.log.entries:
            if entry.kind not in {"tool_result", "tool_error"}:
                continue
            try:
                parsed = json.loads(entry.detail or "null")
            except json.JSONDecodeError:
                parsed = entry.detail
            calls.append(
                ToolCallView(
                    tool=entry.name or "",
                    arguments=entry.arguments or {},
                    status="ok" if entry.kind == "tool_result" else "error",
                    error_code=entry.error_code,
                    truncated=entry.truncated,
                    duration_ms=entry.duration_ms or 0.0,
                    result=parsed,
                )
            )
            structured.info(
                "tool call", event="tool_call", tool=entry.name,
                result_type="Success" if entry.kind == "tool_result" else "ToolError",
                error_code=entry.error_code,
                duration_ms=entry.duration_ms,
                argument_keys=structured.argument_keys(entry.arguments),
            )

        spans = exporter.spans
        records = accounting.from_spans(
            spans, run_id=run_id, identity=getattr(client, "identity", None)
        )
        record = records[0] if records else accounting.RunAccounting(run_id=run_id)
        structured.bind(run_id, record.trace_id)
        structured.info(
            "answered", event="ask_done", iterations=record.iterations,
            tool_calls=record.tool_calls, tokens_in=record.tokens_in,
            tokens_out=record.tokens_out, answer_chars=len(outcome.answer),
        )
        _persist(settings, run_id, outcome, calls, record, spans)
        return AskResponse(
            run_id=run_id, trace_id=record.trace_id, answer=outcome.answer,
            tool_calls=calls, accounting=record,
        )

    # ---- trace -------------------------------------------------------
    @app.get(
        "/v1/runs/{run_id}",
        response_model=AskResponse,
        responses={404: {"model": ErrorResponse}},
    )
    def get_run(run_id: str) -> Any:
        settings: api_config.Settings = app.state.settings
        path = Path(settings.run_store) / f"{run_id}.json"
        # `run_id` reaches the filesystem, so it is checked rather than trusted:
        # a `..` in it would read outside the store.
        if not _safe_run_id(run_id) or not path.exists():
            return _error(ErrorCode.NOT_FOUND, f"no run {run_id!r}")
        stored = json.loads(path.read_text(encoding="utf-8"))
        return AskResponse(**stored)

    # ---- health ------------------------------------------------------
    @app.get("/health", response_model=HealthResponse)
    def health() -> Any:
        settings: api_config.Settings = app.state.settings
        result = _health(settings)
        structured.info(
            "health", event="health", healthy=result.status == "ok",
            provider=result.provider,
        )
        if result.status != "ok":
            return JSONResponse(
                status_code=503, content=json.loads(result.model_dump_json())
            )
        return result

    return app


def _safe_run_id(run_id: str) -> bool:
    return run_id.replace("_", "").replace("-", "").isalnum() and len(run_id) <= 64


def _persist(settings, run_id, outcome, calls, record, spans) -> None:
    """Write the run so `GET /v1/runs/{id}` can serve it. Best effort."""
    store = Path(settings.run_store)
    try:
        store.mkdir(parents=True, exist_ok=True)
        payload = AskResponse(
            run_id=run_id, trace_id=record.trace_id, answer=outcome.answer,
            tool_calls=calls, accounting=record,
        )
        (store / f"{run_id}.json").write_text(
            payload.model_dump_json(indent=2), encoding="utf-8"
        )
        (store / f"{run_id}.spans.json").write_text(
            json.dumps(spans, indent=2), encoding="utf-8"
        )
    except OSError as exc:
        # A read-only store must not fail the answer the caller already has.
        structured.warning("run not persisted", event="persist_failed",
                           detail=type(exc).__name__)


def _health(settings) -> HealthResponse:
    """Is this process able to do its job?

    Two real checks, both cheap and both exercising the thing rather than
    asserting it exists: the database is *opened and queried* read-only, and the
    model artefacts are *loaded*, not stat-ed.

    **The configured provider is reported; whether a credential is present is
    not.** A health endpoint is usually unauthenticated, and "a key is
    configured" is a fact about the deployment that helps an attacker decide
    whether this host is worth more of their time. The operator can read the
    provider from their own environment; they do not need this endpoint to
    confirm their secret arrived.
    """
    checks: dict[str, bool] = {}
    detail: dict[str, str] = {}

    try:
        from src.agent import db as db_module

        path = Path(settings.database) if settings.database else db_module.DEFAULT_DB
        # `read_only` is the same guarded connection the tools use: read-only
        # URI plus the authorizer allowlist. The health check must not open a
        # more permissive connection than the service itself gets.
        with db_module.read_only(path) as connection:
            connection.execute("SELECT 1 FROM machines LIMIT 1").fetchone()
        checks["database_readable"] = True
    except Exception as exc:  # noqa: BLE001 - reported, never raised
        checks["database_readable"] = False
        detail["database"] = type(exc).__name__

    try:
        from src.agent import risk as risk_module

        risk_module._artefacts()
        checks["model_artefacts_load"] = True
    except Exception as exc:  # noqa: BLE001
        checks["model_artefacts_load"] = False
        detail["model"] = type(exc).__name__

    return HealthResponse(
        status="ok" if all(checks.values()) else "degraded",
        checks=checks,
        provider=settings.provider,
        model=settings.model_name,
        detail=detail,
    )


app = create_app()
