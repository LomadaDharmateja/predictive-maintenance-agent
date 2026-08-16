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
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.agent.contracts import ErrorCode, ToolError
from src.api import config as api_config
from src.api import demo
from src.api import errors as api_errors
from src.api import logging as structured
from src.obs import accounting, tracing

#: The demo page's files. Inside the package so the container copies them with
#: the code rather than needing a separate COPY that can be forgotten.
STATIC_DIR = Path(__file__).parent / "static"

# ----------------------------------------------------------------------
# Request and response models -- thin wrappers over the existing contracts
# ----------------------------------------------------------------------


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AskRequest(Strict):
    #: Optional only because `scenario_id` can stand in for it. Exactly one of
    #: the two is required, enforced below rather than by convention.
    question: str | None = Field(default=None, min_length=3, max_length=2000)
    #: The prediction time. Required in practice: without it the model has no
    #: way to know what "now" is and will guess -- a pilot run guessed
    #: `2025-01-01`, outside the dataset entirely.
    as_of: str | None = None
    #: Demo mode only. Names the recorded run to replay. The scenario carries
    #: its own question and prediction time, so both are taken from it and any
    #: `question` sent alongside is ignored -- a replayed run must answer the
    #: question that was recorded, not one supplied at request time.
    scenario_id: str | None = None

    @model_validator(mode="after")
    def _one_of(self) -> "AskRequest":
        if not self.question and not self.scenario_id:
            raise ValueError("one of 'question' or 'scenario_id' is required")
        return self


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
    #: The two fields the risk contract exists to carry -- adequacy of the
    #: warning, and whether the probability is trustworthy -- lifted out of the
    #: tool results so the UI can render them as badges rather than bury them
    #: in JSON. Derived by `demo.highlights` from `tool_calls`; this is a view,
    #: not a second declaration, and it is empty when no risk tool ran.
    highlights: list[dict] = Field(default_factory=list)
    #: True when the model turns came from a recorded transcript.
    replayed: bool = False


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

        # ---- demo mode: replay, never call a provider -------------------
        if settings.demo_mode:
            if not body.scenario_id:
                failure = demo.not_a_preset_error(body.question or "")
                structured.info("demo rejected free text", event="ask_demo_rejected")
                return _error(failure.code, failure.message)

            exporter = tracing.MemorySpanExporter()
            tracing.configure(exporter, reset=True)
            structured.info(
                "ask", event="ask_start", provider="replay",
                scenario_id=body.scenario_id,
            )
            try:
                outcome, identity, replay_client = demo.replay(
                    body.scenario_id,
                    database=Path(settings.database),
                    transcripts=Path(settings.transcripts),
                    scenarios_path=Path(settings.scenarios),
                )
            except demo.DemoUnavailable as exc:
                structured.error("no transcript", event="ask_error", detail=str(exc))
                return _error(ErrorCode.NOT_FOUND, str(exc))
            except Exception as exc:  # noqa: BLE001 - mapped, never swallowed
                structured.error("replay failed", event="ask_error",
                                 detail=type(exc).__name__)
                return _error(ErrorCode.INTERNAL, f"{type(exc).__name__} during replay")

            return _respond(
                settings, run_id, outcome, exporter.spans, identity,
                replayed=True,
            )

        # ---- live: call the configured provider -------------------------
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
            question_chars=len(body.question or ""),
        )
        try:
            agent = Agent(
                client, database=Path(settings.database),
                config=LoopConfig(max_iterations=settings.max_iterations),
            )
            outcome = agent.run(body.question or "", as_of=body.as_of)
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

        return _respond(
            settings, run_id, outcome, exporter.spans,
            getattr(client, "identity", None), replayed=False,
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

    # ---- demo UI -----------------------------------------------------
    @app.get("/", include_in_schema=False)
    def index() -> Any:
        """The demo page. One HTML file, one JS file, no build step, no CDN.

        Served from disk rather than templated so there is nothing to compile
        and nothing to keep in step with a renderer.
        """
        page = STATIC_DIR / "index.html"
        if not page.exists():
            return _error(ErrorCode.NOT_FOUND, "the demo page is not installed")
        return FileResponse(page, media_type="text/html")

    @app.get("/app.js", include_in_schema=False)
    def script() -> Any:
        """The one script. Named explicitly rather than mounted as a directory
        so nothing else in the package can be reached over HTTP."""
        script_path = STATIC_DIR / "app.js"
        if not script_path.exists():
            return _error(ErrorCode.NOT_FOUND, "app.js is not installed")
        return FileResponse(script_path, media_type="application/javascript")

    @app.get("/v1/demo/presets", include_in_schema=False)
    def presets() -> Any:
        settings: api_config.Settings = app.state.settings
        return {
            "demo_mode": settings.demo_mode,
            "provider": "replay" if settings.demo_mode else settings.provider,
            "model": "recorded transcript" if settings.demo_mode else settings.model_name,
            "presets": demo.available(
                transcripts=Path(settings.transcripts),
                scenarios_path=Path(settings.scenarios),
            ),
        }

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


def _respond(settings, run_id, outcome, spans, identity, replayed: bool) -> AskResponse:
    """Turn a finished run into the response body. One path, both modes.

    The demo and live paths differ only in where the model's turns came from.
    Everything after that -- the tool-call view, the accounting, the persisted
    record -- is this function, so a replayed run cannot be shaped differently
    from a live one.
    """
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

    # Accounting is read from the spans, never recomputed here. Milestone 6
    # item 5 made `src/obs/accounting.py` the single place a token is priced,
    # and a UI that did its own arithmetic would be the second.
    records = accounting.from_spans(spans, run_id=run_id, identity=identity)
    record = records[0] if records else accounting.RunAccounting(run_id=run_id)
    structured.bind(run_id, record.trace_id)
    structured.info(
        "answered", event="ask_done", iterations=record.iterations,
        tool_calls=record.tool_calls, tokens_in=record.tokens_in,
        tokens_out=record.tokens_out, answer_chars=len(outcome.answer),
        replayed=replayed,
    )
    payload = AskResponse(
        run_id=run_id, trace_id=record.trace_id, answer=outcome.answer,
        tool_calls=calls, accounting=record,
        highlights=demo.highlights(calls), replayed=replayed,
    )
    _persist(settings, run_id, payload, spans)
    return payload


def _safe_run_id(run_id: str) -> bool:
    return run_id.replace("_", "").replace("-", "").isalnum() and len(run_id) <= 64


def _persist(settings, run_id, payload: "AskResponse", spans) -> None:
    """Write the run so `GET /v1/runs/{id}` can serve it. Best effort.

    The *served* payload is written, not a rebuilt copy. Rebuilding it here is
    how the stored run silently lost `highlights` and `replayed`: a refetched
    run reported itself as live when it had been replayed.
    """
    store = Path(settings.run_store)
    try:
        store.mkdir(parents=True, exist_ok=True)
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
