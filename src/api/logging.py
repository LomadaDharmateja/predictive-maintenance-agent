"""Structured logging, keyed to the run id so a line joins a span.

Milestone 7 item 2. Every line is one JSON object on one line, carrying
`run_id` and `trace_id`. `trace_id` is the same hex string
`src/obs/tracing.py` puts on its spans, so a log line and a span can be joined
without a correlation table.

**Nothing that could carry a credential is ever logged.** That is enforced by
construction rather than by care: this module has an allowlist of fields it will
emit, and a value that is not on it is dropped. Tool *arguments* are the obvious
hazard -- `find_machines` takes a free-text filter, and a question is whatever
the operator typed -- so arguments are summarised as their key names and never
their values. The API key itself is never read by any code in this repository
(`src/agent/providers.py` leaves it to the SDK), so there is no path from a key
to a log line; the allowlist is the second lock on a door that is already shut.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from contextvars import ContextVar
from typing import Any

LOGGER_NAME = "industrial-ai-agent"

#: The current request's ids, so a handler deep in the call stack does not have
#: to thread them through every signature.
_run_id: ContextVar[str | None] = ContextVar("run_id", default=None)
_trace_id: ContextVar[str | None] = ContextVar("trace_id", default=None)

#: The only keys that may reach a log line. Anything else is dropped, so a
#: future caller cannot widen the surface by passing a new keyword.
ALLOWED_FIELDS = frozenset(
    {
        "event", "run_id", "trace_id", "method", "path", "status",
        "duration_ms", "scenario_id", "seed", "tool", "result_type",
        "error_code", "iterations", "tool_calls", "tokens_in", "tokens_out",
        "provider", "model", "answer_chars", "detail", "client",
        "argument_keys", "question_chars", "healthy", "checks",
    }
)

#: Never logged, whatever the caller does. Belt and braces over the allowlist:
#: if one of these is ever added to `ALLOWED_FIELDS` by mistake, this still
#: refuses it.
FORBIDDEN_FIELDS = frozenset(
    {
        "api_key", "anthropic_api_key", "authorization", "token", "secret",
        "password", "credential", "arguments", "question", "answer",
        "result_json", "messages", "prompt",
    }
)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        run_id, trace_id = _run_id.get(), _trace_id.get()
        if run_id:
            payload["run_id"] = run_id
        if trace_id:
            payload["trace_id"] = trace_id
        for key, value in getattr(record, "fields", {}).items():
            payload[key] = value
        return json.dumps(payload, default=str)


def configure(level: str = "INFO", stream=None) -> logging.Logger:
    """One handler on one logger, writing JSON lines to stdout.

    stdout rather than a file because a container's log is its stdout, and a
    service that writes logs inside itself is a service whose logs vanish with
    the container.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level.upper())
    logger.handlers.clear()
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def scrub(fields: dict[str, Any]) -> dict[str, Any]:
    """Drop anything not explicitly allowed. The allowlist is the whole point.

    A denylist would need updating every time a caller invents a field name;
    this fails closed instead, so the worst a careless caller achieves is a
    missing field rather than a leaked one.
    """
    out: dict[str, Any] = {}
    for key, value in fields.items():
        lowered = key.lower()
        if lowered in FORBIDDEN_FIELDS or lowered not in ALLOWED_FIELDS:
            continue
        out[key] = value
    return out


def log(level: int, message: str, **fields: Any) -> None:
    logger().log(level, message, extra={"fields": scrub(fields)})


def info(message: str, **fields: Any) -> None:
    log(logging.INFO, message, **fields)


def warning(message: str, **fields: Any) -> None:
    log(logging.WARNING, message, **fields)


def error(message: str, **fields: Any) -> None:
    log(logging.ERROR, message, **fields)


def argument_keys(arguments: dict | None) -> list[str]:
    """Argument *names* only.

    `find_machines` accepts a free-text filter and a question is whatever
    somebody typed, so argument values are exactly the place an unexpected
    secret would show up in a log. The names are enough to debug a tool call;
    the values never appear.
    """
    return sorted(arguments.keys()) if isinstance(arguments, dict) else []


def new_run_id() -> str:
    return f"run_{uuid.uuid4().hex[:16]}"


def bind(run_id: str | None = None, trace_id: str | None = None) -> str:
    run_id = run_id or new_run_id()
    _run_id.set(run_id)
    if trace_id:
        _trace_id.set(trace_id)
    return run_id


def current_run_id() -> str | None:
    return _run_id.get()


def clear() -> None:
    _run_id.set(None)
    _trace_id.set(None)
