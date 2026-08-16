"""OpenTelemetry tracing for the agent loop.

`docs/MILESTONE_4.md` section 5 required "a structured run log of every call and
result, which Milestone 6 builds on". This is that build: the run log stays as
the in-process record the agent itself reads, and every entry in it now also
becomes a span.

**Offline by default, and that is a design constraint rather than a default.**
The tracer provider ships with no exporter at all unless one is configured, and
the exporters provided here write to a local file or to memory. Nothing in this
module opens a socket. CLAUDE.md's determinism rule -- "no clock, no network, no
environment in the pipeline" -- is about the modelling pipeline, but a tracing
layer that phoned home would make `pytest` require network access, which the
milestone forbids outright.

An OTLP collector can be attached by the caller if one is ever wanted; the
provider is returned so that stays possible without this module importing an
exporter that needs a network stack.

## Span shape

    agent.run                        one per question answered
      model.call                     one per loop iteration
      tool.call                      one per dispatch

Attributes are prefixed `pdm.` where they are this project's own. The generic
token counters follow the GenAI semantic-convention names so a collector that
understands them can aggregate without a mapping.
"""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

TRACER_NAME = "industrial-ai-agent"

#: Span names. Constants because the viewer, the accounting and the tests all
#: key on them, and a typo would silently produce an empty report.
RUN = "agent.run"
MODEL_CALL = "model.call"
TOOL_CALL = "tool.call"

_lock = threading.Lock()
_provider: TracerProvider | None = None


class JsonFileSpanExporter(SpanExporter):
    """Writes finished spans to a JSON-lines file. No network, no buffering.

    One span per line, appended as it finishes, so a crashed run still leaves a
    readable trace of everything that completed before it died -- the same
    reason `evals/record.py` writes each transcript the moment its scenario
    finishes rather than at the end of the sweep.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")

    def export(self, spans) -> SpanExportResult:
        for span in spans:
            self._file.write(json.dumps(to_dict(span)) + "\n")
        self._file.flush()
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        try:
            self._file.close()
        except Exception:  # noqa: BLE001 - shutdown must not raise
            pass

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        self._file.flush()
        return True


class MemorySpanExporter(SpanExporter):
    """Keeps spans in a list. What the tests use, so no file is touched."""

    def __init__(self) -> None:
        self.spans: list[dict] = []

    def export(self, spans) -> SpanExportResult:
        self.spans.extend(to_dict(span) for span in spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True


def to_dict(span: ReadableSpan) -> dict:
    """A span as plain JSON. Ids are hex so a viewer can match parent to child."""
    context = span.get_span_context()
    parent = span.parent
    return {
        "name": span.name,
        "trace_id": f"{context.trace_id:032x}",
        "span_id": f"{context.span_id:016x}",
        "parent_span_id": f"{parent.span_id:016x}" if parent else None,
        "start_unix_nano": span.start_time,
        "end_unix_nano": span.end_time,
        "duration_ms": round((span.end_time - span.start_time) / 1e6, 3)
        if span.end_time and span.start_time
        else None,
        "status": span.status.status_code.name if span.status else "UNSET",
        "attributes": dict(span.attributes or {}),
    }


def configure(exporter: SpanExporter | None = None, *, reset: bool = False) -> TracerProvider:
    """Install a tracer provider. Idempotent unless `reset` is passed.

    `SimpleSpanProcessor` rather than the batching one on purpose: a batch
    processor exports on a background thread and a short-lived CLI can exit
    before it flushes, which loses exactly the spans somebody was running the
    command to see.
    """
    global _provider
    with _lock:
        if _provider is not None and not reset:
            if exporter is not None:
                _provider.add_span_processor(SimpleSpanProcessor(exporter))
            return _provider
        provider = TracerProvider()
        if exporter is not None:
            provider.add_span_processor(SimpleSpanProcessor(exporter))
        # set_tracer_provider only takes effect once per process; the local
        # reference is what this module hands out, so a reset still works.
        try:
            trace.set_tracer_provider(provider)
        except Exception:  # noqa: BLE001 - already set is not an error here
            pass
        _provider = provider
        return provider


def tracer() -> trace.Tracer:
    """The tracer. Safe before `configure`: the API no-ops without a provider."""
    if _provider is not None:
        return _provider.get_tracer(TRACER_NAME)
    return trace.get_tracer(TRACER_NAME)


def shutdown() -> None:
    global _provider
    with _lock:
        if _provider is not None:
            _provider.shutdown()
            _provider = None


def _clean(attributes: dict[str, Any]) -> dict[str, Any]:
    """Drop Nones and coerce to what OTel accepts as an attribute value."""
    out: dict[str, Any] = {}
    for key, value in attributes.items():
        if value is None:
            continue
        if isinstance(value, (str, bool, int, float)):
            out[key] = value
        else:
            out[key] = json.dumps(value, default=str)
    return out


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[trace.Span]:
    """Start a span with attributes, ending it however the block exits."""
    with tracer().start_as_current_span(name, attributes=_clean(attributes)) as current:
        yield current


def set_attributes(current: trace.Span, **attributes: Any) -> None:
    for key, value in _clean(attributes).items():
        current.set_attribute(key, value)
