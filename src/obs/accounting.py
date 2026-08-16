"""Per-run accounting, and the one place a token is priced.

Milestone 6 item 5: the evaluation harness's cost and latency figures must come
from this instrumentation rather than being computed beside it, so there is one
source of truth. `evals/runner.py` previously carried its own price table and
did its own arithmetic; that table now lives here and the runner asks for a
figure rather than working one out.

**Accounting is an aggregate and deliberately holds no per-call detail.**
Milestone 6 item 2 says to persist it *alongside* the existing trace format
rather than duplicating it. The traces file already carries every tool call,
its arguments and its result, and the spans file carries the timing tree. A
third copy of the same rows would be a third thing to keep in step. What is
here is only what neither of those gives you directly: the totals for one run.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from src.obs import tracing

# ----------------------------------------------------------------------
# Pricing -- moved here from evals/runner.py so it is defined once
# ----------------------------------------------------------------------

#: Keyed by model where the model is known, because a tier's prices differ by
#: 5x and a run priced at the wrong tier answers the wrong budgeting question.
#: Standard list prices; any introductory discount is deliberately not encoded,
#: so the figure is the one that survives the promotion ending.
PRICES_PER_1K: dict[str, tuple[float, float]] = {
    "claude-opus-5": (0.005, 0.025),
    "claude-opus-4-8": (0.005, 0.025),
    "claude-sonnet-5": (0.003, 0.015),
    "claude-sonnet-4-6": (0.003, 0.015),
    "claude-haiku-4-5": (0.001, 0.005),
}
PROVIDER_PRICES_PER_1K: dict[str, tuple[float, float]] = {
    "ollama": (0.0, 0.0),  # zero because it is zero
}
DEFAULT_PRICE = (0.003, 0.015)

#: What a cached input token costs relative to an uncached one.
CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_MULTIPLIER = 1.25


def price_for(identity) -> tuple[float, float]:
    """(input, output) price per 1K tokens for the model that produced a run."""
    if identity is None:
        return DEFAULT_PRICE
    provider = getattr(identity, "provider", None)
    if provider in PROVIDER_PRICES_PER_1K:
        return PROVIDER_PRICES_PER_1K[provider]
    model = getattr(identity, "model", "") or ""
    for known, price in PRICES_PER_1K.items():
        if model.startswith(known):
            return price
    return DEFAULT_PRICE


def cost_usd(
    identity,
    tokens_in: int,
    tokens_out: int,
    cache_read: int = 0,
    cache_write: int = 0,
) -> float:
    """Priced with cache tokens at their own rates, not at the input rate.

    Charging a cache read at full price reports a saving as if it had not been
    made, which is the opposite of what the cost column exists for.
    """
    price_in, price_out = price_for(identity)
    return round(
        tokens_in / 1000 * price_in
        + cache_read / 1000 * price_in * CACHE_READ_MULTIPLIER
        + cache_write / 1000 * price_in * CACHE_WRITE_MULTIPLIER
        + tokens_out / 1000 * price_out,
        6,
    )


# ----------------------------------------------------------------------
# The accounting record
# ----------------------------------------------------------------------


class RunAccounting(BaseModel):
    """What one run consumed. Milestone 6 item 2."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    run_id: str
    scenario_id: str | None = None
    seed: int | None = None
    trace_id: str | None = None

    tokens_in: int = 0
    tokens_out: int = 0
    cache_read: int = 0
    cache_write: int = 0
    estimated_cost_usd: float = 0.0

    wall_clock_ms: float = 0.0
    #: Time inside `model.call` spans. The remainder is tool execution and loop
    #: overhead, and the gap between the two is what says where a slow run went.
    model_ms: float = 0.0
    tool_ms: float = 0.0

    tool_calls: int = 0
    tool_errors: int = 0

    iterations: int = 0
    max_iterations: int = 0
    hit_iteration_limit: bool = False

    provider: str | None = None
    model: str | None = None

    @property
    def iteration_headroom(self) -> str:
        """Iterations used against the ceiling. `4/6` reads at a glance."""
        return f"{self.iterations}/{self.max_iterations}"

    @property
    def overhead_ms(self) -> float:
        return round(max(0.0, self.wall_clock_ms - self.model_ms - self.tool_ms), 3)


def from_spans(spans: list[dict], run_id: str, identity=None) -> list[RunAccounting]:
    """Build one accounting record per `agent.run` span.

    The spans are the source. Nothing here re-reads a transcript or a trace,
    so the totals cannot drift from the timing tree they describe.
    """
    runs = [s for s in spans if s["name"] == tracing.RUN]
    by_parent: dict[str, list[dict]] = {}
    for s in spans:
        parent = s.get("parent_span_id")
        if parent:
            by_parent.setdefault(parent, []).append(s)

    out: list[RunAccounting] = []
    for root in runs:
        children = by_parent.get(root["span_id"], [])
        model_calls = [c for c in children if c["name"] == tracing.MODEL_CALL]
        tool_calls = [c for c in children if c["name"] == tracing.TOOL_CALL]
        attributes = root["attributes"]

        def total(key: str) -> int:
            return int(sum(c["attributes"].get(key, 0) for c in model_calls))

        record = RunAccounting(
            run_id=run_id,
            scenario_id=attributes.get("pdm.scenario_id"),
            seed=attributes.get("pdm.seed"),
            trace_id=root["trace_id"],
            tokens_in=total("gen_ai.usage.input_tokens"),
            tokens_out=total("gen_ai.usage.output_tokens"),
            cache_read=total("pdm.cache_read_tokens"),
            cache_write=total("pdm.cache_write_tokens"),
            wall_clock_ms=root.get("duration_ms") or 0.0,
            model_ms=round(sum(c.get("duration_ms") or 0.0 for c in model_calls), 3),
            tool_ms=round(sum(c.get("duration_ms") or 0.0 for c in tool_calls), 3),
            tool_calls=len(tool_calls),
            tool_errors=sum(
                1 for c in tool_calls if c["attributes"].get("pdm.result_type") == "ToolError"
            ),
            iterations=int(attributes.get("pdm.iterations", 0)),
            max_iterations=int(attributes.get("pdm.max_iterations", 0)),
            hit_iteration_limit=bool(attributes.get("pdm.hit_iteration_limit", False)),
            provider=getattr(identity, "provider", None),
            model=getattr(identity, "model", None),
        )
        record.estimated_cost_usd = cost_usd(
            identity,
            record.tokens_in,
            record.tokens_out,
            record.cache_read,
            record.cache_write,
        )
        out.append(record)
    return out


def write(records: list[RunAccounting], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([json.loads(r.model_dump_json()) for r in records], indent=2),
        encoding="utf-8",
    )
    return path


def read(path: Path) -> list[RunAccounting]:
    return [RunAccounting(**r) for r in json.loads(path.read_text(encoding="utf-8"))]


def totals(records: list[RunAccounting]) -> dict:
    """Suite-level roll-up. Used by the viewer and the summary line."""
    return {
        "runs": len(records),
        "tokens_in": sum(r.tokens_in for r in records),
        "tokens_out": sum(r.tokens_out for r in records),
        "cache_read": sum(r.cache_read for r in records),
        "cache_write": sum(r.cache_write for r in records),
        "estimated_cost_usd": round(sum(r.estimated_cost_usd for r in records), 6),
        "wall_clock_ms": round(sum(r.wall_clock_ms for r in records), 3),
        "tool_calls": sum(r.tool_calls for r in records),
        "tool_errors": sum(r.tool_errors for r in records),
        "hit_iteration_limit": sum(1 for r in records if r.hit_iteration_limit),
    }
