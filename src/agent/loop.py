"""The agent loop. Written here rather than adopted, so every decision in it can
be defended line by line.

`docs/MILESTONE_4.md` section 5. v1 used `AgentExecutor` with
`ConversationBufferMemory`, which grows without limit, and
`max_execution_time=None`. The control flow was somebody else's and its failure
modes were inherited rather than chosen.

What this loop guarantees:

- **Bounded iterations.** `max_iterations` with a *defined* terminal behaviour:
  on exhaustion it returns a stated limitation naming what it was still doing,
  never a silently truncated answer.
- **Bounded state.** Transcript trimmed to `max_history_messages`, oldest tool
  exchanges dropped first, with the drop recorded in the run log so a short
  answer is never mistaken for a complete one.
- **Typed failure handling.** A `ToolError` is a first-class branch. The loop may
  retry a `retryable` error once, but a validation error is never retried -- the
  same arguments will fail identically and a retry only burns an iteration.
- **Provider independence.** Everything model-shaped goes through `LLMClient`, so
  swapping provider touches one file. A local open-weights configuration is
  included because data sovereignty is a live constraint for the employers this
  project targets, and designing for it costs nothing now.
- **A structured run log** of every call and result, which Milestone 6 builds on.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

from src.agent.contracts import ErrorCode, ToolError
from src.agent import tools as tools_module
from src.agent.tools import REGISTRY, serialise
from src.obs import tracing

PROMPT_PATH = Path(__file__).parent / "prompts" / "system_prompt.md"


# ----------------------------------------------------------------------
# Provider-agnostic model interface
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict


@dataclass(frozen=True)
class LLMResponse:
    """Either a final answer or a request to call tools, never both."""

    text: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class LLMClient(Protocol):
    """The only surface the loop knows about.

    Implementations exist for a hosted provider and for a local open-weights
    server; the loop cannot tell them apart. Tests use a scripted stub, which is
    why the suite needs no network.
    """

    def complete(self, messages: list[dict], tools: list[dict]) -> LLMResponse: ...


@dataclass(frozen=True)
class ModelConfig:
    """Generation settings. Temperature 0 and a seed wherever the provider
    honours one -- an agent whose tool choices vary run to run cannot be
    evaluated."""

    provider: str = "anthropic"
    model: str = "claude-opus-5"
    #: Requested, not guaranteed. Anthropic removed sampling parameters on its
    #: newest models, so `src/agent/providers.py` omits this rather than
    #: sending a value the API rejects -- and records that it did.
    temperature: float = 0.0
    seed: int | None = 20240608
    timeout_seconds: float = 120.0
    #: Room for thinking plus the answer. `max_tokens` caps both together, and
    #: a tight budget truncates the answer rather than the reasoning.
    max_tokens: int = 8192

    #: The local open-weights path. Exercised by `evals/record.py --provider
    #: ollama`, because "can this run entirely on our own hardware" is a
    #: question that gets asked before a pilot, not after.
    local_base_url: str | None = "http://127.0.0.1:11434"
    local_model: str | None = "qwen3:4b-q4_K_M"


@dataclass
class LoopConfig:
    max_iterations: int = 6
    max_history_messages: int = 24
    max_tool_retries: int = 1
    retry_backoff_seconds: float = 0.5


# ----------------------------------------------------------------------
# Run log
# ----------------------------------------------------------------------


@dataclass
class LogEntry:
    iteration: int
    kind: str
    name: str | None = None
    arguments: dict | None = None
    status: str | None = None
    error_code: str | None = None
    truncated: bool = False
    duration_ms: float | None = None
    detail: str | None = None


@dataclass
class RunLog:
    entries: list[LogEntry] = field(default_factory=list)

    def record(self, **kwargs) -> None:
        self.entries.append(LogEntry(**kwargs))

    def to_json(self) -> str:
        return json.dumps([asdict(e) for e in self.entries], indent=2)


@dataclass
class AgentResult:
    answer: str
    iterations: int
    hit_iteration_limit: bool
    log: RunLog
    messages_dropped: int = 0


# ----------------------------------------------------------------------


def load_system_prompt(path: Path = PROMPT_PATH) -> str:
    """Loaded from disk, never duplicated inline.

    `tests/test_agent_loop.py` asserts the file is read and that no module
    embeds a copy, so the prompt can be versioned and diffed like anything else.
    """
    if not path.exists():
        raise FileNotFoundError(f"system prompt missing: {path}")
    return path.read_text(encoding="utf-8")


def tool_schemas() -> list[dict]:
    """JSON schemas for the registered tools, generated from the Pydantic models
    so a schema cannot drift from the function it describes."""
    return [
        {
            "name": name,
            "description": (function.__doc__ or "").strip().split("\n")[0],
            "input_schema": model.model_json_schema(),
        }
        for name, (model, function) in sorted(REGISTRY.items())
    ]


def _trim(messages: list[dict], limit: int) -> tuple[list[dict], int]:
    """Keep the system message and the most recent exchanges.

    The system message is pinned: dropping it would silently remove every rule
    the agent is meant to follow, which is the worst possible thing to lose to a
    length cap.
    """
    if len(messages) <= limit:
        return messages, 0
    head, tail = messages[:1], messages[1:]
    keep = limit - 1
    return head + tail[-keep:], len(tail) - keep


class Agent:
    def __init__(
        self,
        client: LLMClient,
        database: Path | None = None,
        config: LoopConfig | None = None,
        prompt_path: Path = PROMPT_PATH,
    ) -> None:
        self.client = client
        self.config = config or LoopConfig()
        self.database = database
        self.system_prompt = load_system_prompt(prompt_path)

    #: How the caller tells the agent what "now" is. Appended to the versioned
    #: system prompt rather than written into it, because it is the one piece
    #: of genuinely per-run context and the prompt file has to stay diffable.
    AS_OF_TEMPLATE = (
        "\n\n## Prediction time\n\n"
        "The current time is {as_of}. Every tool that takes an `as_of` argument "
        "must be given exactly this value. Do not substitute today's date, and "
        "do not infer a time from the question: this system reasons about a "
        "fixed historical point, and a different `as_of` answers a different "
        "question.\n"
    )

    def run(
        self,
        question: str,
        as_of=None,
        scenario_id: str | None = None,
        seed: int | None = None,
    ) -> AgentResult:
        """Answer `question` as of `as_of`.

        `as_of` is not optional in practice, only in signature. Without it the
        model has no way to know what "now" is, and it will guess -- the pilot
        run against a hosted model guessed `2025-01-01`, which is outside the
        dataset entirely. The scenario always carried the prediction time; the
        harness simply never passed it on.
        """
        from src.agent import db as db_module

        database = self.database or db_module.DEFAULT_DB
        log = RunLog()
        # `scenario_id` and `seed` are labels only -- the loop never behaves
        # differently for them. They exist so a span can be found again.
        self._run_span_labels = {
            "pdm.scenario_id": scenario_id,
            "pdm.seed": seed,
        }
        system = self.system_prompt
        if as_of is not None:
            system += self.AS_OF_TEMPLATE.format(
                as_of=as_of.isoformat() if hasattr(as_of, "isoformat") else as_of
            )
        messages: list[dict] = [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ]
        dropped_total = 0
        schemas = tool_schemas()

        with tracing.span(
            tracing.RUN,
            **{
                "pdm.question": question,
                "pdm.as_of": as_of.isoformat() if hasattr(as_of, "isoformat") else as_of,
                "pdm.scenario_id": scenario_id,
                "pdm.seed": seed,
                "pdm.max_iterations": self.config.max_iterations,
            },
        ) as run_span:
            return self._iterate(
                question, messages, schemas, database, log, dropped_total, run_span
            )

    def _iterate(
        self, question, messages, schemas, database, log, dropped_total, run_span
    ) -> AgentResult:
        """The loop proper. Split out so the root span wraps the whole run
        including the terminal branches, without indenting the original body
        past readability."""
        for iteration in range(1, self.config.max_iterations + 1):
            messages, dropped = _trim(messages, self.config.max_history_messages)
            if dropped:
                dropped_total += dropped
                log.record(
                    iteration=iteration,
                    kind="history_trimmed",
                    detail=f"dropped {dropped} older messages",
                )

            started = time.perf_counter()
            with tracing.span(tracing.MODEL_CALL, **{"pdm.iteration": iteration}) as call_span:
                response = self.client.complete(messages, schemas)
                # Usage comes off the adapter rather than being counted here:
                # the provider is the only thing that knows what it billed, and
                # a locally re-tokenised estimate would be a second number that
                # disagrees with the invoice.
                usage = getattr(self.client, "last_exchange", None) or {}
                tracing.set_attributes(
                    call_span,
                    **{
                        "gen_ai.usage.input_tokens": usage.get("tokens_in"),
                        "gen_ai.usage.output_tokens": usage.get("tokens_out"),
                        "pdm.cache_read_tokens": usage.get("cache_read"),
                        "pdm.cache_write_tokens": usage.get("cache_write"),
                        "pdm.stop_reason": usage.get("stop_reason"),
                        "pdm.wants_tools": response.wants_tools,
                        "pdm.tool_call_count": len(response.tool_calls),
                    },
                )
            log.record(
                iteration=iteration,
                kind="model_call",
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
                detail=f"{len(response.tool_calls)} tool call(s)"
                if response.wants_tools
                else "final answer",
            )

            if not response.wants_tools:
                tracing.set_attributes(
                    run_span,
                    **{
                        "pdm.iterations": iteration,
                        "pdm.hit_iteration_limit": False,
                        "pdm.answer_chars": len(response.text or ""),
                    },
                )
                return AgentResult(
                    answer=response.text or "",
                    iterations=iteration,
                    hit_iteration_limit=False,
                    log=log,
                    messages_dropped=dropped_total,
                )

            # The assistant's own turn goes into the transcript before its
            # results do. Without it the conversation is a user message
            # followed by tool results that answer nothing -- which a scripted
            # stub tolerates and a real provider rejects, because a tool result
            # has to refer back to the call that produced it.
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"name": call.name, "arguments": call.arguments}
                        for call in response.tool_calls
                    ],
                }
            )
            for call in response.tool_calls:
                rendered = self._call_tool(call, database, log, iteration)
                messages.append(
                    {"role": "tool", "name": call.name, "content": rendered}
                )

        # Terminal behaviour on exhaustion is defined, and it is not silence.
        tracing.set_attributes(
            run_span,
            **{
                "pdm.iterations": self.config.max_iterations,
                "pdm.hit_iteration_limit": True,
            },
        )
        log.record(
            iteration=self.config.max_iterations,
            kind="iteration_limit",
            detail="max_iterations reached",
        )
        return AgentResult(
            answer=(
                f"I reached the {self.config.max_iterations}-step limit for this "
                "question without arriving at a complete answer. I have not "
                "guessed the remainder. What I retrieved is in the run log; "
                "please narrow the question or raise the step limit."
            ),
            iterations=self.config.max_iterations,
            hit_iteration_limit=True,
            log=log,
            messages_dropped=dropped_total,
        )

    def _call_tool(self, call: ToolCall, database: Path, log: RunLog, iteration: int) -> str:
        attempts = 0
        while True:
            started = time.perf_counter()
            # Resolved through the module on every call, never bound at import;
            # see the note that used to live here, now in the dispatch call.
            with tracing.span(
                tracing.TOOL_CALL,
                **{
                    "pdm.tool": call.name,
                    "pdm.arguments": call.arguments,
                    "pdm.iteration": iteration,
                    "pdm.attempt": attempts + 1,
                },
            ) as tool_span:
                result = tools_module.dispatch(call.name, call.arguments, database)
                # The discriminated union is the interesting attribute: a
                # `ToolError` that read as a success is the v1 defect this
                # project was rebuilt to prevent, and a trace that did not
                # distinguish them would hide it again.
                tracing.set_attributes(
                    tool_span,
                    **{
                        "pdm.result_type": type(result).__name__,
                        "pdm.error_code": getattr(getattr(result, "code", None), "value", None),
                        "pdm.truncated": bool(getattr(result, "truncated", False)),
                    },
                )
            elapsed = round((time.perf_counter() - started) * 1000, 2)

            if isinstance(result, ToolError):
                log.record(
                    iteration=iteration,
                    kind="tool_error",
                    name=call.name,
                    arguments=call.arguments,
                    status="error",
                    error_code=result.code.value,
                    duration_ms=elapsed,
                    detail=result.message,
                )
                # Retry transient failures only. A validation error retried is a
                # wasted iteration with an identical outcome.
                if (
                    result.retryable
                    and result.code is not ErrorCode.INVALID_INPUT
                    and attempts < self.config.max_tool_retries
                ):
                    attempts += 1
                    time.sleep(self.config.retry_backoff_seconds * attempts)
                    continue
                rendered, _ = serialise(result)
                return rendered

            rendered, truncated = serialise(result)
            log.record(
                iteration=iteration,
                kind="tool_result",
                name=call.name,
                arguments=call.arguments,
                status="ok",
                truncated=truncated or getattr(result, "truncated", False),
                duration_ms=elapsed,
                # The rendered result, so an evaluation harness can check the
                # answer's figures against what the model actually saw. Bounded
                # by the same character budget the model received.
                detail=rendered,
            )
            return rendered
