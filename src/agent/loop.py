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
from src.agent.tools import REGISTRY, dispatch, serialise

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
    model: str = "claude-sonnet-4-5"
    temperature: float = 0.0
    seed: int | None = 20240608
    timeout_seconds: float = 30.0
    max_tokens: int = 2048

    #: A local open-weights path, configured but not exercised here. Present
    #: because "can this run entirely on our own hardware" is a question that
    #: gets asked before a pilot, not after.
    local_base_url: str | None = None
    local_model: str | None = "qwen2.5-14b-instruct"


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

    def run(self, question: str) -> AgentResult:
        from src.agent import db as db_module

        database = self.database or db_module.DEFAULT_DB
        log = RunLog()
        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]
        dropped_total = 0
        schemas = tool_schemas()

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
            response = self.client.complete(messages, schemas)
            log.record(
                iteration=iteration,
                kind="model_call",
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
                detail=f"{len(response.tool_calls)} tool call(s)"
                if response.wants_tools
                else "final answer",
            )

            if not response.wants_tools:
                return AgentResult(
                    answer=response.text or "",
                    iterations=iteration,
                    hit_iteration_limit=False,
                    log=log,
                    messages_dropped=dropped_total,
                )

            for call in response.tool_calls:
                rendered = self._call_tool(call, database, log, iteration)
                messages.append(
                    {"role": "tool", "name": call.name, "content": rendered}
                )

        # Terminal behaviour on exhaustion is defined, and it is not silence.
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
            result = dispatch(call.name, call.arguments, database)
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
