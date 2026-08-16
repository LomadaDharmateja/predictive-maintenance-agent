"""Concrete `LLMClient` implementations. One interface, two providers.

`docs/MILESTONE_4.md` section 5 requires "a model-agnostic interface so the
provider can be swapped in one place", and a local open-weights path because
data sovereignty is asked about before a pilot rather than after. Until now that
interface had no implementations: `LLMClient` in `src/agent/loop.py` was a
`Protocol`, the only things satisfying it were a transcript replayer and test
stubs, and `evals/runner.py --live` raised. This module is the missing half.

Two providers sit behind the Protocol:

- **Ollama** -- local, free, no key, nothing leaves the machine. Used to build
  and shake out the recording path.
- **Anthropic** -- hosted, for the run whose numbers get reported.

The loop cannot tell them apart. Selection is `ModelConfig.provider` and
nothing else.

**Credentials come from the environment and are never handled here.** The
Anthropic SDK resolves `ANTHROPIC_API_KEY` (or an `ant auth login` profile)
itself; this module never reads, stores, logs or forwards a key value.
`scripts/fetch_data.py` treats Kaggle the same way, for the same reason.

**Temperature 0 where the provider accepts it, and stated where it does not.**
Anthropic removed `temperature`, `top_p` and `top_k` on its newest models --
sending one is a 400. So the adapter omits the parameter on those models and
records `temperature: null` with a note in `ModelIdentity`. A transcript must
never claim a setting that was not sent; that is the same rule as
`Success[T] | ToolError`, applied to generation settings.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from pydantic import BaseModel, ConfigDict

from src.agent.loop import LLMResponse, ModelConfig, ToolCall

# ----------------------------------------------------------------------
# Model identity
# ----------------------------------------------------------------------


class ModelIdentity(BaseModel):
    """Which model produced a transcript, and under what settings.

    Recorded on every transcript and carried into `RunMetadata`. A transcript
    that cannot be attributed to a model is not usable: two runs that disagree
    are indistinguishable from one run of a model that changed underneath you.
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    provider: str
    model: str
    #: The provider's own resolved identifier -- the `model` field Anthropic
    #: echoes back, or Ollama's manifest digest. `model` above is what was
    #: asked for; this is what answered.
    version: str
    temperature: float | None
    seed: int | None
    max_tokens: int
    #: Why `temperature` is null, when it is. Empty otherwise.
    sampling_note: str = ""

    def label(self) -> str:
        return f"{self.provider}/{self.model}@{self.version}"


#: Anthropic models that removed `temperature`, `top_p` and `top_k`. Sending a
#: sampling parameter to one of these is a 400, not a warning. Matched by
#: prefix so dated snapshots of the same family are covered.
NO_SAMPLING_PARAMS = (
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-sonnet-5",
    "claude-fable-5",
    "claude-mythos-5",
)

SAMPLING_REMOVED_NOTE = (
    "temperature is not accepted by this Anthropic model (sampling parameters "
    "were removed on this family); the request omitted it rather than sending a "
    "value the API would reject"
)


def accepts_sampling_params(model: str) -> bool:
    return not any(model.startswith(prefix) for prefix in NO_SAMPLING_PARAMS)


# ----------------------------------------------------------------------
# Retry policy
# ----------------------------------------------------------------------


class ProviderError(RuntimeError):
    """A provider call that failed after the retry policy was exhausted."""


class RateLimited(RuntimeError):
    """Raised internally so the backoff loop can distinguish it from a bug."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


#: Exponential backoff, no jitter. Deterministic on purpose: recording is the
#: one place in this project that touches the network, and a reproducible wait
#: schedule is easier to reason about in a log than a randomised one.
BACKOFF_BASE_SECONDS = 2.0
BACKOFF_MAX_SECONDS = 60.0
MAX_ATTEMPTS = 6


def backoff_seconds(attempt: int) -> float:
    """Seconds to wait before attempt `attempt` (1-indexed retries)."""
    return min(BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)), BACKOFF_MAX_SECONDS)


def _with_backoff(call, describe: str, sleep=time.sleep):
    """Run `call`, retrying rate limits and transient failures.

    Retries a rate limit or a 5xx; never retries a request the provider
    rejected as invalid, because the identical request will be rejected
    identically and a retry only burns quota. Same rule the agent loop applies
    to `ToolError`.
    """
    last: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return call()
        except RateLimited as exc:
            last = exc
            if attempt == MAX_ATTEMPTS:
                break
            wait = exc.retry_after if exc.retry_after else backoff_seconds(attempt)
            print(f"  rate limited on {describe}; waiting {wait:.0f}s "
                  f"(attempt {attempt}/{MAX_ATTEMPTS})")
            sleep(wait)
    raise ProviderError(
        f"{describe} failed after {MAX_ATTEMPTS} attempts: {last}"
    ) from last


# ----------------------------------------------------------------------
# Tool schema handling
# ----------------------------------------------------------------------


def inline_refs(schema: dict) -> dict:
    """Resolve `$ref`/`$defs` into a self-contained schema.

    Pydantic emits `$defs` for nested models. Anthropic tolerates them; local
    servers are less consistent about it. Inlining costs nothing and removes a
    provider-specific failure mode from the recording path.
    """
    defs = schema.get("$defs", {})
    if not defs:
        return schema

    def resolve(node, depth: int = 0):
        if depth > 8:
            return node
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                target = defs.get(ref.removeprefix("#/$defs/"))
                if target is not None:
                    merged = {k: v for k, v in node.items() if k != "$ref"}
                    return {**resolve(target, depth + 1), **merged}
            return {k: resolve(v, depth + 1) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [resolve(item, depth + 1) for item in node]
        return node

    return resolve(schema)


# ----------------------------------------------------------------------
# Message translation
#
# The loop speaks a provider-neutral dialect: a system message, a user message,
# an assistant message carrying `tool_calls`, and one `tool` message per
# result. Each adapter renders that into its provider's shape. Nothing
# provider-specific leaks back into the loop.
# ----------------------------------------------------------------------


def _tool_use_id(turn: int, index: int) -> str:
    """A stable id pairing an assistant tool call with its result.

    Derived from position rather than randomly generated, so re-rendering the
    same conversation produces the same request bytes.
    """
    return f"toolu_{turn:02d}_{index:02d}"


def to_anthropic(messages: list[dict]) -> tuple[str, list[dict]]:
    """Split the loop's messages into (system prompt, Anthropic messages)."""
    system_parts: list[str] = []
    rendered: list[dict] = []
    turn = 0
    pending: list[str] = []  # tool_use ids awaiting their results

    for message in messages:
        role = message.get("role")
        if role == "system":
            system_parts.append(str(message.get("content", "")))
        elif role == "user":
            rendered.append({"role": "user", "content": str(message.get("content", ""))})
        elif role == "assistant":
            calls = message.get("tool_calls") or []
            turn += 1
            pending = [_tool_use_id(turn, i) for i in range(len(calls))]
            rendered.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": pending[i],
                            "name": call["name"],
                            "input": call.get("arguments") or {},
                        }
                        for i, call in enumerate(calls)
                    ],
                }
            )
        elif role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": pending.pop(0) if pending else _tool_use_id(turn, 0),
                "content": str(message.get("content", "")),
            }
            # Consecutive results belong in one user turn: splitting them
            # teaches the model to stop making parallel calls.
            if rendered and rendered[-1]["role"] == "user" and isinstance(
                rendered[-1]["content"], list
            ):
                rendered[-1]["content"].append(block)
            else:
                rendered.append({"role": "user", "content": [block]})

    return "\n\n".join(system_parts), rendered


def to_ollama(messages: list[dict]) -> list[dict]:
    """Render the loop's messages into Ollama's `/api/chat` shape."""
    rendered: list[dict] = []
    for message in messages:
        role = message.get("role")
        if role in {"system", "user"}:
            rendered.append({"role": role, "content": str(message.get("content", ""))})
        elif role == "assistant":
            rendered.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": call["name"],
                                "arguments": call.get("arguments") or {},
                            }
                        }
                        for call in (message.get("tool_calls") or [])
                    ],
                }
            )
        elif role == "tool":
            rendered.append(
                {
                    "role": "tool",
                    "tool_name": message.get("name", ""),
                    "content": str(message.get("content", "")),
                }
            )
    return rendered


# ----------------------------------------------------------------------
# Adapters
# ----------------------------------------------------------------------


class _Adapter:
    """Shared surface. `last_exchange` is what `RecordingClient` writes down."""

    identity: ModelIdentity

    def __init__(self) -> None:
        self.last_exchange: dict[str, Any] = {}

    def complete(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        raise NotImplementedError


class AnthropicClient(_Adapter):
    """The hosted path. Credentials resolved by the SDK from the environment."""

    def __init__(self, config: ModelConfig, sleep=time.sleep) -> None:
        super().__init__()
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - dependency is pinned
            raise ProviderError(
                "the `anthropic` package is required for provider='anthropic'; "
                "it is pinned in requirements.txt"
            ) from exc

        self._anthropic = anthropic
        # max_retries=0: the backoff policy is this module's, stated and
        # tested, not the SDK's default hidden behind a constructor argument.
        self._client = anthropic.Anthropic(
            timeout=config.timeout_seconds, max_retries=0
        )
        self._config = config
        self._sleep = sleep
        self._sampling = accepts_sampling_params(config.model)
        self.identity = ModelIdentity(
            provider="anthropic",
            model=config.model,
            version="unresolved",
            temperature=config.temperature if self._sampling else None,
            seed=None,  # Anthropic exposes no seed; determinism is the recording
            max_tokens=config.max_tokens,
            sampling_note="" if self._sampling else SAMPLING_REMOVED_NOTE,
        )

    #: Splits the system prompt at the prediction-time heading the agent loop
    #: appends. Everything before it is stable across the whole suite and is
    #: what the cache breakpoint is placed on.
    STABLE_SYSTEM_SPLIT = "\n\n## Prediction time\n\n"

    def _system_blocks(self, system: str) -> list[dict]:
        stable, _, volatile = system.partition(self.STABLE_SYSTEM_SPLIT)
        blocks = [
            {
                "type": "text",
                "text": stable,
                "cache_control": {"type": "ephemeral"},
            }
        ]
        if volatile:
            blocks.append(
                {"type": "text", "text": self.STABLE_SYSTEM_SPLIT.strip() + "\n\n" + volatile}
            )
        return blocks

    def complete(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        system, rendered = to_anthropic(messages)
        rendered_tools = [
            {
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": inline_refs(tool["input_schema"]),
            }
            for tool in tools
        ]
        payload: dict[str, Any] = {
            "model": self._config.model,
            "max_tokens": self._config.max_tokens,
            "messages": rendered,
        }
        if rendered_tools:
            # Prompt caching, breakpoint 1. Render order is tools -> system ->
            # messages, so a breakpoint on the last tool caches the whole tool
            # block -- which is byte-identical across every call in the suite,
            # agent and scenario alike.
            rendered_tools[-1] = {
                **rendered_tools[-1],
                "cache_control": {"type": "ephemeral"},
            }
            payload["tools"] = rendered_tools
        if system:
            # Omitted rather than sent empty: the judge asks a flat question
            # with no system prompt, and an empty string is a rejected request.
            #
            # Breakpoint 2, on the *first* system block. `to_anthropic` splits
            # the system prompt so the versioned file is its own block and the
            # per-scenario prediction time is a second one after it. Caching
            # only the first keeps the prefix identical across all 123 runs;
            # putting the breakpoint after the `as_of` line would make every
            # scenario a fresh cache write and save nothing across the suite.
            payload["system"] = self._system_blocks(system)
        if self._sampling:
            payload["temperature"] = self._config.temperature

        def call():
            try:
                return self._client.messages.create(**payload)
            except self._anthropic.RateLimitError as exc:
                header = getattr(getattr(exc, "response", None), "headers", {}) or {}
                try:
                    retry_after = float(header.get("retry-after"))
                except (TypeError, ValueError):
                    retry_after = None
                raise RateLimited(str(exc), retry_after) from exc
            except self._anthropic.APIStatusError as exc:
                if exc.status_code >= 500:
                    raise RateLimited(f"{exc.status_code}: {exc}") from exc
                # A 4xx is a rejected request. Retrying sends the identical
                # bytes and gets the identical rejection.
                raise ProviderError(f"{exc.status_code}: {exc}") from exc
            except self._anthropic.APIConnectionError as exc:
                raise RateLimited(f"connection error: {exc}") from exc

        response = _with_backoff(call, f"anthropic {self._config.model}", self._sleep)

        self.identity = self.identity.model_copy(update={"version": response.model})
        tool_calls = tuple(
            ToolCall(block.name, dict(block.input or {}))
            for block in response.content
            if block.type == "tool_use"
        )
        text = "".join(
            block.text for block in response.content if block.type == "text"
        )
        usage = response.usage
        self.last_exchange = {
            "request": {k: v for k, v in payload.items() if k != "tools"},
            "response": json.loads(response.model_dump_json()),
            "tokens_in": usage.input_tokens,
            "tokens_out": usage.output_tokens,
            # Billed differently from `tokens_in`: reads at ~0.1x, writes at
            # ~1.25x. Recorded separately so the cost column can price them
            # separately instead of quietly charging a cache read at full rate.
            "cache_read": getattr(usage, "cache_read_input_tokens", 0) or 0,
            "cache_write": getattr(usage, "cache_creation_input_tokens", 0) or 0,
            "stop_reason": response.stop_reason,
        }
        if response.stop_reason == "refusal":
            # A refusal is not an answer. Surfacing it as empty text would let
            # the harness score a refusal as a (very ungrounded) response.
            raise ProviderError(
                f"the model declined this request (stop_reason=refusal, "
                f"{getattr(response, 'stop_details', None)})"
            )
        # The loop's contract is an answer or tool calls, never both. A
        # preamble alongside tool calls is kept in the raw exchange above.
        if tool_calls:
            return LLMResponse(tool_calls=tool_calls)
        return LLMResponse(text=text)


class OllamaClient(_Adapter):
    """The local path. No key, no network beyond localhost."""

    def __init__(self, config: ModelConfig, seed: int | None = None, sleep=time.sleep) -> None:
        super().__init__()
        import httpx

        self._httpx = httpx
        self._base = (config.local_base_url or "http://127.0.0.1:11434").rstrip("/")
        self._model = config.local_model or "qwen3:4b-q4_K_M"
        self._config = config
        self._sleep = sleep
        self._client = httpx.Client(timeout=config.timeout_seconds)
        self.identity = ModelIdentity(
            provider="ollama",
            model=self._model,
            version=self._resolve_digest(),
            temperature=config.temperature,
            seed=seed if seed is not None else config.seed,
            max_tokens=config.max_tokens,
        )

    def _resolve_digest(self) -> str:
        """The manifest digest of the pulled weights.

        A tag like `qwen3:4b-q4_K_M` is mutable -- `ollama pull` can move it.
        The digest is what actually answered.
        """
        try:
            reply = self._client.get(f"{self._base}/api/tags")
            reply.raise_for_status()
            for entry in reply.json().get("models", []):
                if entry.get("name") == self._model or entry.get("model") == self._model:
                    return str(entry.get("digest", "unknown"))[:19]
        except Exception:  # noqa: BLE001 - identity is best effort, never fatal
            return "unresolved"
        return "unknown"

    def complete(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        payload = {
            "model": self._model,
            "messages": to_ollama(messages),
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": inline_refs(tool["input_schema"]),
                    },
                }
                for tool in tools
            ],
            "stream": False,
            "think": False,
            "options": {
                "temperature": self._config.temperature,
                "seed": self.identity.seed,
                "num_predict": self._config.max_tokens,
            },
        }

        def call():
            try:
                reply = self._client.post(f"{self._base}/api/chat", json=payload)
            except self._httpx.HTTPError as exc:
                raise RateLimited(f"connection error: {exc}") from exc
            if reply.status_code == 429 or reply.status_code >= 500:
                raise RateLimited(f"{reply.status_code}: {reply.text[:200]}")
            if reply.status_code >= 400:
                raise ProviderError(f"{reply.status_code}: {reply.text[:400]}")
            return reply.json()

        body = _with_backoff(call, f"ollama {self._model}", self._sleep)

        message = body.get("message") or {}
        raw_calls = message.get("tool_calls") or []
        tool_calls = tuple(
            ToolCall(
                call["function"]["name"],
                _coerce_arguments(call["function"].get("arguments")),
            )
            for call in raw_calls
            if call.get("function", {}).get("name")
        )
        self.last_exchange = {
            "request": {k: v for k, v in payload.items() if k != "tools"},
            "response": body,
            "tokens_in": int(body.get("prompt_eval_count") or 0),
            "tokens_out": int(body.get("eval_count") or 0),
            "stop_reason": body.get("done_reason"),
        }
        if tool_calls:
            return LLMResponse(tool_calls=tool_calls)
        return LLMResponse(text=str(message.get("content", "")))


def _coerce_arguments(arguments) -> dict:
    """Ollama returns arguments as an object; some models emit a JSON string."""
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


# ----------------------------------------------------------------------
# Selection
# ----------------------------------------------------------------------

PROVIDERS = ("anthropic", "ollama")


def build_client(config: ModelConfig, seed: int | None = None) -> _Adapter:
    """The one place a provider is chosen. `docs/MILESTONE_4.md` section 5."""
    if config.provider == "anthropic":
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            # Named, never printed. The SDK also resolves an `ant auth login`
            # profile, so absence of the variable is a warning, not an error.
            print(
                "  note: neither ANTHROPIC_API_KEY nor ANTHROPIC_AUTH_TOKEN is set; "
                "the SDK will fall back to a configured auth profile"
            )
        return AnthropicClient(config)
    if config.provider == "ollama":
        return OllamaClient(config, seed=seed)
    raise ProviderError(
        f"unknown provider {config.provider!r}; expected one of {PROVIDERS}"
    )
