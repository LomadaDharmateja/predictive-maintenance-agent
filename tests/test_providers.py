"""The provider adapters. No network call is made anywhere in this file.

`src/agent/providers.py` is the only code in this project that can talk to a
model, so the parts that can be tested without one are tested here: message
translation, schema flattening, the sampling-parameter decision, the backoff
schedule, and the rule that a key value is never touched.
"""

from __future__ import annotations

import pytest

from src.agent.loop import ModelConfig
from src.agent.providers import (
    NO_SAMPLING_PARAMS,
    PROVIDERS,
    ModelIdentity,
    ProviderError,
    RateLimited,
    _with_backoff,
    accepts_sampling_params,
    backoff_seconds,
    build_client,
    inline_refs,
    to_anthropic,
    to_ollama,
)

# ----------------------------------------------------------------------
# Message translation
#
# The loop speaks one dialect; each provider speaks its own. A tool result has
# to refer back to the call that produced it, and the loop's transcript is
# where that pairing comes from.
# ----------------------------------------------------------------------

CONVERSATION = [
    {"role": "system", "content": "you are a maintenance planner"},
    {"role": "user", "content": "should I order a comp1 part for machine 30?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"name": "get_failure_risk", "arguments": {"machine_id": 30}},
            {"name": "get_parts_position", "arguments": {"component": "comp1"}},
        ],
    },
    {"role": "tool", "name": "get_failure_risk", "content": '{"status":"ok"}'},
    {"role": "tool", "name": "get_parts_position", "content": '{"stock":22}'},
]


def test_anthropic_translation_pairs_every_result_with_its_call():
    system, messages = to_anthropic(CONVERSATION)
    assert system == "you are a maintenance planner"

    assistant = messages[1]
    assert assistant["role"] == "assistant"
    ids = [block["id"] for block in assistant["content"]]
    assert len(ids) == 2 and len(set(ids)) == 2

    results = messages[2]
    assert results["role"] == "user"
    assert [block["tool_use_id"] for block in results["content"]] == ids, (
        "a tool_result must name the tool_use it answers"
    )


def test_anthropic_translation_keeps_parallel_results_in_one_turn():
    """Splitting them teaches the model to stop making parallel calls."""
    _, messages = to_anthropic(CONVERSATION)
    result_turns = [m for m in messages if isinstance(m["content"], list)
                    and m["role"] == "user"]
    assert len(result_turns) == 1
    assert len(result_turns[0]["content"]) == 2


def test_translation_is_deterministic():
    """Re-rendering the same conversation must produce the same bytes."""
    assert to_anthropic(CONVERSATION) == to_anthropic(CONVERSATION)
    assert to_ollama(CONVERSATION) == to_ollama(CONVERSATION)


def test_ollama_translation_carries_tool_calls_and_names_results():
    messages = to_ollama(CONVERSATION)
    assistant = next(m for m in messages if m["role"] == "assistant")
    assert [c["function"]["name"] for c in assistant["tool_calls"]] == [
        "get_failure_risk",
        "get_parts_position",
    ]
    results = [m for m in messages if m["role"] == "tool"]
    assert [m["tool_name"] for m in results] == [
        "get_failure_risk",
        "get_parts_position",
    ]


def test_neither_adapter_drops_content_the_other_keeps():
    """The two renderings differ in shape, never in what the model is shown."""
    system, anthropic_messages = to_anthropic(CONVERSATION)
    ollama_messages = to_ollama(CONVERSATION)

    def texts(blob) -> str:
        return repr(blob)

    for fragment in (
        "you are a maintenance planner",
        "should I order a comp1 part for machine 30?",
        "get_failure_risk",
        "get_parts_position",
        '{"status":"ok"}',
        '{"stock":22}',
    ):
        assert fragment in texts(system) + texts(anthropic_messages), fragment
        assert fragment in texts(ollama_messages), fragment


# ----------------------------------------------------------------------
# Tool schemas
# ----------------------------------------------------------------------


def test_refs_are_inlined_so_no_provider_has_to_resolve_them():
    schema = {
        "type": "object",
        "properties": {"filters": {"$ref": "#/$defs/Filters"}},
        "$defs": {"Filters": {"type": "object", "properties": {"model": {"type": "string"}}}},
    }
    flattened = inline_refs(schema)
    assert "$defs" not in flattened
    assert flattened["properties"]["filters"]["properties"]["model"]["type"] == "string"


def test_a_schema_without_refs_is_returned_unchanged():
    schema = {"type": "object", "properties": {"machine_id": {"type": "integer"}}}
    assert inline_refs(schema) == schema


def test_every_shipped_tool_schema_survives_flattening():
    from src.agent.loop import tool_schemas

    for tool in tool_schemas():
        flattened = inline_refs(tool["input_schema"])
        assert "$defs" not in flattened
        assert flattened.get("type") == "object"


# ----------------------------------------------------------------------
# Temperature 0, and saying so honestly
# ----------------------------------------------------------------------


@pytest.mark.parametrize("model", NO_SAMPLING_PARAMS)
def test_models_that_removed_sampling_params_are_recognised(model):
    assert accepts_sampling_params(model) is False
    assert accepts_sampling_params(f"{model}-20260101") is False


@pytest.mark.parametrize(
    "model", ["claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-6"]
)
def test_models_that_accept_temperature_are_recognised(model):
    assert accepts_sampling_params(model) is True


def test_identity_never_claims_a_temperature_that_was_not_sent():
    """The Success/ToolError rule, applied to generation settings."""
    unavailable = ModelIdentity(
        provider="anthropic", model="claude-opus-5", version="claude-opus-5",
        temperature=None, seed=None, max_tokens=8192,
        sampling_note="sampling parameters were removed on this family",
    )
    assert unavailable.temperature is None
    assert unavailable.sampling_note

    sent = ModelIdentity(
        provider="ollama", model="qwen3:4b-q4_K_M", version="abc",
        temperature=0.0, seed=1, max_tokens=8192,
    )
    assert sent.temperature == 0.0
    assert sent.sampling_note == ""


def test_the_loop_default_requests_temperature_zero():
    assert ModelConfig().temperature == 0.0


# ----------------------------------------------------------------------
# Backoff
# ----------------------------------------------------------------------


def test_backoff_is_exponential_and_capped():
    waits = [backoff_seconds(i) for i in range(1, 8)]
    assert waits[:4] == [2.0, 4.0, 8.0, 16.0]
    assert all(b >= a for a, b in zip(waits, waits[1:])), "monotonic"
    assert max(waits) == 60.0, "capped, so a stuck run does not sleep for an hour"


def test_a_rate_limit_is_retried_then_succeeds():
    attempts = {"n": 0}
    slept: list[float] = []

    def call():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RateLimited("429")
        return "ok"

    assert _with_backoff(call, "test", slept.append) == "ok"
    assert attempts["n"] == 3
    assert slept == [2.0, 4.0]


def test_a_rate_limit_that_never_clears_raises_rather_than_looping_forever():
    slept: list[float] = []

    def call():
        raise RateLimited("429")

    with pytest.raises(ProviderError, match="after 6 attempts"):
        _with_backoff(call, "test", slept.append)
    assert len(slept) == 5


def test_a_retry_after_header_wins_over_the_schedule():
    slept: list[float] = []
    attempts = {"n": 0}

    def call():
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RateLimited("429", retry_after=37.0)
        return "ok"

    _with_backoff(call, "test", slept.append)
    assert slept == [37.0]


def test_a_rejected_request_is_not_retried():
    """The identical request gets the identical rejection; a retry burns quota."""
    attempts = {"n": 0}

    def call():
        attempts["n"] += 1
        raise ProviderError("400: invalid tool schema")

    with pytest.raises(ProviderError, match="400"):
        _with_backoff(call, "test", lambda _: None)
    assert attempts["n"] == 1


# ----------------------------------------------------------------------
# Selection and credentials
# ----------------------------------------------------------------------


def test_an_unknown_provider_is_named_not_guessed():
    with pytest.raises(ProviderError, match="unknown provider"):
        build_client(ModelConfig(provider="not-a-provider"))


def test_both_providers_are_reachable_through_one_call():
    assert set(PROVIDERS) == {"anthropic", "ollama"}


def test_no_credential_value_appears_anywhere_in_the_module(monkeypatch):
    """The SDK resolves the key itself; this module must never read the value."""
    import inspect

    import src.agent.providers as providers

    source = inspect.getsource(providers)
    # The module may *name* the variables to tell the user which are checked,
    # but must never bind, format or log a value read from them.
    assert "os.environ.get(\"ANTHROPIC_API_KEY\")" in source, (
        "presence is checked, which is allowed"
    )
    for forbidden in ("api_key=", "print(os.environ", "f\"{os.environ"):
        assert forbidden not in source, f"module must not handle a key value: {forbidden}"


def test_ollama_needs_no_credential_at_all():
    """The data-sovereignty path: nothing to leak because nothing is sent."""
    import inspect

    from src.agent.providers import OllamaClient

    assert "environ" not in inspect.getsource(OllamaClient)
