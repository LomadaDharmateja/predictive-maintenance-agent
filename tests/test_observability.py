"""Milestone 6: tracing, accounting, replay and the viewer.

Every test here runs offline. One of them enforces that: `tracing` must not be
able to open a socket, which is checked by refusing the socket module rather
than by trusting the code to behave.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from src.agent.loop import Agent, LLMResponse, LoopConfig, ToolCall
from src.obs import accounting, tracing, viewer

REPO = Path(__file__).resolve().parents[1]


class ScriptedClient:
    """A stub model that also reports usage, the way a real adapter does."""

    def __init__(self, script: list[LLMResponse], usage: dict | None = None) -> None:
        self.script = list(script)
        self.last_exchange = usage or {
            "tokens_in": 1000,
            "tokens_out": 120,
            "cache_read": 400,
            "cache_write": 0,
            "stop_reason": "end_turn",
        }

    def complete(self, messages, tools):
        return self.script.pop(0) if self.script else LLMResponse(text="done")


@pytest.fixture
def spans(built_db):
    """Run the agent once under a memory exporter and return its spans."""
    exporter = tracing.MemorySpanExporter()
    tracing.configure(exporter, reset=True)
    client = ScriptedClient(
        [
            LLMResponse(
                tool_calls=(
                    ToolCall("get_machine_profile", {"machine_id": 14}),
                    ToolCall("get_parts_position", {"component": "comp3"}),
                )
            ),
            LLMResponse(text="Machine 14 is a model3."),
        ]
    )
    agent = Agent(client, database=Path(built_db), config=LoopConfig())
    agent.run(
        "How old is machine 14?", scenario_id="scn-1", seed=1,
    )
    tracing.shutdown()
    return exporter.spans


# ----------------------------------------------------------------------
# Span shape
# ----------------------------------------------------------------------


def test_one_root_span_per_run_with_children_for_every_call(spans):
    roots = [s for s in spans if s["name"] == tracing.RUN]
    assert len(roots) == 1, "one span per run"

    root = roots[0]
    assert root["parent_span_id"] is None
    children = [s for s in spans if s["parent_span_id"] == root["span_id"]]
    assert [s["name"] for s in children].count(tracing.MODEL_CALL) == 2
    assert [s["name"] for s in children].count(tracing.TOOL_CALL) == 2
    assert all(s["trace_id"] == root["trace_id"] for s in spans), "one trace"


def test_the_run_span_carries_its_labels_and_iteration_accounting(spans):
    root = next(s for s in spans if s["name"] == tracing.RUN)
    a = root["attributes"]
    assert a["pdm.scenario_id"] == "scn-1"
    assert a["pdm.seed"] == 1
    assert a["pdm.iterations"] == 2
    assert a["pdm.max_iterations"] == LoopConfig().max_iterations
    assert a["pdm.hit_iteration_limit"] is False


def test_tool_spans_carry_name_arguments_and_result_type(spans):
    tools = [s for s in spans if s["name"] == tracing.TOOL_CALL]
    names = {s["attributes"]["pdm.tool"] for s in tools}
    assert names == {"get_machine_profile", "get_parts_position"}
    for s in tools:
        assert json.loads(s["attributes"]["pdm.arguments"])
        assert s["attributes"]["pdm.result_type"] in {"Success", "ToolError"}
        assert s["duration_ms"] is not None


def test_model_spans_carry_tokens_and_cache_counters(spans):
    calls = [s for s in spans if s["name"] == tracing.MODEL_CALL]
    assert calls
    for s in calls:
        assert s["attributes"]["gen_ai.usage.input_tokens"] == 1000
        assert s["attributes"]["gen_ai.usage.output_tokens"] == 120
        assert s["attributes"]["pdm.cache_read_tokens"] == 400


def test_a_tool_error_is_distinguishable_in_the_trace(built_db, monkeypatch):
    """A ToolError that read as a success is the v1 defect; the trace must
    never blur the two."""
    from src.agent import tools as tools_module
    from src.agent.contracts import ErrorCode, ToolError

    exporter = tracing.MemorySpanExporter()
    tracing.configure(exporter, reset=True)
    monkeypatch.setattr(
        tools_module, "dispatch",
        lambda name, args, db=None: ToolError(
            code=ErrorCode.TIMEOUT, message="injected", tool=name, retryable=False
        ),
    )
    client = ScriptedClient(
        [
            LLMResponse(tool_calls=(ToolCall("get_machine_profile", {"machine_id": 1}),)),
            LLMResponse(text="The call failed."),
        ]
    )
    Agent(client, database=Path(built_db)).run("q", scenario_id="s", seed=1)
    tracing.shutdown()

    tools = [s for s in exporter.spans if s["name"] == tracing.TOOL_CALL]
    assert tools and all(
        s["attributes"]["pdm.result_type"] == "ToolError" for s in tools
    )
    assert tools[0]["attributes"]["pdm.error_code"] == "timeout"


# ----------------------------------------------------------------------
# Accounting
# ----------------------------------------------------------------------


def test_accounting_totals_match_the_span_tree(spans):
    records = accounting.from_spans(spans, run_id="test-run")
    assert len(records) == 1
    r = records[0]

    model_calls = [s for s in spans if s["name"] == tracing.MODEL_CALL]
    tool_calls = [s for s in spans if s["name"] == tracing.TOOL_CALL]
    root = next(s for s in spans if s["name"] == tracing.RUN)

    assert r.tokens_in == sum(
        s["attributes"]["gen_ai.usage.input_tokens"] for s in model_calls
    )
    assert r.tokens_out == sum(
        s["attributes"]["gen_ai.usage.output_tokens"] for s in model_calls
    )
    assert r.tool_calls == len(tool_calls)
    assert r.iterations == root["attributes"]["pdm.iterations"]
    assert r.max_iterations == root["attributes"]["pdm.max_iterations"]
    assert r.trace_id == root["trace_id"]
    assert r.scenario_id == "scn-1" and r.seed == 1


#: Every span duration is rounded to three decimal places independently, so a
#: sum of children can exceed a parent by pure rounding: each child may round up
#: by 0.0005 ms and the parent may round down by the same. With four children
#: that is 0.0025 ms of slack, and a scripted client's spans are short enough
#: for it to matter -- this assertion failed about one run in three at a 1e-6
#: tolerance, which is a flaky test rather than a real inversion.
ROUNDING_SLACK_MS = 0.005


def test_wall_clock_covers_model_and_tool_time(spans):
    """The root span must contain its children. Absolute durations are not
    asserted: a scripted client returns in well under a millisecond, so the
    only honest invariant here is the containment relation."""
    r = accounting.from_spans(spans, run_id="test-run")[0]
    assert r.model_ms >= 0 and r.tool_ms >= 0
    assert r.wall_clock_ms >= r.model_ms + r.tool_ms - ROUNDING_SLACK_MS
    assert r.overhead_ms >= 0
    assert r.iteration_headroom == f"{r.iterations}/{r.max_iterations}"


def test_the_containment_relation_holds_on_the_unrounded_spans(spans):
    """The same invariant without the rounding, checked on raw nanoseconds so
    it is exact rather than tolerant."""
    root = next(s for s in spans if s["name"] == tracing.RUN)
    children = [s for s in spans if s["parent_span_id"] == root["span_id"]]
    assert children
    for child in children:
        assert child["start_unix_nano"] >= root["start_unix_nano"]
        assert child["end_unix_nano"] <= root["end_unix_nano"]


def test_accounting_round_trips_through_disk(spans, tmp_path):
    records = accounting.from_spans(spans, run_id="test-run")
    path = accounting.write(records, tmp_path / "acc.json")
    assert accounting.read(path) == records


# ----------------------------------------------------------------------
# Pricing: one source of truth
# ----------------------------------------------------------------------


class _Identity:
    def __init__(self, provider, model):
        self.provider, self.model = provider, model


def test_the_harness_prices_through_the_accounting_module():
    """Item 5. `evals/runner.py` must not carry its own price table."""
    import inspect

    import evals.runner as runner

    source = inspect.getsource(runner)
    assert "PRICES_PER_1K" not in source, "the price table moved to src/obs/accounting"
    assert "accounting.cost_usd" in source


def test_a_local_run_costs_nothing_and_a_hosted_one_does_not():
    local = _Identity("ollama", "qwen3:4b-q4_K_M")
    hosted = _Identity("anthropic", "claude-sonnet-5")
    assert accounting.cost_usd(local, 10_000, 1_000) == 0.0
    assert accounting.cost_usd(hosted, 10_000, 1_000) > 0.0


def test_cache_tokens_are_cheaper_than_uncached_ones():
    hosted = _Identity("anthropic", "claude-sonnet-5")
    uncached = accounting.cost_usd(hosted, 10_000, 0)
    cached = accounting.cost_usd(hosted, 0, 0, cache_read=10_000)
    assert cached == pytest.approx(uncached * accounting.CACHE_READ_MULTIPLIER)
    written = accounting.cost_usd(hosted, 0, 0, cache_write=10_000)
    assert written == pytest.approx(uncached * accounting.CACHE_WRITE_MULTIPLIER)


# ----------------------------------------------------------------------
# Offline guarantee
# ----------------------------------------------------------------------


def test_no_exporter_opens_a_socket(built_db, monkeypatch):
    """Enforced rather than trusted: a run under a poisoned socket module must
    still trace. Milestone 6 requires the tests to need no network."""

    def refuse(*args, **kwargs):
        raise AssertionError("the observability layer must not open a socket")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)

    exporter = tracing.MemorySpanExporter()
    tracing.configure(exporter, reset=True)
    Agent(ScriptedClient([LLMResponse(text="ok")]), database=Path(built_db)).run("q")
    tracing.shutdown()
    assert any(s["name"] == tracing.RUN for s in exporter.spans)


def test_the_file_exporter_writes_json_lines_and_nothing_else(tmp_path, built_db):
    path = tmp_path / "spans.jsonl"
    exporter = tracing.JsonFileSpanExporter(path)
    tracing.configure(exporter, reset=True)
    Agent(ScriptedClient([LLMResponse(text="ok")]), database=Path(built_db)).run("q")
    tracing.shutdown()

    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert lines and all("span_id" in s and "attributes" in s for s in lines)


def test_tracing_imports_no_network_exporter():
    """Checked against the module's actual imports, not its prose -- the
    docstring mentions OTLP precisely to say it is not wired up."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(tracing))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    for name in imported:
        lowered = name.lower()
        for forbidden in ("otlp", "requests", "httpx", "urllib", "socket", "http"):
            assert forbidden not in lowered, (
                f"tracing imports {name!r}, which can reach the network"
            )


# ----------------------------------------------------------------------
# Viewer
# ----------------------------------------------------------------------


def _trace(**overrides) -> dict:
    base = {
        "scenario_id": "scn-1", "seed": 1, "question": "How old is machine 14?",
        "answer": "Machine 14 is a model3.", "iterations": 2,
        "hit_iteration_limit": False, "messages_dropped": 0,
        "tokens_in": 2000, "tokens_out": 240, "cache_read": 800, "cache_write": 0,
        "wall_clock_ms": 12.5, "estimated_cost_usd": 0.0123,
        "tool_calls": [
            {"tool": "get_machine_profile", "arguments": {"machine_id": 14},
             "status": "ok", "error_code": None, "truncated": False,
             "duration_ms": 2.0, "result_json": '{"status":"ok","data":{"age_years":1}}'},
        ],
    }
    base.update(overrides)
    return base


def test_the_viewer_page_is_self_contained():
    page = viewer.render("run-1", [_trace()])
    for forbidden in ("http://", "https://", "<script", "src=", "@import"):
        assert forbidden not in page, f"the page must not reference {forbidden}"
    assert "<style>" in page, "styling is inline"


def test_the_viewer_shows_the_question_calls_answer_and_cost():
    page = viewer.render("run-1", [_trace()])
    assert "How old is machine 14?" in page
    assert "get_machine_profile" in page
    assert "Machine 14 is a model3." in page
    assert "age_years" in page, "the tool result is shown, not just its name"
    assert "$0.0123" in page
    assert "12.5 ms" in page


def test_the_viewer_marks_a_failed_tool_call():
    failing = _trace(tool_calls=[{
        "tool": "get_failure_risk", "arguments": {"machine_id": 1}, "status": "error",
        "error_code": "timeout", "truncated": False, "duration_ms": 1.0,
        "result_json": '{"status":"error"}',
    }])
    page = viewer.render("run-1", [failing])
    assert "timeout" in page and 'class="step err"' in page


def test_the_viewer_escapes_answer_text_rather_than_rendering_it():
    """An answer is untrusted text as far as the page is concerned."""
    page = viewer.render("run-1", [_trace(answer="<img onerror=alert(1)>")])
    assert "<img" not in page
    assert "&lt;img" in page


# ----------------------------------------------------------------------
# Replay determinism
# ----------------------------------------------------------------------

RESULTS = REPO / "evals" / "results"


def _latest_run_id() -> str | None:
    """A run's sidecars share its id and add a suffix, so a dot in the stem is
    what separates `<run>.json` from `<run>.traces.json`."""
    runs = sorted(p for p in RESULTS.glob("*.json") if "." not in p.stem)
    return runs[-1].stem if runs else None


@pytest.mark.skipif(_latest_run_id() is None, reason="no recorded run on disk")
def test_replaying_a_recorded_run_reproduces_it_exactly(built_db):
    """Item 3. No model is called; the stored answers must come back byte for
    byte, or the divergence is named."""
    from src.obs.replay import replay_run

    run_id = _latest_run_id()
    report, records, spans = replay_run(
        run_id, RESULTS, REPO / "evals" / "scenarios.yaml", Path(built_db),
        transcripts=REPO / "evals" / "transcripts",
    )
    assert report.replayed > 0
    assert report.deterministic, [d.model_dump() for d in report.divergences]
    assert report.identical == report.replayed
    assert len(records) == report.replayed, "one accounting record per replayed run"
    assert any(s["name"] == tracing.RUN for s in spans)


def test_a_changed_answer_is_reported_as_a_divergence():
    """Anti-vacuity: the check must bite, not merely pass."""
    from src.obs.replay import _compare

    stored = {
        "scenario_id": "scn-1", "seed": 1, "answer": "the stored answer",
        "iterations": 2, "hit_iteration_limit": False,
        "tokens_in": 10, "tokens_out": 2, "tool_calls": [],
    }

    class Fresh:
        answer = "a different answer"
        iterations, hit_iteration_limit = 2, False
        tokens_in, tokens_out, tool_calls = 10, 2, []

    divergence = _compare(stored, Fresh())
    assert divergence is not None
    assert divergence.field == "answer"
    assert "stored" in divergence.diff and "replayed" in divergence.diff


def test_a_changed_tool_argument_is_reported_as_a_divergence():
    from src.obs.replay import _compare

    stored = {
        "scenario_id": "scn-1", "seed": 1, "answer": "same", "iterations": 1,
        "hit_iteration_limit": False, "tokens_in": 1, "tokens_out": 1,
        "tool_calls": [{"tool": "get_failure_risk",
                        "arguments": {"machine_id": 30}, "status": "ok"}],
    }

    class Fresh:
        answer, iterations, hit_iteration_limit = "same", 1, False
        tokens_in, tokens_out = 1, 1

        class _Call:
            tool, arguments, status = "get_failure_risk", {"machine_id": 42}, "ok"

        tool_calls = [_Call()]

    divergence = _compare(stored, Fresh())
    assert divergence is not None and divergence.field == "tool_calls"


def test_timing_differences_are_not_treated_as_divergence():
    """Wall clock differs between any two runs of the same code and is not
    evidence of anything."""
    from src.obs.replay import _compare

    stored = {
        "scenario_id": "scn-1", "seed": 1, "answer": "same", "iterations": 1,
        "hit_iteration_limit": False, "tokens_in": 1, "tokens_out": 1,
        "tool_calls": [], "wall_clock_ms": 5.0, "estimated_cost_usd": 0.01,
    }

    class Fresh:
        answer, iterations, hit_iteration_limit = "same", 1, False
        tokens_in, tokens_out, tool_calls = 1, 1, []
        wall_clock_ms, estimated_cost_usd = 999.0, 0.02

    assert _compare(stored, Fresh()) is None
