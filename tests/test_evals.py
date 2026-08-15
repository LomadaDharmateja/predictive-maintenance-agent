"""The harness machinery. Scenarios themselves are the project owner's work.

`docs/MILESTONE_5.md` sections 1, 3, 4 and 5. Every check here runs offline: the
runner replays recorded transcripts and no judge client is wired up, which is the
whole point of the recorded mode.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from evals import diff as diff_module
from evals.record import RecordingClient, record_suite
from evals.transcript import (
    TranscriptInvalid,
    TranscriptMissing,
    ValidationWindowError,
    assert_validation_window,
    expected_machine_ids,
    read_transcript,
    transcript_path,
    validate_transcript,
    window_guarded,
)
from evals.judge import KAPPA_FLOOR, Judge, calibrate, load_prompt, prompt_version
from evals.metrics import (
    _states_limitation,
    check_assertions,
    check_grounding,
    check_tool_selection,
    cohens_kappa,
    percentile,
    score_scenario,
)
from evals.report import render as render_report
from evals.runner import (
    ReplayClient,
    load_scenarios,
    run_suite,
)
from evals.schema import (
    MINIMUM_SCENARIOS,
    REQUIRED_DISTRIBUTION,
    Category,
    RunMetadata,
    RunResults,
    Scenario,
    ScenarioTrace,
    ToolCallTrace,
)
from evals.validate_scenarios import report as validate_report

REPO = Path(__file__).resolve().parents[1]
SCENARIOS = REPO / "evals" / "scenarios.yaml"
TRANSCRIPTS = REPO / "evals" / "transcripts"


def TRANSCRIPTS_FOR_TEST(directory: Path, scenario_id: str, seed: int) -> Path:
    return transcript_path(scenario_id, seed, directory)


# ----------------------------------------------------------------------
# A stub provider, so the recording path is exercised without a model
#
# It satisfies the same surface `AnthropicClient` and `OllamaClient` do:
# `identity`, `last_exchange`, `complete`. Arguments are built from the
# scenario itself, which is exactly what the transcript-consistency invariant
# checks -- so these transcripts are valid by construction, and the tests that
# corrupt one prove the check bites.
# ----------------------------------------------------------------------

COMPONENT = re.compile(r"(comp[1-4])")


class StubAdapter:
    """Calls the scenario's required tools, then answers. No network."""

    def __init__(self, scenario: Scenario, seed: int) -> None:
        from src.agent.providers import ModelIdentity

        self.scenario = scenario
        self.turn = 0
        self.last_exchange: dict = {}
        self.identity = ModelIdentity(
            provider="stub",
            model="scripted",
            version="1",
            temperature=0.0,
            seed=seed,
            max_tokens=1024,
        )

    def _arguments(self, tool: str) -> dict:
        from src.agent.tools import REGISTRY

        model, _ = REGISTRY[tool]
        machines = expected_machine_ids(self.scenario) or {1}
        found = COMPONENT.search(self.scenario.id)
        candidates = {
            "machine_id": sorted(machines)[0],
            "as_of": self.scenario.as_of.isoformat() if self.scenario.as_of else None,
            "component": found.group(1) if found else "comp1",
        }
        return {
            key: value
            for key, value in candidates.items()
            if key in model.model_fields and value is not None
        }

    def complete(self, messages, tools):
        from src.agent.loop import LLMResponse, ToolCall

        self.turn += 1
        self.last_exchange = {"tokens_in": 100, "tokens_out": 20, "request": {}, "response": {}}
        if self.turn == 1 and self.scenario.required_tools:
            return LLMResponse(
                tool_calls=tuple(
                    ToolCall(tool, self._arguments(tool))
                    for tool in self.scenario.required_tools
                )
            )
        return LLMResponse(
            text=(
                "The warning adequacy is insufficient and I could not establish "
                "enough to support an order; stock and consumption are the basis."
            )
        )


def stub_factory(config, scenario, seed):
    return StubAdapter(scenario, seed)


@pytest.fixture(scope="module")
def recorded_suite(tmp_path_factory):
    """Two scenarios recorded through the real recording path, into a tmp dir."""
    directory = tmp_path_factory.mktemp("transcripts")
    scenarios = load_scenarios(SCENARIOS)[:2]
    summary = record_suite(
        scenarios,
        (1, 2, 3),
        None,
        REPO / "data" / "pdm.db",
        directory,
        resume=False,
        client_factory=stub_factory,
    )
    assert not summary["failed"], summary["failed"]
    return scenarios, directory


def make_trace(**overrides) -> ScenarioTrace:
    base = dict(
        scenario_id="scn-1",
        seed=1,
        answer="Stock is 21 units.",
        tool_calls=[
            ToolCallTrace(
                tool="get_parts_position",
                arguments={"component": "comp3"},
                status="ok",
                duration_ms=1.0,
                result_json=json.dumps({"parts": [{"stock_quantity": 21}]}),
            )
        ],
        iterations=2,
        hit_iteration_limit=False,
        messages_dropped=0,
        tokens_in=100,
        tokens_out=20,
        wall_clock_ms=5.0,
        estimated_cost_usd=0.001,
    )
    base.update(overrides)
    return ScenarioTrace(**base)


# ======================================================================
# Section 1: metrics
# ======================================================================


def test_grounded_answer_has_no_hallucinations():
    scenario = Scenario(id="scn-1", category=Category.PARTS_POSITION, question="what is the stock position?")
    grounded, found = check_grounding(scenario, make_trace())
    assert grounded and found == []


def test_a_fabricated_figure_is_detected_and_listed():
    scenario = Scenario(id="scn-1", category=Category.PARTS_POSITION, question="what is the stock position?")
    trace = make_trace(answer="Stock is 21 units and 87 are on order.")
    grounded, found = check_grounding(scenario, trace)
    assert not grounded
    assert [h.value for h in found] == ["87"]
    assert "on order" in found[0].context


def test_a_percentage_rendering_of_a_probability_counts_as_grounded():
    """0.0704 reported as 7% is the same figure, not a fabrication."""
    scenario = Scenario(id="scn-1", category=Category.RISK_ADEQUATE, question="what is the risk here?")
    trace = make_trace(
        answer="The probability is about 7%.",
        tool_calls=[
            ToolCallTrace(
                tool="get_failure_risk",
                arguments={},
                status="ok",
                duration_ms=1.0,
                result_json=json.dumps({"calibrated_probability": 0.0704}),
            )
        ],
    )
    grounded, found = check_grounding(scenario, trace)
    assert grounded, [h.value for h in found]


def test_forbidden_tool_calls_are_counted_absolutely():
    scenario = Scenario(
        id="scn-1",
        category=Category.PARTS_POSITION,
        question="what is the stock position?",
        required_tools=["get_parts_position"],
        forbidden_tools=["get_failure_risk"],
    )
    trace = make_trace(
        tool_calls=[
            ToolCallTrace(
                tool="get_failure_risk",
                arguments={"machine_id": 4},
                status="ok",
                duration_ms=1.0,
                result_json="{}",
            )
        ]
    )
    selection, violations = check_tool_selection(scenario, trace)
    assert selection.forbidden_called == ["get_failure_risk"]
    assert len(violations) == 1
    assert violations[0].arguments == {"machine_id": 4}
    assert selection.required_missing == ["get_parts_position"]


def test_tool_selection_precision_and_recall():
    scenario = Scenario(
        id="scn-1",
        category=Category.MULTI_STEP,
        question="a question long enough to validate",
        required_tools=["get_parts_position", "get_recent_errors"],
    )
    selection, _ = check_tool_selection(scenario, make_trace())
    assert selection.precision == 1.0
    assert selection.recall == 0.5


def test_a_scenario_with_a_forbidden_call_cannot_pass():
    scenario = Scenario(
        id="scn-1",
        category=Category.PARTS_POSITION,
        question="what is the stock position?",
        forbidden_tools=["get_parts_position"],
    )
    result, violations, _ = score_scenario(scenario, make_trace())
    assert violations and result.passed is False


def test_an_unjudged_assertion_is_recorded_unsatisfied_not_passing():
    """A missing check must never read as a passing one."""
    scenario = Scenario(
        id="scn-1",
        category=Category.LOOKUP,
        question="a question long enough to validate",
        must_contain=["some_free_text_judgement"],
    )
    outcomes = check_assertions(scenario, make_trace(), judge=None)
    assert outcomes[0].satisfied is False
    assert outcomes[0].method == "judge"


def test_literal_assertions_resolve_deterministically():
    scenario = Scenario(
        id="scn-1",
        category=Category.LOOKUP,
        question="a question long enough to validate",
        must_contain=["literal: 21 units"],
        must_not_contain=["literal: absolutely certain"],
    )
    outcomes = check_assertions(scenario, make_trace(), judge=None)
    assert all(o.method == "deterministic" for o in outcomes)
    assert all(o.satisfied for o in outcomes)


def test_percentile_reports_the_tail():
    values = [1.0, 2.0, 3.0, 100.0]
    assert percentile(values, 50) <= percentile(values, 95)
    assert percentile(values, 95) == 100.0


# ----------------------------------------------------------------------
# states_limitation
#
# The predicate matched the bare word "failed", so "comp2 failed" satisfied
# `states_limitation` while stating no limitation at all. These pin both halves:
# a fact about the plant is not a limitation, and the phrasings the
# `tool_failure` and `unanswerable` scenarios actually need still count.
# ----------------------------------------------------------------------

NOT_A_LIMITATION = [
    "comp2 failed on machine 1 on 2015-10-11; the probability is 1.0000.",
    "Machine 53 failed four times this year: comp1 twice, comp4 twice.",
    "Machine 16 has three comp3 replacements; the last followed a failed comp3.",
    "PN-COMP4-001 holds 4 units. Larger quantities are not available from "
    "this supplier.",
    "No. The part could not arrive before the predicted failure, so I would "
    "not order it.",
    "comp4 warning adequacy is marginal: 335.0h against a 288h lead time.",
]

IS_A_LIMITATION = [
    "The parts lookup failed with a timeout, so I cannot give you a stock figure.",
    "The database call returned an error, so I could not retrieve the risk scores.",
    "get_failure_risk came back empty, so the ranking covers three components.",
    "get_parts_position is unavailable on this run; treat stock as unknown.",
    "I cannot determine the root cause; nothing in the data attributes a "
    "failure to a cause.",
    "That is outside the 14-day horizon, so I have no basis for it.",
    "The inventory does not record supplier reliability.",
    "No cost data exists in this system. I cannot compute a downtime cost.",
    "I have no way to know which error caused it.",
    "The warning cannot support an ordering decision at any of these lead times.",
]


@pytest.mark.parametrize("answer", NOT_A_LIMITATION)
def test_a_fact_about_the_plant_is_not_a_statement_of_limitation(answer):
    assert _states_limitation(answer, make_trace()) is False


@pytest.mark.parametrize("answer", IS_A_LIMITATION)
def test_the_phrasings_the_scenarios_need_still_count(answer):
    assert _states_limitation(answer, make_trace()) is True


def test_the_bare_word_failed_no_longer_carries_the_assertion():
    """The exact regression: the word alone used to be enough."""
    scenario = Scenario(
        id="tf-1",
        category=Category.TOOL_FAILURE,
        question="a question long enough to validate",
        must_contain=["states_limitation"],
    )
    trace = make_trace(answer="comp2 failed on 2015-10-11 and was replaced.")
    outcomes = check_assertions(scenario, trace, judge=None)
    assert outcomes[0].method == "deterministic"
    assert outcomes[0].satisfied is False


# ======================================================================
# Section 3: judge calibration
# ======================================================================


def test_kappa_corrects_for_chance():
    """Raw agreement is misleading when one label dominates."""
    judge = [True] * 9 + [False]
    human = [True] * 10
    assert sum(a == b for a, b in zip(judge, human)) / 10 == 0.9
    assert cohens_kappa(judge, human) < 0.5


def test_perfect_disagreement_is_negative():
    assert cohens_kappa([True, False, True, False], [False, True, False, True]) < 0


def test_kappa_is_undefined_when_both_raters_are_constant():
    assert cohens_kappa([True] * 5, [True] * 5) != cohens_kappa([True] * 5, [True] * 5)


def test_calibration_flags_an_inadequate_rubric():
    agreement = calibrate([True] * 9 + [False], [True] * 10, "1.0.0")
    assert agreement.adequate is False
    assert str(KAPPA_FLOOR) in agreement.note


def test_judge_prompt_is_a_versioned_file():
    assert "Version:" in load_prompt()
    assert prompt_version() == "1.0.0"


def test_an_unparseable_judge_reply_fails_the_assertion():
    """The safe direction: a judge that cannot be parsed has not judged."""

    class Broken:
        def complete(self, prompt: str) -> str:
            return "I think probably yes"

    verdict = Judge(Broken()).assess("does it hold", "some answer")
    assert verdict.holds is False and verdict.confidence == 0.0


def test_judge_parses_a_well_formed_reply():
    class Fine:
        def complete(self, prompt: str) -> str:
            return 'noise {"holds": true, "confidence": 0.9, "reason": "said it"} noise'

    verdict = Judge(Fine()).assess("does it hold", "answer")
    assert verdict.holds is True and verdict.confidence == 0.9


# ======================================================================
# Section 4: runner, traces, diff
# ======================================================================


def test_the_shipped_scenarios_parse():
    """The suite is written. This pinned `== 2` while it was still a stub."""
    scenarios = load_scenarios(SCENARIOS)
    assert len(scenarios) == MINIMUM_SCENARIOS
    assert {s.category for s in scenarios} == set(REQUIRED_DISTRIBUTION)


def test_every_shipped_scenario_states_what_a_wrong_answer_would_say():
    """`must_not_contain` is where the value is, per docs/HOW_TO_WRITE_SCENARIOS.md.

    A scenario with only `must_contain` cannot catch an answer that is correct
    and unsafe at once -- the one that reports every figure accurately and then
    recommends an order the system cannot justify.
    """
    bare = [s.id for s in load_scenarios(SCENARIOS) if not s.must_not_contain]
    assert bare == []


def test_every_shipped_scenario_records_why_the_answer_is_correct():
    empty = [s.id for s in load_scenarios(SCENARIOS) if not s.notes.strip()]
    assert empty == []


def test_parts_scenario_forbids_the_risk_tool():
    """The Milestone 3B conclusion, expressed in the scenario set."""
    scenarios = {s.id: s for s in load_scenarios(SCENARIOS)}
    parts = scenarios["parts-position-comp3-01"]
    assert "get_failure_risk" in parts.forbidden_tools


def test_replay_client_makes_no_network_call_and_replays_in_order():
    client = ReplayClient(
        [
            {"tool_calls": [{"name": "get_machine_profile", "arguments": {"machine_id": 1}}]},
            {"text": "done", "tokens_in": 10, "tokens_out": 5},
        ]
    )
    first = client.complete([], [])
    assert first.wants_tools
    second = client.complete([], [])
    assert second.text == "done"


def test_running_past_the_end_of_a_transcript_raises():
    """An improvised turn would silently turn a regression into a pass."""
    client = ReplayClient([{"text": "done"}])
    client.complete([], [])
    with pytest.raises(TranscriptMissing, match="Re-record"):
        client.complete([], [])


def test_a_missing_transcript_names_the_file():
    with pytest.raises(TranscriptMissing, match="not found"):
        read_transcript("no-such-scenario", 1, TRANSCRIPTS)


@pytest.mark.skipif(
    not any(TRANSCRIPTS.glob("*.json")),
    reason="no transcripts recorded yet; see evals/record.py",
)
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_every_recorded_transcript_belongs_to_its_scenario(seed):
    """Every transcript on disk must pass both invariants for its scenario."""
    for scenario in load_scenarios(SCENARIOS):
        path = TRANSCRIPTS / f"{scenario.id}.seed{seed}.json"
        if not path.exists():
            continue
        validate_transcript(scenario, json.loads(path.read_text(encoding="utf-8")))


def test_suite_runs_offline_and_deterministically(built_db, recorded_suite):
    scenarios, transcripts = recorded_suite
    first, _ = run_suite(scenarios, Path(built_db), transcripts=transcripts)
    second, _ = run_suite(scenarios, Path(built_db), transcripts=transcripts)

    assert [r.passed for r in first.results] == [r.passed for r in second.results]
    assert [r.scenario_id for r in first.results] == [
        r.scenario_id for r in second.results
    ]
    assert len(first.results) == len(scenarios) * 3


# ----------------------------------------------------------------------
# Transcript / scenario consistency, and the validation-window guard
#
# The defect these pin: a transcript calling get_failure_risk(machine_id=42,
# as_of="2015-11-15T06:00:00") replayed against a scenario rewritten to machine
# 30 at 2015-10-14. It succeeded, read the test split, and scored passed=True
# for answering a different question.
# ----------------------------------------------------------------------


def a_scenario(**overrides) -> Scenario:
    base = dict(
        id="scn-window",
        category=Category.RISK_INADEQUATE,
        question="Machine 30 - should I order a comp1 replacement?",
        as_of=datetime(2015, 10, 14, 13, 0, 0),
        required_tools=["get_failure_risk"],
    )
    base.update(overrides)
    return Scenario(**base)


def a_transcript(tool_arguments: dict, **overrides) -> dict:
    base = {
        "scenario_id": "scn-window",
        "seed": 1,
        "model": {
            "provider": "stub",
            "model": "scripted",
            "version": "1",
            "temperature": 0.0,
            "seed": 1,
            "max_tokens": 1024,
        },
        "turns": [
            {"tokens_in": 1, "tokens_out": 1,
             "tool_calls": [{"name": "get_failure_risk", "arguments": tool_arguments}]},
            {"tokens_in": 1, "tokens_out": 1, "text": "an answer"},
        ],
    }
    base.update(overrides)
    return base


def test_a_transcript_matching_its_scenario_validates():
    """Anti-vacuity: a check that always fails would pass the tests below."""
    validate_transcript(
        a_scenario(),
        a_transcript({"machine_id": 30, "as_of": "2015-10-14T13:00:00"}),
    )


def test_a_transcript_with_the_wrong_as_of_is_invalid():
    with pytest.raises(TranscriptInvalid, match="does not match the scenario"):
        validate_transcript(
            a_scenario(),
            a_transcript({"machine_id": 30, "as_of": "2015-10-15T13:00:00"}),
        )


def test_a_transcript_with_the_wrong_machine_is_invalid():
    with pytest.raises(TranscriptInvalid, match="not named in the"):
        validate_transcript(
            a_scenario(),
            a_transcript({"machine_id": 42, "as_of": "2015-10-14T13:00:00"}),
        )


def test_a_transcript_reading_the_test_split_is_refused():
    """The exact regression: machine 42 at 2015-11-15, a test-split timestamp."""
    with pytest.raises(ValidationWindowError, match="outside the validation window"):
        validate_transcript(
            a_scenario(as_of=datetime(2015, 11, 15, 6, 0, 0)),
            a_transcript({"machine_id": 30, "as_of": "2015-11-15T06:00:00"}),
        )


def test_a_transcript_with_no_model_block_is_invalid():
    transcript = a_transcript({"machine_id": 30, "as_of": "2015-10-14T13:00:00"})
    del transcript["model"]
    with pytest.raises(TranscriptInvalid, match="cannot be attributed to a model"):
        validate_transcript(a_scenario(), transcript)


def test_a_fleet_question_names_no_machine_and_constrains_none():
    """The agent's own selection is the thing under test on fleet questions."""
    fleet = a_scenario(question="Which machines should I be looking at this week?")
    assert expected_machine_ids(fleet) is None
    validate_transcript(
        fleet, a_transcript({"machine_id": 77, "as_of": "2015-10-14T13:00:00"})
    )


@pytest.mark.parametrize(
    "moment",
    ["2015-09-30T23:00:00", "2015-10-18T00:00:00", "2015-11-15T06:00:00",
     "2015-12-17T06:00:00"],
)
def test_the_window_guard_refuses_everything_outside_validation(moment):
    with pytest.raises(ValidationWindowError):
        assert_validation_window(moment, "test")


@pytest.mark.parametrize("moment", ["2015-10-01T00:00:00", "2015-10-17T23:00:00"])
def test_the_window_guard_admits_both_edges(moment):
    """Anti-vacuity: the guard must not be so blunt that valid work fails."""
    assert assert_validation_window(moment, "test")


def test_the_window_guard_stops_a_live_call_before_it_reads_the_test_split():
    """During recording the model chooses the arguments; this is the only layer
    that runs before the read."""
    calls = []
    guarded = window_guarded(lambda name, args, db=None: calls.append(name))

    guarded("get_failure_risk", {"machine_id": 1, "as_of": "2015-10-05T00:00:00"})
    assert calls == ["get_failure_risk"]

    with pytest.raises(ValidationWindowError):
        guarded("get_failure_risk", {"machine_id": 1, "as_of": "2015-12-01T00:00:00"})
    assert calls == ["get_failure_risk"], "the refused call must not have executed"


def test_every_shipped_scenario_sits_inside_the_validation_window():
    for scenario in load_scenarios(SCENARIOS):
        if scenario.as_of is not None:
            assert_validation_window(scenario.as_of, scenario.id)


# ----------------------------------------------------------------------
# Recording: resumable, and one failure does not cost the sweep
# ----------------------------------------------------------------------


def test_recording_resumes_rather_than_re_recording_what_succeeded(
    built_db, tmp_path
):
    scenarios = load_scenarios(SCENARIOS)[:1]
    first = record_suite(
        scenarios, (1,), None, Path(built_db), tmp_path, resume=False,
        client_factory=stub_factory,
    )
    assert first["recorded"] and not first["skipped"]

    second = record_suite(
        scenarios, (1,), None, Path(built_db), tmp_path, resume=True,
        client_factory=stub_factory,
    )
    assert second["skipped"] and not second["recorded"]

    forced = record_suite(
        scenarios, (1,), None, Path(built_db), tmp_path, resume=False,
        client_factory=stub_factory,
    )
    assert forced["recorded"] and not forced["skipped"]


def test_one_failing_scenario_does_not_cost_the_rest_of_the_sweep(built_db, tmp_path):
    scenarios = load_scenarios(SCENARIOS)[:2]

    def flaky(config, scenario, seed):
        if scenario.id == scenarios[0].id:
            raise RuntimeError("provider exploded")
        return StubAdapter(scenario, seed)

    summary = record_suite(
        scenarios, (1,), None, Path(built_db), tmp_path, resume=False,
        client_factory=flaky,
    )
    assert len(summary["failed"]) == 1
    assert len(summary["recorded"]) == 1
    assert "provider exploded" in summary["failed"][0]["error"]


def test_a_recorded_transcript_carries_its_model_and_every_exchange(built_db, tmp_path):
    scenario = load_scenarios(SCENARIOS)[0]
    from evals.record import record_scenario

    transcript = record_scenario(
        scenario, 1, None, Path(built_db), tmp_path, client_factory=stub_factory
    )
    assert transcript["model"]["provider"] == "stub"
    assert transcript["turns"], "a replayable turn list"
    assert len(transcript["exchanges"]) == len(transcript["turns"])
    assert all("messages_sent" in e for e in transcript["exchanges"]), (
        "every request must be written down, not just every response"
    )


def test_a_run_records_which_model_produced_it(built_db, recorded_suite):
    """A transcript that cannot be attributed to a model is not usable."""
    scenarios, transcripts = recorded_suite
    run, _ = run_suite(scenarios, Path(built_db), transcripts=transcripts)
    assert run.metadata.model is not None
    assert run.metadata.model.provider == "stub"
    assert "stub" in run.metadata.model.label()


def test_a_run_refuses_transcripts_from_two_different_models(
    built_db, recorded_suite, tmp_path
):
    """Blending models would make every figure a weighted average of two systems."""
    import shutil

    scenarios, transcripts = recorded_suite
    scratch = tmp_path / "mixed"
    shutil.copytree(transcripts, scratch)

    path = TRANSCRIPTS_FOR_TEST(scratch, scenarios[0].id, 1)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["model"]["model"] = "a-different-model"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(TranscriptInvalid, match="more than one model"):
        run_suite(scenarios, Path(built_db), transcripts=scratch)


def test_traces_capture_the_tool_results_grounding_needs(built_db, recorded_suite):
    scenarios, transcripts = recorded_suite
    _, traces = run_suite(scenarios, Path(built_db), transcripts=transcripts)
    assert traces
    for trace in traces:
        assert trace.tool_calls
        for call in trace.tool_calls:
            assert call.result_json, "grounding cannot be checked without the result"


def _run(passed: dict[str, bool], forbidden=(), hallucinated=()) -> RunResults:
    from evals.schema import (
        ForbiddenCall,
        Hallucination,
        ScenarioResult,
        ToolSelection,
    )

    return RunResults(
        metadata=RunMetadata(
            run_id="r",
            git_sha="abc1234",
            utc=datetime.now(timezone.utc),
            mode="recorded",
            seeds=[1],
            n_scenarios=len(passed),
            harness_version="1.0.0",
        ),
        results=[
            ScenarioResult(
                scenario_id=name,
                category=Category.LOOKUP,
                seed=1,
                tool_selection=ToolSelection(
                    precision=1.0, recall=1.0, required_missing=[], forbidden_called=[]
                ),
                grounded=True,
                hallucinations=[],
                assertions=[],
                passed=value,
                tokens_in=10,
                tokens_out=5,
                wall_clock_ms=1.0,
                estimated_cost_usd=0.0001,
            )
            for name, value in passed.items()
        ],
        forbidden_calls=[
            ForbiddenCall(scenario_id=s, seed=1, tool=t, arguments={})
            for s, t in forbidden
        ],
        hallucinations=[
            Hallucination(scenario_id=s, seed=1, value=v, context="")
            for s, v in hallucinated
        ],
    )


def test_diff_separates_regressions_from_improvements():
    before = _run({"a": True, "b": False, "c": True})
    after = _run({"a": False, "b": True, "c": True})
    result = diff_module.compare(before, after)

    assert result.regressions == ["a#seed1"]
    assert result.improvements == ["b#seed1"]
    assert result.unchanged_pass == ["c#seed1"]
    assert result.has_regressions is True


def test_a_net_zero_change_is_still_a_regression():
    """Three fixed and three broken is not neutral. Netting them off is how a
    regression ships."""
    before = _run({"a": True, "b": True, "c": True, "d": False, "e": False, "f": False})
    after = _run({"a": False, "b": False, "c": False, "d": True, "e": True, "f": True})
    result = diff_module.compare(before, after)

    assert len(result.regressions) == 3
    assert len(result.improvements) == 3
    assert result.has_regressions is True


def test_diff_detects_a_new_forbidden_call():
    before = _run({"a": True})
    after = _run({"a": True}, forbidden=[("a", "get_failure_risk")])
    result = diff_module.compare(before, after)
    assert result.new_forbidden_calls == ["a#seed1:get_failure_risk"]
    assert result.has_regressions is True


def test_diff_detects_a_new_hallucination():
    before = _run({"a": True})
    after = _run({"a": True}, hallucinated=[("a", "87")])
    result = diff_module.compare(before, after)
    assert result.new_hallucinations == ["a#seed1:87"]
    assert result.has_regressions is True


def test_diff_reports_added_and_removed_scenarios():
    result = diff_module.compare(_run({"a": True}), _run({"b": True}))
    assert result.added == ["b#seed1"] and result.removed == ["a#seed1"]


# ======================================================================
# Section 5: report, and the scenario validator
# ======================================================================


def test_report_lists_failures_individually_not_as_averages():
    run = _run(
        {"a": True, "b": False},
        forbidden=[("b", "get_failure_risk")],
        hallucinated=[("b", "87")],
    )
    text = render_report(run)
    assert "get_failure_risk" in text
    assert "`87`" in text
    assert "p95" in text
    assert "Known failure modes" in text


def test_report_says_plainly_when_no_judge_was_configured():
    text = render_report(_run({"a": True}))
    assert "No judge was configured" in text
    assert "0.7" in text


def test_report_warns_that_an_incomplete_suite_is_not_a_quality_measure():
    text = render_report(_run({"a": True}))
    assert "not the full suite" in text


def _short_of_the_distribution() -> list[Scenario]:
    """One scenario in each of two categories, so every other category is short.

    These two tests used to run against `evals/scenarios.yaml` itself, which was
    a valid check only while that file was a two-scenario stub. Now that the
    suite is written the shortfall path needs its own incomplete input, or the
    tests would be asserting that the milestone was never delivered.
    """
    return [
        Scenario(
            id="stub-risk-inadequate",
            category=Category.RISK_INADEQUATE,
            question="question about machine 1",
        ),
        Scenario(
            id="stub-parts",
            category=Category.PARTS_POSITION,
            question="question about comp3 stock",
            forbidden_tools=["get_failure_risk"],
        ),
    ]


def test_validator_reports_the_shortfall_by_category(capsys):
    problems = validate_report(_short_of_the_distribution(), SCENARIOS)
    printed = capsys.readouterr().out

    assert problems, "two scenarios cannot satisfy a 41-scenario distribution"
    assert "risk_inadequate_warning" in printed
    assert "write" in printed
    assert any(str(MINIMUM_SCENARIOS) in p for p in problems)


def test_validator_names_the_two_categories_the_milestone_calls_the_heart():
    problems = " ".join(validate_report(_short_of_the_distribution(), SCENARIOS))
    assert "risk_inadequate_warning" in problems
    assert "unanswerable" in problems


def test_validator_accepts_the_shipped_suite(capsys):
    """The complement of the two above, against the real file."""
    assert validate_report(load_scenarios(SCENARIOS), SCENARIOS) == []


def test_validator_accepts_a_complete_set(capsys):
    """Anti-vacuity: a validator that always fails would pass the tests above."""
    complete = []
    for category, count in REQUIRED_DISTRIBUTION.items():
        for index in range(count):
            complete.append(
                Scenario(
                    id=f"{category.value}-{index}",
                    category=category,
                    question=f"question {index} about machine {index + 1}",
                    forbidden_tools=(
                        ["get_failure_risk"]
                        if category is Category.PARTS_POSITION
                        else []
                    ),
                    injected_failure=(
                        {"tool": "get_failure_risk", "code": "timeout"}
                        if category is Category.TOOL_FAILURE
                        else None
                    ),
                )
            )
    assert validate_report(complete, SCENARIOS) == []


def test_validator_rejects_a_parts_scenario_that_permits_the_risk_tool():
    scenarios = [
        Scenario(
            id=f"parts-{i}",
            category=Category.PARTS_POSITION,
            question="what is the stock position?",
            forbidden_tools=[],
        )
        for i in range(5)
    ]
    problems = " ".join(validate_report(scenarios, SCENARIOS))
    assert "must forbid get_failure_risk" in problems


def test_validator_rejects_a_tool_failure_scenario_without_an_injection():
    scenarios = [
        Scenario(id=f"tf-{i}", category=Category.TOOL_FAILURE, question="a question long enough to validate")
        for i in range(5)
    ]
    problems = " ".join(validate_report(scenarios, SCENARIOS))
    assert "injected_failure" in problems


def test_required_distribution_sums_to_the_stated_minimum():
    assert sum(REQUIRED_DISTRIBUTION.values()) == MINIMUM_SCENARIOS == 41
