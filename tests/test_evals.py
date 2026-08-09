"""The harness machinery. Scenarios themselves are the project owner's work.

`docs/MILESTONE_5.md` sections 1, 3, 4 and 5. Every check here runs offline: the
runner replays recorded transcripts and no judge client is wired up, which is the
whole point of the recorded mode.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from evals import diff as diff_module
from evals.judge import KAPPA_FLOOR, Judge, calibrate, load_prompt, prompt_version
from evals.metrics import (
    check_assertions,
    check_grounding,
    check_tool_selection,
    cohens_kappa,
    percentile,
    score_scenario,
)
from evals.report import render as render_report
from evals.runner import (
    RecordedClient,
    TranscriptMissing,
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
    scenarios = load_scenarios(SCENARIOS)
    assert len(scenarios) == 2
    assert {s.category for s in scenarios} == {
        Category.RISK_INADEQUATE,
        Category.PARTS_POSITION,
    }


def test_parts_scenario_forbids_the_risk_tool():
    """The Milestone 3B conclusion, expressed in the scenario set."""
    scenarios = {s.id: s for s in load_scenarios(SCENARIOS)}
    parts = scenarios["parts-position-comp3-01"]
    assert "get_failure_risk" in parts.forbidden_tools


def test_recorded_client_makes_no_network_call_and_replays_in_order():
    client = RecordedClient(
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
    client = RecordedClient([{"text": "done"}])
    client.complete([], [])
    with pytest.raises(TranscriptMissing, match="Re-record"):
        client.complete([], [])


def test_a_missing_transcript_names_the_file():
    from evals.runner import load_transcript

    with pytest.raises(TranscriptMissing, match="not found"):
        load_transcript("no-such-scenario", 1, TRANSCRIPTS)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_every_shipped_scenario_has_a_transcript_for_every_seed(seed):
    for scenario in load_scenarios(SCENARIOS):
        assert (TRANSCRIPTS / f"{scenario.id}.seed{seed}.json").exists()


def test_suite_runs_offline_and_deterministically(built_db):
    scenarios = load_scenarios(SCENARIOS)
    first, _ = run_suite(scenarios, Path(built_db), transcripts=TRANSCRIPTS)
    second, _ = run_suite(scenarios, Path(built_db), transcripts=TRANSCRIPTS)

    assert [r.passed for r in first.results] == [r.passed for r in second.results]
    assert [r.scenario_id for r in first.results] == [
        r.scenario_id for r in second.results
    ]
    assert len(first.results) == len(scenarios) * 3


def test_traces_capture_the_tool_results_grounding_needs(built_db):
    _, traces = run_suite(
        load_scenarios(SCENARIOS), Path(built_db), transcripts=TRANSCRIPTS
    )
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


def test_validator_reports_the_shortfall_by_category(capsys):
    scenarios = load_scenarios(SCENARIOS)
    problems = validate_report(scenarios, SCENARIOS)
    printed = capsys.readouterr().out

    assert problems, "two scenarios cannot satisfy a 41-scenario distribution"
    assert "risk_inadequate_warning" in printed
    assert "write" in printed
    assert any(str(MINIMUM_SCENARIOS) in p for p in problems)


def test_validator_names_the_two_categories_the_milestone_calls_the_heart():
    problems = " ".join(validate_report(load_scenarios(SCENARIOS), SCENARIOS))
    assert "risk_inadequate_warning" in problems
    assert "unanswerable" in problems


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
