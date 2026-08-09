"""The four metric families. Reported separately, never collapsed into a score.

`docs/MILESTONE_5.md` section 1. A single number would let a good grounding score
hide a forbidden-tool call, and the forbidden-tool call is the one that matters:
`get_failure_risk` inside a pure parts question is a design violation, because
Milestone 3B established that parts reasoning cannot come from predictions.

The four are:

1. **Tool selection** -- precision and recall over tool calls, with forbidden
   calls counted absolutely.
2. **Grounding** -- every figure in the answer checked against the tool results
   for that run. An unmatched number is a hallucination and is listed
   individually.
3. **Required assertions** -- deterministic where the check can be deterministic,
   judged only where it cannot.
4. **Cost and latency** -- reported as a distribution, because the tail is what
   breaks in production.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from evals.schema import (
    AssertionOutcome,
    ForbiddenCall,
    Hallucination,
    Scenario,
    ScenarioResult,
    ScenarioTrace,
    ToolSelection,
)

#: Numbers in the answer worth checking. Deliberately not every digit: a bare
#: "14" that is the horizon, or "2015" in a date, would otherwise be flagged
#: constantly and the signal would be lost in the noise.
NUMBER = re.compile(r"(?<![\w.])(\d+(?:[.,]\d+)?)(?![\w])")

#: Figures that appear in the system prompt or are structural to the domain, and
#: so are not evidence of grounding either way.
DOMAIN_CONSTANTS = {
    "1", "2", "3", "4", "14", "24", "10", "100", "0", "9",
}

#: A tolerance for matching a formatted number against a raw one: an answer
#: saying "7%" for a probability of 0.0704 is grounded, not fabricated.
RELATIVE_TOLERANCE = 0.02


def _numbers_in(text: str) -> list[str]:
    return [m.group(1) for m in NUMBER.finditer(text)]


def _candidate_values(trace: ScenarioTrace) -> list[float]:
    """Every numeric value the tools actually returned, flattened."""
    values: list[float] = []

    def walk(node) -> None:
        if isinstance(node, bool):
            return
        if isinstance(node, (int, float)):
            values.append(float(node))
        elif isinstance(node, dict):
            for item in node.values():
                walk(item)
        elif isinstance(node, list):
            for item in node:
                walk(item)
        elif isinstance(node, str):
            for token in _numbers_in(node):
                try:
                    values.append(float(token.replace(",", "")))
                except ValueError:
                    pass

    for call in trace.tool_calls:
        try:
            walk(json.loads(call.result_json))
        except json.JSONDecodeError:
            walk(call.result_json)
    return values


def _matches_any(value: float, candidates: Sequence[float]) -> bool:
    for candidate in candidates:
        if candidate == value:
            return True
        # A probability rendered as a percentage is the same figure.
        for scaled in (candidate, candidate * 100, candidate / 100):
            if scaled == 0:
                continue
            if abs(scaled - value) <= abs(scaled) * RELATIVE_TOLERANCE:
                return True
        # Rounding: 0.0704 reported as 0.07.
        for places in (0, 1, 2, 3):
            if round(candidate, places) == value or round(candidate * 100, places) == value:
                return True
    return False


def check_grounding(
    scenario: Scenario, trace: ScenarioTrace
) -> tuple[bool, list[Hallucination]]:
    """Every figure in the answer must trace to a tool result."""
    candidates = _candidate_values(trace)
    found: list[Hallucination] = []

    for token in _numbers_in(trace.answer):
        if token in DOMAIN_CONSTANTS:
            continue
        try:
            value = float(token.replace(",", ""))
        except ValueError:
            continue
        if _matches_any(value, candidates):
            continue
        index = trace.answer.find(token)
        found.append(
            Hallucination(
                scenario_id=scenario.id,
                seed=trace.seed,
                value=token,
                context=trace.answer[max(0, index - 45) : index + 45].strip(),
            )
        )

    return not found, found


def check_tool_selection(
    scenario: Scenario, trace: ScenarioTrace
) -> tuple[ToolSelection, list[ForbiddenCall]]:
    """Precision and recall over tool calls, plus forbidden calls listed."""
    called = [call.tool for call in trace.tool_calls]
    called_set = set(called)
    required = set(scenario.required_tools)
    forbidden = set(scenario.forbidden_tools)

    hit = called_set & required
    # Precision counts distinct tools, not repeated calls: calling the right tool
    # twice is inefficiency, not a selection error, and the latency metric is
    # where that belongs.
    precision = len(hit) / len(called_set) if called_set else None
    recall = len(hit) / len(required) if required else None

    violations = [
        ForbiddenCall(
            scenario_id=scenario.id,
            seed=trace.seed,
            tool=call.tool,
            arguments=call.arguments,
        )
        for call in trace.tool_calls
        if call.tool in forbidden
    ]

    return (
        ToolSelection(
            precision=precision,
            recall=recall,
            required_missing=sorted(required - called_set),
            forbidden_called=sorted({v.tool for v in violations}),
        ),
        violations,
    )


#: Assertions the harness can settle without a model. Each maps to a predicate
#: over the answer and the trace. Anything not here goes to the judge, and the
#: judge's agreement with the owner's labels is reported alongside.
def _reported_adequacy(answer: str, trace: ScenarioTrace) -> bool:
    return any(
        word in answer.lower()
        for word in ("insufficient", "marginal", "sufficient", "warning adequacy")
    )


def _refuses_order(answer: str, trace: ScenarioTrace) -> bool:
    lowered = answer.lower()
    refusal = any(
        phrase in lowered
        for phrase in ("cannot support", "not enough warning", "too short", "cannot order", "does not support")
    )
    return refusal


def _order_recommendation(answer: str, trace: ScenarioTrace) -> bool:
    lowered = answer.lower()
    return any(
        phrase in lowered
        for phrase in ("you should order", "place an order", "i recommend ordering", "order the part now")
    )


def _states_limitation(answer: str, trace: ScenarioTrace) -> bool:
    lowered = answer.lower()
    return any(
        phrase in lowered
        for phrase in ("could not", "failed", "unable to", "cannot answer", "no data", "not available")
    )


def _flags_uncalibrated(answer: str, trace: ScenarioTrace) -> bool:
    lowered = answer.lower()
    return "calibrat" in lowered or "base rate" in lowered


DETERMINISTIC_ASSERTIONS = {
    "warning_adequacy_reported": _reported_adequacy,
    "refuses_order_recommendation": _refuses_order,
    "order_recommendation": _order_recommendation,
    "states_limitation": _states_limitation,
    "flags_uncalibrated_probability": _flags_uncalibrated,
}


def check_assertions(
    scenario: Scenario, trace: ScenarioTrace, judge=None
) -> list[AssertionOutcome]:
    """Deterministic where possible; the judge only for what cannot be."""
    outcomes: list[AssertionOutcome] = []

    for assertion in scenario.must_contain:
        outcomes.append(_evaluate(assertion, scenario, trace, judge, expected=True))
    for assertion in scenario.must_not_contain:
        outcomes.append(_evaluate(assertion, scenario, trace, judge, expected=False))

    return outcomes


def _evaluate(
    assertion: str, scenario: Scenario, trace: ScenarioTrace, judge, expected: bool
) -> AssertionOutcome:
    predicate = DETERMINISTIC_ASSERTIONS.get(assertion)
    if predicate is not None:
        present = predicate(trace.answer, trace)
        return AssertionOutcome(
            assertion=assertion,
            satisfied=(present is expected),
            method="deterministic",
        )

    # A literal string is still deterministic; only free-text judgements are not.
    if assertion.startswith("literal:"):
        needle = assertion.removeprefix("literal:").strip()
        present = needle.lower() in trace.answer.lower()
        return AssertionOutcome(
            assertion=assertion,
            satisfied=(present is expected),
            method="deterministic",
        )

    if judge is None:
        # No judge configured: record it as unsatisfied rather than passing it
        # silently. A missing check must never read as a passing one.
        return AssertionOutcome(
            assertion=assertion,
            satisfied=False,
            method="judge",
            judge_confidence=None,
        )

    verdict = judge.assess(assertion, trace.answer)
    return AssertionOutcome(
        assertion=assertion,
        satisfied=(verdict.holds is expected),
        method="judge",
        judge_confidence=verdict.confidence,
    )


def score_scenario(
    scenario: Scenario, trace: ScenarioTrace, judge=None
) -> tuple[ScenarioResult, list[ForbiddenCall], list[Hallucination]]:
    selection, violations = check_tool_selection(scenario, trace)
    grounded, hallucinations = check_grounding(scenario, trace)
    assertions = check_assertions(scenario, trace, judge)

    passed = (
        not violations
        and grounded
        and not selection.required_missing
        and all(a.satisfied for a in assertions)
    )

    result = ScenarioResult(
        scenario_id=scenario.id,
        category=scenario.category,
        seed=trace.seed,
        tool_selection=selection,
        grounded=grounded,
        hallucinations=hallucinations,
        assertions=assertions,
        passed=passed,
        tokens_in=trace.tokens_in,
        tokens_out=trace.tokens_out,
        wall_clock_ms=trace.wall_clock_ms,
        estimated_cost_usd=trace.estimated_cost_usd,
    )
    return result, violations, hallucinations


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q / 100 * (len(ordered) - 1)))))
    return ordered[index]


def cohens_kappa(judge_labels: Sequence[bool], human_labels: Sequence[bool]) -> float:
    """Agreement corrected for chance.

    Raw agreement is misleading when one label dominates: a judge that always
    says "satisfied" agrees 90% of the time on a suite that is 90% satisfied and
    has learned nothing.
    """
    if len(judge_labels) != len(human_labels) or not judge_labels:
        raise ValueError("label sequences must be the same non-zero length")

    n = len(judge_labels)
    observed = sum(a == b for a, b in zip(judge_labels, human_labels)) / n

    judge_true = sum(judge_labels) / n
    human_true = sum(human_labels) / n
    expected = judge_true * human_true + (1 - judge_true) * (1 - human_true)

    if expected == 1.0:
        # Both raters constant and identical: agreement is total but kappa is
        # undefined. Reporting 1.0 would overstate it; 0.0 would understate it.
        return float("nan")
    return (observed - expected) / (1 - expected)
