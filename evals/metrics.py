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
    # The safety factor `sufficient` requires over the part lead time. No tool
    # returns it -- it is the rule the adequacy verdict is computed by, not a
    # field -- but an answer explaining why a `marginal` verdict is not a
    # `sufficient` one has to be able to name it. Without this entry the
    # risk_adequate_warning scenarios cannot satisfy
    # `margin_below_safety_factor_stated` and avoid `fabricated_figure` at once.
    "1.25",
    # Window widths that live in field *names* rather than values --
    # `observed_consumption_per_30d`, `error1_count_7d`. An answer saying
    # "1.5 replacements per 30 days" is quoting the tool exactly; without these
    # the 30 has nothing to match against and reads as invented.
    "30", "7",
}

#: A tolerance for matching a formatted number against a raw one: an answer
#: saying "7%" for a probability of 0.0704 is grounded, not fabricated.
RELATIVE_TOLERANCE = 0.02


def _numbers_in(text: str) -> list[str]:
    return [m.group(1) for m in NUMBER.finditer(text)]


def _candidate_values(trace: ScenarioTrace, successful_only: bool = False) -> list[float]:
    """Every numeric value the tools actually returned, flattened.

    `successful_only` drops the results of failed calls. `check_grounding` keeps
    them, because a figure quoted out of an error message is traceable even if it
    is useless; the tool-failure assertions below need the stricter set, since the
    whole question there is whether the answer invented what the failed tool would
    have supplied.
    """
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
        if successful_only and call.status != "ok":
            continue
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


#: Verbs that make an inability a limit on the *answer* rather than a finding
#: about the plant. "cannot answer" is a limitation; "the part cannot arrive in
#: time" is a conclusion the agent reached, and a conclusion is not a limitation.
_ANSWERING_VERBS = (
    r"answer|address|tell|say|state|report|provide|give|offer|"
    r"retrieve|return|read|load|fetch|access|obtain|look ?up|query|search|"
    r"determine|establish|estimate|calculate|compute|measure|quantify|"
    r"confirm|verify|validate|check|assess|evaluate|rank|compare|"
    r"attribute|isolate|identify|distinguish|separate|diagnose|explain|"
    r"support|justify|complete|resolve|know"
)

#: What counts as a statement of limitation.
#:
#: This predicate used to match the bare word "failed", which any answer
#: satisfies by saying "comp2 failed on 2015-10-11" -- a fact about a machine,
#: not a statement about the boundary of the answer. That is precisely backwards
#: for the scenarios that assert it: `tool_failure` and `unanswerable` both turn
#: on the agent saying what it could *not* establish, and an answer that reported
#: a component failure and stopped satisfied the assertion while stating no
#: limitation at all. "could not" and "not available" had a milder version of the
#: same defect ("the part is not available" is a stock fact).
#:
#: Three shapes count, and in each the subject is the system, never the plant:
#:
#: 1. **Inability** -- a marker of inability governing a verb of answering.
#: 2. **A call that failed** -- the thing that failed is a tool, a query or a
#:    lookup, named as such.
#: 3. **Absence** -- what the answer would need is not in the data, not returned,
#:    or outside the system's scope.
LIMITATION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.I)
    for pattern in (
        # 1. Inability. Up to three words of slack for adverbs and objects:
        #    "cannot reliably tell you", "could not, on this data, determine".
        rf"\b(?:can ?not|can'?t|could ?n'?t|could not|cou?ldn'?t|"
        rf"unable to|not able to|no way (?:for me )?to|there is no way to|"
        rf"have no way to)"
        rf"(?:\W+\w+){{0,3}}\W+(?:{_ANSWERING_VERBS})\b",
        # 2. The subject that failed is a call, not a component.
        rf"\b(?:tool|call|query|request|lookup|look-up|retrieval|search|"
        rf"function|database)\b(?:\W+\w+){{0,3}}\W+"
        rf"(?:failed|errored|timed out|returned an error|returned nothing|"
        rf"did not return|didn'?t return|came back empty|is unavailable)\b",
        rf"\bfailed to(?:\W+\w+){{0,2}}\W+(?:{_ANSWERING_VERBS})\b",
        # Phrasings only a call can be the subject of, so the noun is not needed:
        # a component does not come back empty or time out.
        r"\b(?:came back empty|returned an error|returned nothing|"
        r"returned no (?:data|rows?|results?)|timed out|"
        r"did not return|didn'?t return)\b",
        # A snake_case identifier as the subject is a tool or a field name --
        # nothing in the plant is spelled `get_parts_position`.
        r"\b\w+_\w+\b(?:\W+\w+){0,2}\W+"
        r"(?:is|was) unavailable\b",
        # 3. Absence of what the answer would need.
        r"\bno (?:\w+ ){0,3}(?:data|record|records|figure|figures|"
        r"information|history|coverage|basis|source|tool|field|column|"
        r"way of knowing)\b",
        r"\bno (?:record|records|trace) of\b",
        # "not available" only where the thing missing is information. "Larger
        # quantities are not available from this supplier" is a stock fact.
        r"\b(?:data|figures?|information|records?|numbers?|values?|results?|"
        r"scores?|history|breakdown|field|column|tools?)\b(?:\W+\w+){0,3}\W+"
        r"(?:not|no longer) available\b",
        r"\b(?:not|no longer) available (?:in|from) (?:the |this )?"
        r"(?:data|dataset|database|tool|tools|system|inventory|record|records)\b",
        r"\b(?:data|figures?|information|records?|results?|scores?|tools?)\b"
        r"(?:\W+\w+){0,3}\W+(?:is|are|was|were) unavailable\b",
        r"\b(?:do not|don'?t|does not|doesn'?t) (?:have|record|capture|measure|"
        r"track|model|contain|include|store|expose|return|cover)\b",
        r"\b(?:is|are) not (?:measured|recorded|captured|modelled|modeled|"
        r"tracked|stored|returned|exposed|something)\b",
        r"\b(?:outside|beyond) (?:the |what |anything )?"
        r"(?:\w+[- ]){0,3}(?:horizon|window|scope|split|coverage|data|dataset|"
        r"tools?|system|model)\b",
        r"\bthis (?:data|dataset|system|model|harness|agent) "
        r"(?:can ?not|can'?t|does not|doesn'?t)\b",
    )
)


def _states_limitation(answer: str, trace: ScenarioTrace) -> bool:
    """Does the answer state a limit on what it could establish?

    Not "does the word 'failed' appear". See `LIMITATION_PATTERNS`.
    """
    return any(pattern.search(answer) for pattern in LIMITATION_PATTERNS)


def _flags_uncalibrated(answer: str, trace: ScenarioTrace) -> bool:
    lowered = answer.lower()
    return "calibrat" in lowered or "base rate" in lowered


# ----------------------------------------------------------------------
# Trace-grounded predicates
#
# The five predicates above match keywords in the answer. The ones below check
# the answer against *what the tools returned on this run*, which is a stronger
# claim: "the answer quotes comp2's probability" means the number in the answer
# is the number `get_failure_risk` produced, not that the word "comp2" appears
# near a digit. Anything that cannot be settled this way stays with the judge and
# is listed in docs/SECURITY.md's companion table in the milestone report.
# ----------------------------------------------------------------------


def _result_objects(trace: ScenarioTrace, tool: str | None = None):
    """Parsed results of successful calls, optionally for one tool."""
    for call in trace.tool_calls:
        if call.status != "ok":
            continue
        if tool is not None and call.tool != tool:
            continue
        try:
            yield json.loads(call.result_json)
        except json.JSONDecodeError:
            continue


def _values_under(trace: ScenarioTrace, key: str, tool: str | None = None) -> list:
    """Every value stored under `key`, at any depth, across successful results."""
    out: list = []

    def walk(node) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if k == key:
                    out.append(v)
                walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    for obj in _result_objects(trace, tool):
        walk(obj)
    return out


def _numeric(values: Sequence) -> list[float]:
    return [
        float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]


def _renders_as(target: float, value: float) -> bool:
    """Is `value` a reasonable rendering of `target`?

    Exact, rounded to any of four places, or -- for a figure that is already a
    proportion -- written as a percentage. The percentage form is only offered for
    targets in [0, 1] so that a stock level of 22 is not matched by a stray 2200.
    """
    forms = {target}
    if 0.0 <= target <= 1.0:
        forms.add(target * 100.0)
    for form in list(forms):
        for places in (0, 1, 2, 3, 4):
            forms.add(round(form, places))
    return any(abs(form - value) <= 1e-9 for form in forms)


def _reports_number(answer: str, targets: Sequence[float]) -> bool:
    if not targets:
        return False
    for token in _numbers_in(answer):
        try:
            value = float(token.replace(",", ""))
        except ValueError:
            continue
        if any(_renders_as(target, value) for target in targets):
            return True
    return False


def _ungrounded_numbers(answer: str, trace: ScenarioTrace) -> list[str]:
    """Figures in the answer that no *successful* tool call produced."""
    candidates = _candidate_values(trace, successful_only=True)
    loose: list[str] = []
    for token in _numbers_in(answer):
        if token in DOMAIN_CONSTANTS:
            continue
        try:
            value = float(token.replace(",", ""))
        except ValueError:
            continue
        if not _matches_any(value, candidates):
            loose.append(token)
    return loose


def _component_probability_reported(component: str):
    """The answer names the component and quotes the probability the tool returned."""

    def predicate(answer: str, trace: ScenarioTrace) -> bool:
        if component not in answer.lower():
            return False
        probabilities: list[float] = []
        for obj in _result_objects(trace, "get_failure_risk"):
            for risk in _objects_with(obj, "component", "calibrated_probability"):
                if risk.get("component") == component:
                    probabilities.extend(_numeric([risk["calibrated_probability"]]))
        return _reports_number(answer, probabilities)

    return predicate


def _objects_with(node, *required_keys: str) -> list[dict]:
    found: list[dict] = []

    def walk(n) -> None:
        if isinstance(n, dict):
            if all(k in n for k in required_keys):
                found.append(n)
            for v in n.values():
                walk(v)
        elif isinstance(n, list):
            for item in n:
                walk(item)

    walk(node)
    return found


def _adequacy_word(word: str):
    def predicate(answer: str, trace: ScenarioTrace) -> bool:
        return word in answer.lower()

    return predicate


#: `\bsufficient\b` does not match inside "insufficient" -- there is no word
#: boundary between "in" and "sufficient" -- so the two verdicts do not collide.
SUFFICIENT = re.compile(r"\bsufficient\b", re.I)
NEGATORS = ("not", "n't", "never", "far from", "barely", "hardly", "isn't", "no ")


def _claims_sufficient_warning(answer: str, trace: ScenarioTrace) -> bool:
    for match in SUFFICIENT.finditer(answer):
        preceding = answer[max(0, match.start() - 25) : match.start()].lower()
        if any(negator in preceding for negator in NEGATORS):
            continue
        return True
    return False


def _lead_time_mentioned(answer: str, trace: ScenarioTrace) -> bool:
    # Present in both PartPosition and PartAdequacy, so this survives the parts
    # tool failing while the risk tool succeeds.
    return _reports_number(answer, _numeric(_values_under(trace, "lead_time_days")))


def _stock_mentioned(answer: str, trace: ScenarioTrace) -> bool:
    return _reports_number(answer, _numeric(_values_under(trace, "stock_quantity")))


def _days_of_cover_reported(answer: str, trace: ScenarioTrace) -> bool:
    return _reports_number(answer, _numeric(_values_under(trace, "days_of_cover")))


def _consumption_reported(answer: str, trace: ScenarioTrace) -> bool:
    return _reports_number(
        answer, _numeric(_values_under(trace, "observed_consumption_per_30d"))
    )


def _machine_age_reported(answer: str, trace: ScenarioTrace) -> bool:
    return _reports_number(answer, _numeric(_values_under(trace, "age_years")))


def _machine_model_reported(answer: str, trace: ScenarioTrace) -> bool:
    lowered = answer.lower()
    models = {v for v in _values_under(trace, "model") if isinstance(v, str)}
    return any(model.lower() in lowered for model in models)


def _supplier_ids_reported(answer: str, trace: ScenarioTrace) -> bool:
    lowered = answer.lower()
    suppliers = {v for v in _values_under(trace, "supplier_id") if isinstance(v, str)}
    return any(supplier.lower() in lowered for supplier in suppliers)


def _error_counts_reported(answer: str, trace: ScenarioTrace) -> bool:
    lowered = answer.lower()
    ids = [v for v in _values_under(trace, "error_id") if isinstance(v, str)]
    if not any(error_id.lower() in lowered for error_id in ids):
        return False
    figures = _numeric(_values_under(trace, "count")) + _numeric(
        _values_under(trace, "total")
    )
    return _reports_number(answer, figures)


ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _replacement_date_reported(answer: str, trace: ScenarioTrace) -> bool:
    dates = {
        str(v)[:10]
        for v in _values_under(trace, "replaced_at")
        if isinstance(v, str) and ISO_DATE.match(str(v))
    }
    return any(date in answer for date in dates)


def _fabricated_figure(answer: str, trace: ScenarioTrace) -> bool:
    return bool(_ungrounded_numbers(answer, trace))


CURRENCY = re.compile(
    r"(?:[$£€]\s?(\d+(?:[.,]\d+)*))"
    r"|(?:(\d+(?:[.,]\d+)*)\s?(?:dollars|pounds|euros|usd|eur|gbp)\b)",
    re.I,
)


def _fabricated_currency_figure(answer: str, trace: ScenarioTrace) -> bool:
    loose = set(_ungrounded_numbers(answer, trace))
    for match in CURRENCY.finditer(answer):
        token = match.group(1) or match.group(2)
        if token and token in loose:
            return True
    return False


def _tool_failure_acknowledged(answer: str, trace: ScenarioTrace) -> bool:
    if not any(call.status == "error" for call in trace.tool_calls):
        return False
    lowered = answer.lower()
    return any(
        phrase in lowered
        for phrase in (
            "failed", "could not", "unable", "unavailable", "timed out", "timeout",
            "no data", "did not return", "not available", "error",
        )
    )


def _answers_from_failed_tool(answer: str, trace: ScenarioTrace) -> bool:
    """A figure in the answer that no successful call produced, while one failed.

    This is the v1 `search_manual` defect stated mechanically: the retrieval came
    back empty and the answer carried numbers anyway.
    """
    if not any(call.status == "error" for call in trace.tool_calls):
        return False
    return bool(_ungrounded_numbers(answer, trace))


def _claims_out_of_stock(answer: str, trace: ScenarioTrace) -> bool:
    lowered = answer.lower()
    return any(
        phrase in lowered
        for phrase in (
            "out of stock", "no stock", "zero stock", "none in stock",
            "nothing in stock", "stock is 0", "stocked out",
        )
    )


def _claims_all_machines_healthy(answer: str, trace: ScenarioTrace) -> bool:
    lowered = answer.lower()
    return any(
        phrase in lowered
        for phrase in (
            "all machines are healthy", "all machines healthy",
            "every machine is healthy", "fleet is healthy",
            "all machines as healthy",
        )
    )


BARE_VERDICT = re.compile(r"^\W*(yes|no)\b", re.I)


def _bare_yes_or_no(answer: str, trace: ScenarioTrace) -> bool:
    """A verdict with no room for the reasoning that makes it defensible.

    Catches the literal collapse -- an answer that opens on yes/no and stops. An
    answer that opens on "No" and then explains for a paragraph is not bare, and
    is not what the scenario forbids.
    """
    stripped = answer.strip()
    return len(stripped) <= 120 and BARE_VERDICT.match(stripped) is not None


DETERMINISTIC_ASSERTIONS = {
    # Keyword predicates over the answer.
    "warning_adequacy_reported": _reported_adequacy,
    "refuses_order_recommendation": _refuses_order,
    "order_recommendation": _order_recommendation,
    "states_limitation": _states_limitation,
    "flags_uncalibrated_probability": _flags_uncalibrated,
    "warning_adequacy_insufficient_stated": _adequacy_word("insufficient"),
    "warning_adequacy_marginal_stated": _adequacy_word("marginal"),
    "claims_sufficient_warning": _claims_sufficient_warning,
    "claims_out_of_stock": _claims_out_of_stock,
    "claims_all_machines_healthy": _claims_all_machines_healthy,
    "bare_yes_or_no": _bare_yes_or_no,
    # Checked against what the tools returned on this run.
    "comp1_probability_reported": _component_probability_reported("comp1"),
    "comp2_probability_reported": _component_probability_reported("comp2"),
    "comp3_probability_reported": _component_probability_reported("comp3"),
    "comp4_probability_reported": _component_probability_reported("comp4"),
    "part_lead_time_mentioned": _lead_time_mentioned,
    "stock_on_hand_mentioned": _stock_mentioned,
    "days_of_cover_reported": _days_of_cover_reported,
    "consumption_rate_reported": _consumption_reported,
    "machine_age_reported": _machine_age_reported,
    "machine_model_reported": _machine_model_reported,
    "supplier_ids_reported": _supplier_ids_reported,
    "error_counts_reported": _error_counts_reported,
    "replacement_date_reported": _replacement_date_reported,
    # Grounding, stated as assertions.
    "fabricated_figure": _fabricated_figure,
    "fabricated_currency_figure": _fabricated_currency_figure,
    "answers_from_failed_tool": _answers_from_failed_tool,
    "tool_failure_acknowledged": _tool_failure_acknowledged,
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
