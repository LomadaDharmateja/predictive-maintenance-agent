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
from datetime import datetime
from itertools import combinations

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
#:
#: The trailing `[a-zA-Z%]*` is a **unit suffix**, and it is load-bearing. The
#: pattern used to end `(?![\w])` with no suffix allowed, so "28.7d" could not
#: match as `28.7` -- the `d` is a word character -- and the engine backtracked
#: to the shorter alternative `28`, which is a figure no tool ever returned.
#: The grounding check then reported a fabricated figure against an answer that
#: had quoted the tool exactly. A pilot run against a hosted model produced
#: precisely that on "28.7d cover", and "15d" and "31d" are the same shape.
#:
#: Consuming the whole suffix rather than one or two letters matters too:
#: allowing `[a-zA-Z]{1,3}` would fail on "15days" and drop the number
#: entirely, and a *missed* figure in a grounding check is the dangerous
#: direction -- a fabrication that is never tokenised is never caught.
NUMBER = re.compile(r"(?<![\w.])(\d+(?:[.,]\d+)?)[a-zA-Z%]*(?![\w])")

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


#: Identifiers are names, not quantities. `PN-COMP3-003` is one token meaning
#: one part; the `003` in it is not a figure anybody quoted, on either side of
#: the comparison.
#:
#: Stripping these before tokenising fixes a false positive at its root. Part
#: and supplier ids were contributing `1.0`, `2.0` and `3.0` to the candidate
#: set, and a candidate of `1.0` grounded anything in 98-102 through the
#: percentage form below -- so `PN-COMP3-001` appearing anywhere in a tool
#: result silently vouched for a fabricated "100 units on order".
#:
#: Applied to the answer and the tool results alike, so an answer naming a part
#: number is not then charged for the digits inside it.
IDENTIFIER = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)+\b")


def _numbers_in(text: str) -> list[str]:
    return [m.group(1) for m in NUMBER.finditer(IDENTIFIER.sub(" ", text))]


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


# ----------------------------------------------------------------------
# Derived figures: arithmetic over grounded values
#
# THE DECISION: arithmetic over grounded values counts as grounded, but only
# for a closed, enumerated set of derivations. General arithmetic does not.
#
# The question arose because an answer computed "~15-16 day intervals" from two
# grounded replacement dates and the harness called it fabrication. Refusing all
# derivation is clearly wrong: this system exists to compare a detection lead
# against a part lead time, and `margin_below_safety_factor_stated` *requires*
# the agent to do arithmetic the tools do not return. An answer that may only
# echo field values cannot do the job.
#
# Admitting *general* arithmetic is worse. A parts result carries around thirty
# numeric values. Allowing arbitrary pairwise sums, differences and ratios adds
# on the order of 2,700 candidates, and against a 2% relative tolerance almost
# any two-digit figure then finds a match by chance. That does not make the
# check permissive, it makes it vacuous -- and the two error directions are not
# symmetric. A false positive costs a reviewer a minute. A false negative ships
# an invented number labelled grounded, which is the exact v1 defect this
# project was rebuilt to prevent.
#
# So: three derivations, each applied once and never composed.
#
#   1. Differences between two timestamps, in days and in hours. Timestamps are
#      a small closed set, so this stays discriminating.
#   2. Hours <-> days conversion. Detection lead is reported in hours and lead
#      time in days; comparing them requires converting one.
#   3. The 1.25 safety factor, multiplied and divided. It is the rule the
#      adequacy verdict is computed by, and an answer explaining a `marginal`
#      verdict has to be able to apply it.
#
# Composition is excluded because it reintroduces the combinatorics through the
# back door: two rounds of the rules above would generate the same unfalsifiable
# candidate space.
#
# The empirical check that this is discriminating rather than a rubber stamp:
# on the pilot's maintenance history the real intervals are 15, 30, 45, 75 and
# 79 days, so an answer quoting 15 or 45 is admitted -- and the "16" that
# started this stays flagged, because it is not an interval between any two
# replacement dates. It was a hedge the answer invented around a real 15.
# `tests/test_evals.py` pins both halves of that.
# ----------------------------------------------------------------------

ISO_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?)?")

#: The safety factor `sufficient` requires over the part lead time.
SAFETY_FACTOR = 1.25

#: Refuse to walk a candidate set past this. If a trace ever produces more,
#: the derivation rules have grown past the point of discriminating and the
#: check needs revisiting rather than silently widening.
MAX_DERIVED_CANDIDATES = 4000


def _timestamps_in(trace: ScenarioTrace) -> list[datetime]:
    """Every ISO timestamp any successful tool returned."""
    moments: set[datetime] = set()
    for call in trace.tool_calls:
        if call.status != "ok":
            continue
        for match in ISO_TIMESTAMP.finditer(call.result_json):
            try:
                moments.add(datetime.fromisoformat(match.group(0).replace(" ", "T")))
            except ValueError:
                continue
    return sorted(moments)


def _derived_candidates(
    direct: Sequence[float], moments: Sequence[datetime]
) -> list[float]:
    """The three derivations above. Applied once each, never composed."""
    derived: set[float] = set()

    # 1. Intervals between timestamps, in days and hours.
    for earlier, later in combinations(moments, 2):
        seconds = abs((later - earlier).total_seconds())
        derived.add(seconds / 86400.0)
        derived.add(seconds / 3600.0)

    # 2 and 3. Unit conversion and the safety factor.
    for value in direct:
        if value == 0:
            continue
        derived.update(
            (
                value / 24.0,
                value * 24.0,
                value * SAFETY_FACTOR,
                value / SAFETY_FACTOR,
            )
        )

    out = sorted(derived)
    if len(out) > MAX_DERIVED_CANDIDATES:
        raise ValueError(
            f"derivation produced {len(out)} candidates, past the "
            f"{MAX_DERIVED_CANDIDATES} ceiling. The rules have stopped "
            "discriminating; revisit them rather than raising the ceiling."
        )
    return out


def _matches_derived(value: float, candidates: Sequence[float]) -> bool:
    """Strict matching for derived candidates: exact, or rounded. Nothing else.

    Deliberately not `_matches_any`. That function also tries the value scaled
    by 100 -- correct for a raw probability rendered as a percentage -- and
    applies a 2% relative tolerance. Both are far too loose once arithmetic has
    widened the candidate set. The existing unit test caught it immediately:
    a stock of 21, converted 21/24 = 0.875 days, scaled by 100 to 87.5, lands
    within 2% of a fabricated "87 on order". A derived figure is already in its
    natural unit and has no percentage form, so it gets no scaling and no
    tolerance band.
    """
    for candidate in candidates:
        if candidate == value:
            return True
        for places in (0, 1, 2, 3):
            if round(candidate, places) == value:
                return True
    return False


def _matches_any(value: float, candidates: Sequence[float]) -> bool:
    """Is `value` a reasonable rendering of any candidate?

    **The percentage form is offered only for candidates in [0, 1].** It was
    previously offered for every candidate, which meant a stock level of 22
    grounded a fabricated 2,200 and -- worse, because it was common -- a
    candidate of `1.0` grounded anything in 98 to 102. That mattered because
    part-id digits were entering the candidate set as 1.0, 2.0 and 3.0, so any
    trace touching `get_parts_position` vouched for three whole bands of
    invented figures. `_renders_as` already carried this guard; `_matches_any`
    did not, and it is the one the grounding check actually uses.

    `candidate / 100` is dropped entirely. No tool in this project returns a
    percentage, so dividing by 100 only ever widened the net.
    """
    for candidate in candidates:
        if candidate == value:
            return True
        forms = [candidate]
        # A probability rendered as a percentage is the same figure -- but only
        # a probability has a percentage form.
        if 0.0 <= candidate <= 1.0:
            forms.append(candidate * 100)
        for scaled in forms:
            if scaled == 0:
                continue
            if abs(scaled - value) <= abs(scaled) * RELATIVE_TOLERANCE:
                return True
        # Rounding: 0.0704 reported as 0.07.
        for form in forms:
            for places in (0, 1, 2, 3):
                if round(form, places) == value:
                    return True
    return False


def check_grounding(
    scenario: Scenario, trace: ScenarioTrace
) -> tuple[bool, list[Hallucination]]:
    """Every figure in the answer must trace to a tool result.

    Directly, or through one of the three derivations above. A figure admitted
    only by derivation is reported as such by `derived_figures`, so a reader can
    see how much of an answer's grounding rests on arithmetic rather than on
    quotation.
    """
    grounded, found, _ = _grounding(scenario, trace)
    return grounded, found


def derived_figures(scenario: Scenario, trace: ScenarioTrace) -> list[str]:
    """Figures the answer got right by arithmetic rather than by quoting."""
    return _grounding(scenario, trace)[2]


def _harness_supplied_values(scenario: Scenario) -> list[float]:
    """Figures the harness itself put in front of the agent.

    An answer cannot fabricate a number it was handed. Two sources:

    **The prediction time.** `Agent.run` writes `as_of` into the system prompt,
    because without it the model has no way to know what "now" is and will
    guess. An answer that then writes "as of 2015-10-16 06:00" is quoting its
    own instructions. The grounding check had no way to see that, so a
    `tool_failure` scenario -- where the only tool call errors and there are no
    successful results to match against -- flagged the year and the clock
    digits as fabricated.

    **The question.** "Machine 30 - should I order a comp1 replacement?" puts
    30 in front of the agent. Repeating it back is not invention.

    The line is deliberately drawn at *what the agent was shown*, not at what is
    plausible. Nothing here is inferred, derived or looked up; it is the two
    inputs the harness controls, and both are checkable from the scenario file.
    """
    supplied: list[float] = []
    for text in (
        # `sep=" "` on purpose: "2015-10-16T06:00" hides the 16 and the 06
        # behind the "T", because a letter cannot separate two figures.
        scenario.as_of.isoformat(sep=" ") if scenario.as_of else "",
        scenario.question,
    ):
        for token in _numbers_in(text):
            try:
                supplied.append(float(token.replace(",", "")))
            except ValueError:
                pass
    return supplied


def _grounding(
    scenario: Scenario, trace: ScenarioTrace
) -> tuple[bool, list[Hallucination], list[str]]:
    direct = _candidate_values(trace) + _harness_supplied_values(scenario)
    derived: list[float] | None = None  # built lazily; most answers never need it
    found: list[Hallucination] = []
    by_arithmetic: list[str] = []

    for token in _numbers_in(trace.answer):
        if token in DOMAIN_CONSTANTS:
            continue
        try:
            value = float(token.replace(",", ""))
        except ValueError:
            continue
        if _matches_any(value, direct):
            continue
        if derived is None:
            derived = _derived_candidates(direct, _timestamps_in(trace))
        if _matches_derived(value, derived):
            by_arithmetic.append(token)
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

    return not found, found, by_arithmetic


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


#: What counts as declining to recommend an order.
#:
#: **Widened after seeing an answer, which is a hazard and is recorded as one.**
#: The original list only matched refusals phrased as a statement about the
#: *system's capability* -- "cannot support", "does not support". A hosted model
#: answered `risk-inadequate-comp1-01` with "Short answer: no - don't order
#: comp1 based on this risk score", then reported adequacy `insufficient`,
#: flagged `calibrated: false`, and quoted both lead times. That is the verdict
#: the scenario exists to elicit, in the register a maintenance planner actually
#: uses, and the predicate failed it.
#:
#: This is a gap in the predicate rather than the test being fitted to the
#: output, and the distinction rests on something checkable: the scenario's
#: `notes` field, written before any model ran, says the correct answer "states
#: that the prediction cannot support an ordering decision". A direct refusal to
#: order is that statement. The predicate was enumerating one dialect of it.
#:
#: The safeguard is that nothing here weakens the assertion: every phrase added
#: is an explicit refusal, none of them can be satisfied by an answer that
#: recommends ordering, and `order_recommendation` still has to be absent
#: independently. An answer that says nothing about ordering still fails.
REFUSAL_PHRASES = (
    # The system cannot support the decision -- the original framing.
    "cannot support", "can't support", "does not support", "doesn't support",
    "not enough warning", "too short", "cannot order", "can't order",
    "cannot inform", "does not justify", "doesn't justify",
    # A direct refusal, which is the same verdict stated plainly.
    "do not order", "don't order", "would not order", "wouldn't order",
    "should not order", "shouldn't order", "not order",
    "do not reorder", "don't reorder", "no reorder trigger",
    "hold off", "against ordering", "no immediate reorder",
    "not recommend ordering", "would not recommend an order",
)


def _refuses_order(answer: str, trace: ScenarioTrace) -> bool:
    lowered = answer.lower()
    return any(phrase in lowered for phrase in REFUSAL_PHRASES)


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


def _ungrounded_numbers(
    answer: str, trace: ScenarioTrace, scenario: Scenario | None = None
) -> list[str]:
    """Figures in the answer that no *successful* tool call produced.

    Harness-supplied values count here too, and this is where it matters most:
    a `tool_failure` scenario has no successful call at all, so without them the
    prediction time the harness wrote into the system prompt has nothing to
    trace to and the answer is charged for quoting its own instructions.
    """
    candidates = _candidate_values(trace, successful_only=True)
    if scenario is not None:
        candidates = candidates + _harness_supplied_values(scenario)
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


def _fabricated_figure(answer, trace, scenario=None) -> bool:
    return bool(_ungrounded_numbers(answer, trace, scenario))


_fabricated_figure.needs_scenario = True


CURRENCY = re.compile(
    r"(?:[$£€]\s?(\d+(?:[.,]\d+)*))"
    r"|(?:(\d+(?:[.,]\d+)*)\s?(?:dollars|pounds|euros|usd|eur|gbp)\b)",
    re.I,
)


def _fabricated_currency_figure(answer, trace, scenario=None) -> bool:
    loose = set(_ungrounded_numbers(answer, trace, scenario))
    for match in CURRENCY.finditer(answer):
        token = match.group(1) or match.group(2)
        if token and token in loose:
            return True
    return False


_fabricated_currency_figure.needs_scenario = True


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


def _answers_from_failed_tool(answer, trace, scenario=None) -> bool:
    """A figure in the answer that no successful call produced, while one failed.

    This is the v1 `search_manual` defect stated mechanically: the retrieval came
    back empty and the answer carried numbers anyway.
    """
    if not any(call.status == "error" for call in trace.tool_calls):
        return False
    return bool(_ungrounded_numbers(answer, trace, scenario))


_answers_from_failed_tool.needs_scenario = True


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
        # Most predicates need only the answer and the trace. The three that
        # check grounding also need the scenario, because what the harness put
        # in front of the agent is not something the agent invented.
        present = (
            predicate(trace.answer, trace, scenario)
            if getattr(predicate, "needs_scenario", False)
            else predicate(trace.answer, trace)
        )
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
    grounded, hallucinations, derived = _grounding(scenario, trace)
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
        derived_figures=derived,
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
