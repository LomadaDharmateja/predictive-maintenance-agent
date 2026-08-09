"""Schemas for scenarios, traces and results.

`docs/MILESTONE_5.md` sections 2, 4 and 5. Everything the harness reads or writes
is a Pydantic model, for the same reason the agent's tools are: a schema that is
checked is a schema that cannot drift from what produced it.

**Scenarios are the project owner's work, not the harness's.** This file defines
the format and the required distribution; `evals/scenarios.yaml` holds two worked
examples so the shape is unambiguous, and `evals/validate_scenarios.py` refuses a
file that does not meet the distribution. An agent that writes its own
correctness criteria measures nothing.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ----------------------------------------------------------------------
# Scenarios
# ----------------------------------------------------------------------


class Category(str, Enum):
    LOOKUP = "lookup"
    RISK_ADEQUATE = "risk_adequate_warning"
    RISK_INADEQUATE = "risk_inadequate_warning"
    PARTS_POSITION = "parts_position"
    MULTI_STEP = "multi_step"
    TOOL_FAILURE = "tool_failure"
    PROMPT_INJECTION = "prompt_injection"
    UNANSWERABLE = "unanswerable"


#: The distribution `docs/MILESTONE_5.md` section 2 requires. The eight
#: `risk_inadequate_warning` scenarios and the four `unanswerable` ones are the
#: heart of the suite: Milestone 3B established that one part in nine can be
#: ordered in time, and an agent that hides that is worse than no agent.
REQUIRED_DISTRIBUTION: dict[Category, int] = {
    Category.LOOKUP: 5,
    Category.RISK_ADEQUATE: 5,
    Category.RISK_INADEQUATE: 8,
    Category.PARTS_POSITION: 5,
    Category.MULTI_STEP: 5,
    Category.TOOL_FAILURE: 5,
    Category.PROMPT_INJECTION: 4,
    Category.UNANSWERABLE: 4,
}

MINIMUM_SCENARIOS = sum(REQUIRED_DISTRIBUTION.values())  # 41


class InjectedFailure(Strict):
    """A tool failure to force, for `tool_failure` scenarios.

    The harness injects it rather than waiting for one to happen, so the check is
    deterministic. v1's `search_manual` returned its error text as a normal
    result and the agent kept answering; these scenarios exist to prove that
    cannot recur.
    """

    tool: str
    code: str
    message: str = "injected failure"


class Scenario(Strict):
    id: str = Field(min_length=3)
    category: Category
    question: str = Field(min_length=5)
    as_of: datetime | None = None

    #: Tools the answer cannot be correct without.
    required_tools: list[str] = Field(default_factory=list)
    #: Tools whose use is a design violation. One call to `get_failure_risk`
    #: inside a pure parts question is not a small error.
    forbidden_tools: list[str] = Field(default_factory=list)

    #: Deterministic where possible: a machine id, a stock count, the literal
    #: string `insufficient`. Judged only where it genuinely cannot be checked.
    must_contain: list[str] = Field(default_factory=list)
    must_not_contain: list[str] = Field(default_factory=list)

    injected_failure: InjectedFailure | None = None
    notes: str = ""


class ScenarioSuite(Strict):
    scenarios: list[Scenario]


# ----------------------------------------------------------------------
# Traces
# ----------------------------------------------------------------------


class ToolCallTrace(Strict):
    """One tool call: what was asked, what came back, what it cost."""

    tool: str
    arguments: dict
    status: Literal["ok", "error"]
    error_code: str | None = None
    truncated: bool = False
    duration_ms: float
    #: The serialised result. Kept in full so grounding can be checked against
    #: what the model actually saw, not against a summary of it.
    result_json: str


class ScenarioTrace(Strict):
    scenario_id: str
    seed: int
    answer: str
    tool_calls: list[ToolCallTrace]
    iterations: int
    hit_iteration_limit: bool
    messages_dropped: int
    tokens_in: int
    tokens_out: int
    wall_clock_ms: float
    estimated_cost_usd: float


# ----------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------


class ForbiddenCall(Strict):
    """Reported individually, never averaged. Section 1 is explicit about this."""

    scenario_id: str
    seed: int
    tool: str
    arguments: dict


class Hallucination(Strict):
    """A figure in the answer that appears in no tool result for that run."""

    scenario_id: str
    seed: int
    value: str
    context: str


class ToolSelection(Strict):
    precision: float | None
    recall: float | None
    required_missing: list[str]
    forbidden_called: list[str]


class AssertionOutcome(Strict):
    assertion: str
    satisfied: bool
    method: Literal["deterministic", "judge"]
    judge_confidence: float | None = None


class ScenarioResult(Strict):
    scenario_id: str
    category: Category
    seed: int
    tool_selection: ToolSelection
    grounded: bool
    hallucinations: list[Hallucination]
    assertions: list[AssertionOutcome]
    passed: bool
    tokens_in: int
    tokens_out: int
    wall_clock_ms: float
    estimated_cost_usd: float

    @property
    def failed_assertions(self) -> list[str]:
        return [a.assertion for a in self.assertions if not a.satisfied]


class RunMetadata(Strict):
    run_id: str
    git_sha: str
    utc: datetime
    mode: Literal["recorded", "live"]
    seeds: list[int]
    n_scenarios: int
    harness_version: str


class RunResults(Strict):
    metadata: RunMetadata
    results: list[ScenarioResult]
    forbidden_calls: list[ForbiddenCall]
    hallucinations: list[Hallucination]
    judge_agreement: "JudgeAgreement | None" = None


class JudgeAgreement(Strict):
    """Cohen's kappa between judge labels and the owner's hand labels.

    Reported always. An uncalibrated judge is a random number generator with good
    manners, and a harness that does not say so is worse than none.
    """

    kappa: float
    n_labelled: int
    n_agreements: int
    judge_prompt_version: str
    adequate: bool
    note: str = ""


RunResults.model_rebuild()
