"""Demo mode: the UI, driven by recorded transcripts instead of a model.

Milestone 9 item 5. A visitor with no API key, no Ollama and no network gets
the real agent loop, the real tools and the real database -- only the model's
turns come from disk rather than from a provider. That is the same substitution
the eval harness makes, and it is made here by importing that machinery rather
than by reimplementing it:

    ReplayClient, load_validated, window_guarded, identity_of  <- evals

**There is deliberately no second replay implementation.** A demo that replayed
transcripts its own way could drift from the harness and show a visitor
something the evaluation never scored. Importing the harness costs a slightly
odd dependency direction (`src` -> `evals`) and buys the guarantee that what
the page shows is what was measured. The import is local to each function so
the API package still starts without the eval extras present.

**Demo mode answers presets only, and says so.** A free-text question has no
recorded transcript, so there is nothing to replay. Rather than quietly falling
back to a live provider -- which would cost money on a page advertised as free
-- it returns a typed `invalid_input`, and the UI explains why.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.agent.contracts import ErrorCode, ToolError

#: Which recorded run each preset button plays. Seed 1 throughout: the seed is
#: an arbitrary recorded occasion, and pinning it keeps the page reproducible.
DEMO_SEED = 1


@dataclass(frozen=True)
class Preset:
    """One example question, and what a visitor is meant to notice in it.

    `takeaway` exists because the point of a preset is rarely the answer. The
    interesting scenarios are the ones where the system declines, and a visitor
    who does not know that reads a refusal as a failure.
    """

    scenario_id: str
    label: str
    question: str
    takeaway: str
    #: Grouping for the button row, not a scoring category.
    kind: str


#: Eight presets covering the six shapes the milestone asks for. Every
#: `scenario_id` is an owner-written scenario from `evals/scenarios.yaml` with a
#: recorded transcript on disk; `available()` drops any whose recording is
#: missing rather than offering a button that will error.
PRESETS: tuple[Preset, ...] = (
    Preset(
        scenario_id="lookup-machine-age-01",
        label="How old is machine 14?",
        question="How old is machine 14, and what model is it?",
        takeaway="A plain lookup. One tool call, one factual answer, nothing hedged.",
        kind="lookup",
    ),
    Preset(
        scenario_id="lookup-last-replacement-01",
        label="When was comp2 last replaced on machine 12?",
        question="When was comp2 last replaced on machine 12?",
        takeaway="Maintenance history, read straight from the record.",
        kind="lookup",
    ),
    Preset(
        scenario_id="risk-adequate-comp4-01",
        label="comp4 on machine 8 — is there time to use the part?",
        question=(
            "comp4 on machine 8 is showing high. The 12-day part is the one I can "
            "actually get — does the warning give me enough time to use it?"
        ),
        takeaway=(
            "The one component whose warning is long enough to act on. Watch "
            "warning_adequacy come back marginal rather than sufficient — it "
            "clears the lead time and fails the 1.25 safety factor."
        ),
        kind="risk",
    ),
    Preset(
        scenario_id="risk-inadequate-comp1-01",
        label="Machine 30 — should I order a comp1 replacement?",
        question="Machine 30 — should I order a comp1 replacement?",
        takeaway=(
            "The core finding, live. comp1's warning is 24 hours; the part takes "
            "10 days. The agent is required to refuse the order recommendation "
            "rather than produce one the supply chain cannot execute."
        ),
        kind="risk",
    ),
    Preset(
        scenario_id="risk-inadequate-uncalibrated-01",
        label="comp3 on machine 16 is at the top of the scale",
        question=(
            "comp3 on machine 16 is reading at the top of the scale. How "
            "seriously should I take that?"
        ),
        takeaway=(
            "comp3's probability is not established as better than the base "
            "rate. `calibrated: false` has to appear in the same breath as the "
            "number, not in a footnote."
        ),
        kind="risk",
    ),
    Preset(
        scenario_id="parts-position-comp3-01",
        label="What is our stock position on comp3 parts?",
        question="What is our stock position on comp3 parts, and how long would it last?",
        takeaway=(
            "Answered from stock and consumption alone. `get_failure_risk` is "
            "never called here — the separation is enforced by the import "
            "graph, not by the prompt."
        ),
        kind="parts",
    ),
    Preset(
        scenario_id="multi-step-weekly-review-01",
        label="Which machines should I be looking at this week?",
        question="Which machines should I be looking at this week?",
        takeaway="Fleet-level triage across several tool calls in sequence.",
        kind="fleet",
    ),
    Preset(
        scenario_id="unanswerable-root-cause-01",
        label="What caused the comp2 problem on machine 39?",
        question="What caused the comp2 problem on machine 39?",
        takeaway=(
            "The system cannot attribute cause — it has no diagnostic model. "
            "The honest answer is to say so, not to assemble a plausible story "
            "from correlations."
        ),
        kind="refusal",
    ),
)


class DemoUnavailable(Exception):
    """No recorded transcript backs this request."""


def scenarios_by_id(path: Path = Path("evals/scenarios.yaml")) -> dict:
    from evals.runner import load_scenarios

    return {s.id: s for s in load_scenarios(path)}


def available(
    transcripts: Path | None = None,
    scenarios_path: Path = Path("evals/scenarios.yaml"),
) -> list[dict]:
    """The presets that actually have a recording, in declaration order.

    A button whose transcript is missing is worse than no button: it advertises
    a capability and then errors. This filters instead.
    """
    from evals.transcript import TRANSCRIPTS, transcript_path

    directory = transcripts or TRANSCRIPTS
    try:
        known = scenarios_by_id(scenarios_path)
    except (OSError, ValueError):
        return []

    out: list[dict] = []
    for preset in PRESETS:
        scenario = known.get(preset.scenario_id)
        if scenario is None:
            continue
        if not transcript_path(preset.scenario_id, DEMO_SEED, directory).exists():
            continue
        out.append(
            {
                "scenario_id": preset.scenario_id,
                "label": preset.label,
                "question": preset.question,
                "takeaway": preset.takeaway,
                "kind": preset.kind,
                "as_of": str(scenario.as_of),
            }
        )
    return out


def replay(
    scenario_id: str,
    database: Path,
    transcripts: Path | None = None,
    scenarios_path: Path = Path("evals/scenarios.yaml"),
    seed: int = DEMO_SEED,
):
    """Run the agent against a recorded transcript. No network, no cost.

    Returns `(outcome, identity, tokens)` where `tokens` carries the recorded
    usage. The caller builds accounting from the emitted spans, exactly as the
    live path does -- this function does not price anything.
    """
    from evals.runner import ReplayClient, _inject
    from evals.transcript import TRANSCRIPTS, identity_of, load_validated, window_guarded
    from src.agent import tools as tools_module
    from src.agent.loop import Agent, LoopConfig

    directory = transcripts or TRANSCRIPTS
    known = scenarios_by_id(scenarios_path)
    scenario = known.get(scenario_id)
    if scenario is None:
        raise DemoUnavailable(f"no scenario {scenario_id!r}")

    try:
        transcript = load_validated(scenario, seed, directory)
    except Exception as exc:  # noqa: BLE001 - re-raised as a typed demo failure
        raise DemoUnavailable(
            f"no valid transcript for {scenario_id!r} seed {seed}: {type(exc).__name__}"
        ) from exc

    client = ReplayClient(transcript["turns"])
    agent = Agent(client, database=database, config=LoopConfig())

    # The same two wrappers the harness uses: the injected failure for
    # `tool_failure` scenarios, and the guard refusing an `as_of` outside the
    # validation window. Resolved through the module so the patch is visible to
    # the loop -- binding `dispatch` directly is how it went unnoticed before.
    original = tools_module.dispatch
    tools_module.dispatch = window_guarded(_inject(scenario, database))
    try:
        outcome = agent.run(
            scenario.question,
            as_of=scenario.as_of,
            scenario_id=scenario.id,
            seed=seed,
        )
    finally:
        tools_module.dispatch = original

    return outcome, identity_of(transcript), client


def not_a_preset_error(question: str) -> ToolError:
    """What free text gets in demo mode. Typed, and explains the alternative."""
    return ToolError(
        code=ErrorCode.INVALID_INPUT,
        message=(
            "Demo mode replays recorded runs, so it can only answer the preset "
            "questions. Nothing here calls a model, which is what makes the page "
            "free and offline. To ask your own question, configure a provider "
            "and set PDM_DEMO_MODE=0."
        ),
        tool="demo",
        retryable=False,
    )


# ----------------------------------------------------------------------
# Surfacing the two fields that matter
# ----------------------------------------------------------------------


def highlights(tool_calls: list[Any]) -> list[dict]:
    """Pull `warning_adequacy` and `calibrated` out of the risk results.

    Milestone 9 item 4: these two fields are the point of the system, and in a
    wall of JSON they are two lines among sixty. This lifts them so the page can
    render them as badges.

    **Derived, never authored.** Every value here is copied out of a
    `ComponentRisk` that a tool already returned. Nothing is recomputed and
    nothing is defaulted -- a missing field yields no badge rather than a
    reassuring one.
    """
    out: list[dict] = []
    for call in tool_calls:
        tool = getattr(call, "tool", None) or (
            call.get("tool") if isinstance(call, dict) else None
        )
        if tool != "get_failure_risk":
            continue
        status = getattr(call, "status", None) or (
            call.get("status") if isinstance(call, dict) else None
        )
        if status != "ok":
            continue
        result = getattr(call, "result", None)
        if result is None and isinstance(call, dict):
            result = call.get("result")
        if not isinstance(result, dict):
            continue
        # Tool results arrive wrapped: `Success[FailureRisk]` serialises as
        # `{"status": "ok", "data": {...}, "truncated": ...}`. Unwrap rather
        # than reaching for `components` at the top level, where it is not.
        payload = result.get("data") if isinstance(result.get("data"), dict) else result

        machine_id = payload.get("machine_id")
        for component in payload.get("components") or []:
            if not isinstance(component, dict):
                continue
            if "warning_adequacy" not in component and "calibrated" not in component:
                continue
            out.append(
                {
                    "machine_id": machine_id,
                    "component": component.get("component"),
                    "probability": component.get("calibrated_probability"),
                    # Carried under the contract's own names: an abbreviation
                    # to `ci_low` here would re-create at the API boundary the
                    # exact ambiguity the rename removed.
                    "model_prauc_ci_low": component.get("model_prauc_ci_low"),
                    "model_prauc_ci_high": component.get("model_prauc_ci_high"),
                    "calibrated": component.get("calibrated"),
                    "warning_adequacy": component.get("warning_adequacy"),
                    "exceeds_threshold": component.get("exceeds_threshold"),
                    "parts": [
                        {
                            "part_id": part.get("part_id"),
                            "lead_time_days": part.get("lead_time_days"),
                            "adequacy": part.get("adequacy"),
                            "detection_lead_hours": part.get("detection_lead_hours"),
                        }
                        for part in (component.get("per_part_adequacy") or [])
                        if isinstance(part, dict)
                    ],
                }
            )
    return out
