"""Transcript storage, and the two invariants a transcript has to satisfy.

A recorded transcript is the harness's ground truth: everything
`docs/AGENT_EVALUATION.md` reports is computed by replaying one. That makes an
unchecked transcript worse than a missing one -- a missing transcript raises,
and an unchecked one scores.

Two invariants, both hard failures:

**1. The transcript must answer the scenario's question.** Its tool calls must
carry the scenario's `as_of` and, where the question names a machine, that
machine. This is not hypothetical: the two hand-written fixtures this module
replaces called `get_failure_risk(machine_id=42, as_of="2015-11-15T06:00:00")`
against a scenario that had been rewritten to machine 30 at
`2015-10-14T13:00:00`. The call succeeded, and the run scored `passed=True`
for answering a different question.

**2. No prediction time may fall outside the validation window.** The same
defect read the *test* split, which CLAUDE.md's standing rule opens once, at
the end of a milestone, after every modelling decision is final. An evaluation
harness is not a modelling decision and has no business there. The guard is
mechanical because the rule cannot rely on nobody making that mistake twice.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from evals.schema import Scenario
from src.agent.providers import ModelIdentity

TRANSCRIPTS = Path("evals") / "transcripts"

#: The validation split, from `data/generated/build_manifest.json` under
#: `features.val`: 408 prediction times, 2015-10-01 00:00 to 2015-10-17 23:00.
#: `docs/FEATURES.md` "Splits" carries the same figures and their derivation.
VALIDATION_START = datetime(2015, 10, 1, 0, 0, 0)
VALIDATION_END = datetime(2015, 10, 17, 23, 0, 0)

MACHINE_IN_QUESTION = re.compile(r"\bmachine\s+(\d+)\b", re.I)


class TranscriptInvalid(RuntimeError):
    """A transcript that does not belong to its scenario. Never recoverable."""


class ValidationWindowError(RuntimeError):
    """A prediction time outside the validation split. Never recoverable."""


class TranscriptMissing(RuntimeError):
    """No recording exists for this scenario and seed."""


# ----------------------------------------------------------------------
# Guards
# ----------------------------------------------------------------------


def parse_as_of(value) -> datetime:
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", ""))
    except ValueError as exc:
        raise ValidationWindowError(f"unparseable as_of {value!r}") from exc


def assert_validation_window(value, where: str) -> datetime:
    """Refuse any prediction time outside 2015-10-01 00:00 .. 2015-10-17 23:00."""
    moment = parse_as_of(value)
    if not (VALIDATION_START <= moment <= VALIDATION_END):
        raise ValidationWindowError(
            f"{where}: as_of {moment.isoformat()} is outside the validation "
            f"window ({VALIDATION_START.isoformat()} .. "
            f"{VALIDATION_END.isoformat()}). The test split is opened once, at "
            "the end of a milestone; an eval run is not that occasion."
        )
    return moment


def expected_machine_ids(scenario: Scenario) -> set[int] | None:
    """Machines the scenario's question names, or None if it names none.

    Fleet-level questions ("which machines should I be looking at this week")
    name no machine and constrain nothing -- the agent's selection is the thing
    under test there, so this returns None and the check does not apply.
    """
    found = {int(m.group(1)) for m in MACHINE_IN_QUESTION.finditer(scenario.question)}
    return found or None


def tool_calls_in(turns: list[dict]):
    for index, turn in enumerate(turns):
        for call in turn.get("tool_calls") or []:
            yield index, call


def validate_transcript(scenario: Scenario, transcript: dict) -> None:
    """Both invariants, as hard failures. Called on record and on replay."""
    identity = transcript.get("model")
    if not identity:
        raise TranscriptInvalid(
            f"{scenario.id}: transcript carries no `model` block. A transcript "
            "that cannot be attributed to a model is not usable; re-record it."
        )

    recorded_id = transcript.get("scenario_id")
    if recorded_id and recorded_id != scenario.id:
        raise TranscriptInvalid(
            f"transcript is for scenario {recorded_id!r}, not {scenario.id!r}"
        )

    machines = expected_machine_ids(scenario)

    for index, call in tool_calls_in(transcript.get("turns") or []):
        arguments = call.get("arguments") or {}
        where = f"{scenario.id} turn {index} call {call.get('name')!r}"

        if "as_of" in arguments:
            moment = assert_validation_window(arguments["as_of"], where)
            if scenario.as_of is not None and moment != scenario.as_of:
                raise TranscriptInvalid(
                    f"{where}: as_of {moment.isoformat()} does not match the "
                    f"scenario's {scenario.as_of.isoformat()}. The transcript "
                    "answers a different question than the scenario asks."
                )

        if "machine_id" in arguments and machines is not None:
            called = arguments["machine_id"]
            if called not in machines:
                raise TranscriptInvalid(
                    f"{where}: machine_id {called} is not named in the "
                    f"scenario's question (which names {sorted(machines)}). "
                    "The transcript answers a different question."
                )


def validate_scenario_window(scenario: Scenario) -> None:
    """A scenario whose own `as_of` is outside the window cannot be recorded."""
    if scenario.as_of is not None:
        assert_validation_window(scenario.as_of, f"scenario {scenario.id}")


# ----------------------------------------------------------------------
# Storage
# ----------------------------------------------------------------------


def transcript_path(scenario_id: str, seed: int, directory: Path = TRANSCRIPTS) -> Path:
    return directory / f"{scenario_id}.seed{seed}.json"


def write_transcript(transcript: dict, path: Path) -> Path:
    """Written atomically, so an interrupted recording leaves no half file.

    Resumability depends on this: the sweep decides what to skip by asking
    whether a valid transcript is already on disk, and a truncated file would
    be skipped as if it were complete.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(transcript, indent=2), encoding="utf-8")
    temporary.replace(path)
    return path


def read_transcript(scenario_id: str, seed: int, directory: Path = TRANSCRIPTS) -> dict:
    path = transcript_path(scenario_id, seed, directory)
    if not path.exists():
        raise TranscriptMissing(
            f"{path} not found. Record it with "
            f"`python -m evals.record --only {scenario_id} --seeds {seed}`."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_validated(scenario: Scenario, seed: int, directory: Path = TRANSCRIPTS) -> dict:
    transcript = read_transcript(scenario.id, seed, directory)
    validate_transcript(scenario, transcript)
    return transcript


def is_recorded(scenario: Scenario, seed: int, directory: Path = TRANSCRIPTS) -> bool:
    """Is there a complete, valid transcript already? Drives `--resume`."""
    try:
        load_validated(scenario, seed, directory)
    except (TranscriptMissing, TranscriptInvalid, ValidationWindowError, ValueError):
        return False
    return True


def identity_of(transcript: dict) -> ModelIdentity:
    return ModelIdentity(**transcript["model"])


# ----------------------------------------------------------------------
# The guard at the point of execution
# ----------------------------------------------------------------------


def window_guarded(dispatch):
    """Wrap tool dispatch so no eval run can read outside the validation split.

    `validate_transcript` already refuses an out-of-window `as_of` on replay,
    and `validate_scenario_window` refuses one in a scenario. This is the third
    layer, and the only one that holds while a *live* model is choosing the
    arguments: during recording the model can invent any timestamp it likes,
    and the first two guards run too late to stop the read.

    It raises rather than returning a `ToolError` deliberately. A `ToolError`
    would be recorded, and a transcript containing a refused test-split call is
    a transcript that documents the attempt rather than preventing it. The
    recording sweep catches the raise, marks that scenario failed, and carries
    on with the rest.
    """

    def guarded(name: str, arguments: dict, *args, **kwargs):
        if isinstance(arguments, dict) and arguments.get("as_of") is not None:
            assert_validation_window(arguments["as_of"], f"tool call {name!r}")
        return dispatch(name, arguments, *args, **kwargs)

    return guarded
