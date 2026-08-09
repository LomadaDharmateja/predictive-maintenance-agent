"""The judge, and its calibration against the owner's labels.

`docs/MILESTONE_5.md` section 3. A judge that has not been checked against human
labels is a random number generator with good manners, so the agreement figure is
reported with every run whether or not anyone asks for it.

The prompt is a versioned file (`evals/prompts/judge_assertion_v1.md`), not a
string literal, for the same reason the agent's system prompt is: it is the thing
most likely to need revision, and a change to it must be visible in a diff.

No judge model is wired up here. Milestone 5's acceptance requires the suite to
run offline, and a judge that calls a provider would break that on every run. The
interface and the calibration arithmetic exist; supplying a client is the owner's
step, taken alongside the hand labels the kappa needs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from evals.metrics import cohens_kappa
from evals.schema import JudgeAgreement

PROMPT_DIR = Path(__file__).parent / "prompts"
DEFAULT_PROMPT = PROMPT_DIR / "judge_assertion_v1.md"

#: Below this the rubric is inadequate and must be revised, with before and after
#: both reported. Section 3 sets the number.
KAPPA_FLOOR = 0.7


@dataclass(frozen=True)
class Verdict:
    holds: bool
    confidence: float
    reason: str = ""


class JudgeClient(Protocol):
    """Same shape as the agent's LLM interface, for the same reason."""

    def complete(self, prompt: str) -> str: ...


def load_prompt(path: Path = DEFAULT_PROMPT) -> str:
    if not path.exists():
        raise FileNotFoundError(f"judge prompt missing: {path}")
    return path.read_text(encoding="utf-8")


def prompt_version(path: Path = DEFAULT_PROMPT) -> str:
    for line in load_prompt(path).splitlines():
        if line.lower().startswith("version:"):
            return line.split(":", 1)[1].strip()
    return "unversioned"


class Judge:
    """Wraps a client with the versioned rubric and a strict output contract."""

    def __init__(self, client: JudgeClient, prompt_path: Path = DEFAULT_PROMPT) -> None:
        self.client = client
        self.prompt_path = prompt_path
        self.rubric = load_prompt(prompt_path)
        self.version = prompt_version(prompt_path)

    def assess(self, assertion: str, answer: str) -> Verdict:
        rendered = (
            f"{self.rubric}\n\n---\n\nASSERTION: {assertion}\n\nANSWER:\n{answer}\n"
        )
        raw = self.client.complete(rendered)
        try:
            payload = json.loads(raw[raw.index("{") : raw.rindex("}") + 1])
        except (ValueError, json.JSONDecodeError):
            # A judge that cannot be parsed has not judged. Recording it as
            # "does not hold, zero confidence" is the safe direction: it fails
            # the assertion rather than passing it on a malformed reply.
            return Verdict(holds=False, confidence=0.0, reason="unparseable judge reply")
        return Verdict(
            holds=bool(payload.get("holds", False)),
            confidence=float(payload.get("confidence", 0.0)),
            reason=str(payload.get("reason", "")),
        )


def calibrate(
    judge_labels: Sequence[bool],
    human_labels: Sequence[bool],
    version: str,
) -> JudgeAgreement:
    """Cohen's kappa between the judge and the owner's hand labels."""
    kappa = cohens_kappa(judge_labels, human_labels)
    agreements = sum(a == b for a, b in zip(judge_labels, human_labels))
    adequate = kappa == kappa and kappa >= KAPPA_FLOOR  # NaN-safe

    if kappa != kappa:
        note = (
            "Kappa is undefined: both raters gave a constant label, so there is no "
            "agreement beyond chance to measure. Label a set with both outcomes."
        )
    elif not adequate:
        note = (
            f"Kappa {kappa:.3f} is below the {KAPPA_FLOOR} floor. The rubric is "
            "inadequate; revise it and report the before and after."
        )
    else:
        note = f"Kappa {kappa:.3f} clears the {KAPPA_FLOOR} floor."

    return JudgeAgreement(
        kappa=kappa if kappa == kappa else 0.0,
        n_labelled=len(judge_labels),
        n_agreements=agreements,
        judge_prompt_version=version,
        adequate=adequate,
        note=note,
    )
