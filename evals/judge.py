"""The judge, and its calibration against the owner's labels.

`docs/MILESTONE_5.md` section 3. A judge that has not been checked against human
labels is a random number generator with good manners, so the agreement figure is
reported with every run whether or not anyone asks for it.

The prompt is a versioned file (`evals/prompts/judge_assertion_v1.md`), not a
string literal, for the same reason the agent's system prompt is: it is the thing
most likely to need revision, and a change to it must be visible in a diff.

**Verdicts are recorded, exactly as transcripts are.** Milestone 5's acceptance
requires the suite to run offline and free, and a judge that calls a provider on
every run breaks both. So a judge call is made once, against a live model, and
written to `evals/judgements/<prompt version>.json`; thereafter the run replays
it. A cache miss with no client raises rather than passing, for the same reason a
missing transcript raises: a check that did not run must never read as one that
passed.

The cache is keyed on the rubric version as well as the assertion and the answer.
Revising the rubric therefore invalidates every verdict it produced, which is the
point -- section 3 requires the before and after of a rubric change to be
reported, and that is impossible if the old verdicts survive the edit.

**The judge's model is recorded separately from the agent's.** They are two
different measurements and may be two different models: the agent's model is what
is being evaluated, the judge's is part of the instrument. A report that named
only one of them would let a judge change be mistaken for an agent change.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from evals.metrics import cohens_kappa
from evals.schema import JudgeAgreement
from src.agent.providers import ModelIdentity

PROMPT_DIR = Path(__file__).parent / "prompts"
DEFAULT_PROMPT = PROMPT_DIR / "judge_assertion_v1.md"
JUDGEMENTS_DIR = Path("evals") / "judgements"

#: Below this the rubric is inadequate and must be revised, with before and after
#: both reported. Section 3 sets the number.
KAPPA_FLOOR = 0.7


@dataclass(frozen=True)
class Verdict:
    holds: bool
    confidence: float
    reason: str = ""


class JudgeClient(Protocol):
    """Same shape as the agent's LLM interface, for the same reason.

    `system` carries the rubric. It is a separate argument rather than part of
    `prompt` so a provider can cache it: the rubric is byte-identical across
    every judge call in a run, and the assertion and answer are not.
    """

    def complete(self, prompt: str, system: str | None = None) -> str: ...


def load_prompt(path: Path = DEFAULT_PROMPT) -> str:
    if not path.exists():
        raise FileNotFoundError(f"judge prompt missing: {path}")
    return path.read_text(encoding="utf-8")


def prompt_version(path: Path = DEFAULT_PROMPT) -> str:
    for line in load_prompt(path).splitlines():
        if line.lower().startswith("version:"):
            return line.split(":", 1)[1].strip()
    return "unversioned"


class JudgementMissing(RuntimeError):
    """No recorded verdict, and no client to produce one.

    The judge equivalent of a missing transcript. Passing the assertion would
    record a check that never ran as one that passed.
    """


class VerdictCache:
    """Recorded verdicts, keyed by (rubric version, judge model, assertion, answer).

    Keyed on the answer's full text rather than the scenario id: two seeds that
    produced the same answer deserve the same verdict, and re-recording one
    scenario should not silently re-grade the others.

    **The judge model is part of the key.** Without it, switching the judge from
    Sonnet to Haiku replays every Sonnet verdict while the report header names
    Haiku -- the same attribution failure that `RunMetadata.model` exists to
    prevent, one level down. A verdict is a measurement made by a specific
    instrument; served under another instrument's name it is a fabrication.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.entries: dict[str, dict] = {}
        self.hits = 0
        self.misses = 0
        if path.exists():
            self.entries = json.loads(path.read_text(encoding="utf-8")).get("verdicts", {})

    @staticmethod
    def key(version: str, model_key: str, assertion: str, answer: str) -> str:
        digest = hashlib.sha256(
            "\x00".join((version, model_key, assertion, answer)).encode("utf-8")
        )
        return digest.hexdigest()[:32]

    def models(self) -> set[str]:
        """Judge models that have verdicts here, for a helpful miss message."""
        return {e.get("model_key", "unknown") for e in self.entries.values()}

    def get(
        self, version: str, model_key: str, assertion: str, answer: str
    ) -> Verdict | None:
        entry = self.entries.get(self.key(version, model_key, assertion, answer))
        if entry is None:
            self.misses += 1
            return None
        self.hits += 1
        return Verdict(
            holds=bool(entry["holds"]),
            confidence=float(entry["confidence"]),
            reason=str(entry.get("reason", "")),
        )

    def put(
        self,
        version: str,
        model_key: str,
        assertion: str,
        answer: str,
        verdict: Verdict,
        model: ModelIdentity | None,
    ) -> None:
        self.entries[self.key(version, model_key, assertion, answer)] = {
            "holds": verdict.holds,
            "confidence": verdict.confidence,
            "reason": verdict.reason,
            "assertion": assertion,
            # `model_key` is what the key is built from (provider/model, stable
            # across re-pulls and version bumps); `model` is the full identity
            # including the resolved version, kept for audit.
            "model_key": model_key,
            "model": model.label() if model else "unknown",
        }

    def save(self, identity: ModelIdentity | None = None) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {
                    "judge_model": identity.model_dump() if identity else None,
                    "verdicts": self.entries,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return self.path


class ProviderJudgeClient:
    """Bridges a `src.agent.providers` adapter to the judge's flat interface.

    The judge asks one question and reads one string; the agent's adapters take
    a conversation and a tool list. Same provider code underneath -- the judge
    gets no separate credential path, no separate retry policy, and no separate
    way to fail.
    """

    def __init__(self, adapter) -> None:
        self.adapter = adapter
        self.tokens_in = 0
        self.tokens_out = 0
        self.cache_read = 0
        self.cache_write = 0
        self.calls = 0

    @property
    def identity(self) -> ModelIdentity:
        return self.adapter.identity

    def complete(self, prompt: str, system: str | None = None) -> str:
        messages = [{"role": "user", "content": prompt}]
        if system:
            messages.insert(0, {"role": "system", "content": system})
        response = self.adapter.complete(messages, [])
        exchange = self.adapter.last_exchange or {}
        self.tokens_in += int(exchange.get("tokens_in", 0))
        self.tokens_out += int(exchange.get("tokens_out", 0))
        self.cache_read += int(exchange.get("cache_read", 0))
        self.cache_write += int(exchange.get("cache_write", 0))
        self.calls += 1
        return response.text or ""


class Judge:
    """Wraps a client with the versioned rubric and a strict output contract.

    `client=None` is the offline mode: verdicts come from the cache and a miss
    raises. A client is only needed the first time an answer is graded.
    """

    def __init__(
        self,
        client: JudgeClient | None = None,
        prompt_path: Path = DEFAULT_PROMPT,
        cache: VerdictCache | None = None,
        model_key: str | None = None,
    ) -> None:
        self.client = client
        self.prompt_path = prompt_path
        self.rubric = load_prompt(prompt_path)
        self.version = prompt_version(prompt_path)
        self.cache = cache
        self._model_key = model_key

    @property
    def identity(self) -> ModelIdentity | None:
        return getattr(self.client, "identity", None)

    @property
    def model_key(self) -> str:
        """Which judge model's verdicts this instance reads and writes.

        Taken from the live client when there is one, and from `--judge-model`
        when replaying offline. `unspecified` never matches a recorded verdict,
        so an offline run that forgets to name a judge misses rather than
        silently reading somebody else's verdicts.
        """
        if self._model_key:
            return self._model_key
        identity = self.identity
        if identity is not None:
            return f"{identity.provider}/{identity.model}"
        return "unspecified"

    def assess(self, assertion: str, answer: str) -> Verdict:
        if self.cache is not None:
            cached = self.cache.get(self.version, self.model_key, assertion, answer)
            if cached is not None:
                return cached

        if self.client is None:
            available = sorted(self.cache.models()) if self.cache else []
            raise JudgementMissing(
                f"no recorded verdict for assertion {assertion!r} under rubric "
                f"version {self.version} from judge {self.model_key!r}, and no "
                "judge client is configured. "
                + (
                    f"Verdicts exist from: {available}. Select one with "
                    "--judge-model, or re-run with --live-judge to record new ones."
                    if available
                    else "Re-run with --live-judge to record it."
                )
            )

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
            verdict = Verdict(False, 0.0, "unparseable judge reply")
        else:
            verdict = Verdict(
                holds=bool(payload.get("holds", False)),
                confidence=float(payload.get("confidence", 0.0)),
                reason=str(payload.get("reason", "")),
            )

        if self.cache is not None:
            self.cache.put(
                self.version, self.model_key, assertion, answer, verdict, self.identity
            )
        return verdict


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
