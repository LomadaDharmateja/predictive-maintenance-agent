"""The offline runner. Deterministic and free by default.

`docs/MILESTONE_5.md` section 4. The suite runs against a **recorded model
transcript**, so a full run costs nothing, needs no network, and produces the
same answer every time. `--live` re-records against a real provider; nothing else
about the harness changes, which is what makes the recorded mode trustworthy.

Three seeds per scenario, with variance reported. A single run of a stochastic
system is an anecdote. In recorded mode the three seeds replay three recorded
responses, so seed-to-seed variance measures what the model actually did on those
three occasions rather than pretending to resample it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from evals.metrics import score_scenario
from evals.schema import (
    Category,
    RunMetadata,
    RunResults,
    Scenario,
    ScenarioTrace,
    ToolCallTrace,
)
from evals.transcript import (
    TRANSCRIPTS,
    TranscriptInvalid,
    TranscriptMissing,
    ValidationWindowError,
    identity_of,
    load_validated,
    validate_scenario_window,
    window_guarded,
)
from src.agent.contracts import ErrorCode, ToolError
from src.agent.loop import Agent, LLMResponse, LoopConfig, ToolCall
from src.agent.providers import ModelIdentity

HARNESS_VERSION = "1.1.0"

EVALS = Path("evals")
SCENARIOS = EVALS / "scenarios.yaml"
RESULTS_DIR = EVALS / "results"

DEFAULT_SEEDS = (1, 2, 3)

#: Per-1K-token prices in USD, used only for the cost column, keyed by the
#: provider that produced the transcript. Stated as an assumption: these are
#: published list prices, not a bill anyone has paid.
#:
#: Ollama is zero because it is zero. Pricing a local run at hosted rates --
#: which this harness did until transcripts carried a provider -- reports a
#: cost that was never incurred, and the cost column exists to inform a
#: decision about what a real run would cost.
#: Keyed by model where the model is known, because a tier's prices differ by
#: 5x and a run priced at the wrong tier answers the wrong budgeting question.
#: Standard list prices; any introductory discount is deliberately not encoded,
#: so the figure is the one that survives the promotion ending.
PRICES_PER_1K = {
    "claude-opus-5": (0.005, 0.025),
    "claude-opus-4-8": (0.005, 0.025),
    "claude-sonnet-5": (0.003, 0.015),
    "claude-sonnet-4-6": (0.003, 0.015),
    "claude-haiku-4-5": (0.001, 0.005),
}
PROVIDER_PRICES_PER_1K = {
    "ollama": (0.0, 0.0),  # zero because it is zero
}
DEFAULT_PRICE = (0.003, 0.015)

#: What a cached input token costs relative to an uncached one.
CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_MULTIPLIER = 1.25


def price_for(identity) -> tuple[float, float]:
    if identity.provider in PROVIDER_PRICES_PER_1K:
        return PROVIDER_PRICES_PER_1K[identity.provider]
    for model, price in PRICES_PER_1K.items():
        if identity.model.startswith(model):
            return price
    return DEFAULT_PRICE


class ReplayClient:
    """Replays a recorded exchange. Makes no network call.

    A transcript is a list of turns, each either a set of tool calls or a final
    answer -- the same shape `LLMClient.complete` returns. If the loop asks for
    more turns than were recorded, that is a mismatch between the transcript and
    the current agent, and it raises rather than improvising: an improvised turn
    would silently turn a regression into a pass.
    """

    def __init__(self, turns: list[dict]) -> None:
        self.turns = turns
        self.index = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.cache_read = 0
        self.cache_write = 0

    def complete(self, messages, tools) -> LLMResponse:
        if self.index >= len(self.turns):
            raise TranscriptMissing(
                f"transcript has {len(self.turns)} turn(s); the loop asked for "
                f"{self.index + 1}. Re-record with --live."
            )
        turn = self.turns[self.index]
        self.index += 1

        # Token counts come from the recording, not from a live tokeniser, so
        # the cost column is reproducible.
        self.tokens_in += int(turn.get("tokens_in", 0))
        self.tokens_out += int(turn.get("tokens_out", 0))
        self.cache_read += int(turn.get("cache_read", 0))
        self.cache_write += int(turn.get("cache_write", 0))

        if turn.get("tool_calls"):
            return LLMResponse(
                tool_calls=tuple(
                    ToolCall(call["name"], call.get("arguments", {}))
                    for call in turn["tool_calls"]
                )
            )
        return LLMResponse(text=turn.get("text", ""))


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def load_scenarios(path: Path = SCENARIOS) -> list[Scenario]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Scenarios are written by the project owner; see "
            "docs/MILESTONE_5.md section 2."
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    scenarios = [Scenario(**entry) for entry in raw]
    # A scenario whose own prediction time sits in the test split would send
    # every transcript recorded from it into the test split too.
    for scenario in scenarios:
        validate_scenario_window(scenario)
    return scenarios


def _inject(scenario: Scenario, database: Path):
    """Wrap dispatch so a `tool_failure` scenario gets its forced failure.

    Injected rather than waited for, so the check is deterministic.
    """
    from src.agent import tools as tools_module

    original = tools_module.dispatch
    failure = scenario.injected_failure
    if failure is None:
        return original

    def patched(name: str, arguments: dict, db_path=database):
        if name == failure.tool:
            return ToolError(
                code=ErrorCode(failure.code),
                message=failure.message,
                tool=name,
                retryable=False,
            )
        return original(name, arguments, db_path)

    return patched


def run_scenario(
    scenario: Scenario,
    seed: int,
    database: Path,
    transcripts: Path = TRANSCRIPTS,
) -> tuple[ScenarioTrace, ModelIdentity]:
    from src.agent import tools as tools_module

    # Validated before it is replayed: a transcript whose tool calls do not
    # match its scenario answers a different question, and scoring it would
    # score that other question. See `evals/transcript.py`.
    transcript = load_validated(scenario, seed, transcripts)
    client = ReplayClient(transcript["turns"])
    agent = Agent(client, database=database, config=LoopConfig())

    original = tools_module.dispatch
    tools_module.dispatch = window_guarded(_inject(scenario, database))
    started = time.perf_counter()
    try:
        outcome = agent.run(scenario.question, as_of=scenario.as_of)
    finally:
        tools_module.dispatch = original
    elapsed = (time.perf_counter() - started) * 1000

    calls = [
        ToolCallTrace(
            tool=entry.name or "",
            arguments=entry.arguments or {},
            status="ok" if entry.kind == "tool_result" else "error",
            error_code=entry.error_code,
            truncated=entry.truncated,
            duration_ms=entry.duration_ms or 0.0,
            result_json=entry.detail or "",
        )
        for entry in outcome.log.entries
        if entry.kind in {"tool_result", "tool_error"}
    ]

    identity = identity_of(transcript)
    price_in, price_out = price_for(identity)
    # Cached tokens are not `tokens_in` -- the provider reports them in their
    # own fields and bills them at ~0.1x for a read and ~1.25x for a write.
    # Charging them at the input rate would report a saving that was made as
    # if it had not been.
    cost = (
        client.tokens_in / 1000 * price_in
        + client.cache_read / 1000 * price_in * CACHE_READ_MULTIPLIER
        + client.cache_write / 1000 * price_in * CACHE_WRITE_MULTIPLIER
        + client.tokens_out / 1000 * price_out
    )
    trace = ScenarioTrace(
        scenario_id=scenario.id,
        seed=seed,
        answer=outcome.answer,
        tool_calls=calls,
        iterations=outcome.iterations,
        hit_iteration_limit=outcome.hit_iteration_limit,
        messages_dropped=outcome.messages_dropped,
        tokens_in=client.tokens_in,
        tokens_out=client.tokens_out,
        cache_read=client.cache_read,
        cache_write=client.cache_write,
        wall_clock_ms=round(elapsed, 2),
        estimated_cost_usd=round(cost, 6),
    )
    return trace, identity


def run_suite(
    scenarios: list[Scenario],
    database: Path,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    transcripts: Path = TRANSCRIPTS,
    judge=None,
    mode: str = "recorded",
) -> tuple[RunResults, list[ScenarioTrace]]:
    results, forbidden, hallucinations, traces = [], [], [], []
    identities: dict[str, ModelIdentity] = {}

    for scenario in scenarios:
        for seed in seeds:
            trace, identity = run_scenario(scenario, seed, database, transcripts)
            identities[identity.label()] = identity
            traces.append(trace)
            result, violations, fabricated = score_scenario(scenario, trace, judge)
            results.append(result)
            forbidden.extend(violations)
            hallucinations.extend(fabricated)

    if len(identities) > 1:
        # Blending two models inside one results file makes every per-category
        # figure a weighted average of two different systems, and nothing in
        # the report would say so.
        raise TranscriptInvalid(
            "transcripts in this run come from more than one model: "
            f"{sorted(identities)}. A run is one model; re-record the odd ones out."
        )

    sha = git_sha()
    stamp = datetime.now(timezone.utc)
    metadata = RunMetadata(
        run_id=f"{stamp.strftime('%Y%m%dT%H%M%SZ')}-{sha}",
        git_sha=sha,
        utc=stamp,
        mode=mode,
        seeds=list(seeds),
        n_scenarios=len(scenarios),
        harness_version=HARNESS_VERSION,
        model=next(iter(identities.values())) if identities else None,
        judge_model=getattr(judge, "identity", None) if judge else None,
    )
    return (
        RunResults(
            metadata=metadata,
            results=results,
            forbidden_calls=forbidden,
            hallucinations=hallucinations,
        ),
        traces,
    )


def write_results(run: RunResults, directory: Path = RESULTS_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{run.metadata.run_id}.json"
    path.write_text(run.model_dump_json(indent=2), encoding="utf-8")
    return path


def write_traces(traces: list[ScenarioTrace], run_id: str, directory: Path = RESULTS_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{run_id}.traces.json"
    path.write_text(
        json.dumps([json.loads(t.model_dump_json()) for t in traces], indent=2),
        encoding="utf-8",
    )
    return path


def build_judge(args):
    """Assemble the judge from the CLI flags. Returns (judge, cache).

    Offline by default: verdicts are replayed from
    `evals/judgements/<rubric version>.json` and a miss raises. `--live-judge`
    supplies a client so a miss is recorded instead.
    """
    if args.no_judge:
        return None, None

    from dataclasses import replace as _replace

    from evals.judge import (
        DEFAULT_PROMPT,
        JUDGEMENTS_DIR,
        Judge,
        ProviderJudgeClient,
        VerdictCache,
        prompt_version,
    )

    version = prompt_version(DEFAULT_PROMPT)
    cache = VerdictCache(JUDGEMENTS_DIR / f"{version}.json")

    # Which judge model's verdicts to read. Offline this comes from
    # --judge-model, because the cache holds verdicts per judge and replaying
    # the wrong one would attribute Sonnet's grades to Haiku.
    provider = args.judge_provider or args.provider
    model_key = f"{provider}/{args.judge_model}" if args.judge_model else None

    client = None
    if args.live_judge:
        from src.agent.loop import ModelConfig
        from src.agent.providers import build_client as build_provider

        provider = args.judge_provider or args.provider
        config = ModelConfig(provider=provider)
        if args.judge_model:
            config = _replace(
                config,
                **(
                    {"model": args.judge_model}
                    if provider == "anthropic"
                    else {"local_model": args.judge_model}
                ),
            )
        client = ProviderJudgeClient(build_provider(config))
        print(f"judge: live against {client.identity.label()}")
    elif model_key:
        print(f"judge: replaying recorded verdicts from {model_key}")

    return Judge(client=client, cache=cache, model_key=model_key), cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=Path, default=SCENARIOS)
    parser.add_argument("--transcripts", type=Path, default=TRANSCRIPTS)
    parser.add_argument("--database", type=Path, default=Path("data/pdm.db"))
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--live",
        action="store_true",
        help="re-record against a live provider first, then score the recording",
    )
    parser.add_argument("--provider", choices=("ollama", "anthropic"), default="ollama")
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="with --live, re-record even scenarios that already have a transcript",
    )
    parser.add_argument("--throttle", type=float, default=0.0)
    parser.add_argument(
        "--live-judge",
        action="store_true",
        help="grade uncached free-text assertions against a live model and record "
        "the verdicts; without it, an ungraded assertion raises",
    )
    parser.add_argument(
        "--judge-provider", choices=("ollama", "anthropic"), default=None,
        help="defaults to --provider",
    )
    parser.add_argument("--judge-model", default=None)
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="score without a judge; every free-text assertion records unsatisfied",
    )
    args = parser.parse_args()

    scenarios = load_scenarios(args.scenarios)

    if args.live:
        # --live re-records, then scores the recording. Scoring never reads a
        # live model: the numbers in a report always come from a transcript on
        # disk, whether it was written a minute ago or a month ago.
        from dataclasses import replace as _replace

        from evals.record import record_suite
        from src.agent.loop import ModelConfig

        config = ModelConfig(provider=args.provider)
        if args.model:
            config = _replace(
                config,
                **(
                    {"model": args.model}
                    if args.provider == "anthropic"
                    else {"local_model": args.model}
                ),
            )
        print(f"--live: recording against {args.provider} before scoring")
        summary = record_suite(
            scenarios,
            DEFAULT_SEEDS,
            config,
            args.database,
            args.transcripts,
            resume=not args.force,
            throttle=args.throttle,
        )
        print(
            f"  {len(summary['recorded'])} recorded, {len(summary['skipped'])} skipped, "
            f"{len(summary['failed'])} failed"
        )
        if summary["failed"]:
            raise SystemExit(
                f"{len(summary['failed'])} scenario(s) failed to record; the suite "
                "is not scored on a partial recording. Fix or re-run them first:\n  "
                + "\n  ".join(f"{e['key']}: {e['error']}" for e in summary["failed"])
            )

    judge, cache = build_judge(args)

    print(f"{len(scenarios)} scenario(s), seeds {DEFAULT_SEEDS}")
    run, traces = run_suite(
        scenarios,
        args.database,
        transcripts=args.transcripts,
        judge=judge,
        mode="live" if args.live else "recorded",
    )

    if cache is not None:
        path = cache.save(getattr(judge, "identity", None))
        print(f"  judge: {cache.hits} cached verdict(s), {cache.misses} new -> {path}")

    results_path = write_results(run, args.results)
    traces_path = write_traces(traces, run.metadata.run_id, args.results)

    passed = sum(r.passed for r in run.results)
    if run.metadata.model:
        print(f"  model: {run.metadata.model.label()}")
    print(f"  {passed}/{len(run.results)} scenario-seed runs passed")
    print(f"  {len(run.forbidden_calls)} forbidden tool call(s)")
    print(f"  {len(run.hallucinations)} hallucinated figure(s)")
    print(f"wrote {results_path}")
    print(f"wrote {traces_path}")


if __name__ == "__main__":
    main()
