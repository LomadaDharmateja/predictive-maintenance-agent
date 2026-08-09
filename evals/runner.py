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
from src.agent.contracts import ErrorCode, ToolError
from src.agent.loop import Agent, LLMResponse, LoopConfig, ToolCall

HARNESS_VERSION = "1.0.0"

EVALS = Path("evals")
SCENARIOS = EVALS / "scenarios.yaml"
TRANSCRIPTS = EVALS / "transcripts"
RESULTS_DIR = EVALS / "results"

DEFAULT_SEEDS = (1, 2, 3)

#: Rough per-token prices in USD, used only for the cost column. Stated as an
#: assumption: they are provider list prices, not a bill anyone has paid.
COST_PER_1K_INPUT = 0.003
COST_PER_1K_OUTPUT = 0.015


class TranscriptMissing(RuntimeError):
    """No recording exists for this scenario and seed."""


class RecordedClient:
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
    return [Scenario(**entry) for entry in raw]


def load_transcript(scenario_id: str, seed: int, directory: Path = TRANSCRIPTS) -> list[dict]:
    path = directory / f"{scenario_id}.seed{seed}.json"
    if not path.exists():
        raise TranscriptMissing(
            f"{path} not found. Record it with `make eval-record` (--live), or "
            "check the scenario id."
        )
    return json.loads(path.read_text(encoding="utf-8"))["turns"]


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
) -> ScenarioTrace:
    from src.agent import tools as tools_module

    client = RecordedClient(load_transcript(scenario.id, seed, transcripts))
    agent = Agent(client, database=database, config=LoopConfig())

    original = tools_module.dispatch
    tools_module.dispatch = _inject(scenario, database)
    started = time.perf_counter()
    try:
        outcome = agent.run(scenario.question)
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

    cost = (
        client.tokens_in / 1000 * COST_PER_1K_INPUT
        + client.tokens_out / 1000 * COST_PER_1K_OUTPUT
    )
    return ScenarioTrace(
        scenario_id=scenario.id,
        seed=seed,
        answer=outcome.answer,
        tool_calls=calls,
        iterations=outcome.iterations,
        hit_iteration_limit=outcome.hit_iteration_limit,
        messages_dropped=outcome.messages_dropped,
        tokens_in=client.tokens_in,
        tokens_out=client.tokens_out,
        wall_clock_ms=round(elapsed, 2),
        estimated_cost_usd=round(cost, 6),
    )


def run_suite(
    scenarios: list[Scenario],
    database: Path,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    transcripts: Path = TRANSCRIPTS,
    judge=None,
    mode: str = "recorded",
) -> tuple[RunResults, list[ScenarioTrace]]:
    results, forbidden, hallucinations, traces = [], [], [], []

    for scenario in scenarios:
        for seed in seeds:
            trace = run_scenario(scenario, seed, database, transcripts)
            traces.append(trace)
            result, violations, fabricated = score_scenario(scenario, trace, judge)
            results.append(result)
            forbidden.extend(violations)
            hallucinations.extend(fabricated)

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=Path, default=SCENARIOS)
    parser.add_argument("--transcripts", type=Path, default=TRANSCRIPTS)
    parser.add_argument("--database", type=Path, default=Path("data/pdm.db"))
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--live",
        action="store_true",
        help="re-record against a live provider instead of replaying",
    )
    args = parser.parse_args()

    if args.live:
        raise SystemExit(
            "--live is the re-recording path and needs a configured provider. "
            "It is deliberately not wired to one here: Milestone 5's acceptance "
            "requires the suite to run offline, and a live default would make "
            "every run cost money and drift. Configure src.agent.loop.ModelConfig "
            "and implement a recording client before using it."
        )

    scenarios = load_scenarios(args.scenarios)
    print(f"{len(scenarios)} scenario(s), seeds {DEFAULT_SEEDS}")
    run, traces = run_suite(scenarios, args.database, transcripts=args.transcripts)

    results_path = write_results(run, args.results)
    traces_path = write_traces(traces, run.metadata.run_id, args.results)

    passed = sum(r.passed for r in run.results)
    print(f"  {passed}/{len(run.results)} scenario-seed runs passed")
    print(f"  {len(run.forbidden_calls)} forbidden tool call(s)")
    print(f"  {len(run.hallucinations)} hallucinated figure(s)")
    print(f"wrote {results_path}")
    print(f"wrote {traces_path}")


if __name__ == "__main__":
    main()
