"""Recording. The one path in this project that touches a network.

`docs/MILESTONE_5.md` section 4: the suite "runs offline against a **recorded
model transcript** by default, so the suite is deterministic and free. A
`--live` flag re-records." This module is that flag's implementation.

`RecordingClient` wraps a provider adapter and writes down every request and
every response. Nothing else about the agent changes -- the loop, the tools and
the database posture are identical to a replayed run, which is what makes the
replayed numbers mean anything.

Three properties the sweep has to have, because a 123-run recording is long
enough that all three will be exercised:

**Resumable.** Each transcript is written the moment its scenario finishes, and
a rerun skips any scenario that already has a valid one on disk. A crash at run
100 costs one run, not a hundred. `--force` re-records regardless.

**Throttled.** Exponential backoff on rate limits, in
`src/agent/providers.py`, plus a configurable gap between scenarios so a free
tier's per-minute limit is respected rather than discovered.

**Bounded blast radius.** One scenario that fails -- a refusal, a malformed
tool call, an out-of-window prediction time -- is recorded as a failure and the
sweep moves on. It does not take the other 122 with it.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from evals.schema import Scenario
from evals.transcript import (
    TRANSCRIPTS,
    is_recorded,
    transcript_path,
    validate_scenario_window,
    validate_transcript,
    window_guarded,
    write_transcript,
)
from src.agent.loop import Agent, LLMResponse, LoopConfig, ModelConfig
from src.agent.providers import ModelIdentity, build_client

RECORDER_VERSION = "1.0.0"

#: Seconds between scenarios. Zero locally; raise it for a hosted free tier
#: whose limit is per-minute rather than per-request.
DEFAULT_THROTTLE_SECONDS = 0.0


class RecordingClient:
    """Wraps an adapter and writes down everything that crossed the wire.

    Satisfies `LLMClient` exactly as `ReplayClient` does, so the agent under
    recording is the agent under replay.
    """

    def __init__(self, inner, scenario_id: str, seed: int) -> None:
        self.inner = inner
        self.scenario_id = scenario_id
        self.seed = seed
        self.turns: list[dict] = []
        self.exchanges: list[dict] = []
        self.tokens_in = 0
        self.tokens_out = 0

    def complete(self, messages, tools) -> LLMResponse:
        started = time.perf_counter()
        response = self.inner.complete(messages, tools)
        elapsed = round((time.perf_counter() - started) * 1000, 2)

        exchange = dict(self.inner.last_exchange)
        tokens_in = int(exchange.get("tokens_in", 0))
        tokens_out = int(exchange.get("tokens_out", 0))
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out

        # The replayable shape: exactly what `ReplayClient` needs to hand the
        # loop back, and nothing else.
        turn: dict = {
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            # Billed at ~0.1x (read) and ~1.25x (write), not at the input rate.
            "cache_read": int(exchange.get("cache_read", 0)),
            "cache_write": int(exchange.get("cache_write", 0)),
        }
        if response.wants_tools:
            turn["tool_calls"] = [
                {"name": call.name, "arguments": call.arguments}
                for call in response.tool_calls
            ]
        else:
            turn["text"] = response.text or ""
        self.turns.append(turn)

        # The audit shape: every request and every response, in full. Never
        # read by the replayer -- kept so a disputed score can be traced back
        # to what the model was actually shown.
        exchange["latency_ms"] = elapsed
        exchange["messages_sent"] = messages
        self.exchanges.append(exchange)
        return response

    def transcript(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "model": self.inner.identity.model_dump(),
            "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "recorder_version": RECORDER_VERSION,
            "note": (
                "Recorded against a live provider. Replayed offline thereafter; "
                "no network call is made on replay."
            ),
            "turns": self.turns,
            "exchanges": self.exchanges,
        }


def default_client_factory(config: ModelConfig, scenario: Scenario, seed: int):
    return build_client(config, seed=seed)


def record_scenario(
    scenario: Scenario,
    seed: int,
    config: ModelConfig,
    database: Path,
    directory: Path = TRANSCRIPTS,
    client_factory=default_client_factory,
) -> dict:
    """Record one scenario at one seed and write it. Raises on any failure.

    `client_factory` is injected rather than constructed so the recording path
    itself can be tested without a provider -- the tests drive it with a
    scripted adapter, which is the only way to prove the sweep's resume,
    validation and failure handling work without spending money to find out.
    """
    from src.agent import tools as tools_module

    validate_scenario_window(scenario)

    client = RecordingClient(client_factory(config, scenario, seed), scenario.id, seed)
    agent = Agent(client, database=database, config=LoopConfig())

    original = tools_module.dispatch
    tools_module.dispatch = window_guarded(_inject(scenario, database, original))
    try:
        agent.run(scenario.question, as_of=scenario.as_of)
    finally:
        tools_module.dispatch = original

    transcript = client.transcript()
    validate_transcript(scenario, transcript)
    write_transcript(transcript, transcript_path(scenario.id, seed, directory))
    return transcript


def _inject(scenario: Scenario, database: Path, original):
    """Force a `tool_failure` scenario's error, exactly as replay does.

    The injection has to happen while recording too, or the recorded answer is
    the model's response to a tool that worked -- and the scenario is asking
    what it says when one doesn't.
    """
    from src.agent.contracts import ErrorCode, ToolError

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


def record_suite(
    scenarios: list[Scenario],
    seeds: tuple[int, ...],
    config: ModelConfig,
    database: Path,
    directory: Path = TRANSCRIPTS,
    resume: bool = True,
    throttle: float = DEFAULT_THROTTLE_SECONDS,
    sleep=time.sleep,
    client_factory=default_client_factory,
) -> dict:
    """The resumable sweep. Returns a summary; never raises for one bad run."""
    recorded, skipped, failed = [], [], []
    identities: set[str] = set()

    for scenario in scenarios:
        for seed in seeds:
            key = f"{scenario.id}.seed{seed}"
            if resume and is_recorded(scenario, seed, directory):
                skipped.append(key)
                print(f"  skip     {key} (already recorded)")
                continue
            try:
                transcript = record_scenario(
                    scenario, seed, config, database, directory, client_factory
                )
            except Exception as exc:  # noqa: BLE001
                # Deliberately broad, and deliberately not silent. CLAUDE.md
                # forbids a bare `except Exception` that *returns a value*;
                # this one returns nothing and hides nothing. The failure is
                # named, counted, printed, and carried in the summary --
                # `--live` refuses to score a partial recording because of it.
                # The alternative is one bad scenario costing the other 122.
                failed.append({"key": key, "error": f"{type(exc).__name__}: {exc}"})
                print(f"  FAILED   {key}: {type(exc).__name__}: {exc}")
                continue
            identities.add(ModelIdentity(**transcript["model"]).label())
            recorded.append(key)
            print(
                f"  recorded {key}  "
                f"{len(transcript['turns'])} turn(s), "
                f"{sum(t['tokens_in'] for t in transcript['turns'])} in / "
                f"{sum(t['tokens_out'] for t in transcript['turns'])} out"
            )
            if throttle:
                sleep(throttle)

    return {
        "recorded": recorded,
        "skipped": skipped,
        "failed": failed,
        "models": sorted(identities),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=Path, default=Path("evals/scenarios.yaml"))
    parser.add_argument("--transcripts", type=Path, default=TRANSCRIPTS)
    parser.add_argument("--database", type=Path, default=Path("data/pdm.db"))
    parser.add_argument("--provider", choices=("ollama", "anthropic"), default="ollama")
    parser.add_argument("--model", default=None, help="override the provider's model")
    parser.add_argument("--base-url", default=None, help="Ollama base URL")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--only", nargs="+", default=None, help="scenario id(s)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-record scenarios that already have a valid transcript",
    )
    parser.add_argument("--throttle", type=float, default=DEFAULT_THROTTLE_SECONDS)
    args = parser.parse_args()

    from evals.runner import load_scenarios

    scenarios = load_scenarios(args.scenarios)
    if args.only:
        wanted = set(args.only)
        scenarios = [s for s in scenarios if s.id in wanted]
        missing = wanted - {s.id for s in scenarios}
        if missing:
            raise SystemExit(f"no such scenario(s): {sorted(missing)}")

    config = ModelConfig(provider=args.provider)
    if args.model:
        config = replace(
            config,
            **({"model": args.model} if args.provider == "anthropic" else {"local_model": args.model}),
        )
    if args.base_url:
        config = replace(config, local_base_url=args.base_url)

    print(
        f"recording {len(scenarios)} scenario(s) x {len(args.seeds)} seed(s) "
        f"against {args.provider}"
    )
    summary = record_suite(
        scenarios,
        tuple(args.seeds),
        config,
        args.database,
        args.transcripts,
        resume=not args.force,
        throttle=args.throttle,
    )
    print(
        f"\n{len(summary['recorded'])} recorded, {len(summary['skipped'])} skipped, "
        f"{len(summary['failed'])} failed"
    )
    for entry in summary["failed"]:
        print(f"  {entry['key']}: {entry['error']}")
    if summary["models"]:
        print(f"model(s): {', '.join(summary['models'])}")


if __name__ == "__main__":
    main()
