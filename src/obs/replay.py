"""Deterministic replay of a recorded run, with a divergence check.

Milestone 6 item 3: given a run id, replay it from stored traces without
calling a model, reusing the evaluation harness's `ReplayClient` rather than
building a second path. There is exactly one replayer in this project and this
module drives it.

**Replay is a determinism check, not a rendering step.** The point is not to
produce something for the viewer -- the viewer can read the stored traces
directly. The point is that replaying a run should reproduce it exactly, and
saying so out loud is only worth anything if the claim is tested. A replay that
diverges means one of:

- the agent loop changed since the run was recorded,
- a tool's output changed because the database moved underneath it,
- the transcript no longer matches its scenario,

and all three are things you want to hear about loudly rather than discover
when a report disagrees with itself. Divergence is reported per scenario, with
the first differing field named.
"""

from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from src.obs import accounting, tracing


class Divergence(BaseModel):
    """One scenario whose replay did not reproduce the stored run."""

    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    seed: int
    field: str
    stored: str
    replayed: str
    diff: str = ""


class ReplayReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    replayed: int
    identical: int
    divergences: list[Divergence] = []

    @property
    def deterministic(self) -> bool:
        return not self.divergences


def _short(value: str, limit: int = 160) -> str:
    value = str(value)
    return value if len(value) <= limit else value[:limit] + "…"


def replay_run(
    run_id: str,
    results_dir: Path = Path("evals/results"),
    scenarios_path: Path = Path("evals/scenarios.yaml"),
    database: Path = Path("data/pdm.db"),
    transcripts: Path | None = None,
    exporter: tracing.SpanExporter | None = None,
) -> tuple[ReplayReport, list[accounting.RunAccounting], list[dict]]:
    """Replay every scenario-seed in `run_id`. Makes no network call.

    Returns the divergence report, the per-run accounting rebuilt from the
    spans this replay emitted, and those spans.
    """
    # Imported here rather than at module scope: `evals` depends on `src`, and
    # importing it the other way round at import time would make the
    # observability layer require the eval harness to exist.
    from evals.runner import TRANSCRIPTS, load_scenarios, run_scenario

    traces_path = results_dir / f"{run_id}.traces.json"
    if not traces_path.exists():
        raise SystemExit(
            f"{traces_path} not found. `python -m evals.runner` writes it "
            "alongside the results file."
        )
    stored = {
        (t["scenario_id"], t["seed"]): t
        for t in json.loads(traces_path.read_text(encoding="utf-8"))
    }
    scenarios = {s.id: s for s in load_scenarios(scenarios_path)}

    memory = exporter or tracing.MemorySpanExporter()
    tracing.configure(memory, reset=True)

    divergences: list[Divergence] = []
    identical = 0
    for (scenario_id, seed), trace in sorted(stored.items()):
        scenario = scenarios.get(scenario_id)
        if scenario is None:
            divergences.append(
                Divergence(
                    scenario_id=scenario_id, seed=seed, field="scenario",
                    stored="present in the run", replayed="no longer in scenarios.yaml",
                )
            )
            continue
        fresh, _ = run_scenario(
            scenario, seed, database, transcripts or TRANSCRIPTS
        )
        mismatch = _compare(trace, fresh)
        if mismatch is None:
            identical += 1
        else:
            divergences.append(mismatch)

    spans = getattr(memory, "spans", [])
    identity = None
    records = accounting.from_spans(spans, run_id, identity)
    return (
        ReplayReport(
            run_id=run_id,
            replayed=len(stored),
            identical=identical,
            divergences=divergences,
        ),
        records,
        spans,
    )


def _compare(stored: dict, fresh) -> Divergence | None:
    """First differing field, or None. Timings are excluded on purpose.

    Wall-clock and cost-from-wall-clock differ between any two runs of the same
    code and are not evidence of anything. What must reproduce is the *content*:
    the answer, the tool calls made, and the arguments they were made with.
    """
    checks: list[tuple[str, object, object]] = [
        ("answer", stored["answer"], fresh.answer),
        ("iterations", stored["iterations"], fresh.iterations),
        ("hit_iteration_limit", stored["hit_iteration_limit"], fresh.hit_iteration_limit),
        ("tokens_in", stored["tokens_in"], fresh.tokens_in),
        ("tokens_out", stored["tokens_out"], fresh.tokens_out),
        (
            "tool_calls",
            [(c["tool"], c["arguments"], c["status"]) for c in stored["tool_calls"]],
            [(c.tool, c.arguments, c.status) for c in fresh.tool_calls],
        ),
    ]
    for field, was, now in checks:
        if was == now:
            continue
        diff = ""
        if field == "answer":
            diff = "\n".join(
                list(
                    difflib.unified_diff(
                        str(was).splitlines(), str(now).splitlines(),
                        fromfile="stored", tofile="replayed", lineterm="", n=1,
                    )
                )[:40]
            )
        return Divergence(
            scenario_id=stored["scenario_id"], seed=stored["seed"], field=field,
            stored=_short(was), replayed=_short(now), diff=diff,
        )
    return None


def render(report: ReplayReport, records: list[accounting.RunAccounting]) -> str:
    out: list[str] = []
    add = out.append
    add(f"run          {report.run_id}")
    add(f"replayed     {report.replayed}")
    add(f"identical    {report.identical}")
    add(f"divergent    {len(report.divergences)}")
    add("")
    if report.deterministic:
        add("DETERMINISTIC — every scenario reproduced its stored answer, tool")
        add("calls and token counts exactly, with no model called.")
    else:
        add("DIVERGED — the replay did not reproduce the stored run. Either the")
        add("agent loop changed, a tool's data moved underneath it, or a")
        add("transcript no longer matches its scenario.")
        add("")
        for d in report.divergences:
            add(f"  {d.scenario_id} seed {d.seed}: `{d.field}` differs")
            add(f"    stored:   {d.stored}")
            add(f"    replayed: {d.replayed}")
            if d.diff:
                for line in d.diff.splitlines():
                    add(f"    | {line}")
            add("")
    if records:
        t = accounting.totals(records)
        add("")
        add(f"accounting   {t['runs']} run(s), {t['tokens_in']} in / "
            f"{t['tokens_out']} out, {t['tool_calls']} tool call(s), "
            f"{t['tool_errors']} error(s)")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", help="a run id from evals/results/")
    parser.add_argument("--results", type=Path, default=Path("evals/results"))
    parser.add_argument("--database", type=Path, default=Path("data/pdm.db"))
    parser.add_argument(
        "--spans", type=Path, default=None, help="write the replay's spans here"
    )
    parser.add_argument(
        "--accounting", type=Path, default=None, help="write per-run accounting here"
    )
    args = parser.parse_args()

    report, records, spans = replay_run(args.run_id, args.results, database=args.database)
    print(render(report, records))

    if args.spans:
        args.spans.parent.mkdir(parents=True, exist_ok=True)
        args.spans.write_text(json.dumps(spans, indent=2), encoding="utf-8")
        print(f"\nwrote {args.spans}")
    if args.accounting:
        accounting.write(records, args.accounting)
        print(f"wrote {args.accounting}")

    raise SystemExit(0 if report.deterministic else 1)


if __name__ == "__main__":
    main()
