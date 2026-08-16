"""A single self-contained HTML file showing one run, readable offline.

Milestone 6 item 4. **No network, no CDN, no external asset.** The CSS is
inline, there is no JavaScript beyond `<details>` (which is HTML, not script),
and no font, image or stylesheet is fetched. Opening the file from a USB stick
on a machine with no route to the internet shows the same thing it shows here —
which is the point, because the plants this targets do not let a browser out.

It reads the stored traces and the run's results. It does **not** re-run the
agent: `src/obs/replay.py` is the thing that replays, and keeping the two apart
means looking at a trace can never change it.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from src.obs import accounting

CSS = """
:root{--bg:#fbfaf8;--fg:#1c1a17;--muted:#6b6560;--line:#e2ded8;--card:#fff;
--ok:#1f7a4d;--err:#a52f24;--accent:#8a5a2b}
@media(prefers-color-scheme:dark){:root{--bg:#171614;--fg:#eae7e2;--muted:#9c958d;
--line:#302d29;--card:#1f1e1b;--ok:#5fbf8e;--err:#e08078;--accent:#d9a066}}
*{box-sizing:border-box}
body{margin:0;padding:2rem 1.25rem 4rem;background:var(--bg);color:var(--fg);
font:15px/1.6 Georgia,"Iowan Old Style",serif;max-width:70rem;margin-inline:auto}
h1{font-size:1.7rem;margin:0 0 .25rem}
h2{font-size:1.15rem;margin:0}
.sub{color:var(--muted);margin:0 0 1.5rem;font-size:.9rem}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(9rem,1fr));gap:.6rem;margin:1rem 0 2rem}
.tile{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:.7rem .85rem}
.tile b{display:block;font:600 1.25rem/1.2 system-ui,sans-serif}
.tile span{color:var(--muted);font-size:.78rem;text-transform:uppercase;letter-spacing:.04em}
details{background:var(--card);border:1px solid var(--line);border-radius:6px;
margin:0 0 .6rem;padding:.7rem .9rem}
summary{cursor:pointer;font:600 1rem/1.4 system-ui,sans-serif;display:flex;
gap:.6rem;align-items:baseline;flex-wrap:wrap}
summary::marker{color:var(--muted)}
.tag{font:500 .72rem/1 system-ui,sans-serif;padding:.25rem .45rem;border-radius:4px;
border:1px solid var(--line);color:var(--muted);white-space:nowrap}
.pass{color:var(--ok);border-color:currentColor}
.fail{color:var(--err);border-color:currentColor}
.q{font-style:italic;color:var(--muted);margin:.6rem 0 1rem}
.step{border-left:3px solid var(--line);padding:.1rem 0 .1rem .9rem;margin:.9rem 0}
.step.err{border-left-color:var(--err)}
.step h3{font:600 .95rem/1.4 ui-monospace,SFMono-Regular,Menlo,monospace;margin:0 0 .3rem;color:var(--accent)}
pre{background:var(--bg);border:1px solid var(--line);border-radius:4px;
padding:.6rem .7rem;overflow-x:auto;font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;margin:.35rem 0}
.answer{background:var(--bg);border:1px solid var(--line);border-left:3px solid var(--accent);
border-radius:4px;padding:.8rem 1rem;white-space:pre-wrap}
table{border-collapse:collapse;width:100%;font-size:.86rem;margin:.5rem 0}
th,td{text-align:left;padding:.35rem .5rem;border-bottom:1px solid var(--line)}
th{font:600 .75rem/1.4 system-ui,sans-serif;text-transform:uppercase;
letter-spacing:.04em;color:var(--muted)}
td.n{text-align:right;font-variant-numeric:tabular-nums}
footer{color:var(--muted);font-size:.8rem;margin-top:2.5rem;border-top:1px solid var(--line);padding-top:1rem}
"""


def _e(value) -> str:
    return html.escape(str(value), quote=False)


def _pretty(raw: str, limit: int = 4000) -> str:
    try:
        text = json.dumps(json.loads(raw), indent=2)
    except (json.JSONDecodeError, TypeError):
        text = str(raw)
    if len(text) > limit:
        text = text[:limit] + f"\n… ({len(text) - limit} more characters)"
    return _e(text)


def render(
    run_id: str,
    traces: list[dict],
    results: dict | None = None,
    records: list[accounting.RunAccounting] | None = None,
) -> str:
    meta = (results or {}).get("metadata", {})
    by_key = {(r["scenario_id"], r["seed"]): r for r in (results or {}).get("results", [])}
    by_run = {(r.scenario_id, r.seed): r for r in (records or [])}

    total_cost = sum(t.get("estimated_cost_usd", 0) for t in traces)
    total_in = sum(t.get("tokens_in", 0) for t in traces)
    total_out = sum(t.get("tokens_out", 0) for t in traces)
    total_calls = sum(len(t.get("tool_calls", [])) for t in traces)
    errors = sum(
        1 for t in traces for c in t.get("tool_calls", []) if c.get("status") == "error"
    )
    passed = sum(1 for r in by_key.values() if r.get("passed"))

    out: list[str] = []
    a = out.append
    a("<!-- Self-contained: no network request is made to display this file. -->")
    a(f"<title>Trace — {_e(run_id)}</title>")
    a(f"<style>{CSS}</style>")
    a(f"<h1>Run trace</h1>")
    agent = meta.get("model") or {}
    judge = meta.get("judge_model") or {}
    a(f'<p class="sub"><code>{_e(run_id)}</code> · '
      f'{len(traces)} run(s) · mode <code>{_e(meta.get("mode", "?"))}</code> · '
      f'git <code>{_e(meta.get("git_sha", "?"))}</code><br>'
      f'agent <code>{_e(agent.get("model", "not recorded"))}</code>'
      + (f' · judge <code>{_e(judge.get("model"))}</code>' if judge else "")
      + "</p>")

    a('<div class="grid">')
    for label, value in (
        ("scenario-seeds", len(traces)),
        ("passed", f"{passed}/{len(by_key)}" if by_key else "—"),
        ("tool calls", total_calls),
        ("tool errors", errors),
        ("tokens in", f"{total_in:,}"),
        ("tokens out", f"{total_out:,}"),
        ("cost (USD)", f"${total_cost:.4f}"),
    ):
        a(f'<div class="tile"><b>{_e(value)}</b><span>{_e(label)}</span></div>')
    a("</div>")

    for trace in sorted(traces, key=lambda t: (t["scenario_id"], t["seed"])):
        key = (trace["scenario_id"], trace["seed"])
        result = by_key.get(key)
        record = by_run.get(key)
        calls = trace.get("tool_calls", [])
        verdict = ""
        if result is not None:
            good = result.get("passed")
            verdict = (f'<span class="tag {"pass" if good else "fail"}">'
                       f'{"passed" if good else "failed"}</span>')
        a("<details>")
        a("<summary>"
          f'<h2>{_e(trace["scenario_id"])}</h2>'
          f'<span class="tag">seed {_e(trace["seed"])}</span>'
          f'{verdict}'
          f'<span class="tag">{len(calls)} tool call(s)</span>'
          f'<span class="tag">{_e(trace.get("iterations"))}'
          f'{"/" + str(record.max_iterations) if record and record.max_iterations else ""}'
          " iteration(s)</span>"
          f'<span class="tag">${trace.get("estimated_cost_usd", 0):.5f}</span>'
          "</summary>")

        if result is not None and result.get("assertions"):
            failed = [x["assertion"] for x in result["assertions"] if not x["satisfied"]]
            if failed:
                a(f'<p class="sub">failing: <code>{_e(", ".join(failed))}</code></p>')

        a("<h3>Question</h3>")
        a(f'<p class="q">{_e(trace.get("question", "(not stored in the trace)"))}</p>')

        for index, call in enumerate(calls, start=1):
            bad = call.get("status") == "error"
            a(f'<div class="step{" err" if bad else ""}">')
            a(f'<h3>{index}. {_e(call["tool"])}'
              f'<span class="tag {"fail" if bad else "pass"}">{_e(call["status"])}'
              + (f' · {_e(call.get("error_code"))}' if call.get("error_code") else "")
              + f'</span> <span class="tag">{call.get("duration_ms", 0):.1f} ms</span></h3>')
            a(f"<pre>{_pretty(json.dumps(call.get('arguments', {})), 1200)}</pre>")
            a(f"<pre>{_pretty(call.get('result_json', ''))}</pre>")
            a("</div>")

        a("<h3>Final answer</h3>")
        a(f'<div class="answer">{_e(trace.get("answer", ""))}</div>')

        a("<h3>Accounting</h3>")
        a("<table><tr><th>Measure</th><th>Value</th></tr>")
        rows = [
            ("tokens in", f'{trace.get("tokens_in", 0):,}'),
            ("tokens out", f'{trace.get("tokens_out", 0):,}'),
            ("cache read", f'{trace.get("cache_read", 0):,}'),
            ("cache write", f'{trace.get("cache_write", 0):,}'),
            ("estimated cost", f'${trace.get("estimated_cost_usd", 0):.5f}'),
            ("wall clock", f'{trace.get("wall_clock_ms", 0):.1f} ms'),
            ("iterations used", trace.get("iterations")),
            ("hit iteration limit", trace.get("hit_iteration_limit")),
            ("messages dropped", trace.get("messages_dropped")),
        ]
        if record is not None:
            rows += [
                ("model time", f"{record.model_ms:.1f} ms"),
                ("tool time", f"{record.tool_ms:.1f} ms"),
                ("loop overhead", f"{record.overhead_ms:.1f} ms"),
                ("iterations against ceiling", record.iteration_headroom),
            ]
        for label, value in rows:
            a(f'<tr><td>{_e(label)}</td><td class="n">{_e(value)}</td></tr>')
        a("</table>")
        a("</details>")

    a("<footer>Generated by <code>python -m src.obs.viewer</code> from stored "
      "traces. No model was called to produce this page, and the page makes no "
      "network request to display itself.</footer>")
    return "\n".join(out) + "\n"


def load(run_id: str, results_dir: Path) -> tuple[list[dict], dict | None]:
    traces_path = results_dir / f"{run_id}.traces.json"
    if not traces_path.exists():
        raise SystemExit(f"{traces_path} not found")
    traces = json.loads(traces_path.read_text(encoding="utf-8"))
    results_path = results_dir / f"{run_id}.json"
    results = (
        json.loads(results_path.read_text(encoding="utf-8"))
        if results_path.exists()
        else None
    )
    return traces, results


def attach_questions(traces: list[dict], scenarios_path: Path) -> list[dict]:
    """The trace stores the answer but not the question; the scenario has it."""
    try:
        from evals.runner import load_scenarios

        questions = {s.id: s.question for s in load_scenarios(scenarios_path)}
    except Exception:  # noqa: BLE001 - the viewer must render without the harness
        return traces
    for trace in traces:
        trace.setdefault("question", questions.get(trace["scenario_id"], ""))
    return traces


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--results", type=Path, default=Path("evals/results"))
    parser.add_argument("--scenarios", type=Path, default=Path("evals/scenarios.yaml"))
    parser.add_argument("--accounting", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    traces, results = load(args.run_id, args.results)
    traces = attach_questions(traces, args.scenarios)
    records = accounting.read(args.accounting) if args.accounting else None
    out = args.out or Path(f"evals/results/{args.run_id}.trace.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(args.run_id, traces, results, records), encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB, {len(traces)} run(s))")


if __name__ == "__main__":
    main()
