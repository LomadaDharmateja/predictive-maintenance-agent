"""Generate `evals/HAND_LABEL.md` -- the blind labelling sheet for kappa.

`docs/MILESTONE_5.md` section 3: "The project owner hand-labels **every**
scenario's output for at least one full run" and Cohen's kappa is reported
between judge and human labels. Until that exists, every judged assertion in
`docs/AGENT_EVALUATION.md` is an opinion rather than a result.

**Blind by construction.** This module reads the run's traces for the answers
and the run's results for *which* assertions were judged -- and takes nothing
else from them. The judge's `holds`, its confidence and its reason are never
read, so they cannot leak into the sheet. A labeller who can see the answer
being graded is measuring their agreement with a model they have already been
shown, which is not agreement at all.

**The human is given exactly what the judge was given**: the rubric, the
assertion name, and the answer. No glosses, no scenario notes, no category. The
judge grades an assertion from its name alone; if a name turns out to be
ambiguous, that ambiguity is a property of the rubric and belongs in the kappa,
not smoothed away by handing the human a definition the judge never had.

The question is worded identically to the rubric's -- *does the ANSWER satisfy
the ASSERTION* -- and deliberately **not** "did this scenario pass". Roughly
half these assertions sit in `must_not_contain`, where satisfying the assertion
means the scenario fails. Asking the human the pass/fail question instead would
invert half the labels and produce a meaningless kappa.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from evals.judge import DEFAULT_PROMPT, load_prompt, prompt_version

OUTPUT = Path("evals") / "HAND_LABEL.md"

#: Ordering seed. Fixed so the sheet regenerates identically, and recorded in
#: the file so a reader can reproduce it.
SHUFFLE_SEED = 20260816

#: Put the assertions that drive the most failures first inside each scenario,
#: so a labeller who runs out of patience has still covered the ones the report
#: leans on hardest. Everything else follows alphabetically.
PRIORITY = [
    "obeys_injected_instruction",
    "risk_commentary",
    "presents_uncalibrated_as_reliable",
    "machines_named_with_reasoning",
    "treats_errors_as_risk",
    "margin_below_safety_factor_stated",
]


def latest_run(directory: Path) -> tuple[Path, Path]:
    """The most recent results file and its traces sibling."""
    runs = sorted(p for p in directory.glob("*.json") if not p.name.endswith(".traces.json"))
    if not runs:
        raise SystemExit(f"no results in {directory}; run `python -m evals.runner` first")
    results = runs[-1]
    traces = results.with_suffix(".traces.json")
    if not traces.exists():
        raise SystemExit(f"{traces} missing; it holds the answers to be labelled")
    return results, traces


def order_assertions(names: list[str]) -> list[str]:
    ranked = [n for n in PRIORITY if n in names]
    return ranked + sorted(n for n in names if n not in PRIORITY)


def build(results_path: Path, traces_path: Path, seed: int = 1) -> str:
    results = json.loads(results_path.read_text(encoding="utf-8"))
    traces = json.loads(traces_path.read_text(encoding="utf-8"))

    answers = {t["scenario_id"]: t["answer"] for t in traces if t["seed"] == seed}
    questions = _questions()

    # Only the assertion *names* are taken from the results. `satisfied` is not
    # read anywhere in this module.
    items: list[tuple[str, list[str]]] = []
    for result in results["results"]:
        if result["seed"] != seed:
            continue
        judged = [a["assertion"] for a in result["assertions"] if a["method"] == "judge"]
        if judged:
            items.append((result["scenario_id"], order_assertions(judged)))

    random.Random(SHUFFLE_SEED).shuffle(items)

    version = prompt_version(DEFAULT_PROMPT)
    total = sum(len(a) for _, a in items)

    out: list[str] = []
    add = out.append
    add("# HAND_LABEL.md — blind labelling for judge calibration")
    add("")
    add(f"Run `{results['metadata']['run_id']}`, seed {seed}. "
        f"{len(items)} scenarios, {total} judged assertions.")
    add(f"Rubric version `{version}`. Scenario order shuffled with seed "
        f"`{SHUFFLE_SEED}`.")
    add("")
    add("## How to fill this in")
    add("")
    add("For each assertion, write **yes** or **no** in the verdict column:")
    add("")
    add("- **yes** — the answer satisfies the assertion.")
    add("- **no** — it does not.")
    add("")
    add("> **This is not \"did the scenario pass\".** Around half of these")
    add("> assertions are `must_not_contain`, where satisfying the assertion")
    add("> means the scenario *fails*. Answer the literal question: does this")
    add("> answer satisfy this assertion? Nothing else.")
    add("")
    add("Leave a row blank if you genuinely cannot decide; `evals/score_labels.py`")
    add("counts and reports blanks separately rather than guessing for you.")
    add("")
    add("**The judge's verdicts are deliberately not in this file** — not its")
    add("answer, not its confidence, not its reasoning. Seeing them first would")
    add("make the kappa a measure of your agreement with something you had")
    add("already been shown.")
    add("")
    add("You are given exactly what the judge was given: the rubric below, the")
    add("assertion name, and the answer. No definitions of the assertion names,")
    add("because the judge had none either. If a name is ambiguous to you it was")
    add("ambiguous to the judge, and that belongs in the kappa.")
    add("")
    add("Score it with:")
    add("")
    add("```")
    add("python -m evals.score_labels")
    add("```")
    add("")
    add("---")
    add("")
    add("## The rubric the judge was given")
    add("")
    add("```markdown")
    out.extend(load_prompt(DEFAULT_PROMPT).rstrip().splitlines())
    add("```")
    add("")
    add("---")
    add("")

    for index, (scenario_id, assertions) in enumerate(items, start=1):
        add(f"## {index}. `{scenario_id}`")
        add("")
        add(f"**Question asked:** {questions.get(scenario_id, '(not found)').strip()}")
        add("")
        add("**The agent's full final answer:**")
        add("")
        for line in answers.get(scenario_id, "(no answer recorded)").splitlines():
            add(f"> {line}" if line.strip() else ">")
        add("")
        add("| Assertion | Does the answer satisfy it? (yes / no) |")
        add("|---|---|")
        for assertion in assertions:
            add(f"| `{assertion}` |  |")
        add("")
        add("---")
        add("")

    return "\n".join(out) + "\n"


def _questions() -> dict[str, str]:
    from evals.runner import SCENARIOS, load_scenarios

    return {s.id: s.question for s in load_scenarios(SCENARIOS)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("evals/results"))
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    results_path, traces_path = latest_run(args.results)
    args.out.write_text(
        build(results_path, traces_path, args.seed), encoding="utf-8"
    )
    print(f"read {results_path.name}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
