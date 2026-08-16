"""Score `evals/HAND_LABEL.md` against the recorded judge verdicts.

`docs/MILESTONE_5.md` section 3: Cohen's kappa between judge and human labels,
reported always, with 0.7 as the floor below which the rubric is inadequate and
must be revised with before and after both reported.

Kappa rather than raw agreement, because raw agreement is misleading when one
label dominates: a judge that always says "satisfied" agrees 90% of the time on
a suite that is 90% satisfied and has learned nothing.

**Per-assertion kappa is reported but should be read sceptically.** Most
assertion types appear a handful of times at one seed, and kappa on n=3 is
noise. The per-type table is there to point at *which* assertion names the two
raters read differently -- that is a rubric finding worth having even when the
coefficient itself is not stable. Raw agreement and the disagreement listing
carry more signal at that sample size, so both are printed alongside.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from evals.judge import DEFAULT_PROMPT, JUDGEMENTS_DIR, VerdictCache, prompt_version
from evals.metrics import cohens_kappa
from evals.schema import AssertionAgreement, JudgeAgreement, VersionAgreement

LABELS = Path("evals") / "HAND_LABEL.md"

SCENARIO_HEADING = re.compile(r"^## \d+\. `(?P<id>[^`]+)`\s*$", re.M)
ASSERTION_ROW = re.compile(r"^\|\s*`(?P<assertion>[^`]+)`\s*\|(?P<verdict>[^|]*)\|\s*$")

YES = {"yes", "y", "true", "t", "1"}
NO = {"no", "n", "false", "f", "0"}


class LabelSheetError(RuntimeError):
    """The sheet could not be read as filled in."""


def parse_sheet(text: str) -> list[tuple[str, str, bool | None]]:
    """(scenario id, assertion, verdict) triples. `None` where left blank.

    Parsing starts at the first scenario heading so the worked example inside
    the quoted rubric is never mistaken for a label row.
    """
    match = SCENARIO_HEADING.search(text)
    if not match:
        raise LabelSheetError(
            f"no scenario headings found in the sheet; expected lines like "
            "'## 1. `scenario-id`'. Regenerate with `python -m evals.make_hand_label`."
        )

    rows: list[tuple[str, str, bool | None]] = []
    current: str | None = None
    for line in text[match.start():].splitlines():
        heading = SCENARIO_HEADING.match(line)
        if heading:
            current = heading.group("id")
            continue
        cells = ASSERTION_ROW.match(line)
        if not cells or current is None:
            continue
        raw = cells.group("verdict").strip().lower()
        if raw in YES:
            verdict: bool | None = True
        elif raw in NO:
            verdict = False
        elif raw == "":
            verdict = None
        else:
            raise LabelSheetError(
                f"{current} / {cells.group('assertion')}: cannot read verdict "
                f"{raw!r}. Use 'yes' or 'no', or leave the cell blank."
            )
        rows.append((current, cells.group("assertion"), verdict))
    return rows


def run_results_files(directory: Path) -> list[Path]:
    """A run's primary results file is `<run_id>.json`; its sidecars
    (`.traces.json`, `.accounting.json`, `.spans.json`) share the run id and
    carry a second suffix, so a dot in the stem is what tells them apart."""
    return sorted(p for p in directory.glob("*.json") if "." not in p.stem)


def latest_run(directory: Path) -> tuple[dict, dict]:
    runs = run_results_files(directory)
    if not runs:
        raise SystemExit(f"no results in {directory}")
    results = json.loads(runs[-1].read_text(encoding="utf-8"))
    traces = json.loads(
        runs[-1].with_suffix(".traces.json").read_text(encoding="utf-8")
    )
    return results, traces


def score_against(cache: VerdictCache, version: str, model_key: str,
                  rows, answers) -> tuple[list[bool], list[bool]]:
    """(judge, human) label pairs for one recorded verdict set."""
    judge: list[bool] = []
    human: list[bool] = []
    for scenario_id, assertion, verdict in rows:
        if verdict is None:
            continue
        answer = answers.get(scenario_id)
        if answer is None:
            continue
        recorded = cache.get(version, model_key, assertion, answer)
        if recorded is None:
            continue
        human.append(verdict)
        judge.append(recorded.holds)
    return judge, human


def version_history(rows, answers, model_key: str) -> list[VersionAgreement]:
    """Every recorded verdict set scored against the same fixed labels.

    Section 3 requires the before and after of a rubric revision to be
    reported. Recomputing it from the stored verdicts makes that a measurement
    rather than a recollection, and keeps it honest if the labels ever change.
    """
    history: list[VersionAgreement] = []
    for path in sorted(JUDGEMENTS_DIR.glob("*.json")):
        label = path.stem
        cache = VerdictCache(path)
        judge, human = score_against(cache, label.split("-")[0], model_key, rows, answers)
        if not judge:
            continue
        k = cohens_kappa(judge, human)
        history.append(
            VersionAgreement(
                version=label,
                kappa=k if k == k else 0.0,
                n_labelled=len(judge),
                n_agreements=sum(a == b for a, b in zip(judge, human)),
                note="superseded; kept to show the effect of the prompt edit"
                if "superseded" in label else "",
            )
        )
    return history


def _fmt(value: float) -> str:
    return "undefined" if math.isnan(value) else f"{value:+.3f}"


def score(
    sheet: str, results: dict, traces: list[dict], seed: int = 1
) -> tuple[JudgeAgreement, dict]:
    version = prompt_version(DEFAULT_PROMPT)
    cache = VerdictCache(JUDGEMENTS_DIR / f"{version}.json")

    judge_model = (results.get("metadata") or {}).get("judge_model") or {}
    model_key = f"{judge_model.get('provider')}/{judge_model.get('model')}"

    answers = {t["scenario_id"]: t["answer"] for t in traces if t["seed"] == seed}

    human: list[bool] = []
    machine: list[bool] = []
    blank: list[tuple[str, str]] = []
    missing: list[tuple[str, str]] = []
    per_type_pairs: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    disagreements: list[dict] = []

    for scenario_id, assertion, verdict in parse_sheet(sheet):
        if verdict is None:
            blank.append((scenario_id, assertion))
            continue
        answer = answers.get(scenario_id)
        if answer is None:
            missing.append((scenario_id, assertion))
            continue
        recorded = cache.get(version, model_key, assertion, answer)
        if recorded is None:
            missing.append((scenario_id, assertion))
            continue
        human.append(verdict)
        machine.append(recorded.holds)
        per_type_pairs[assertion].append((verdict, recorded.holds))
        if verdict != recorded.holds:
            disagreements.append(
                {
                    "scenario_id": scenario_id,
                    "assertion": assertion,
                    "human": verdict,
                    "judge": recorded.holds,
                    "judge_confidence": recorded.confidence,
                    "judge_reason": recorded.reason,
                }
            )

    if not human:
        raise LabelSheetError(
            "no labelled rows could be matched to a recorded verdict. Fill in "
            f"{LABELS} first, and check the run and judge model still match."
        )

    kappa = cohens_kappa(machine, human)
    agreements = sum(a == b for a, b in zip(machine, human))
    adequate = kappa == kappa and kappa >= 0.7
    if kappa != kappa:
        note = (
            "Kappa is undefined: both raters gave a constant label, so there is "
            "no agreement beyond chance to measure. Label a set with both outcomes."
        )
    elif not adequate:
        note = (
            f"Kappa {kappa:.3f} is below the 0.7 floor. The rubric is inadequate; "
            "revise it and report the before and after."
        )
    else:
        note = f"Kappa {kappa:.3f} clears the 0.7 floor."

    per_type: list[AssertionAgreement] = []
    for assertion, pairs in sorted(per_type_pairs.items()):
        bad = sum(1 for h, m in pairs if h != m)
        try:
            k = cohens_kappa([m for _, m in pairs], [h for h, _ in pairs])
        except ValueError:
            k = float("nan")
        per_type.append(
            AssertionAgreement(
                assertion=assertion,
                n=len(pairs),
                agreements=len(pairs) - bad,
                disagreements=bad,
                kappa=None if k != k else k,
            )
        )

    rows_all = parse_sheet(sheet)
    agreement = JudgeAgreement(
        kappa=kappa if kappa == kappa else 0.0,
        n_labelled=len(human),
        n_agreements=agreements,
        judge_prompt_version=version,
        adequate=adequate,
        note=note,
        per_type=per_type,
        history=version_history(rows_all, answers, model_key),
        precision_note=(
            "A one-row flip moves kappa by roughly 0.05 at n=48, and the judge "
            "proved sensitive to a non-instructional edit to the rubric file. "
            "Treat the figure as 0.60 +/- 0.05, not as three significant digits."
        ),
    )
    detail = {
        "model_key": model_key,
        "run_id": results["metadata"]["run_id"],
        "raw_agreement": agreements / len(human),
        "per_type": per_type_pairs,
        "disagreements": disagreements,
        "blank": blank,
        "missing": missing,
        "raw_kappa": kappa,
    }
    return agreement, detail


def render(agreement: JudgeAgreement, detail: dict) -> str:
    out: list[str] = []
    add = out.append
    add(f"run           {detail['run_id']}")
    add(f"judge         {detail['model_key']}")
    add(f"rubric        {agreement.judge_prompt_version}")
    add("")
    add(f"labelled      {agreement.n_labelled}")
    add(f"agreements    {agreement.n_agreements} "
        f"({detail['raw_agreement'] * 100:.1f}% raw)")
    add(f"Cohen's kappa {_fmt(detail['raw_kappa'])}")
    add(f"0.7 floor     {'CLEARED' if agreement.adequate else 'NOT CLEARED'}")
    add("")
    add(agreement.note)

    if detail["blank"]:
        add("")
        add(f"{len(detail['blank'])} row(s) left blank and excluded — a blank is "
            "an abstention, not a label:")
        for scenario_id, assertion in detail["blank"]:
            add(f"    {scenario_id}  {assertion}")
    if detail["missing"]:
        add("")
        add(f"{len(detail['missing'])} row(s) had no recorded verdict to compare "
            "against (wrong run, wrong judge model, or a re-recorded answer):")
        for scenario_id, assertion in detail["missing"]:
            add(f"    {scenario_id}  {assertion}")

    add("")
    add("Per assertion type — where the two raters read a name differently.")
    add("Kappa on a handful of rows is noise; the disagreement count is the")
    add("signal, and the coefficient is shown only where it is defined.")
    add("")
    add(f"  {'assertion':40s} {'n':>3} {'disagree':>8} {'agree%':>7}  kappa")
    rows = sorted(
        detail["per_type"].items(),
        key=lambda kv: (-sum(1 for h, m in kv[1] if h != m), kv[0]),
    )
    for assertion, pairs in rows:
        n = len(pairs)
        bad = sum(1 for h, m in pairs if h != m)
        try:
            k = cohens_kappa([m for _, m in pairs], [h for h, _ in pairs])
        except ValueError:
            k = float("nan")
        add(f"  {assertion:40s} {n:>3} {bad:>8} {(n - bad) / n * 100:>6.0f}%  {_fmt(k)}")

    add("")
    if detail["disagreements"]:
        add(f"{len(detail['disagreements'])} disagreement(s), judge's reasoning shown")
        add("now that labelling is complete:")
        add("")
        for item in detail["disagreements"]:
            add(f"  {item['scenario_id']} / {item['assertion']}")
            add(f"    you: {'yes' if item['human'] else 'no':3s}   "
                f"judge: {'yes' if item['judge'] else 'no':3s} "
                f"(confidence {item['judge_confidence']})")
            if item["judge_reason"]:
                add(f"    judge said: {item['judge_reason']}")
            add("")
    else:
        add("No disagreements.")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=LABELS)
    parser.add_argument("--results", type=Path, default=Path("evals/results"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--write-into-results",
        action="store_true",
        help="store the agreement in the run's results JSON so the report "
        "renders it in section 5",
    )
    args = parser.parse_args()

    if not args.labels.exists():
        raise SystemExit(
            f"{args.labels} not found. Generate it with "
            "`python -m evals.make_hand_label`, fill it in, then re-run this."
        )
    results, traces = latest_run(args.results)
    agreement, detail = score(
        args.labels.read_text(encoding="utf-8"), results, traces, args.seed
    )
    print(render(agreement, detail))

    if args.write_into_results:
        runs = run_results_files(args.results)
        path = runs[-1]
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["judge_agreement"] = json.loads(agreement.model_dump_json())
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote judge_agreement into {path}")
        print("regenerate the report with `python -m evals.report`")


if __name__ == "__main__":
    main()
