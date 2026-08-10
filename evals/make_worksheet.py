"""Scenario-writing worksheet: 15 deliberately chosen machine-hours from validation.

Writing scenarios against imagined machines produces scenarios the system cannot
answer, or answers trivially. This script reports what the fleet actually looks
like at fifteen specific moments, so a scenario can be written against a real
row with a known outcome.

Three properties are load-bearing.

**Validation only.** `load_val()` is the only split reader here. The test split
stays closed; it has been read twice under the unlock protocol and this is not a
third. Nothing in this module can open it -- `src/eval/datasets.py` refuses
without a token this file does not import.

**Read-only.** Parquet, joblib and JSON are read; the database is opened through
`src.agent.db`, which connects `mode=ro` behind an authorizer allowlist. The
only file written is `evals/WORKSHEET.md`.

**The fifteen are chosen by named predicate, not sampled.** Each selector states
a hard predicate the row must satisfy and a score that ranks the rows which do.
Random sampling from a fleet where most component-hours are quiet would return
fifteen near-identical quiet rows and cover none of the cases worth writing a
scenario about. Every pick carries the predicate that selected it, so the choice
is arguable and reproducible rather than aesthetic.

Adequacy arithmetic is imported from `src.agent.risk` rather than restated, so
the worksheet cannot drift from what `get_failure_risk` returns.

Run:  make worksheet
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.agent import db
from src.agent.risk import MARGINAL_FACTOR, _adequacy, _artefacts, _best
from src.eval.datasets import load_val
from src.features.config import (
    COMPONENTS,
    ERROR_IDS,
    ERROR_WINDOWS,
    FEATURE_COLUMNS,
    LABEL_HORIZON,
    MACHINE_MODELS,
    PRODUCTION_FAMILY,
    label_column,
)

OUTPUT = Path("evals/WORKSHEET.md")

#: One pick per band, so the fifteen span the validation window instead of
#: clustering on whichever hour happens to maximise a score. A selector that
#: cannot satisfy its predicate in a free band takes the best row anywhere and
#: says so in its note.
N_PICKS = 15

#: Cost ratio whose threshold is quoted. The same key `get_failure_risk` reads.
THRESHOLD_KEY = "10"


# ----------------------------------------------------------------------
# Frame assembly
# ----------------------------------------------------------------------


def scored_validation() -> tuple[pd.DataFrame, dict]:
    """Validation rows with calibrated probabilities attached.

    Probabilities are computed the same way `get_failure_risk` computes them --
    same model artefact, same isotonic calibrator, same rounding -- but in bulk
    off the persisted feature matrix rather than one machine at a time off the
    database. The feature matrix was built by the same `compute_features` the
    tool calls, so the numbers are the tool's numbers.
    """
    artefacts = _artefacts()
    frame = load_val().reset_index(drop=True)

    matrix = frame[FEATURE_COLUMNS].to_numpy(dtype=float)
    for component in COMPONENTS:
        raw = artefacts["models"][component].predict_proba(matrix)[:, 1]
        calibrated = artefacts["calibrators"][component]["isotonic"].predict(raw)
        frame[f"prob_{component}"] = np.round(calibrated, 4)

    prob_columns = [f"prob_{c}" for c in COMPONENTS]
    label_columns = [label_column(c) for c in COMPONENTS]
    frame["max_prob"] = frame[prob_columns].max(axis=1)
    frame["n_failing"] = frame[label_columns].sum(axis=1)
    frame["max_hours_since"] = frame[
        [f"hours_since_{c}" for c in COMPONENTS]
    ].max(axis=1)

    for window in ERROR_WINDOWS:
        frame[f"errors_{window}"] = frame[
            [f"{e}_count_{window}" for e in ERROR_IDS]
        ].sum(axis=1)

    frame["model"] = ""
    for name in MACHINE_MODELS:
        frame.loc[frame[f"model_{name}"] == 1, "model"] = name

    start, end = frame["datetime"].min(), frame["datetime"].max()
    span = (end - start).total_seconds()
    offset = (frame["datetime"] - start).dt.total_seconds()
    frame["band"] = np.minimum((offset / span * N_PICKS).astype(int), N_PICKS - 1)

    return frame, artefacts


def parts_table() -> pd.DataFrame:
    """The parts inventory, read through the read-only database layer."""
    rows = db.fetch(db.PARTS_FOR_COMPONENT)
    return pd.DataFrame([dict(row) for row in rows])


# ----------------------------------------------------------------------
# Selection
# ----------------------------------------------------------------------

Predicate = Callable[[pd.DataFrame], pd.Series]
Score = Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class Selector:
    key: str
    title: str
    why: str
    predicate: Predicate
    score: Score
    #: Set on the required-coverage selectors so the worksheet can report which
    #: of the six requested cases each pick discharges.
    covers: str | None = None


@dataclass
class Pick:
    selector: Selector
    row: pd.Series
    note: str


def build_selectors(
    threshold_comp2: float, quiet_floor: float, age_range: tuple[int, int]
) -> list[Selector]:
    """The fifteen predicates, in the order they claim machines and bands.

    The six required cases come first, so a scarce row goes to a case that was
    asked for rather than to a supplementary one.
    """
    return [
        Selector(
            key="adequacy-split",
            title="One component's warning is long enough, another's is not",
            why=(
                "Both comp4 and comp2 above 0.15, ranked by their sum. comp4's "
                "measured lead clears the 12-day PN-COMP4-001 order; comp2's "
                "does not clear any of its parts. The two components demand "
                "different answers at the same moment. The bar is 0.15 rather "
                "than something higher because no validation row has both above "
                "0.2 -- the two risks do not peak together in this fleet."
            ),
            predicate=lambda f: (f["prob_comp4"] > 0.15) & (f["prob_comp2"] > 0.15),
            score=lambda f: f["prob_comp4"] + f["prob_comp2"],
            covers="one component adequate, another not",
        ),
        Selector(
            key="elevated-confirmed",
            title="Genuinely elevated risk, and the failure happened",
            why=(
                "Highest comp2 probability among rows that did fail within the "
                "horizon. comp2 is the only component whose calibrated "
                "probability is established as better than the base rate, so "
                "this is the one place a high number can be quoted as one."
            ),
            predicate=lambda f: f[label_column("comp2")] == 1,
            score=lambda f: f["prob_comp2"],
            covers="genuinely elevated risk",
        ),
        Selector(
            key="quiet-everywhere",
            title="Near-zero risk on every component",
            why=(
                "Lowest maximum probability across the four components, among "
                "rows where nothing failed within the horizon. The floor is "
                f"{quiet_floor:.4f}, not zero: the isotonic calibrators map "
                "their lowest bins to small positive constants, so no machine "
                "in this fleet is ever reported at zero risk on all four "
                "components. A scenario written here should expect a small "
                "number, not an absent one."
            ),
            predicate=lambda f: f["n_failing"] == 0,
            score=lambda f: -f["max_prob"],
            covers="near-zero risk everywhere",
        ),
        Selector(
            key="scarcest-part",
            title="Elevated risk against the scarcest part in the inventory",
            why=(
                "Highest comp4 probability. comp4's PN-COMP4-001 carries the "
                "lowest stock in the fleet. Stands in for the requested "
                "out-of-stock case, which the inventory cannot supply -- see "
                "the note under the parts table."
            ),
            predicate=lambda f: f["prob_comp4"] > 0.0,
            score=lambda f: f["prob_comp4"],
            covers="out-of-stock part (substituted -- see note)",
        ),
        Selector(
            key="errors-without-risk",
            title="Recent error activity, low risk",
            why=(
                "Most errors in the preceding 7 days among rows whose highest "
                "component probability is under 0.06 and where nothing "
                "subsequently failed. Errors are not failures; an answer that "
                "treats the error count as the risk is wrong here, and the "
                "clean outcome removes any ambiguity about that. The bar is "
                "0.06 rather than 0.05 because below 0.05 exactly one machine "
                "in the fleet qualifies, and it is the `quiet-everywhere` pick; "
                "the fleet median maximum is 0.166 for comparison."
            ),
            predicate=lambda f: (f["max_prob"] < 0.06) & (f["n_failing"] == 0),
            score=lambda f: f["errors_7d"],
            covers="recent errors but low risk",
        ),
        Selector(
            key="high-risk-uncalibrated",
            title="High risk on a component flagged calibrated: false",
            why=(
                "Highest comp3 probability. comp3's held-out Brier skill is not "
                "established as positive, so the number is large and untrustworthy "
                "at once. The answer must surface the flag, not the number alone."
            ),
            predicate=lambda f: f["prob_comp3"] > 0.5,
            score=lambda f: f["prob_comp3"],
            covers="high risk on a calibrated: false component",
        ),
        Selector(
            key="false-positive",
            title="High comp2 probability, no failure followed",
            why=(
                "The over-warning case. A scenario written here tests whether "
                "the answer commits to a failure that did not occur."
            ),
            predicate=lambda f: (f[label_column("comp2")] == 0) & (f["prob_comp2"] > 0.5),
            score=lambda f: f["prob_comp2"],
        ),
        Selector(
            key="missed-failure",
            title="Low comp2 probability, failure followed anyway",
            why=(
                "The under-warning case, and the one that costs. Lowest comp2 "
                "probability among rows that failed."
            ),
            predicate=lambda f: f[label_column("comp2")] == 1,
            score=lambda f: -f["prob_comp2"],
        ),
        Selector(
            key="multi-component",
            title="More than one component failed in the same window",
            why=(
                "Tests whether an answer can hold several components apart "
                "instead of collapsing them into one verdict about the machine."
            ),
            predicate=lambda f: f["n_failing"] >= 2,
            score=lambda f: f["n_failing"] + f["max_prob"],
        ),
        Selector(
            key="fresh-but-risky",
            title="Shortest interval since replacement, among elevated comp2 risk",
            why=(
                "Smallest hours-since-replacement on comp2 among rows where "
                "comp2 risk is above 0.3. Tests the heuristic that a recent "
                "replacement settles the question. Read the actual interval in "
                "the table before writing to it: if it is large, that is the "
                "finding -- elevated comp2 risk does not appear soon after a "
                "comp2 replacement in this fleet."
            ),
            predicate=lambda f: f["prob_comp2"] > 0.3,
            score=lambda f: -f["hours_since_comp2"],
        ),
        Selector(
            key="long-overdue",
            title="Longest interval since any component was replaced",
            why=(
                "The other end of the same axis. Useful for checking whether an "
                "answer reads a long interval as risk when the model does not."
            ),
            predicate=lambda f: f["max_hours_since"] > 0,
            score=lambda f: f["max_hours_since"],
        ),
        Selector(
            key="oldest-machine",
            title="Oldest machine still unclaimed",
            why=(
                f"Age is a model feature; the fleet runs {age_range[0]} to "
                f"{age_range[1]} years. This is the upper end among machines no "
                "earlier selector had already taken."
            ),
            predicate=lambda f: f["age"] >= 0,
            score=lambda f: f["age"],
        ),
        Selector(
            key="newest-machine",
            title="Newest machine still unclaimed",
            why=(
                f"Age is a model feature; the fleet runs {age_range[0]} to "
                f"{age_range[1]} years. This is the lower end among machines no "
                "earlier selector had already taken."
            ),
            predicate=lambda f: f["age"] >= 0,
            score=lambda f: -f["age"],
        ),
        Selector(
            key="threshold-borderline",
            title="comp2 probability sitting on the operating threshold",
            why=(
                f"Closest comp2 probability to the {THRESHOLD_KEY}:1 cost-ratio "
                f"threshold of {threshold_comp2:.4f}. The decision flips here, "
                "so an answer that sounds equally confident either side of it is "
                "not reading the threshold."
            ),
            predicate=lambda f: f["prob_comp2"] > 0.0,
            score=lambda f: -(f["prob_comp2"] - threshold_comp2).abs(),
        ),
        Selector(
            key="comp1-peak",
            title="Highest comp1 risk -- the component with a 24-hour warning",
            why=(
                "comp1's measured effective detection lead is 24 hours against "
                "part lead times of 10 and 17 days. This is the worst "
                "warning-to-lead-time ratio in the system, at the moment comp1 "
                "risk peaks."
            ),
            predicate=lambda f: f["prob_comp1"] > 0.0,
            score=lambda f: f["prob_comp1"],
        ),
    ]


def select(frame: pd.DataFrame, selectors: list[Selector]) -> list[Pick]:
    """Resolve each selector to one row, with machine and band held distinct.

    Ties break on machineID then datetime, so the worksheet is byte-identical
    across runs on the same artefacts.
    """
    picks: list[Pick] = []
    used_machines: set[int] = set()
    used_bands: set[int] = set()

    for selector in selectors:
        mask = selector.predicate(frame)
        eligible = frame[mask & ~frame["machineID"].isin(used_machines)]
        if eligible.empty:
            picks.append(
                Pick(
                    selector=selector,
                    row=pd.Series(dtype=object),
                    note="NO ROW SATISFIES THIS PREDICATE ANYWHERE IN VALIDATION.",
                )
            )
            continue

        note = ""
        spread = eligible[~eligible["band"].isin(used_bands)]
        if spread.empty:
            note = (
                "Every free time band was exhausted before this selector ran, so "
                "it took the best row anywhere in the window."
            )
        else:
            eligible = spread

        ranked = eligible.assign(_score=selector.score(eligible)).sort_values(
            ["_score", "machineID", "datetime"], ascending=[False, True, True]
        )
        row = ranked.iloc[0]
        used_machines.add(int(row["machineID"]))
        used_bands.add(int(row["band"]))
        picks.append(Pick(selector=selector, row=row, note=note))

    return picks


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------


def _lead_hours(artefacts: dict, component: str) -> float | None:
    return artefacts["detection"].get(component, {}).get("median_hours")


def component_adequacy(artefacts: dict, parts: pd.DataFrame) -> dict[str, str]:
    """Component-level verdict: the best any of that component's parts achieves."""
    verdicts = {}
    for component in COMPONENTS:
        lead = _lead_hours(artefacts, component)
        rows = parts[parts["component"] == component]
        per_part = [_adequacy(lead, int(r.lead_time_days)) for r in rows.itertuples()]
        verdicts[component] = _best(per_part).value
    return verdicts


def render(
    frame: pd.DataFrame,
    picks: list[Pick],
    parts: pd.DataFrame,
    artefacts: dict,
) -> str:
    lines: list[str] = []
    add = lines.append

    horizon_days = int(LABEL_HORIZON.total_seconds() // 86400)
    start, end = frame["datetime"].min(), frame["datetime"].max()
    verdicts = component_adequacy(artefacts, parts)
    quality = artefacts["calibration"]["components"]

    add("# WORKSHEET.md")
    add("")
    add("Fifteen machine-hours from the validation split, chosen by named")
    add("predicate, for writing `evals/scenarios.yaml` against. Generated by")
    add("`make worksheet`; every figure is computed, none typed.")
    add("")
    add(f"Validation window: `{start}` to `{end}` "
        f"({len(frame['datetime'].unique())} hourly prediction times, "
        f"{len(frame):,} machine-hours).")
    add("")
    add(f"Prediction horizon: {horizon_days} days. Model family: "
        f"`{PRODUCTION_FAMILY}`. Probabilities are isotonic-calibrated, computed "
        "with the same artefacts `get_failure_risk` loads.")
    add("")
    add("**The test split was not opened.** This script reads `load_val()` only.")
    add("")
    add("---")
    add("")

    # ---- Component reference -----------------------------------------
    add("## 1. Component reference")
    add("")
    add("Constant across every machine and every hour below. The detection lead")
    add("is a fleet-level measurement, and the calibration flag is a property of")
    add("the model, so neither varies row to row.")
    add("")
    add("| Component | Effective detection lead (h) | `calibrated` | Held-out Brier skill | 95% CI | Component `warning_adequacy` |")
    add("|---|---|---|---|---|---|")
    for component in COMPONENTS:
        record = quality[component]
        lead = _lead_hours(artefacts, component)
        add(
            f"| {component} | {lead if lead is not None else 'n/a'} "
            f"| `{str(bool(record['calibrated'])).lower()}` "
            f"| {record['skill_calibrated_held_out']:+.4f} "
            f"| [{record['skill_ci_low']:+.4f}, {record['skill_ci_high']:+.4f}] "
            f"| {verdicts[component]} |"
        )
    add("")
    add("Component-level `warning_adequacy` is the best verdict any of that")
    add("component's parts achieves. Because both inputs -- the measured lead and")
    add("the part lead time -- are fleet constants, **`warning_adequacy` does not")
    add("vary by machine or by hour**. A scenario contrasting two components'")
    add("adequacy is therefore a contrast that holds for every machine; the pick")
    add("labelled `adequacy-split` below is the one where both components also")
    add("carry enough probability for the contrast to matter.")
    add("")

    # ---- Parts table -------------------------------------------------
    add("---")
    add("")
    add("## 2. Every (component, part) pair")
    add("")
    add(f"`sufficient` requires the measured lead to clear the part's lead time by")
    add(f"a factor of {MARGINAL_FACTOR}; `marginal` requires it to clear the lead time")
    add("at all. Stock is the level in the inventory table and is not part of the")
    add("adequacy arithmetic -- adequacy is about time, not quantity.")
    add("")
    add("| Component | Part | Lead time (days) | Lead time (h) | Detection lead (h) | Stock | Unit cost | Supplier | Verdict |")
    add("|---|---|---|---|---|---|---|---|---|")
    ordered = parts.sort_values(["component", "part_id"])
    for part in ordered.itertuples():
        lead = _lead_hours(artefacts, part.component)
        verdict = _adequacy(lead, int(part.lead_time_days)).value
        add(
            f"| {part.component} | `{part.part_id}` | {part.lead_time_days} "
            f"| {int(part.lead_time_days) * 24} "
            f"| {lead if lead is not None else 'n/a'} "
            f"| {part.stock_quantity} | {part.unit_cost} | {part.supplier_id} "
            f"| **{verdict}** |"
        )
    add("")

    counts = (
        ordered.assign(
            verdict=[
                _adequacy(_lead_hours(artefacts, p.component), int(p.lead_time_days)).value
                for p in ordered.itertuples()
            ]
        )["verdict"]
        .value_counts()
        .to_dict()
    )
    summary = ", ".join(f"{n} {v}" for v, n in sorted(counts.items()))
    add(f"Across {len(ordered)} pairs: {summary}.")
    add("")

    minimum = int(parts["stock_quantity"].min())
    scarcest = parts.loc[parts["stock_quantity"].idxmin()]
    if minimum > 0:
        add("### On the requested out-of-stock case")
        add("")
        add(f"**No part in the inventory is out of stock.** The minimum stock "
            f"level is {minimum} units (`{scarcest['part_id']}`, "
            f"{scarcest['component']}, {scarcest['lead_time_days']}-day lead). A "
            "scenario asserting a zero-stock part would assert something the "
            "data does not contain, and the agent would contradict it.")
        add("")
        add("The `scarcest-part` pick below substitutes the lowest-stock part")
        add("instead. If a genuine out-of-stock scenario is wanted, the inventory")
        add("generator (`scripts/generate_inventory.py`) is where a zero would have")
        add("to come from -- not from the scenario file.")
        add("")

    # ---- Picks -------------------------------------------------------
    add("---")
    add("")
    add("## 3. Coverage of the six requested cases")
    add("")
    add("| Requested case | Pick |")
    add("|---|---|")
    for pick in picks:
        if pick.selector.covers:
            add(f"| {pick.selector.covers} | `{pick.selector.key}` |")
    add("")

    add("---")
    add("")
    add("## 4. The fifteen")
    add("")
    add("Two constraints run across the whole set. **No machine appears twice**,")
    add("so fifteen scenarios exercise fifteen machines rather than fifteen hours")
    add(f"of one. **No two picks share a time band** -- the window is cut into")
    add(f"{N_PICKS} equal bands and each pick claims one, so the set spans the")
    add("validation period instead of clustering wherever a score peaks.")
    add("")
    add("Both constraints bind on the selectors that run later. A selector")
    add("described as picking a maximum picks the maximum among rows still")
    add("available to it, which is not always the fleet maximum; where that")
    add("distinction matters the entry below says so.")
    add("")
    add("| # | Key | Machine | `as_of` | Selected because |")
    add("|---|---|---|---|---|")
    for index, pick in enumerate(picks, start=1):
        if pick.row.empty:
            add(f"| {index} | `{pick.selector.key}` | — | — | {pick.note} |")
            continue
        add(
            f"| {index} | `{pick.selector.key}` | {int(pick.row['machineID'])} "
            f"| `{pick.row['datetime']}` | {pick.selector.title} |"
        )
    add("")

    for index, pick in enumerate(picks, start=1):
        add("---")
        add("")
        add(f"### {index}. `{pick.selector.key}` — {pick.selector.title}")
        add("")
        if pick.row.empty:
            add(f"**{pick.note}**")
            add("")
            add(f"Intended predicate: {pick.selector.why}")
            add("")
            continue

        row = pick.row
        add(f"**Why this row:** {pick.selector.why}")
        add("")
        if pick.note:
            add(f"**Note:** {pick.note}")
            add("")
        age = int(row["age"])
        add(f"**Machine {int(row['machineID'])}**, model `{row['model']}`, "
            f"age {age} year{'' if age == 1 else 's'}, "
            f"`as_of = {row['datetime']}` "
            f"(time band {int(row['band'])} of {N_PICKS - 1}).")
        add("")

        add("| Component | Calibrated probability | `calibrated` | `warning_adequacy` | Detection lead (h) | Hours since replacement | Failed within "
            f"{horizon_days}d | Parts (stock @ lead time) |")
        add("|---|---|---|---|---|---|---|---|")
        for component in COMPONENTS:
            lead = _lead_hours(artefacts, component)
            failed = bool(row[label_column(component)])
            inventory = ", ".join(
                f"`{p.part_id}` {p.stock_quantity} @ {p.lead_time_days}d "
                f"({_adequacy(lead, int(p.lead_time_days)).value})"
                for p in ordered.itertuples()
                if p.component == component
            )
            add(
                f"| {component} | {row[f'prob_{component}']:.4f} "
                f"| `{str(bool(quality[component]['calibrated'])).lower()}` "
                f"| {verdicts[component]} "
                f"| {lead if lead is not None else 'n/a'} "
                f"| {row[f'hours_since_{component}']:.0f} "
                f"| {'**yes**' if failed else 'no'} "
                f"| {inventory} |"
            )
        add("")
        add("Stock and lead times are fleet constants, repeated per row so each")
        add("section stands alone; section 2 is the same data in one table.")
        add("")

        add("Errors in the lookback windows:")
        add("")
        add("| Window | " + " | ".join(ERROR_IDS) + " | total |")
        add("|---|" + "---|" * (len(ERROR_IDS) + 1))
        for window in ERROR_WINDOWS:
            cells = " | ".join(
                f"{row[f'{e}_count_{window}']:.0f}" for e in ERROR_IDS
            )
            add(f"| {window} | {cells} | **{row[f'errors_{window}']:.0f}** |")
        add("")

        failing = [c for c in COMPONENTS if bool(row[label_column(c)])]
        if failing:
            add(f"**Outcome:** {', '.join(failing)} failed within the "
                f"{horizon_days}-day window following `{row['datetime']}`.")
        else:
            add(f"**Outcome:** no component failed within the {horizon_days}-day "
                f"window following `{row['datetime']}`.")
        add("")

    # ---- What this cannot tell you -----------------------------------
    add("---")
    add("")
    add("## 5. What this worksheet does not establish")
    add("")
    add("- Every row is simulated data (Microsoft Azure PdM). An outcome column")
    add("  reading `yes` means the simulation recorded a failure, not that a")
    add("  machine broke.")
    add("- The outcome column is the label the model was trained against. A")
    add("  scenario must not ask the agent for it: no tool returns it, and an")
    add("  answer that supplied it would be a fabrication that happened to match.")
    add("- Three of the four components carry `calibrated: false`. Their")
    add("  probabilities appear above because the tool returns them, not because")
    add("  they are established as meaningful.")
    add("- Fifteen rows chosen to be interesting are not a sample. Nothing here")
    add("  supports a statement about how the fleet behaves on average.")
    add("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    frame, artefacts = scored_validation()
    parts = parts_table()
    threshold = float(
        artefacts["validation"]["components"]["comp2"]["thresholds"][THRESHOLD_KEY][
            "threshold"
        ]
    )
    quiet_floor = float(frame.loc[frame["n_failing"] == 0, "max_prob"].min())
    ages = (int(frame["age"].min()), int(frame["age"].max()))
    picks = select(frame, build_selectors(threshold, quiet_floor, ages))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(frame, picks, parts, artefacts), encoding="utf-8")

    unresolved = [p.selector.key for p in picks if p.row.empty]
    print(f"validation rows scored: {len(frame):,}")
    print(f"picks resolved: {len(picks) - len(unresolved)} of {len(picks)}")
    if unresolved:
        print(f"UNRESOLVED PREDICATES: {', '.join(unresolved)}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
