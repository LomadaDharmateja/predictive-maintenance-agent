"""Check a scenario file against the required distribution.

`docs/MILESTONE_5.md` section 2. The scenarios are the project owner's work; this
is the gate that says whether the set is complete enough to run.

It fails loudly and specifically. "12 scenarios, expected 41" is not useful; the
message below names every category that is short and by how many, so the next
piece of work is obvious without opening the spec.

Run:  make eval-validate
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import yaml
from pydantic import ValidationError

from evals.schema import (
    MINIMUM_SCENARIOS,
    REQUIRED_DISTRIBUTION,
    Category,
    Scenario,
)

DEFAULT_PATH = Path("evals/scenarios.yaml")


class ScenarioSetIncomplete(SystemExit):
    pass


def load(path: Path) -> list[Scenario]:
    if not path.exists():
        raise ScenarioSetIncomplete(f"{path} not found.")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    if not isinstance(raw, list):
        raise ScenarioSetIncomplete(f"{path} must contain a list of scenarios.")

    scenarios, problems = [], []
    for index, entry in enumerate(raw):
        try:
            scenarios.append(Scenario(**entry))
        except ValidationError as exc:
            name = entry.get("id", f"entry {index}") if isinstance(entry, dict) else index
            problems.append(f"  {name}: {exc.error_count()} schema error(s)")
            for error in exc.errors()[:3]:
                problems.append(
                    f"      {'.'.join(str(p) for p in error['loc'])}: {error['msg']}"
                )
    if problems:
        raise ScenarioSetIncomplete(
            f"{path} has scenarios that do not match the schema:\n" + "\n".join(problems)
        )
    return scenarios


def report(scenarios: list[Scenario], path: Path) -> list[str]:
    """Returns the list of problems. Empty means the set is complete."""
    problems: list[str] = []
    counts = Counter(s.category for s in scenarios)

    duplicates = [i for i, n in Counter(s.id for s in scenarios).items() if n > 1]
    if duplicates:
        problems.append(f"duplicate scenario ids: {', '.join(sorted(duplicates))}")

    lines = ["", f"{'Category':<26}{'have':>6}{'need':>6}{'short':>7}"]
    shortfall = 0
    for category, required in REQUIRED_DISTRIBUTION.items():
        have = counts.get(category, 0)
        missing = max(0, required - have)
        shortfall += missing
        flag = "" if missing == 0 else f"  <-- write {missing} more"
        lines.append(f"{category.value:<26}{have:>6}{required:>6}{missing:>7}{flag}")
    lines.append(f"{'TOTAL':<26}{len(scenarios):>6}{MINIMUM_SCENARIOS:>6}{shortfall:>7}")
    print("\n".join(lines))

    if shortfall:
        problems.append(
            f"{shortfall} scenario(s) short of the required distribution "
            f"({len(scenarios)} of {MINIMUM_SCENARIOS})"
        )

    # The two categories the milestone calls the heart of the suite.
    for category in (Category.RISK_INADEQUATE, Category.UNANSWERABLE):
        if counts.get(category, 0) < REQUIRED_DISTRIBUTION[category]:
            problems.append(
                f"{category.value} is short. Milestone 3B established that one part "
                "in nine can be ordered in time; these scenarios are what prove the "
                "agent does not hide that."
            )

    # Section 2's writing guidance, checked where it can be.
    machines = [s.question for s in scenarios]
    if len(scenarios) >= 10 and sum("42" in q for q in machines) > len(scenarios) / 3:
        problems.append(
            "machine 42 appears in more than a third of the questions; section 2 "
            "asks for varied machines and timestamps"
        )

    parts_without_forbidden = [
        s.id
        for s in scenarios
        if s.category is Category.PARTS_POSITION
        and "get_failure_risk" not in s.forbidden_tools
    ]
    if parts_without_forbidden:
        problems.append(
            "parts_position scenarios must forbid get_failure_risk: "
            + ", ".join(parts_without_forbidden)
        )

    failures_without_injection = [
        s.id
        for s in scenarios
        if s.category is Category.TOOL_FAILURE and s.injected_failure is None
    ]
    if failures_without_injection:
        problems.append(
            "tool_failure scenarios need an injected_failure to be deterministic: "
            + ", ".join(failures_without_injection)
        )

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_PATH)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="report the shortfall but exit 0; for work in progress",
    )
    args = parser.parse_args()

    scenarios = load(args.scenarios)
    problems = report(scenarios, args.scenarios)

    if not problems:
        print(f"\n{args.scenarios} is complete: {len(scenarios)} scenarios.")
        return 0

    print("\nINCOMPLETE:")
    for problem in problems:
        print(f"  - {problem}")
    print(
        "\nScenarios are written by the project owner, not generated. "
        "See docs/MILESTONE_5.md section 2 for the format and the guidance."
    )
    return 0 if args.allow_incomplete else 1


if __name__ == "__main__":
    sys.exit(main())
