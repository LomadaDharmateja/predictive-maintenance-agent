"""Compare two runs scenario by scenario.

`docs/MILESTONE_5.md` section 4: version-to-version comparison is the point of
the harness. A single run tells you the current state; only a diff tells you
whether a change helped.

Regressions and improvements are reported separately and never netted off. A run
that fixes three scenarios and breaks three is not neutral -- it is three new
failures, and averaging them away is how a regression ships.

Run:  make eval-diff BEFORE=... AFTER=...
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

from evals.schema import RunResults


@dataclass
class Diff:
    regressions: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)
    unchanged_pass: list[str] = field(default_factory=list)
    unchanged_fail: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    new_forbidden_calls: list[str] = field(default_factory=list)
    new_hallucinations: list[str] = field(default_factory=list)

    @property
    def has_regressions(self) -> bool:
        return bool(
            self.regressions or self.new_forbidden_calls or self.new_hallucinations
        )


def load(path: Path) -> RunResults:
    return RunResults(**json.loads(path.read_text(encoding="utf-8")))


def _key(result) -> str:
    return f"{result.scenario_id}#seed{result.seed}"


def compare(before: RunResults, after: RunResults) -> Diff:
    old = {_key(r): r for r in before.results}
    new = {_key(r): r for r in after.results}
    diff = Diff()

    for key in sorted(set(old) | set(new)):
        was, now = old.get(key), new.get(key)
        if was is None:
            diff.added.append(key)
        elif now is None:
            diff.removed.append(key)
        elif was.passed and not now.passed:
            diff.regressions.append(key)
        elif not was.passed and now.passed:
            diff.improvements.append(key)
        elif was.passed:
            diff.unchanged_pass.append(key)
        else:
            diff.unchanged_fail.append(key)

    old_forbidden = {
        f"{c.scenario_id}#seed{c.seed}:{c.tool}" for c in before.forbidden_calls
    }
    diff.new_forbidden_calls = sorted(
        {f"{c.scenario_id}#seed{c.seed}:{c.tool}" for c in after.forbidden_calls}
        - old_forbidden
    )

    old_fabricated = {
        f"{h.scenario_id}#seed{h.seed}:{h.value}" for h in before.hallucinations
    }
    diff.new_hallucinations = sorted(
        {f"{h.scenario_id}#seed{h.seed}:{h.value}" for h in after.hallucinations}
        - old_fabricated
    )
    return diff


def render(before: RunResults, after: RunResults, diff: Diff) -> str:
    lines = [
        f"{before.metadata.run_id}  ->  {after.metadata.run_id}",
        f"  git {before.metadata.git_sha} -> {after.metadata.git_sha}",
        "",
        f"  regressions          {len(diff.regressions)}",
        f"  improvements         {len(diff.improvements)}",
        f"  unchanged (pass)     {len(diff.unchanged_pass)}",
        f"  unchanged (fail)     {len(diff.unchanged_fail)}",
        f"  added                {len(diff.added)}",
        f"  removed              {len(diff.removed)}",
        f"  new forbidden calls  {len(diff.new_forbidden_calls)}",
        f"  new hallucinations   {len(diff.new_hallucinations)}",
    ]
    for title, items in (
        ("REGRESSIONS", diff.regressions),
        ("NEW FORBIDDEN TOOL CALLS", diff.new_forbidden_calls),
        ("NEW HALLUCINATIONS", diff.new_hallucinations),
        ("IMPROVEMENTS", diff.improvements),
    ):
        if items:
            lines += ["", title]
            lines += [f"  {item}" for item in items]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    args = parser.parse_args()

    before, after = load(args.before), load(args.after)
    diff = compare(before, after)
    print(render(before, after, diff))
    # Non-zero on regression, so this can gate a merge.
    return 1 if diff.has_regressions else 0


if __name__ == "__main__":
    raise SystemExit(main())
