"""Regenerate docs/EVALUATION.md from persisted results.

Both halves of the document come from JSON on disk:

- the validation half from `data/generated/validation_results.json`
- the test half from `data/generated/test_results.json`, which
  `src/eval/test_evaluation.py` writes on every run

This exists so that a wording change to the report never requires re-reading the
locked split. Earlier the only way to re-render while keeping the test section
was to slice it back out of the markdown that was about to be overwritten --
which meant the document was its own source of truth, and a botched edit was
unrecoverable. That extraction path is gone.

Run:  python -m src.eval.render
"""

from __future__ import annotations

import json
from pathlib import Path

from src.eval.report import append_test_report, write_validation_report
from src.eval.validate import EVALUATION, RESULTS

TEST_RESULTS = Path("data/generated/test_results.json")


def render_all(path: Path = EVALUATION, quiet: bool = False) -> None:
    def say(message: str) -> None:
        if not quiet:
            print(message)

    if not RESULTS.exists():
        raise FileNotFoundError(
            f"{RESULTS} not found. Run `make evaluate` first."
        )

    write_validation_report(
        json.loads(RESULTS.read_text(encoding="utf-8")), path
    )
    say(f"rendered the validation sections of {path}")

    if not TEST_RESULTS.exists():
        say(
            f"{TEST_RESULTS} not found, so the test section is absent. That is "
            "correct if `make evaluate-test` has not run at this horizon; it is "
            "not something to work around by editing the markdown."
        )
        return

    payload = json.loads(TEST_RESULTS.read_text(encoding="utf-8"))
    append_test_report(payload["section"], path)
    runs = payload.get("results", {}).get("runs", [])
    say(f"appended the test section ({len(runs)} recorded run(s))")


if __name__ == "__main__":
    render_all()
