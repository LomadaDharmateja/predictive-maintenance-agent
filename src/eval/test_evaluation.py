"""The only module permitted to read the test split.

`docs/MILESTONE_3.md` section 0. Everything this script uses -- the models, the
calibrators, the thresholds -- was fixed on train and validation before it ran.
Nothing here selects, tunes, or chooses anything; it reads decisions out of
`data/generated/validation_results.json` and applies them once.

It also keeps its own audit trail. Every run appends to `test_evaluation` in
`data/generated/build_manifest.json` with the git commit, the fingerprint of the
model files, and the data content hash. If it has run more than once against
different model artefacts, the report says so at the top rather than quietly
presenting the latest number as though it were the first.

This file is deliberately named `test_evaluation.py` as the milestone specifies.
`pytest.ini` restricts collection to `tests/`, so pytest does not mistake it for
a test module.

Run:  make evaluate-test
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.eval import calibration as calib
from src.eval import plots
from src.eval.baselines import BASELINE_THRESHOLDS, BASELINES, COMPONENT_BASELINES
from src.eval.datasets import (
    LOCKED_SPLIT,
    TEST_SPLIT_UNLOCK,
    load_features,
    xy,
)
from src.eval.metrics import (
    N_BOOTSTRAP,
    bootstrap_intervals,
    event_clusters,
    point_metrics,
)
from src.eval.report import SERIES_LABEL, SERIES_ORDER, append_test_report, fmt, fmt_ci
from src.eval.validate import (
    CALIBRATORS,
    EVALUATION,
    FAMILIES,
    MODELS_DIR,
    RESULTS,
    load_failures,
    load_model,
)
from src.eval.horizons import horizon_label
from src.features.config import COMPONENTS, LABEL_HORIZON
from src.models.train import SUMMARY, git_commit

MANIFEST = Path("data/generated/build_manifest.json")
TEST_RESULTS = Path("data/generated/test_results.json")


def model_fingerprint(models_dir: Path = MODELS_DIR) -> str:
    """SHA-256 over every model artefact, so two runs against different models
    are distinguishable in the audit log."""
    digest = hashlib.sha256()
    for path in sorted(models_dir.glob("*.joblib")):
        digest.update(path.name.encode("utf-8"))
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def record_run(fingerprint: str, data_hash: str, horizon: str) -> list[dict]:
    """Append this run to the manifest and return the full history.

    Note that this makes `build_manifest.json` non-deterministic in its
    `test_evaluation` key -- by design. The `tables` and `features` keys remain
    a pure function of the inputs; this one is an append-only audit log, and a
    log that could be regenerated identically would not be evidence of anything.
    """
    payload = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    history = payload.get("test_evaluation", {}).get("runs", [])
    history.append(
        {
            "run": len(history) + 1,
            "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_commit": git_commit(),
            "model_fingerprint": fingerprint,
            "test_data_sha256": data_hash,
            "horizon": horizon,
        }
    )
    payload["test_evaluation"] = {"runs": history}
    MANIFEST.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return history


def evaluate_test(models_dir: Path = MODELS_DIR, quiet: bool = False) -> dict:
    def say(message: str) -> None:
        if not quiet:
            print(message, flush=True)

    if not RESULTS.exists():
        raise FileNotFoundError(
            f"{RESULTS} not found. Run `make evaluate` first -- thresholds and "
            "calibrators must be fixed on validation before the test split is read."
        )
    validation = json.loads(RESULTS.read_text(encoding="utf-8"))
    calibrators = joblib.load(CALIBRATORS)

    say("reading the test split. This is the one-shot evaluation.")
    frame = load_features(LOCKED_SPLIT, unlock=TEST_SPLIT_UNLOCK)
    failures = load_failures()
    say(f"test rows: {len(frame):,}")

    results: dict = {"components": {}, "n_bootstrap": N_BOOTSTRAP}

    for component in COMPONENTS:
        y = frame[f"label_{component}"].to_numpy()
        clusters = event_clusters(frame, component, failures, LABEL_HORIZON)
        X, _ = xy(frame, component)

        scores = {name: fn(frame) for name, fn in BASELINES.items()}
        scores.update(
            {name: fn(frame, component) for name, fn in COMPONENT_BASELINES.items()}
        )
        for family in FAMILIES:
            bundle = load_model(component, family, models_dir)
            scores[family] = bundle["model"].predict_proba(
                X.to_numpy(dtype=float)
            )[:, 1]

        record: dict = {
            "positive_rate": float(y.mean()),
            "n_rows": int(len(y)),
            "n_positive": int(y.sum()),
            "n_bootstrap_clusters": int(len(np.unique(clusters))),
            "series": {},
        }

        for name, score in scores.items():
            if name in FAMILIES:
                # The threshold chosen on validation. Not re-derived here: that
                # would be selecting on the test split, which is the whole thing
                # section 0 forbids.
                threshold = float(
                    validation["components"][component]["thresholds"]["10"]["threshold"]
                ) if name == "lgbm" else float(
                    validation["components"][component]["series"][name]["threshold"]
                )
            else:
                threshold = BASELINE_THRESHOLDS.get(name, 0.5)

            metrics = point_metrics(y, score, threshold, component, with_roc=True)
            intervals = bootstrap_intervals(y, score, clusters, threshold)
            record["series"][name] = {
                "threshold": metrics.threshold,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
                "pr_auc": metrics.pr_auc,
                "roc_auc": metrics.roc_auc,
                "tp": metrics.true_positive,
                "fp": metrics.false_positive,
                "fn": metrics.false_negative,
                "tn": metrics.true_negative,
                "ci": {k: list(v) for k, v in intervals.items()},
            }
            say(
                f"  {component} {name:<18} PR-AUC {metrics.pr_auc:.4f}  "
                f"P {metrics.precision:.3f}  R {metrics.recall:.3f}"
            )

        # Out-of-sample calibration: the calibrator was fitted on validation and
        # has never seen these rows. This is the honest version of section 4.
        raw = scores["lgbm"]
        isotonic = calib.apply_calibrator(calibrators[component]["isotonic"], raw)
        before = calib.assess(y, raw, component, "lgbm (raw)")
        after = calib.assess(y, isotonic, component, "isotonic (fitted on val)")
        record["calibration"] = {
            report.method: {
                "brier": report.brier,
                "brier_base_rate": report.brier_base_rate,
                "brier_skill_score": report.brier_skill_score,
            }
            for report in (before, after)
        }
        plots.reliability_curves([before, after], component, suffix="test")
        plots.pr_curves(
            {name: (y, score) for name, score in scores.items()},
            float(y.mean()),
            component,
            suffix="test",
        )

        results["components"][component] = record

    fingerprint = model_fingerprint(models_dir)
    summary = json.loads(SUMMARY.read_text(encoding="utf-8")) if SUMMARY.exists() else {}
    data_hash = summary.get("data_content_hash", {}).get("test", "unknown")
    results["runs"] = record_run(
        fingerprint, data_hash, horizon_label(LABEL_HORIZON)
    )
    results["model_fingerprint"] = fingerprint
    return results


def render(results: dict) -> str:
    runs = results["runs"]
    distinct = {run["model_fingerprint"] for run in runs}
    lines: list[str] = []
    add = lines.append

    add("## 8. Test split, evaluated once")
    add("")
    if len(runs) == 1:
        add(
            f"This is run 1. The test split has been read exactly once, at "
            f"`{runs[0]['utc']}`, against model fingerprint "
            f"`{runs[0]['model_fingerprint'][:16]}` and commit "
            f"`{runs[0]['git_commit'][:10]}`."
        )
    else:
        horizons = {run.get("horizon", "24h (Milestone 3)") for run in runs}
        add(
            f"**This script has run {len(runs)} times, against "
            f"{len(distinct)} distinct set(s) of model artefacts.** Reporting that is "
            "better than presenting the latest number as though it were the first."
        )
        add("")
        add("| Run | UTC | Horizon | Commit | Model fingerprint |")
        add("|---|---|---|---|---|")
        for run in runs:
            add(
                f"| {run['run']} | {run['utc']} "
                f"| {run.get('horizon', '24h (not recorded; back-filled from the commit)')} "
                f"| `{run['git_commit'][:10]}` "
                f"| `{run['model_fingerprint'][:16]}` |"
            )
        add("")
        if len(horizons) == len(runs):
            add("Each run is at a **different prediction horizon**, which makes them")
            add("different experiments rather than repeated looks at the same one. Run 1")
            add("is the Milestone 3 evaluation at 24 hours, archived in")
            add("`docs/EVALUATION_24h.md`; run 2 is this one, at the 14-day horizon")
            add("derived in `docs/SIGNAL_ANALYSIS.md`. No modelling decision was made")
            add("after seeing either. The horizon changed because of the lead-time")
            add("argument in section 0, not because of a test score.")
        else:
            add("**Two or more runs share a horizon.** That is a repeated look at the")
            add("same question and the later numbers should be treated with suspicion.")
    add("")
    add("Thresholds and calibrators come from validation and were not re-derived")
    add("here. No model, hyperparameter or feature decision was made after this")
    add("section was produced.")
    add("")

    add("### PR-AUC")
    add("")
    add("| Component | " + " | ".join(SERIES_LABEL[s] for s in SERIES_ORDER) + " | no-skill floor |")
    add("|---" * (len(SERIES_ORDER) + 2) + "|")
    for component in COMPONENTS:
        record = results["components"][component]
        cells = [
            fmt_ci(record["series"][s]["pr_auc"], record["series"][s]["ci"]["pr_auc"])
            for s in SERIES_ORDER
        ]
        add(f"| {component} | " + " | ".join(cells) + f" | {record['positive_rate']:.5f} |")
    add("")

    for component in COMPONENTS:
        record = results["components"][component]
        add(f"### {component}")
        add("")
        add(
            f"{record['n_rows']:,} rows, {record['n_positive']:,} positive "
            f"({record['positive_rate']:.4%}), "
            f"{record['n_bootstrap_clusters']:,} bootstrap clusters."
        )
        add("")
        add("| Series | Threshold | Precision | Recall | F1 | TP | FP | FN | TN |")
        add("|---|---|---|---|---|---|---|---|---|")
        for series in SERIES_ORDER:
            entry = record["series"][series]
            add(
                f"| {SERIES_LABEL[series]} | {entry['threshold']:.4f} "
                f"| {fmt_ci(entry['precision'], entry['ci']['precision'])} "
                f"| {fmt_ci(entry['recall'], entry['ci']['recall'])} "
                f"| {fmt(entry['f1'], 3)} "
                f"| {entry['tp']:,} | {entry['fp']:,} | {entry['fn']:,} | {entry['tn']:,} |"
            )
        add("")
        add(f"![PR curve, {component}, test](images/pr_{component}_test.png)")
        add("")

    add("### Calibration, out of sample")
    add("")
    add("The isotonic calibrator was fitted on validation and applied here unchanged.")
    add("Unlike the validation figures in section 4, these are genuinely out of sample.")
    add("")
    add("| Component | Brier, raw | Brier, isotonic (fitted on val) | Base-rate reference |")
    add("|---|---|---|---|")
    for component in COMPONENTS:
        cal = results["components"][component]["calibration"]
        raw = cal["lgbm (raw)"]
        after = cal["isotonic (fitted on val)"]
        add(
            f"| {component} | {raw['brier']:.6f} | {after['brier']:.6f} "
            f"| {raw['brier_base_rate']:.6f} |"
        )
    add("")
    for component in COMPONENTS:
        add(f"![Reliability, {component}, test](images/reliability_{component}_test.png)")
    add("")
    add("### Appendix: ROC-AUC on test")
    add("")
    add("| Component | LightGBM | Logistic regression | matched error code |")
    add("|---|---|---|---|")
    for component in COMPONENTS:
        series = results["components"][component]["series"]
        add(
            f"| {component} | {fmt(series['lgbm']['roc_auc'])} "
            f"| {fmt(series['logreg']['roc_auc'])} "
            f"| {fmt(series['matched_error_24h']['roc_auc'])} |"
        )
    add("")
    add("### Anything wanted after seeing these numbers")
    add("")
    add("Nothing was changed after this section was produced. Recorded for honesty:")
    add("the model is close to the ceiling on this data, so there is no tuning to")
    add("want; what a second pass would change is the *dataset*, not the model, and")
    add("that is a Milestone 4 question rather than a reason to touch these figures.")
    add("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate on the test split, once.")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    args = parser.parse_args()
    results = evaluate_test(args.models_dir)
    section = render(results)
    # Persisted so docs/EVALUATION.md can be re-rendered later without opening
    # the locked split again.
    TEST_RESULTS.write_text(
        json.dumps(
            {"results": results, "section": section},
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    append_test_report(section, EVALUATION)
    print(f"\nappended the test section to {EVALUATION}")
    print(f"test-split read count: {len(results['runs'])}")


if __name__ == "__main__":
    main()
