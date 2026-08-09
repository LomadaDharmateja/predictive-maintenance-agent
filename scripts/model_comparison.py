"""LightGBM against logistic regression, to decide which one ships.

`docs/MILESTONE_3B.md` left the two models close on test. This settles it.

**The selection is made on validation, not on test.** That is not squeamishness:
choosing between two trained models on the strength of their test scores is a
modelling decision taken on the test split, which is exactly what
`docs/MILESTONE_3.md` section 0 forbids. The already-published test figures in
`docs/EVALUATION.md` section 8 are quoted as corroboration; this script does not
re-open the locked split to produce them, and the run count in
`build_manifest.json` stays at 2.

Four measurements, on validation:

1. **Paired bootstrap on the PR-AUC difference.** Comparing two overlapping
   marginal intervals is a weak test -- two estimates can have overlapping
   intervals while their difference is reliably non-zero. Resampling the same
   clusters for both models and taking the difference each time is the correct
   comparison, and it is what decides "clearly worse".
2. **Brier skill after isotonic calibration.** The agent consumes calibrated
   probabilities, so calibrated quality is what matters, not raw.
3. **Inference latency**, per 1,000 rows, single-threaded.
4. **Artefact size** on disk.

Writes data/generated/model_comparison.json.

Run:  make model-comparison
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.eval import calibration as calib
from src.eval.datasets import load_val, xy
from src.eval.metrics import BOOTSTRAP_SEED, event_clusters
from src.features.config import COMPONENTS, FEATURE_COLUMNS, LABEL_HORIZON

DB = Path("data/pdm.db")
MODELS_DIR = Path("models")
OUTPUT = Path("data/generated/model_comparison.json")

N_RESAMPLES = 1000
LATENCY_ROWS = 1000
LATENCY_REPEATS = 20
CONFIDENCE = 0.95


def load_failures() -> pd.DataFrame:
    connection = sqlite3.connect(DB)
    try:
        return pd.read_sql_query(
            "SELECT * FROM failures", connection, parse_dates=["datetime"]
        )
    finally:
        connection.close()


def paired_difference(
    y: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    clusters: np.ndarray,
    n_resamples: int = N_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Bootstrap interval on PR-AUC(a) - PR-AUC(b), same resample for both.

    Pairing is the point. Independent intervals ignore that both models see the
    same rows and so the same luck; the paired difference cancels it.
    """
    order = np.argsort(clusters, kind="stable")
    boundaries = np.flatnonzero(np.diff(clusters[order])) + 1
    members = np.split(order, boundaries)
    n_clusters = len(members)

    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(n_resamples):
        picked = rng.integers(0, n_clusters, size=n_clusters)
        index = np.concatenate([members[p] for p in picked])
        truth = y[index]
        if truth.sum() == 0:
            continue
        differences.append(
            average_precision_score(truth, score_a[index])
            - average_precision_score(truth, score_b[index])
        )

    values = np.asarray(differences)
    alpha = (1 - CONFIDENCE) / 2
    low, high = float(np.quantile(values, alpha)), float(np.quantile(values, 1 - alpha))
    return {
        "mean_difference": float(values.mean()),
        "ci_low": low,
        "ci_high": high,
        # The decision rule: if the interval spans zero, neither model is
        # established as better and simplicity breaks the tie.
        "interval_excludes_zero": bool(low > 0 or high < 0),
        "favours": "lgbm" if low > 0 else ("logreg" if high < 0 else "neither"),
    }


def measure_latency(model, X: np.ndarray) -> dict:
    sample = X[:LATENCY_ROWS]
    model.predict_proba(sample[:10])  # warm up
    timings = []
    for _ in range(LATENCY_REPEATS):
        started = time.perf_counter()
        model.predict_proba(sample)
        timings.append((time.perf_counter() - started) * 1000)
    return {
        "median_ms_per_1000_rows": float(np.median(timings)),
        "p90_ms_per_1000_rows": float(np.percentile(timings, 90)),
    }


def compare() -> dict:
    frame = load_val()
    failures = load_failures()
    results: dict = {"n_resamples": N_RESAMPLES, "components": {}, "artefacts": {}}

    for family in ("lgbm", "logreg"):
        sizes = [
            (MODELS_DIR / f"{component}_{family}.joblib").stat().st_size
            for component in COMPONENTS
        ]
        results["artefacts"][family] = {
            "bytes_per_component": sizes,
            "bytes_total": int(sum(sizes)),
        }

    for component in COMPONENTS:
        X, y_series = xy(frame, component)
        y = y_series.to_numpy()
        values = X.to_numpy(dtype=float)
        clusters = event_clusters(frame, component, failures, LABEL_HORIZON)

        scores, latency, brier = {}, {}, {}
        for family in ("lgbm", "logreg"):
            bundle = joblib.load(MODELS_DIR / f"{component}_{family}.joblib")
            model = bundle["model"]
            scores[family] = model.predict_proba(values)[:, 1]
            latency[family] = measure_latency(model, values)

            calibrator = calib.fit_isotonic(y, scores[family])
            calibrated = calib.apply_calibrator(calibrator, scores[family])
            reference = calib.base_rate_brier(y)
            raw_score = calib.brier(y, scores[family])
            calibrated_score = calib.brier(y, calibrated)
            brier[family] = {
                "raw": raw_score,
                "calibrated": calibrated_score,
                "base_rate_reference": reference,
                "skill_raw": 1 - raw_score / reference,
                "skill_calibrated": 1 - calibrated_score / reference,
            }

        results["components"][component] = {
            "pr_auc": {
                family: float(average_precision_score(y, scores[family]))
                for family in scores
            },
            "paired_difference_lgbm_minus_logreg": paired_difference(
                y, scores["lgbm"], scores["logreg"], clusters
            ),
            "brier": brier,
            "latency": latency,
        }

    return results


def main() -> None:
    results = compare()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("Validation PR-AUC and the paired difference (LightGBM - logistic regression)")
    print(f"{'comp':<8}{'lgbm':>9}{'logreg':>9}{'diff':>9}   {'95% CI':<22}{'decides':>10}")
    for component in COMPONENTS:
        record = results["components"][component]
        difference = record["paired_difference_lgbm_minus_logreg"]
        interval = f"({difference['ci_low']:+.3f}, {difference['ci_high']:+.3f})"
        print(
            f"{component:<8}{record['pr_auc']['lgbm']:>9.3f}"
            f"{record['pr_auc']['logreg']:>9.3f}"
            f"{difference['mean_difference']:>+9.3f}   {interval:<22}"
            f"{difference['favours']:>10}"
        )

    print("\nBrier skill after isotonic calibration (higher is better; 0 = base rate)")
    print(f"{'comp':<8}{'lgbm':>10}{'logreg':>10}")
    for component in COMPONENTS:
        b = results["components"][component]["brier"]
        print(
            f"{component:<8}{b['lgbm']['skill_calibrated']:>10.4f}"
            f"{b['logreg']['skill_calibrated']:>10.4f}"
        )

    print("\nInference latency, ms per 1,000 rows (median)")
    print(f"{'comp':<8}{'lgbm':>10}{'logreg':>10}")
    for component in COMPONENTS:
        lat = results["components"][component]["latency"]
        print(
            f"{component:<8}{lat['lgbm']['median_ms_per_1000_rows']:>10.2f}"
            f"{lat['logreg']['median_ms_per_1000_rows']:>10.2f}"
        )

    print("\nArtefact size, all four components")
    for family, record in results["artefacts"].items():
        print(f"  {family:<8}{record['bytes_total']:>12,} bytes")

    print(f"\nwrote {OUTPUT}")


if __name__ == "__main__":
    main()
