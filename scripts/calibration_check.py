"""Is the shipped model's calibrated probability better than the base rate?

`docs/MILESTONE_5.md` section 0, blocking. The Milestone 4 summary reported
negative Brier skill for comp1 and comp2 under logistic regression without saying
whether that was before or after isotonic calibration. It was **before**. This
answers the question that actually matters: after calibration, on data the
calibrator has not seen.

Method. Isotonic fitted on all of validation and measured on the same rows is
in-sample and flatters itself, so this cross-fits: validation is cut into `K_FOLDS`
contiguous time blocks, and each block is scored by a calibrator fitted on the
other blocks only. Every validation row therefore gets an out-of-fold calibrated
probability, and the skill computed from them is a held-out estimate of the
shipped calibrator's quality.

Contiguous blocks, not random ones: adjacent hours are near-identical
(`docs/DATA.md` section 5.2), and a random fold would let the calibrator see a
row's own neighbours.

The test split is not used. It is locked, has been read twice, and this is a
diagnostic on an already-final model -- not a reason to open it again.

Brier skill = 1 - Brier / (p(1-p)). Zero means "no better than predicting the
base rate". Negative means worse.

Writes data/generated/calibration_check.json.

Run:  make calibration-check
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.eval import calibration as calib
from src.eval.datasets import load_val, xy
from src.eval.metrics import BOOTSTRAP_SEED, event_clusters
from src.features.config import COMPONENTS, LABEL_HORIZON, PRODUCTION_FAMILY

DB = Path("data/pdm.db")
MODELS_DIR = Path("models")
OUTPUT = Path("data/generated/calibration_check.json")

K_FOLDS = 5
N_RESAMPLES = 600
CONFIDENCE = 0.95

#: Below this, the calibrated probability is treated as untrustworthy and the
#: agent must not present it as a probability.
#:
#: Zero is the natural boundary -- it is exactly "no better than the base rate" --
#: but a point estimate that lands at -0.001 is not meaningfully different from
#: one at +0.001. The rule used here is stricter and does not depend on where the
#: point estimate falls: a component is trusted only if its held-out skill is
#: positive **and** its bootstrap interval excludes zero. Anything else is
#: reported as uncalibrated.
SKILL_FLOOR = 0.0


def load_failures() -> pd.DataFrame:
    connection = sqlite3.connect(DB)
    try:
        return pd.read_sql_query(
            "SELECT * FROM failures", connection, parse_dates=["datetime"]
        )
    finally:
        connection.close()


def cross_fitted_probabilities(
    y: np.ndarray, raw: np.ndarray, when: np.ndarray
) -> np.ndarray:
    """Out-of-fold isotonic probabilities over contiguous time blocks."""
    order = np.argsort(when, kind="stable")
    blocks = np.array_split(order, K_FOLDS)
    calibrated = np.empty_like(raw, dtype=float)

    for index, block in enumerate(blocks):
        held_in = np.concatenate([b for i, b in enumerate(blocks) if i != index])
        if y[held_in].sum() == 0 or y[held_in].sum() == len(held_in):
            # A fold with one class cannot fit a calibrator; fall back to the
            # in-fold base rate rather than silently emitting nonsense.
            calibrated[block] = float(y[held_in].mean())
            continue
        model = calib.fit_isotonic(y[held_in], raw[held_in])
        calibrated[block] = calib.apply_calibrator(model, raw[block])

    return calibrated


def skill_interval(
    y: np.ndarray, probabilities: np.ndarray, clusters: np.ndarray
) -> tuple[float, float]:
    """Bootstrap interval on Brier skill, resampled at the failure-event level."""
    order = np.argsort(clusters, kind="stable")
    boundaries = np.flatnonzero(np.diff(clusters[order])) + 1
    members = np.split(order, boundaries)
    n_clusters = len(members)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = []
    for _ in range(N_RESAMPLES):
        index = np.concatenate(
            [members[p] for p in rng.integers(0, n_clusters, size=n_clusters)]
        )
        truth = y[index]
        reference = calib.base_rate_brier(truth)
        if reference == 0:
            continue
        values.append(1 - calib.brier(truth, probabilities[index]) / reference)

    array = np.asarray(values)
    alpha = (1 - CONFIDENCE) / 2
    return float(np.quantile(array, alpha)), float(np.quantile(array, 1 - alpha))


def check() -> dict:
    frame = load_val()
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    failures = load_failures()
    when = frame["datetime"].to_numpy()

    results: dict = {
        "model_family": PRODUCTION_FAMILY,
        "k_folds": K_FOLDS,
        "n_resamples": N_RESAMPLES,
        "skill_floor": SKILL_FLOOR,
        "components": {},
    }

    for component in COMPONENTS:
        X, y_series = xy(frame, component)
        y = y_series.to_numpy()
        bundle = joblib.load(MODELS_DIR / f"{component}_{PRODUCTION_FAMILY}.joblib")
        raw = bundle["model"].predict_proba(X.to_numpy(dtype=float))[:, 1]

        out_of_fold = cross_fitted_probabilities(y, raw, when)
        reference = calib.base_rate_brier(y)

        raw_skill = 1 - calib.brier(y, raw) / reference
        held_out_skill = 1 - calib.brier(y, out_of_fold) / reference
        low, high = skill_interval(
            y, out_of_fold, event_clusters(frame, component, failures, LABEL_HORIZON)
        )

        trusted = held_out_skill > SKILL_FLOOR and low > SKILL_FLOOR
        results["components"][component] = {
            "base_rate": float(y.mean()),
            "brier_base_rate": reference,
            "brier_raw": calib.brier(y, raw),
            "brier_calibrated_held_out": calib.brier(y, out_of_fold),
            "skill_raw": raw_skill,
            "skill_calibrated_held_out": held_out_skill,
            "skill_ci_low": low,
            "skill_ci_high": high,
            "calibrated": bool(trusted),
        }

    return results


def main() -> None:
    results = check()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Brier skill, {PRODUCTION_FAMILY}, {K_FOLDS}-fold cross-fitted isotonic")
    print("(0 = no better than predicting the base rate; negative = worse)\n")
    print(f"{'comp':<8}{'base rate':>11}{'skill raw':>12}{'skill calib':>13}"
          f"   {'95% CI':<22}{'trusted':>9}")
    for component, record in results["components"].items():
        interval = f"({record['skill_ci_low']:+.4f}, {record['skill_ci_high']:+.4f})"
        print(
            f"{component:<8}{record['base_rate']:>11.4f}{record['skill_raw']:>12.4f}"
            f"{record['skill_calibrated_held_out']:>13.4f}   {interval:<22}"
            f"{'yes' if record['calibrated'] else 'NO':>9}"
        )

    untrusted = [c for c, r in results["components"].items() if not r["calibrated"]]
    print()
    if untrusted:
        print(f"NOT TRUSTWORTHY AS PROBABILITIES: {', '.join(untrusted)}")
        print("get_failure_risk must mark these with calibrated=false.")
    else:
        print("All components clear the base-rate floor.")
    print(f"\nwrote {OUTPUT}")


if __name__ == "__main__":
    main()
