"""The three measurements the horizon choice turns on.

`docs/MILESTONE_3B.md` section 4. Run after `make horizon-sweep`.

1. **Bootstrap intervals per horizon.** Constraint 2 asks whether the model beats
   the matched-code baseline by more than the interval, so the interval has to be
   computed, not eyeballed off a point estimate.
2. **A 30-day diagnostic on a re-partitioned split.** At 30 days the embargo
   leaves the shipped validation month with 24 prediction times on a single day,
   which cannot answer anything. Since 30 days is the shortest horizon covering
   the median 23-day lead time, the question has to be answered somewhere: this
   widens validation to two months for that measurement only. It is a diagnostic
   and is never used to select a model or to score the test split.
3. **Effective detection lead time.** The label horizon is an upper bound on
   warning, not the warning itself. What matters operationally is how far ahead
   of a failure the score actually crosses the threshold, because that is the
   number a lead time has to fit inside.

Writes horizon_ci.json, horizon_30d_diagnostic.json and detection_lead_time.json
under data/generated/.

Run:  make horizon-decision
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.eval.baselines import MATCHED_ERROR
from src.eval.datasets import load_train, load_val
from src.eval.horizons import horizon_label, label_at_horizon, rebuild
from src.eval.metrics import bootstrap_intervals, event_clusters
from src.eval.thresholds import DEFAULT_COST_RATIO, select_threshold
from src.features.config import COMPONENTS, FEATURE_COLUMNS, LAST_OBSERVED_FAILURE

DB = Path("data/pdm.db")
GENERATED = Path("data/generated")

SEED = 20240607
N_RESAMPLES = 400

#: Horizons the interval comparison is run at. 30d is excluded here because its
#: validation window is unusable on the shipped splits; it gets the diagnostic.
CI_HORIZONS = [pd.Timedelta(hours=72), pd.Timedelta(days=7), pd.Timedelta(days=14)]

#: Re-partitioned split for the 30-day diagnostic only.
DIAGNOSTIC_HORIZON = pd.Timedelta(days=30)
DIAGNOSTIC_TRAIN = (pd.Timestamp("2015-01-01"), pd.Timestamp("2015-06-30 23:59:59"))
DIAGNOSTIC_VAL = (pd.Timestamp("2015-07-01"), pd.Timestamp("2015-09-30 23:59:59"))

LGBM = dict(
    n_estimators=200,
    num_leaves=31,
    learning_rate=0.05,
    random_state=SEED,
    deterministic=True,
    force_row_wise=True,
    n_jobs=-1,
    verbose=-1,
)


def load_failures() -> pd.DataFrame:
    connection = sqlite3.connect(DB)
    try:
        return pd.read_sql_query(
            "SELECT * FROM failures", connection, parse_dates=["datetime"]
        )
    finally:
        connection.close()


def fit_and_score(train: pd.DataFrame, val: pd.DataFrame, component: str):
    y_train = train[f"label_{component}"].to_numpy()
    model = lgb.LGBMClassifier(**LGBM).fit(
        train[FEATURE_COLUMNS].to_numpy(dtype=float), y_train
    )
    return model.predict_proba(val[FEATURE_COLUMNS].to_numpy(dtype=float))[:, 1]


def matched_code_scores(frame: pd.DataFrame, component: str) -> np.ndarray:
    column = f"{MATCHED_ERROR[component]}_count_24h"
    return (frame[column].to_numpy(dtype=float) > 0).astype(float)


def compare(train, val, component, horizon, failures) -> dict:
    y = val[f"label_{component}"].to_numpy()
    code = matched_code_scores(val, component)
    model = fit_and_score(train, val, component)
    clusters = event_clusters(val, component, failures, horizon)

    ci_code = bootstrap_intervals(y, code, clusters, 0.5, n_resamples=N_RESAMPLES)
    ci_model = bootstrap_intervals(y, model, clusters, 0.5, n_resamples=N_RESAMPLES)
    return {
        "floor": float(y.mean()),
        "n_positive": int(y.sum()),
        "matched": float(average_precision_score(y, code)),
        "ci_matched": list(ci_code["pr_auc"]),
        "lgbm": float(average_precision_score(y, model)),
        "ci_lgbm": list(ci_model["pr_auc"]),
        # Constraint 2 is satisfied only when the intervals do not overlap.
        "intervals_overlap": bool(ci_model["pr_auc"][0] <= ci_code["pr_auc"][1]),
    }


def detection_lead_times(train, val, horizon, failures) -> dict:
    """How far ahead of each failure the score first crosses the threshold."""
    out = {}
    for component in COMPONENTS:
        y = val[f"label_{component}"].to_numpy()
        scores = fit_and_score(train, val, component)
        threshold = select_threshold(
            y, scores, component, DEFAULT_COST_RATIO
        ).threshold

        firing = val[["machineID", "datetime"]].copy()
        firing["score"] = scores
        events = failures[failures["failure"] == component]

        leads, caught, total = [], 0, 0
        for machine, when in zip(events["machineID"], events["datetime"]):
            window = firing[
                (firing["machineID"] == machine)
                & (firing["datetime"] > when - horizon)
                & (firing["datetime"] < when)
            ]
            if window.empty:
                continue
            total += 1
            fired = window[window["score"] >= threshold]
            if not fired.empty:
                caught += 1
                leads.append(
                    (when - fired["datetime"].min()).total_seconds() / 3600
                )

        values = np.asarray(leads, dtype=float)
        out[component] = {
            "threshold": float(threshold),
            "events_in_window": total,
            "events_detected": caught,
            "detection_rate": caught / total if total else float("nan"),
            "p10_hours": float(np.percentile(values, 10)) if len(values) else None,
            "median_hours": float(np.median(values)) if len(values) else None,
            "p90_hours": float(np.percentile(values, 90)) if len(values) else None,
        }
    return out


def main() -> None:
    failures = load_failures()
    train_base, val_base = load_train(), load_val()

    print("1. bootstrap intervals per horizon (shipped splits)")
    intervals = {}
    for horizon in CI_HORIZONS:
        label = horizon_label(horizon)
        train = rebuild(train_base, failures, "train", horizon)
        val = rebuild(val_base, failures, "val", horizon)
        intervals[label] = {
            "val_rows": int(len(val)),
            "val_prediction_times": int(val["datetime"].nunique()),
            "components": {
                component: compare(train, val, component, horizon, failures)
                for component in COMPONENTS
            },
        }
        overlaps = [
            c
            for c, r in intervals[label]["components"].items()
            if r["intervals_overlap"]
        ]
        print(f"   {label:>4}: intervals overlap on {overlaps or 'nothing'}")
    (GENERATED / "horizon_ci.json").write_text(
        json.dumps(intervals, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("\n2. 30-day diagnostic on a re-partitioned split")
    combined = pd.concat([train_base, val_base], ignore_index=True)
    combined["datetime"] = pd.to_datetime(combined["datetime"])

    def slice_for(bounds) -> pd.DataFrame:
        low, high = bounds
        latest = min(
            high - DIAGNOSTIC_HORIZON, LAST_OBSERVED_FAILURE - DIAGNOSTIC_HORIZON
        )
        part = combined[
            (combined["datetime"] >= low) & (combined["datetime"] <= latest)
        ].copy()
        labels = label_at_horizon(part, failures, DIAGNOSTIC_HORIZON)
        for component in COMPONENTS:
            part[f"label_{component}"] = labels[f"label_{component}"].to_numpy()
        return part

    diag_train, diag_val = slice_for(DIAGNOSTIC_TRAIN), slice_for(DIAGNOSTIC_VAL)
    diagnostic = {
        "note": "re-partitioned split, diagnostic only; never used for selection "
        "or for the test evaluation",
        "train_rows": int(len(diag_train)),
        "val_rows": int(len(diag_val)),
        "val_prediction_times": int(diag_val["datetime"].nunique()),
        "components": {
            component: compare(
                diag_train, diag_val, component, DIAGNOSTIC_HORIZON, failures
            )
            for component in COMPONENTS
        },
    }
    overlaps = [
        c for c, r in diagnostic["components"].items() if r["intervals_overlap"]
    ]
    print(f"   30d: intervals overlap on {overlaps or 'nothing'}")
    (GENERATED / "horizon_30d_diagnostic.json").write_text(
        json.dumps(diagnostic, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("\n3. effective detection lead time at 14 days")
    horizon = pd.Timedelta(days=14)
    leads = detection_lead_times(
        rebuild(train_base, failures, "train", horizon),
        rebuild(val_base, failures, "val", horizon),
        horizon,
        failures,
    )
    for component, record in leads.items():
        median = record["median_hours"]
        print(
            f"   {component}: detected {record['events_detected']}/"
            f"{record['events_in_window']}, median lead "
            f"{median:.0f}h" if median else f"   {component}: none detected"
        )
    (GENERATED / "detection_lead_time.json").write_text(
        json.dumps(leads, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("\nwrote horizon_ci.json, horizon_30d_diagnostic.json, detection_lead_time.json")


if __name__ == "__main__":
    main()
