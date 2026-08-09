"""Sweep the prediction horizon and find where the signal stops.

`docs/MILESTONE_3B.md` section 3. Rebuilds labels at 24h, 72h, 7d, 14d and 30d,
refits on train and scores on validation, and runs the three-way ablation from
Milestone 3 -- telemetry-only, errors-only, combined -- because the collapse of
that interaction is the mechanism the sweep is looking for.

A single fixed hyperparameter setting is used at every horizon. The sweep is a
question about horizon, not about tuning, and re-running rolling-origin CV per
horizon would let a tuning difference masquerade as a horizon effect. The chosen
horizon is tuned properly afterwards, in `src/models/train.py`.

Writes `data/generated/horizon_sweep.json` and `docs/images/horizon_sweep.png`.

Run:  make horizon-sweep
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.eval.baselines import MATCHED_ERROR
from src.eval.datasets import load_train, load_val
from src.eval.horizons import SWEEP_HORIZONS, horizon_label, rebuild
from src.features.config import COMPONENTS, FEATURE_COLUMNS

DB = Path("data/pdm.db")
OUTPUT = Path("data/generated/horizon_sweep.json")
PLOT = Path("docs/images/horizon_sweep.png")

SEED = 20240607

#: Fixed for every horizon. See the module docstring.
LGBM_PARAMS = dict(
    n_estimators=200,
    num_leaves=31,
    learning_rate=0.05,
    random_state=SEED,
    deterministic=True,
    force_row_wise=True,
    n_jobs=-1,
    verbose=-1,
)

TELEMETRY = [
    c for c in FEATURE_COLUMNS
    if c.split("_")[0] in ("volt", "rotate", "pressure", "vibration")
]
ERRORS = [c for c in FEATURE_COLUMNS if "_count_" in c]

#: Minimum validation prediction times for a horizon's scores to be reported as
#: meaningful. Below this the embargo has eaten the validation month and the
#: numbers describe a single day, not a month.
MIN_VAL_PREDICTION_TIMES = 100


def fit_lgbm(X_train, y_train, X_val):
    model = lgb.LGBMClassifier(**LGBM_PARAMS).fit(X_train, y_train)
    return model.predict_proba(X_val)[:, 1]


def fit_logreg(X_train, y_train, X_val):
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=1.0, max_iter=500, random_state=SEED)),
        ]
    ).fit(X_train, y_train)
    return model.predict_proba(X_val)[:, 1]


def sweep() -> dict:
    connection = sqlite3.connect(DB)
    failures = pd.read_sql_query(
        "SELECT * FROM failures", connection, parse_dates=["datetime"]
    )
    connection.close()

    train_24h, val_24h = load_train(), load_val()
    results: dict = {"horizons": {}, "min_val_prediction_times": MIN_VAL_PREDICTION_TIMES}

    for horizon in SWEEP_HORIZONS:
        label = horizon_label(horizon)
        started = time.time()
        train = rebuild(train_24h, failures, "train", horizon)
        val = rebuild(val_24h, failures, "val", horizon)

        record: dict = {
            "train_rows": int(len(train)),
            "val_rows": int(len(val)),
            "val_prediction_times": int(val["datetime"].nunique()),
            "val_first": str(val["datetime"].min()),
            "val_last": str(val["datetime"].max()),
            "usable": int(val["datetime"].nunique()) >= MIN_VAL_PREDICTION_TIMES,
            "components": {},
        }

        X_train_all = train[FEATURE_COLUMNS].to_numpy(dtype=float)
        X_val_all = val[FEATURE_COLUMNS].to_numpy(dtype=float)

        for component in COMPONENTS:
            y_train = train[f"label_{component}"].to_numpy()
            y_val = val[f"label_{component}"].to_numpy()
            positive_rate = float(y_val.mean())

            code = f"{MATCHED_ERROR[component]}_count_24h"
            matched = (val[code].to_numpy(dtype=float) > 0).astype(float)

            scores = {
                "majority": np.zeros(len(val)),
                "matched_error_24h": matched,
            }
            if y_train.sum() and y_val.sum():
                scores["logreg"] = fit_logreg(X_train_all, y_train, X_val_all)
                scores["lgbm"] = fit_lgbm(X_train_all, y_train, X_val_all)

            entry = {
                "positive_rate": positive_rate,
                "n_positive": int(y_val.sum()),
                "pr_auc": {
                    name: (
                        float(average_precision_score(y_val, score))
                        if y_val.sum()
                        else float("nan")
                    )
                    for name, score in scores.items()
                },
            }

            # Three-way ablation, the mechanism check.
            if y_train.sum() and y_val.sum():
                ablation = {}
                for name, columns in (
                    ("telemetry_only", TELEMETRY),
                    ("errors_only", ERRORS),
                    ("combined", FEATURE_COLUMNS),
                ):
                    predicted = fit_lgbm(
                        train[columns].to_numpy(dtype=float),
                        y_train,
                        val[columns].to_numpy(dtype=float),
                    )
                    ablation[name] = float(average_precision_score(y_val, predicted))
                entry["ablation"] = ablation

            record["components"][component] = entry

        record["seconds"] = round(time.time() - started, 1)
        results["horizons"][label] = record
        print(
            f"  {label:>4}  train {record['train_rows']:>7,}  "
            f"val {record['val_rows']:>7,} ({record['val_prediction_times']:>5,} times)"
            f"  {'' if record['usable'] else '  VALIDATION TOO SMALL'}"
            f"  [{record['seconds']}s]",
            flush=True,
        )

    return results


def plot(results: dict, path: Path = PLOT) -> None:
    labels = [horizon_label(h) for h in SWEEP_HORIZONS]
    x = np.arange(len(labels))
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.5), dpi=140, sharex=True)

    series = [
        ("lgbm", "LightGBM", "#0072B2", "-", "o"),
        ("logreg", "logistic regression", "#D55E00", "--", "s"),
        ("matched_error_24h", "matched error code", "#009E73", "-.", "^"),
    ]

    for axis, component in zip(axes.ravel(), COMPONENTS):
        floors = []
        for name, legend, colour, dash, marker in series:
            values = [
                results["horizons"][label]["components"][component]["pr_auc"].get(
                    name, float("nan")
                )
                for label in labels
            ]
            axis.plot(x, values, color=colour, linestyle=dash, marker=marker,
                      markersize=4, linewidth=1.6, label=legend)
        floors = [
            results["horizons"][label]["components"][component]["positive_rate"]
            for label in labels
        ]
        axis.plot(x, floors, color="#666666", linestyle=(0, (3, 3)), linewidth=1.2,
                  label="no skill (positive rate)")

        for index, label in enumerate(labels):
            if not results["horizons"][label]["usable"]:
                axis.axvspan(index - 0.5, index + 0.5, color="#cccccc", alpha=0.45,
                             zorder=0)

        axis.set_title(component, fontsize=10)
        axis.set_ylim(-0.03, 1.03)
        axis.set_xticks(x)
        axis.set_xticklabels(labels)
        axis.grid(alpha=0.25, linewidth=0.5)

    axes[0, 0].legend(loc="lower left", fontsize=7, framealpha=0.9)
    for axis in axes[1, :]:
        axis.set_xlabel("prediction horizon")
    for axis in axes[:, 0]:
        axis.set_ylabel("PR-AUC (validation)")

    figure.suptitle(
        "PR-AUC against prediction horizon, validation\n"
        "shaded: embargo leaves too few validation prediction times to interpret",
        fontsize=11,
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)


def main() -> None:
    print("horizon sweep (train -> validation, fixed hyperparameters)")
    results = sweep()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    plot(results)
    print(f"\nwrote {OUTPUT} and {PLOT}")


if __name__ == "__main__":
    main()
