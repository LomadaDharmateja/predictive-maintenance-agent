"""Train one binary classifier per component, selected by rolling-origin CV.

`docs/MILESTONE_3.md` section 2. Four independent binary problems, logistic
regression and gradient boosting, hyperparameters chosen on expanding-window
folds inside the training split. The validation split is not touched here; it is
used by `src/eval/validate.py` for thresholds, calibration and importance. The
test split is not touched by anything except `src/eval/test_evaluation.py`.

Gradient boosting is LightGBM rather than XGBoost. The reason is throughput:
652,200 training rows and 80 fits in the search, where LightGBM's histogram
binning and leaf-wise growth fit in about three seconds against roughly three
times that for XGBoost's default `hist` on this shape. Nothing about the
comparison here depends on which one is used, and the feature matrix is dense
and numeric, so neither library's categorical handling is in play.

Class imbalance is a search dimension, not a decision made in advance.
`scale_pos_weight` takes the value 1 (untouched) or the negative/positive ratio,
and whichever wins on fold PR-AUC is what ships. Weighting is not free: it
inflates predicted probabilities away from the base rate, which is exactly what
section 4 has to measure. No resampling anywhere -- it distorts calibration for
the same reason and, unlike a weight, cannot be undone by a calibrator.

Run:  python -m src.models.train
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.eval.datasets import load_train, xy
from src.eval.folds import assert_folds_are_clean, rolling_origin_folds
from src.features.config import COMPONENTS, FEATURE_COLUMNS

MODELS_DIR = Path("models")
MANIFEST = Path("data/generated/build_manifest.json")
SUMMARY = MODELS_DIR / "training_summary.json"

# MLflow 3.15 put the filesystem store into maintenance mode and refuses to open
# a bare `file:./mlruns`. SQLite is the supported local backend; it stays inside
# mlruns/, which is gitignored, so nothing about the tracking layout leaks into
# the repository.
MLFLOW_DIR = Path("mlruns")
MLFLOW_URI = "sqlite:///mlruns/mlflow.db"
MLFLOW_ARTIFACTS = MLFLOW_DIR / "artifacts"
EXPERIMENT = "pdm-component-failure"

SEED = 20240604

#: Deliberately small. Every configuration costs four fits per component, and
#: with the signal this dataset carries (see docs/EVALUATION.md) the ranking is
#: stable well before a larger grid would change it.
LOGREG_GRID = [
    {"C": 1.0, "class_weight": None},
    {"C": 1.0, "class_weight": "balanced"},
    {"C": 0.05, "class_weight": None},
]

LGBM_GRID = [
    {"n_estimators": 300, "num_leaves": 31, "learning_rate": 0.05, "balanced": False},
    {"n_estimators": 300, "num_leaves": 31, "learning_rate": 0.05, "balanced": True},
    {
        "n_estimators": 500,
        "num_leaves": 63,
        "learning_rate": 0.03,
        "min_child_samples": 100,
        "balanced": False,
    },
]


@dataclass
class FoldScore:
    fold: int
    train_rows: int
    val_rows: int
    val_positives: int
    pr_auc: float


@dataclass
class Candidate:
    component: str
    family: str
    params: dict
    fold_scores: list[FoldScore]
    mean_pr_auc: float
    std_pr_auc: float


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - a missing git is not a training failure
        return "unknown"


def data_content_hash() -> dict[str, str]:
    """The content hashes from the feature build.

    A run that cannot be traced back to an exact dataset state is not a result,
    so this is logged with every MLflow run and written into the summary.
    """
    if not MANIFEST.exists():
        return {}
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {
        split: entry["sha256"]
        for split, entry in payload.get("features", {}).items()
    }


def build_logreg(params: dict) -> Pipeline:
    """Standardisation is required, not cosmetic: `age` spans 0-20 while
    `rotate_mean_24h` sits around 450, and an unscaled L2 penalty would
    effectively regularise them by wildly different amounts."""
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=params["C"],
                    class_weight=params["class_weight"],
                    max_iter=500,
                    solver="lbfgs",
                    random_state=SEED,
                ),
            ),
        ]
    )


def build_lgbm(params: dict, y_train: np.ndarray) -> lgb.LGBMClassifier:
    positive = int(np.sum(y_train))
    negative = len(y_train) - positive
    scale = (negative / positive) if (params["balanced"] and positive) else 1.0
    return lgb.LGBMClassifier(
        n_estimators=params["n_estimators"],
        num_leaves=params["num_leaves"],
        learning_rate=params["learning_rate"],
        min_child_samples=params.get("min_child_samples", 20),
        scale_pos_weight=scale,
        random_state=SEED,
        deterministic=True,
        force_row_wise=True,
        n_jobs=-1,
        verbose=-1,
    )


def fit_family(family: str, params: dict, X, y):
    model = build_logreg(params) if family == "logreg" else build_lgbm(params, y)
    model.fit(X, y)
    return model


def evaluate_candidate(
    family: str, params: dict, component: str, frame: pd.DataFrame, folds
) -> Candidate:
    X, y = xy(frame, component)
    X_values, y_values = X.to_numpy(), y.to_numpy()

    scores: list[FoldScore] = []
    for fold in folds:
        y_fold_val = y_values[fold.val_rows]
        if y_fold_val.sum() == 0:
            # A fold with no positives cannot produce a PR-AUC. Skipping it and
            # saying so beats averaging in a NaN or a zero.
            continue
        model = fit_family(
            family, params, X_values[fold.train_rows], y_values[fold.train_rows]
        )
        proba = model.predict_proba(X_values[fold.val_rows])[:, 1]
        scores.append(
            FoldScore(
                fold=fold.index,
                train_rows=len(fold.train_rows),
                val_rows=len(fold.val_rows),
                val_positives=int(y_fold_val.sum()),
                pr_auc=float(average_precision_score(y_fold_val, proba)),
            )
        )

    values = [s.pr_auc for s in scores]
    return Candidate(
        component=component,
        family=family,
        params=params,
        fold_scores=scores,
        mean_pr_auc=float(np.mean(values)) if values else float("nan"),
        std_pr_auc=float(np.std(values)) if values else float("nan"),
    )


def _is_weighted(candidate: Candidate) -> bool:
    return bool(
        candidate.params.get("balanced") or candidate.params.get("class_weight")
    )


def _capacity(candidate: Candidate) -> float:
    """A rough size ordering, used only to break exact ties."""
    return float(
        candidate.params.get("n_estimators", 0) * candidate.params.get("num_leaves", 1)
    )


def select_best(candidates: list[Candidate]) -> Candidate:
    """Highest CV PR-AUC, with ties broken deliberately rather than by list order.

    Several configurations reach 1.0000 on this dataset, so the tie-break is not
    hypothetical -- it decides what ships. In order:

    1. Highest mean CV PR-AUC.
    2. Prefer the *unweighted* model. Class weighting pushes predicted
       probabilities away from the base rate, and section 4 of the milestone
       depends on those probabilities meaning something. If weighting buys no
       PR-AUC, it costs calibration for nothing.
    3. Prefer the smaller model, on the usual grounds.
    4. Lowest fold-to-fold standard deviation.
    """
    return min(
        candidates,
        key=lambda c: (
            -c.mean_pr_auc,
            _is_weighted(c),
            _capacity(c),
            c.std_pr_auc,
        ),
    )


def train_all(models_dir: Path = MODELS_DIR, quiet: bool = False) -> dict:
    def say(message: str) -> None:
        if not quiet:
            print(message, flush=True)

    frame = load_train()
    folds = rolling_origin_folds(frame["datetime"])
    assert_folds_are_clean(folds)

    say(f"training rows: {len(frame):,}   features: {len(FEATURE_COLUMNS)}")
    say(f"rolling-origin folds: {len(folds)}")
    for fold in folds:
        say(f"  {fold.describe()}")

    MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
    MLFLOW_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(
            EXPERIMENT, artifact_location=MLFLOW_ARTIFACTS.resolve().as_uri()
        )
    mlflow.set_experiment(EXPERIMENT)
    commit = git_commit()
    hashes = data_content_hash()
    models_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "git_commit": commit,
        "data_content_hash": hashes,
        "seed": SEED,
        "n_features": len(FEATURE_COLUMNS),
        "features": FEATURE_COLUMNS,
        "folds": [
            {
                "index": f.index,
                "train_end": str(f.train_end),
                "val_start": str(f.val_start),
                "val_end": str(f.val_end),
                "train_rows": len(f.train_rows),
                "val_rows": len(f.val_rows),
            }
            for f in folds
        ],
        "components": {},
    }

    for component in COMPONENTS:
        say(f"\n=== {component} ===")
        X, y = xy(frame, component)
        candidates: list[Candidate] = []

        for family, grid in (("logreg", LOGREG_GRID), ("lgbm", LGBM_GRID)):
            for params in grid:
                started = time.time()
                candidate = evaluate_candidate(family, params, component, frame, folds)
                candidates.append(candidate)
                say(
                    f"  {family:<7} {str(params):<86} "
                    f"CV PR-AUC {candidate.mean_pr_auc:.4f} "
                    f"+/-{candidate.std_pr_auc:.4f}  ({time.time()-started:.0f}s)"
                )

        component_record = {"candidates": [], "selected": {}}

        for family in ("logreg", "lgbm"):
            family_candidates = [c for c in candidates if c.family == family]
            best = select_best(family_candidates)

            model = fit_family(family, best.params, X.to_numpy(), y.to_numpy())
            path = models_dir / f"{component}_{family}.joblib"
            joblib.dump(
                {
                    "model": model,
                    "feature_columns": FEATURE_COLUMNS,
                    "component": component,
                    "family": family,
                    "params": best.params,
                    "git_commit": commit,
                    "data_content_hash": hashes,
                },
                path,
            )

            with mlflow.start_run(run_name=f"{component}-{family}"):
                mlflow.log_params({f"hp_{k}": v for k, v in best.params.items()})
                mlflow.log_params(
                    {
                        "component": component,
                        "family": family,
                        "seed": SEED,
                        "n_features": len(FEATURE_COLUMNS),
                        "n_train_rows": len(frame),
                        "n_folds": len(folds),
                        "git_commit": commit,
                        **{f"data_sha256_{k}": v for k, v in hashes.items()},
                    }
                )
                mlflow.log_metric("cv_pr_auc_mean", best.mean_pr_auc)
                mlflow.log_metric("cv_pr_auc_std", best.std_pr_auc)
                for score in best.fold_scores:
                    mlflow.log_metric(f"cv_pr_auc_fold{score.fold}", score.pr_auc)
                mlflow.log_dict(
                    {"feature_columns": FEATURE_COLUMNS}, "feature_columns.json"
                )
                mlflow.log_dict(
                    {
                        "candidates": [asdict(c) for c in family_candidates],
                        "selected": asdict(best),
                    },
                    "search.json",
                )
                run_id = mlflow.active_run().info.run_id

            component_record["selected"][family] = {
                "params": best.params,
                "cv_pr_auc_mean": best.mean_pr_auc,
                "cv_pr_auc_std": best.std_pr_auc,
                "fold_scores": [asdict(s) for s in best.fold_scores],
                "mlflow_run_id": run_id,
                "artefact": path.as_posix(),
            }
            say(f"  selected {family}: CV PR-AUC {best.mean_pr_auc:.4f} -> {path}")

        component_record["candidates"] = [asdict(c) for c in candidates]
        summary["components"][component] = component_record

    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    say(f"\nwrote {SUMMARY}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train per-component classifiers.")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    args = parser.parse_args()
    train_all(args.models_dir)


if __name__ == "__main__":
    main()
