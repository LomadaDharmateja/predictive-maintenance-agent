"""Calibrated failure risk, with an honest statement of whether it is actionable.

`docs/MILESTONE_4.md` section 2. Two rules govern this module.

**Only calibrated probabilities leave it.** The raw model score is never placed
on an output model and never returned. On the shipped logistic-regression models
the *raw* Brier skill is negative for comp1 (-0.116) and comp2 (-0.764): those
numbers carry more squared error than predicting the base rate, so they are not
probabilities. Isotonic scaling repairs the sign -- after calibration every
component is at or above zero -- but only comp2's held-out skill is established
as better than the base rate, so the other three carry `calibrated=False`.
`scripts/calibration_check.py` is the measurement; `tests/test_agent.py` asserts
no output schema exposes a raw score.

**Every risk carries a `warning_adequacy`.** A probability an operator cannot act
on is worse than no probability, because it invites an order that will arrive
after the failure. Adequacy compares the component's *measured* effective
detection lead against each part's lead time, and it reads `insufficient` for
most pairs. That is the finding, not a bug to route around.

Features come from `src.features`, using the Milestone 2 `as_of` discipline
unchanged -- `compute_features` is the same function the training set was built
with, so a serving/training skew is not possible by construction.
"""

from __future__ import annotations

import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from src.agent import db
from src.agent.contracts import (
    ComponentRisk,
    ErrorCode,
    FailureRisk,
    FailureRiskInput,
    PartAdequacy,
    Success,
    ToolError,
    ToolResult,
    WarningAdequacy,
)
from src.features.compute import compute_features
from src.features.config import (
    COMPONENTS,
    FEATURE_COLUMNS,
    LABEL_HORIZON,
    PRODUCTION_FAMILY,
)
from src.features.store import FeatureStore, StoreError

TOOL_NAME = "get_failure_risk"

MODELS_DIR = Path("models")
GENERATED = Path("data/generated")
VALIDATION_RESULTS = GENERATED / "validation_results.json"
DETECTION_LEAD = GENERATED / "detection_lead_time.json"
CALIBRATION_CHECK = GENERATED / "calibration_check.json"
CALIBRATORS = MODELS_DIR / "calibrators.joblib"

#: A warning is marginal rather than sufficient if it clears the lead time by
#: less than this factor. 1.25 leaves a quarter of the lead time as slack for
#: raising the order and receiving the part; it is a judgement, stated here so it
#: can be argued with rather than discovered in the code.
MARGINAL_FACTOR = 1.25

CAVEAT = (
    "Calibrated probability that this component fails within 14 days. It does "
    "not say when inside that window, or how severe. Trained on simulated data "
    "(Microsoft Azure PdM); no claim about real plant behaviour is supported. "
    "Check warning_adequacy before treating any of this as actionable, and "
    "check `calibrated`: where it is false, the probability is not established "
    "as better than simply reporting the base rate and must not be quoted as a "
    "reliable likelihood."
)


class RiskUnavailable(RuntimeError):
    """Model artefacts or their supporting measurements are missing."""


@lru_cache(maxsize=1)
def _artefacts() -> dict:
    """Load models, calibrators, thresholds and lead times once.

    Cached: v1 reloaded a 6.9 MB pickle on every single tool call.
    """
    missing = [
        path
        for path in (VALIDATION_RESULTS, DETECTION_LEAD, CALIBRATORS, CALIBRATION_CHECK)
        if not path.exists()
    ]
    if missing:
        raise RiskUnavailable(
            f"missing artefacts: {', '.join(str(p) for p in missing)}. "
            "Run `make train`, `make evaluate` and `make horizon-decision`."
        )

    models = {}
    for component in COMPONENTS:
        path = MODELS_DIR / f"{component}_{PRODUCTION_FAMILY}.joblib"
        if not path.exists():
            raise RiskUnavailable(f"missing model artefact: {path}")
        bundle = joblib.load(path)
        if bundle["feature_columns"] != FEATURE_COLUMNS:
            raise RiskUnavailable(
                f"{path} was trained on a different feature set; retrain."
            )
        models[component] = bundle["model"]

    return {
        "models": models,
        "calibrators": joblib.load(CALIBRATORS),
        "validation": json.loads(VALIDATION_RESULTS.read_text(encoding="utf-8")),
        "detection": json.loads(DETECTION_LEAD.read_text(encoding="utf-8")),
        "calibration": json.loads(CALIBRATION_CHECK.read_text(encoding="utf-8")),
    }


@lru_cache(maxsize=1)
def _store(database: str) -> FeatureStore:
    return FeatureStore.from_database(Path(database))


def _adequacy(detection_hours: float | None, lead_time_days: int) -> WarningAdequacy:
    """Is the measured warning long enough for this part's lead time?"""
    if detection_hours is None:
        return WarningAdequacy.INSUFFICIENT
    needed = lead_time_days * 24
    if detection_hours >= needed * MARGINAL_FACTOR:
        return WarningAdequacy.SUFFICIENT
    if detection_hours >= needed:
        return WarningAdequacy.MARGINAL
    return WarningAdequacy.INSUFFICIENT


def _best(adequacies: list[WarningAdequacy]) -> WarningAdequacy:
    """Component-level adequacy is the best any of its parts achieves.

    Best rather than worst: if one part can be ordered in time, that is a real
    option, and reporting the component as `insufficient` would hide it. The
    per-part breakdown is returned alongside so the summary never stands alone.
    """
    for level in (
        WarningAdequacy.SUFFICIENT,
        WarningAdequacy.MARGINAL,
        WarningAdequacy.INSUFFICIENT,
    ):
        if level in adequacies:
            return level
    return WarningAdequacy.INSUFFICIENT


def get_failure_risk(
    payload: FailureRiskInput, database: Path = db.DEFAULT_DB
) -> ToolResult[FailureRisk]:
    try:
        artefacts = _artefacts()
        store = _store(str(database))
    except RiskUnavailable as exc:
        return ToolError(
            code=ErrorCode.MODEL_UNAVAILABLE, message=str(exc), tool=TOOL_NAME
        )
    except (StoreError, db.DatabaseUnavailable) as exc:
        return ToolError(
            code=ErrorCode.DATABASE_ERROR, message=str(exc), tool=TOOL_NAME
        )

    try:
        features = compute_features(store, pd.Timestamp(payload.as_of))
    except StoreError as exc:
        # An off-grid as_of is a caller error, not a system failure.
        return ToolError(
            code=ErrorCode.INVALID_INPUT, message=str(exc), tool=TOOL_NAME
        )

    if payload.machine_id not in features.index:
        return ToolError(
            code=ErrorCode.NOT_FOUND,
            message=f"machine {payload.machine_id} is not in the fleet",
            tool=TOOL_NAME,
        )

    row = features.loc[[payload.machine_id], FEATURE_COLUMNS]
    parts = db.fetch(db.PARTS_FOR_COMPONENT, db=database)
    validation = artefacts["validation"]["components"]
    detection = artefacts["detection"]

    risks: list[ComponentRisk] = []
    for component in COMPONENTS:
        raw = artefacts["models"][component].predict_proba(
            row.to_numpy(dtype=float)
        )[:, 1]
        # The raw score is consumed here and never leaves this scope.
        calibrated = float(
            artefacts["calibrators"][component]["isotonic"].predict(raw)[0]
        )

        record = validation[component]
        threshold = float(record["thresholds"]["10"]["threshold"])
        interval = record["series"][PRODUCTION_FAMILY]["ci"]["pr_auc"]
        lead_hours = detection.get(component, {}).get("median_hours")
        quality = artefacts["calibration"]["components"][component]

        per_part = [
            PartAdequacy(
                part_id=part["part_id"],
                lead_time_days=part["lead_time_days"],
                adequacy=_adequacy(lead_hours, part["lead_time_days"]),
                detection_lead_hours=lead_hours,
            )
            for part in parts
            if part["component"] == component
        ]

        risks.append(
            ComponentRisk(
                component=component,
                calibrated_probability=round(calibrated, 4),
                calibrated=bool(quality["calibrated"]),
                brier_skill_holdout=round(
                    quality["skill_calibrated_held_out"], 4
                ),
                brier_skill_ci_low=round(quality["skill_ci_low"], 4),
                brier_skill_ci_high=round(quality["skill_ci_high"], 4),
                # The interval is the model's PR-AUC interval, not a per-row
                # predictive interval. Named as such on the schema so it is not
                # mistaken for uncertainty about this machine.
                confidence_interval_low=round(max(0.0, interval[0]), 4),
                confidence_interval_high=round(min(1.0, interval[1]), 4),
                threshold=threshold,
                threshold_cost_ratio=float(
                    record["thresholds"]["10"]["cost_ratio"]
                ),
                exceeds_threshold=calibrated >= threshold,
                warning_adequacy=_best([p.adequacy for p in per_part]),
                per_part_adequacy=per_part,
            )
        )

    return Success(
        data=FailureRisk(
            machine_id=payload.machine_id,
            as_of=payload.as_of,
            horizon_days=int(LABEL_HORIZON.total_seconds() // 86400),
            model_family=PRODUCTION_FAMILY,
            components=risks,
            caveat=CAVEAT,
        )
    )
