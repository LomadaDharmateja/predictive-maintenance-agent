"""Parts position, from stock and consumption. Never from a prediction.

`docs/MILESTONE_4.md` section 2, and the Milestone 3B conclusion expressed as an
import graph rather than as a comment.

**This module imports no model, no feature code, and no risk code.** It reaches
`src.agent.contracts` and `src.agent.db` and nothing else in the project.
`tests/test_agent_parts_independence.py` walks the transitive import closure and
fails if joblib, sklearn, lightgbm, `src.features` or `src.eval` appear anywhere
in it. That test is the point of the module boundary; the docstring is not.

Why: on this data the model cannot buy enough notice to order most parts. Its
effective detection lead is a median of ~14 days for comp2/comp3/comp4 and ~24
hours for comp1, against supplier lead times of 10 to 34 days -- 1 part in 9 can
be ordered in time (`docs/SIGNAL_ANALYSIS.md` section 4). Reordering therefore
has to run off stock on hand and observed consumption, both of which are
measurements available now and need no forecast.
"""

from __future__ import annotations

from pathlib import Path

from src.agent import db
from src.agent.contracts import (
    ErrorCode,
    PartPosition,
    PartsPosition,
    PartsPositionInput,
    Success,
    ToolError,
    ToolResult,
)

TOOL_NAME = "get_parts_position"

#: Consumption is reported per 30 days so it is comparable with lead times,
#: which are quoted in days.
CONSUMPTION_WINDOW_DAYS = 30.0

BASIS = (
    "Stock on hand and consumption observed from the replacement log. "
    "No risk score or prediction is used: the model's effective warning is "
    "shorter than the lead time for 8 of 9 parts, so predictions cannot drive "
    "reordering. See docs/SIGNAL_ANALYSIS.md section 4."
)


def _consumption_rates(database: Path) -> dict[str, float]:
    """Replacements per 30 days per component, from the maintenance log."""
    rows = db.fetch(db.CONSUMPTION_BY_COMPONENT, db=database)
    rates: dict[str, float] = {}
    for row in rows:
        first, last = row["first_seen"], row["last_seen"]
        if not first or not last:
            continue
        from datetime import datetime

        span_days = (
            datetime.fromisoformat(last) - datetime.fromisoformat(first)
        ).total_seconds() / 86400
        if span_days <= 0:
            continue
        rates[row["comp"]] = row["n"] / span_days * CONSUMPTION_WINDOW_DAYS
    return rates


def get_parts_position(
    payload: PartsPositionInput, database: Path = db.DEFAULT_DB
) -> ToolResult[PartsPosition]:
    """Stock, cost, supplier, lead time and observed consumption per part."""
    try:
        rows = db.fetch(db.PARTS_FOR_COMPONENT, db=database)
        rates = _consumption_rates(database)
    except db.DatabaseUnavailable as exc:
        return ToolError(
            code=ErrorCode.DATABASE_ERROR, message=str(exc), tool=TOOL_NAME
        )
    except Exception as exc:  # noqa: BLE001 - classified, then propagated as a typed error
        return ToolError(
            code=ErrorCode.DATABASE_ERROR,
            message=f"parts query failed: {type(exc).__name__}",
            tool=TOOL_NAME,
            retryable=True,
        )

    parts: list[PartPosition] = []
    for row in rows:
        models = row["compatible_models"].split("|")
        if payload.component and row["component"] != payload.component:
            continue
        if payload.model and payload.model not in models:
            continue

        rate = rates.get(row["component"], 0.0)
        # Consumption is fleet-wide per component, not per part, because the
        # maintenance log records which component was replaced and not which
        # part number was fitted. Stated rather than papered over.
        cover = (
            row["stock_quantity"] / rate * CONSUMPTION_WINDOW_DAYS
            if rate > 0
            else None
        )
        parts.append(
            PartPosition(
                part_id=row["part_id"],
                component=row["component"],
                compatible_models=models,
                stock_quantity=row["stock_quantity"],
                unit_cost=row["unit_cost"],
                supplier_id=row["supplier_id"],
                lead_time_days=row["lead_time_days"],
                observed_consumption_per_30d=round(rate, 3),
                days_of_cover=round(cover, 1) if cover is not None else None,
            )
        )

    if not parts:
        return ToolError(
            code=ErrorCode.NO_DATA,
            message=(
                f"no parts match component={payload.component} model={payload.model}"
            ),
            tool=TOOL_NAME,
        )

    return Success(data=PartsPosition(parts=parts, basis=BASIS))
