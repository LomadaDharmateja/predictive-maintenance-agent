"""The remaining tools, and the registry the loop dispatches through.

`docs/MILESTONE_4.md` sections 2 and 4. Every tool takes a validated Pydantic
input and returns `Success[T]` or `ToolError` -- different types, never different
strings. There is no bare `except Exception` that returns a value: exceptions are
caught, classified into an `ErrorCode`, and propagated as a typed error.

`get_parts_position` is imported from `src.agent.parts` and `get_failure_risk`
from `src.agent.risk`; both stay in their own modules so the parts tool's import
closure can be asserted free of any model code.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from pydantic import ValidationError

from src.agent import db
from src.agent.contracts import (
    ErrorCode,
    ErrorCount,
    MachineFilters,
    MachineList,
    MachineProfile,
    MachineProfileInput,
    MachineSummary,
    MaintenanceHistory,
    MaintenanceHistoryInput,
    MaintenanceRecord,
    RecentErrors,
    RecentErrorsInput,
    Success,
    ToolError,
    ToolResult,
    ComponentParts,
)
from src.agent.parts import get_parts_position
from src.agent.risk import get_failure_risk
from src.features.config import COMPONENTS

SQL_FORMAT = "%Y-%m-%d %H:%M:%S"

#: Character budget for a serialised tool result. v1 fed an 8,550,170-character
#: query result straight into model context; one such call ended the session.
RESULT_CHARACTER_BUDGET = 12_000


def _fail(tool: str, exc: Exception, code: ErrorCode, retryable: bool = False) -> ToolError:
    """Classify and propagate. Never returns a value that looks like data."""
    return ToolError(
        code=code, message=f"{type(exc).__name__}: {exc}", tool=tool, retryable=retryable
    )


# ----------------------------------------------------------------------


def get_machine_profile(
    payload: MachineProfileInput, database: Path = db.DEFAULT_DB
) -> ToolResult[MachineProfile]:
    tool = "get_machine_profile"
    try:
        rows = db.fetch(db.MACHINE_PROFILE, (payload.machine_id,), db=database)
        parts = db.fetch(db.PARTS_FOR_COMPONENT, db=database)
    except db.DatabaseUnavailable as exc:
        return _fail(tool, exc, ErrorCode.DATABASE_ERROR)
    except Exception as exc:  # noqa: BLE001
        return _fail(tool, exc, ErrorCode.DATABASE_ERROR, retryable=True)

    if not rows:
        return ToolError(
            code=ErrorCode.NOT_FOUND,
            message=f"machine {payload.machine_id} not found",
            tool=tool,
        )

    machine = rows[0]
    by_component: dict[str, list[str]] = {c: [] for c in COMPONENTS}
    for part in parts:
        if machine["model"] in part["compatible_models"].split("|"):
            by_component[part["component"]].append(part["part_id"])

    return Success(
        data=MachineProfile(
            machine_id=machine["machineID"],
            model=machine["model"],
            age_years=machine["age"],
            components=[
                ComponentParts(component=component, part_ids=sorted(ids))
                for component, ids in sorted(by_component.items())
            ],
        )
    )


def get_maintenance_history(
    payload: MaintenanceHistoryInput, database: Path = db.DEFAULT_DB
) -> ToolResult[MaintenanceHistory]:
    """Replacements strictly at or before `as_of`.

    The `as_of` boundary is the Milestone 2 rule reused unchanged: records at
    exactly `as_of` are visible, anything after is not. docs/DATA.md section 5.1.
    """
    tool = "get_maintenance_history"
    cutoff = payload.as_of.strftime(SQL_FORMAT)
    try:
        if payload.component:
            rows = db.fetch(
                db.MAINTENANCE_HISTORY_FOR_COMPONENT,
                (payload.machine_id, payload.component, cutoff, payload.limit),
                db=database,
            )
            counted = db.fetch(
                db.MAINTENANCE_COUNT_FOR_COMPONENT,
                (payload.machine_id, payload.component, cutoff),
                db=database,
            )
        else:
            rows = db.fetch(
                db.MAINTENANCE_HISTORY,
                (payload.machine_id, cutoff, payload.limit),
                db=database,
            )
            counted = db.fetch(
                db.MAINTENANCE_COUNT, (payload.machine_id, cutoff), db=database
            )
    except db.DatabaseUnavailable as exc:
        return _fail(tool, exc, ErrorCode.DATABASE_ERROR)
    except Exception as exc:  # noqa: BLE001
        return _fail(tool, exc, ErrorCode.DATABASE_ERROR, retryable=True)

    total = counted[0]["n"] if counted else 0
    return Success(
        data=MaintenanceHistory(
            machine_id=payload.machine_id,
            as_of=payload.as_of,
            records=[
                MaintenanceRecord(
                    machine_id=row["machineID"],
                    component=row["comp"],
                    replaced_at=datetime.fromisoformat(row["datetime"]),
                )
                for row in rows
            ],
            total_matching=total,
        ),
        truncated=total > len(rows),
    )


def get_recent_errors(
    payload: RecentErrorsInput, database: Path = db.DEFAULT_DB
) -> ToolResult[RecentErrors]:
    """Error counts in `(as_of - window, as_of]`. Same boundary rule."""
    tool = "get_recent_errors"
    start = (payload.as_of - timedelta(hours=payload.window_hours)).strftime(SQL_FORMAT)
    end = payload.as_of.strftime(SQL_FORMAT)
    try:
        rows = db.fetch(
            db.RECENT_ERRORS, (payload.machine_id, start, end), db=database
        )
        exists = db.fetch(db.MACHINE_EXISTS, (payload.machine_id,), db=database)
    except db.DatabaseUnavailable as exc:
        return _fail(tool, exc, ErrorCode.DATABASE_ERROR)
    except Exception as exc:  # noqa: BLE001
        return _fail(tool, exc, ErrorCode.DATABASE_ERROR, retryable=True)

    if not exists:
        return ToolError(
            code=ErrorCode.NOT_FOUND,
            message=f"machine {payload.machine_id} not found",
            tool=tool,
        )

    counts = [ErrorCount(error_id=row["errorID"], count=row["n"]) for row in rows]
    return Success(
        data=RecentErrors(
            machine_id=payload.machine_id,
            as_of=payload.as_of,
            window_hours=payload.window_hours,
            counts=counts,
            total=sum(c.count for c in counts),
        )
    )


def find_machines(
    payload: MachineFilters, database: Path = db.DEFAULT_DB
) -> ToolResult[MachineList]:
    """Structured filtering. The model supplies values, never syntax."""
    tool = "find_machines"
    if (
        payload.min_age is not None
        and payload.max_age is not None
        and payload.min_age > payload.max_age
    ):
        return ToolError(
            code=ErrorCode.INVALID_INPUT,
            message=f"min_age {payload.min_age} exceeds max_age {payload.max_age}",
            tool=tool,
        )

    try:
        sql, params = db.build_machine_filter(payload)
        rows = db.fetch(sql, params, db=database, limit=payload.limit)
    except db.DatabaseUnavailable as exc:
        return _fail(tool, exc, ErrorCode.DATABASE_ERROR)
    except Exception as exc:  # noqa: BLE001
        return _fail(tool, exc, ErrorCode.DATABASE_ERROR, retryable=True)

    machines = [
        MachineSummary(
            machine_id=row["machineID"],
            model=row["model"],
            age_years=row["age"],
            matched_error_count=row["matched_error_count"],
            matched_replacement_count=row["matched_replacement_count"],
        )
        for row in rows
    ]
    return Success(
        data=MachineList(
            machines=machines,
            total_matching=len(machines),
            filters_applied=payload,
        ),
        truncated=len(machines) >= payload.limit,
    )


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------

REGISTRY: dict[str, tuple[type, object]] = {
    "get_machine_profile": (MachineProfileInput, get_machine_profile),
    "get_failure_risk": (__import__(
        "src.agent.contracts", fromlist=["FailureRiskInput"]
    ).FailureRiskInput, get_failure_risk),
    "get_maintenance_history": (MaintenanceHistoryInput, get_maintenance_history),
    "get_recent_errors": (RecentErrorsInput, get_recent_errors),
    "get_parts_position": (__import__(
        "src.agent.contracts", fromlist=["PartsPositionInput"]
    ).PartsPositionInput, get_parts_position),
    "find_machines": (MachineFilters, find_machines),
}


def dispatch(name: str, arguments: dict, database: Path = db.DEFAULT_DB):
    """Validate arguments against the tool's model, then call it.

    Validation happens **before** dispatch, so a hallucinated `machine_id` of 250
    becomes a `ToolError` and never reaches the database.
    docs/MILESTONE_4.md section 5.
    """
    if name not in REGISTRY:
        return ToolError(
            code=ErrorCode.INVALID_INPUT,
            message=f"unknown tool {name!r}; available: {sorted(REGISTRY)}",
            tool=name,
        )

    model, function = REGISTRY[name]
    try:
        payload = model(**arguments)
    except ValidationError as exc:
        return ToolError(
            code=ErrorCode.INVALID_INPUT,
            message=f"invalid arguments for {name}: {exc.error_count()} error(s); "
            + "; ".join(
                f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}"
                for e in exc.errors()[:5]
            ),
            tool=name,
            retryable=False,
        )
    except TypeError as exc:
        return ToolError(
            code=ErrorCode.INVALID_INPUT, message=str(exc), tool=name
        )

    return function(payload, database)


def serialise(result) -> tuple[str, bool]:
    """Render a tool result for the model, within the character budget."""
    text = result.model_dump_json()
    if len(text) <= RESULT_CHARACTER_BUDGET:
        return text, False
    # Truncation is announced, never silent.
    return text[: RESULT_CHARACTER_BUDGET] + "...[TRUNCATED]", True
