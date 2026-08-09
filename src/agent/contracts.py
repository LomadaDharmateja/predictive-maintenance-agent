"""Pydantic input and output models for every tool.

`docs/MILESTONE_4.md` section 2. No tool returns a bare string and no tool
assembles a dict inline. v1 returned `str(dict)` from three tools, an error
message as a normal result from two more, and a raw DataFrame `to_string()` from
one -- so the model could not tell a result from a failure, and neither could a
reader.

Two constraints here are load-bearing and come from measurement, not taste:

- **No output model carries a raw uncalibrated score.** `docs/EVALUATION.md`
  section 4: Brier skill is negative for comp1 and comp3, so the raw numbers are
  not probabilities. `ComponentRisk` exposes `calibrated_probability` only, and
  `tests/test_agent_contracts.py` asserts by schema inspection that no field
  anywhere is named for a raw score.
- **`PartsPosition` has no risk field.** The Milestone 3B conclusion is that
  parts reasoning cannot come from predictions; expressing that in the type
  system is stronger than writing it in a docstring.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

# Machine ids are 1-100 in this fleet. A hallucinated 250 fails here, before the
# database is touched. docs/MILESTONE_4.md section 5.
MachineId = Annotated[int, Field(ge=1, le=100)]
ComponentName = Literal["comp1", "comp2", "comp3", "comp4"]
MachineModel = Literal["model1", "model2", "model3", "model4"]


class Strict(BaseModel):
    """Base for every model here. `extra="forbid"` so a hallucinated argument is
    a validation error rather than a silently ignored one."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# ----------------------------------------------------------------------
# Errors
# ----------------------------------------------------------------------


class ErrorCode(str, Enum):
    """Machine-readable, so the loop can branch on the kind of failure.

    v1 collapsed every failure into a human-readable string that the model
    consumed as content. An expired key and an empty result were indistinguishable.
    """

    NOT_FOUND = "not_found"
    INVALID_INPUT = "invalid_input"
    NO_DATA = "no_data"
    MODEL_UNAVAILABLE = "model_unavailable"
    DATABASE_ERROR = "database_error"
    TIMEOUT = "timeout"
    INTERNAL = "internal"


class ToolError(Strict):
    """A failure. A *different type* from success, not a different string."""

    status: Literal["error"] = "error"
    code: ErrorCode
    message: str
    tool: str
    # True only for transient conditions. Validation errors are never retried:
    # the same input will fail the same way and a retry just burns an iteration.
    retryable: bool = False


T = TypeVar("T")


class Success(BaseModel, Generic[T]):
    status: Literal["ok"] = "ok"
    data: T
    # Set when a result was cut to fit the character budget. v1 passed an
    # 8.5 MB query result straight into model context; silence about truncation
    # is how a partial answer becomes a confident wrong one.
    truncated: bool = False


ToolResult = Success[T] | ToolError


# ----------------------------------------------------------------------
# get_machine_profile
# ----------------------------------------------------------------------


class MachineProfileInput(Strict):
    machine_id: MachineId


class ComponentParts(Strict):
    component: ComponentName
    part_ids: list[str]


class MachineProfile(Strict):
    machine_id: int
    model: MachineModel
    age_years: int
    components: list[ComponentParts]


# ----------------------------------------------------------------------
# get_failure_risk
# ----------------------------------------------------------------------


class WarningAdequacy(str, Enum):
    """Whether the warning is long enough to act on.

    Not advisory and not optional. Computed by comparing the component's measured
    effective detection lead against the lead time of the part in question. Given
    docs/SIGNAL_ANALYSIS.md section 4 it reads `insufficient` for most pairs, and
    the agent is required to surface that rather than route around it.
    """

    SUFFICIENT = "sufficient"
    MARGINAL = "marginal"
    INSUFFICIENT = "insufficient"


class PartAdequacy(Strict):
    part_id: str
    lead_time_days: int
    adequacy: WarningAdequacy
    #: Measured median hours between the score crossing the threshold and the
    #: failure, from docs/SIGNAL_ANALYSIS.md. None where the component was not
    #: reliably detected at all.
    detection_lead_hours: float | None


class ComponentRisk(Strict):
    component: ComponentName
    #: Isotonic-calibrated. There is deliberately no raw-score field: raw scores
    #: are not probabilities on this model (negative Brier skill for comp1 and
    #: comp3), so exposing one would invite an agent to reason with it.
    calibrated_probability: float = Field(ge=0.0, le=1.0)
    confidence_interval_low: float = Field(ge=0.0, le=1.0)
    confidence_interval_high: float = Field(ge=0.0, le=1.0)
    #: The cost-derived threshold in force, and the ratio it came from.
    threshold: float = Field(ge=0.0, le=1.0)
    threshold_cost_ratio: float
    exceeds_threshold: bool
    warning_adequacy: WarningAdequacy
    per_part_adequacy: list[PartAdequacy]


class FailureRiskInput(Strict):
    machine_id: MachineId
    as_of: datetime


class FailureRisk(Strict):
    machine_id: int
    as_of: datetime
    horizon_days: int
    model_family: str
    components: list[ComponentRisk]
    #: Carried on every risk result so a consumer cannot read a probability
    #: without also seeing what it does not cover.
    caveat: str


# ----------------------------------------------------------------------
# get_maintenance_history
# ----------------------------------------------------------------------


class MaintenanceHistoryInput(Strict):
    machine_id: MachineId
    as_of: datetime
    component: ComponentName | None = None
    limit: int = Field(default=20, ge=1, le=200)


class MaintenanceRecord(Strict):
    machine_id: int
    component: ComponentName
    replaced_at: datetime


class MaintenanceHistory(Strict):
    machine_id: int
    as_of: datetime
    records: list[MaintenanceRecord]
    total_matching: int


# ----------------------------------------------------------------------
# get_recent_errors
# ----------------------------------------------------------------------


class RecentErrorsInput(Strict):
    machine_id: MachineId
    as_of: datetime
    window_hours: int = Field(default=168, ge=1, le=8760)


class ErrorCount(Strict):
    error_id: str
    count: int


class RecentErrors(Strict):
    machine_id: int
    as_of: datetime
    window_hours: int
    counts: list[ErrorCount]
    total: int


# ----------------------------------------------------------------------
# get_parts_position
# ----------------------------------------------------------------------


class PartsPositionInput(Strict):
    """Deliberately has no risk, probability, threshold or prediction field.

    A caller cannot ask "what parts do I need given this risk score", because
    that question is not answerable on this data. docs/SIGNAL_ANALYSIS.md
    section 4: 1 of 9 parts has a lead time short enough to be ordered on the
    model's warning.
    """

    component: ComponentName | None = None
    model: MachineModel | None = None


class PartPosition(Strict):
    part_id: str
    component: ComponentName
    compatible_models: list[MachineModel]
    stock_quantity: int
    unit_cost: float
    supplier_id: str
    lead_time_days: int
    #: Replacements per 30 days, observed from the maintenance log. This is what
    #: drives reordering -- it is a measurement, not a forecast.
    observed_consumption_per_30d: float
    #: Stock divided by consumption rate. None when nothing has been consumed.
    days_of_cover: float | None


class PartsPosition(Strict):
    parts: list[PartPosition]
    basis: str


# ----------------------------------------------------------------------
# find_machines
# ----------------------------------------------------------------------


class MachineFilters(Strict):
    """A typed filter object, not a query string.

    The model chooses values; `src/agent/db.py` builds the parameterised SQL.
    v1's `run_sql_query` took LLM-authored SQL and executed it against a
    read-write connection, where `DROP TABLE logistics` succeeded while the tool
    reported an error. docs/v1/PROJECT_AUDIT.md section 5.4.
    """

    model: MachineModel | None = None
    min_age: int | None = Field(default=None, ge=0, le=100)
    max_age: int | None = Field(default=None, ge=0, le=100)
    component_replaced: ComponentName | None = None
    error_id: str | None = None
    active_since: datetime | None = None
    limit: int = Field(default=25, ge=1, le=100)


class MachineSummary(Strict):
    machine_id: int
    model: MachineModel
    age_years: int
    matched_error_count: int | None = None
    matched_replacement_count: int | None = None


class MachineList(Strict):
    machines: list[MachineSummary]
    total_matching: int
    filters_applied: MachineFilters
