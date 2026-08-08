"""Schema contracts for the Azure PdM source files and the generated inventory.

Every schema is `strict=True` and `coerce=True`. Strict means an unexpected
column is an error, not something to ignore; coerce means a column that cannot
be cast to the declared dtype is an error, not something silently left as
object. No column is nullable. The intent is that a malformed or substituted
input file fails at load time with a named column and a row index, rather than
producing a database that is quietly wrong.

Validation is invoked from `src.data.ingest`. It is also exercised directly by
`tests/test_schemas.py`, which asserts both that real data passes and that
corrupted data fails.
"""

from __future__ import annotations

import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

# Domain values, taken from the observed data and asserted rather than assumed.
# docs/DATA.md section 2 records where these come from.
COMPONENTS = ["comp1", "comp2", "comp3", "comp4"]
ERROR_IDS = ["error1", "error2", "error3", "error4", "error5"]
MACHINE_MODELS = ["model1", "model2", "model3", "model4"]

N_MACHINES = 100
MIN_MACHINE_AGE = 0
MAX_MACHINE_AGE = 20

# The observation window. Anything outside it means the wrong file was supplied.
WINDOW_START = "2014-06-01 06:00:00"
WINDOW_END = "2016-01-01 06:00:00"


def _machine_id() -> Column:
    return Column(
        "int64",
        checks=[Check.ge(1), Check.le(N_MACHINES)],
        nullable=False,
        description="Foreign key to machines.machineID",
    )


def _datetime(nullable: bool = False) -> Column:
    return Column(
        "datetime64[ns]",
        checks=[
            Check.ge(WINDOW_START),
            Check.le(WINDOW_END),
        ],
        nullable=nullable,
        description="Event timestamp, hourly resolution",
    )


MACHINES_SCHEMA = DataFrameSchema(
    {
        "machineID": Column(
            "int64",
            checks=[Check.ge(1), Check.le(N_MACHINES)],
            nullable=False,
            unique=True,
        ),
        "model": Column("str", checks=Check.isin(MACHINE_MODELS), nullable=False),
        "age": Column(
            "int64",
            checks=[Check.ge(MIN_MACHINE_AGE), Check.le(MAX_MACHINE_AGE)],
            nullable=False,
        ),
    },
    strict=True,
    coerce=True,
    name="machines",
)

TELEMETRY_SCHEMA = DataFrameSchema(
    {
        "datetime": _datetime(),
        "machineID": _machine_id(),
        # Bounds are deliberately wide: they catch unit errors and sign flips,
        # not statistical outliers. Narrowing them would discard real readings.
        "volt": Column("float64", checks=[Check.gt(0), Check.lt(1000)], nullable=False),
        "rotate": Column("float64", checks=[Check.gt(0), Check.lt(1000)], nullable=False),
        "pressure": Column("float64", checks=[Check.gt(0), Check.lt(1000)], nullable=False),
        "vibration": Column("float64", checks=[Check.gt(0), Check.lt(1000)], nullable=False),
    },
    strict=True,
    coerce=True,
    unique=["machineID", "datetime"],
    name="telemetry",
)

ERRORS_SCHEMA = DataFrameSchema(
    {
        "datetime": _datetime(),
        "machineID": _machine_id(),
        "errorID": Column("str", checks=Check.isin(ERROR_IDS), nullable=False),
    },
    strict=True,
    coerce=True,
    unique=["machineID", "datetime", "errorID"],
    name="errors",
)

MAINT_SCHEMA = DataFrameSchema(
    {
        "datetime": _datetime(),
        "machineID": _machine_id(),
        "comp": Column("str", checks=Check.isin(COMPONENTS), nullable=False),
    },
    strict=True,
    coerce=True,
    unique=["machineID", "datetime", "comp"],
    name="maint",
)

FAILURES_SCHEMA = DataFrameSchema(
    {
        "datetime": _datetime(),
        "machineID": _machine_id(),
        # The column is named `failure` in the source file but holds a component
        # identifier, not a boolean. Kept under its source name so the raw file
        # and the table agree; documented in docs/DATA.md section 2.
        "failure": Column("str", checks=Check.isin(COMPONENTS), nullable=False),
    },
    strict=True,
    coerce=True,
    unique=["machineID", "datetime", "failure"],
    name="failures",
)

PARTS_INVENTORY_SCHEMA = DataFrameSchema(
    {
        "part_id": Column("str", nullable=False, unique=True),
        "component": Column("str", checks=Check.isin(COMPONENTS), nullable=False),
        # Pipe-separated model list, e.g. "model1|model3". Stored denormalised
        # because SQLite has no array type and the fleet has four models.
        "compatible_models": Column(
            "str",
            checks=Check.str_matches(r"^model[1-4](\|model[1-4])*$"),
            nullable=False,
        ),
        "stock_quantity": Column("int64", checks=Check.ge(0), nullable=False),
        "unit_cost": Column("float64", checks=Check.gt(0), nullable=False),
        "supplier_id": Column(
            "str", checks=Check.str_matches(r"^SUP-\d{3}$"), nullable=False
        ),
        "lead_time_days": Column("int64", checks=Check.ge(1), nullable=False),
    },
    strict=True,
    coerce=True,
    name="parts_inventory",
)

# Table name -> (source filename, schema). Drives both ingestion and tests, so
# adding a table in one place cannot leave the other behind.
SOURCE_SCHEMAS: dict[str, tuple[str, DataFrameSchema]] = {
    "machines": ("PdM_machines.csv", MACHINES_SCHEMA),
    "telemetry": ("PdM_telemetry.csv", TELEMETRY_SCHEMA),
    "errors": ("PdM_errors.csv", ERRORS_SCHEMA),
    "maint": ("PdM_maint.csv", MAINT_SCHEMA),
    "failures": ("PdM_failures.csv", FAILURES_SCHEMA),
}

# Expected row counts, measured from the source files and recorded in
# docs/DATA.md section 2. Ingestion asserts against these so a truncated or
# duplicated download is caught rather than silently ingested.
EXPECTED_ROW_COUNTS: dict[str, int] = {
    "machines": 100,
    "telemetry": 876_100,
    "errors": 3_919,
    "maint": 3_286,
    "failures": 761,
}
