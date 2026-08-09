"""Read-only, parameterised database access. The model never writes SQL.

`docs/MILESTONE_4.md` section 3. This is the direct answer to the most serious
finding in the v1 audit.

v1's `run_sql_query` executed arbitrary LLM-authored SQL against a read-write
connection. `DROP TABLE logistics` **succeeded** while the tool reported
`Error in SQL query: 'NoneType' object is not iterable` -- pandas failing on a
`None` cursor description after SQLite had already auto-committed the DDL. The
tool was destructive precisely for the statements that do the most damage, and
it said so in a way indistinguishable from a normal error.
(`docs/v1/PROJECT_AUDIT.md` section 5.4.)

Three things stop that here, and none of them relies on the model behaving:

1. **Connections open read-only** via `file:...?mode=ro`. SQLite refuses writes
   at the driver level, so a write is an `OperationalError` rather than a
   silently committed change.
2. **Every statement in this module is a literal in this file.** Callers supply
   values, which go through `?` placeholders. There is no code path that
   concatenates caller text into SQL.
3. **Every query is `LIMIT`-bounded** and results are budget-capped, so no
   result can be large enough to matter to a context window.

`tests/test_agent_db.py` asserts each of these, and asserts the database file's
SHA-256 is unchanged after attempting INSERT, UPDATE, DELETE, DROP, ATTACH and
PRAGMA writable_schema through this layer.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

DEFAULT_DB = Path("data/pdm.db")

#: Statement timeout. Every call is local, but an unbounded wait on a locked
#: database is still a hang, and section 4 requires a timeout on every call.
TIMEOUT_SECONDS = 5.0

#: Hard cap on rows returned by any single query, independent of the caller's
#: own limit. A caller cannot raise it.
MAX_ROWS = 500


class DatabaseUnavailable(RuntimeError):
    """The database file is missing or cannot be opened read-only."""


@contextmanager
def read_only(db: Path = DEFAULT_DB) -> Iterator[sqlite3.Connection]:
    """A read-only connection. The only way this module opens the database.

    `mode=ro` is enforced by SQLite itself. `query_only` is set as well: it is
    redundant against `mode=ro` and deliberately so, because the two protect
    against different mistakes -- a future caller passing a plain path would
    lose `mode=ro` but still hit `query_only`.
    """
    if not db.exists():
        raise DatabaseUnavailable(
            f"{db} not found. Run `make data` to build it."
        )

    uri = f"file:{db.resolve().as_posix()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=TIMEOUT_SECONDS)
    try:
        connection.execute("PRAGMA query_only = ON")
        # Installed after the PRAGMA above, because the authorizer denies
        # PRAGMA itself.
        connection.set_authorizer(_authorize)
        connection.row_factory = sqlite3.Row
        yield connection
    finally:
        connection.close()


#: The only operations a query needs. Everything else is denied.
#:
#: An allowlist, not a denylist, and deliberately: `mode=ro` alone turned out to
#: permit both `ATTACH DATABASE` and `PRAGMA writable_schema = ON` without
#: error. Neither could write through this handle, but "did not raise" is
#: exactly the shape of the v1 bug -- a destructive statement accepted while the
#: caller believed it had been refused. A denylist would have needed both of
#: those names added by hand after someone thought of them.
_ALLOWED_ACTIONS = frozenset(
    {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
        sqlite3.SQLITE_RECURSIVE,
    }
)


def _authorize(action: int, arg1, arg2, database_name, trigger) -> int:
    """SQLite's own authorizer hook. Denies at prepare time, before execution."""
    return sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY


def fetch(
    sql: str,
    params: tuple = (),
    db: Path = DEFAULT_DB,
    limit: int = MAX_ROWS,
) -> list[sqlite3.Row]:
    """Run one of this module's literal statements and return capped rows.

    `sql` is always a literal defined in `src/agent/`, never caller text. The
    assertion below is a tripwire for a future edit that forgets that.
    """
    if ";" in sql.strip().rstrip(";"):
        raise ValueError("multiple statements are not permitted")

    with read_only(db) as connection:
        cursor = connection.execute(sql, params)
        return cursor.fetchmany(min(limit, MAX_ROWS))


# ----------------------------------------------------------------------
# Statements. Every one is a literal; values arrive as ? parameters.
# ----------------------------------------------------------------------

MACHINE_PROFILE = """
SELECT machineID, model, age
FROM machines
WHERE machineID = ?
"""

PARTS_FOR_COMPONENT = """
SELECT part_id, component, compatible_models, stock_quantity,
       unit_cost, supplier_id, lead_time_days
FROM parts_inventory
ORDER BY part_id
"""

MAINTENANCE_HISTORY = """
SELECT machineID, comp, datetime
FROM maint
WHERE machineID = ?
  AND datetime <= ?
ORDER BY datetime DESC
LIMIT ?
"""

MAINTENANCE_HISTORY_FOR_COMPONENT = """
SELECT machineID, comp, datetime
FROM maint
WHERE machineID = ?
  AND comp = ?
  AND datetime <= ?
ORDER BY datetime DESC
LIMIT ?
"""

MAINTENANCE_COUNT = """
SELECT COUNT(*) AS n
FROM maint
WHERE machineID = ?
  AND datetime <= ?
"""

MAINTENANCE_COUNT_FOR_COMPONENT = """
SELECT COUNT(*) AS n
FROM maint
WHERE machineID = ?
  AND comp = ?
  AND datetime <= ?
"""

RECENT_ERRORS = """
SELECT errorID, COUNT(*) AS n
FROM errors
WHERE machineID = ?
  AND datetime > ?
  AND datetime <= ?
GROUP BY errorID
ORDER BY errorID
"""

#: Replacements per component across the whole log, for the consumption rate.
CONSUMPTION_BY_COMPONENT = """
SELECT comp, COUNT(*) AS n,
       MIN(datetime) AS first_seen,
       MAX(datetime) AS last_seen
FROM maint
GROUP BY comp
"""

MACHINE_EXISTS = "SELECT 1 FROM machines WHERE machineID = ? LIMIT 1"


def build_machine_filter(filters) -> tuple[str, tuple]:
    """Assemble `find_machines`' SQL from a validated filter object.

    Every fragment below is a literal chosen by branching on typed fields. The
    only caller-derived content is the parameter tuple. Pydantic has already
    constrained `model` and `component_replaced` to their enums, so even the
    branch keys cannot be arbitrary.
    """
    clauses: list[str] = []
    params: list = []

    if filters.model is not None:
        clauses.append("m.model = ?")
        params.append(filters.model)
    if filters.min_age is not None:
        clauses.append("m.age >= ?")
        params.append(filters.min_age)
    if filters.max_age is not None:
        clauses.append("m.age <= ?")
        params.append(filters.max_age)

    error_join = ""
    error_select = "NULL AS matched_error_count"
    if filters.error_id is not None:
        error_select = "COALESCE(e.n, 0) AS matched_error_count"
        since = filters.active_since.strftime("%Y-%m-%d %H:%M:%S") if filters.active_since else "0000-01-01 00:00:00"
        error_join = (
            "LEFT JOIN (SELECT machineID, COUNT(*) AS n FROM errors "
            "WHERE errorID = ? AND datetime >= ? GROUP BY machineID) e "
            "ON e.machineID = m.machineID"
        )
        params = [filters.error_id, since] + params
        clauses.append("COALESCE(e.n, 0) > 0")

    replacement_join = ""
    replacement_select = "NULL AS matched_replacement_count"
    if filters.component_replaced is not None:
        replacement_select = "COALESCE(r.n, 0) AS matched_replacement_count"
        replacement_join = (
            "LEFT JOIN (SELECT machineID, COUNT(*) AS n FROM maint "
            "WHERE comp = ? GROUP BY machineID) r ON r.machineID = m.machineID"
        )
        params = params + [filters.component_replaced]
        clauses.append("COALESCE(r.n, 0) > 0")

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = (
        f"SELECT m.machineID, m.model, m.age, {error_select}, {replacement_select} "
        f"FROM machines m {error_join} {replacement_join} {where} "
        "ORDER BY m.machineID LIMIT ?"
    )
    params.append(min(filters.limit, MAX_ROWS))
    return sql, tuple(params)
