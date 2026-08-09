"""The database is provably read-only through the tool layer.

`docs/MILESTONE_4.md` section 3. v1's `run_sql_query` executed
`DROP TABLE logistics` successfully while reporting an error, because SQLite
auto-commits DDL before pandas fails on the empty result
(`docs/v1/PROJECT_AUDIT.md` section 5.4). These tests assert that cannot recur,
and verify it the only way that actually settles the question: by hashing the
file before and after.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from src.agent import db

WRITE_STATEMENTS = [
    "INSERT INTO machines (machineID, model, age) VALUES (9999, 'model1', 3)",
    "UPDATE machines SET age = 99 WHERE machineID = 1",
    "DELETE FROM machines WHERE machineID = 1",
    "DROP TABLE maint",
    "ALTER TABLE machines ADD COLUMN sneaky TEXT",
    "CREATE TABLE evil (x INTEGER)",
    "PRAGMA writable_schema = ON",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@pytest.fixture
def database(built_db) -> Path:
    return Path(built_db)


def test_connection_is_read_only(database):
    """Two independent protections, so the test checks both.

    `PRAGMA query_only` cannot be *read back* here, because the authorizer denies
    PRAGMA outright -- which is itself the assertion. Reads still work.
    """
    with db.read_only(database) as connection:
        with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
            connection.execute("PRAGMA query_only")

        rows = connection.execute("SELECT COUNT(*) FROM machines").fetchone()
        assert rows[0] == 100


def test_the_authorizer_allows_only_reads(database):
    """An allowlist, not a denylist. `mode=ro` alone permitted both ATTACH and
    PRAGMA writable_schema without raising."""
    assert sqlite3.SQLITE_SELECT in db._ALLOWED_ACTIONS
    assert sqlite3.SQLITE_READ in db._ALLOWED_ACTIONS
    for denied in (
        sqlite3.SQLITE_INSERT,
        sqlite3.SQLITE_UPDATE,
        sqlite3.SQLITE_DELETE,
        sqlite3.SQLITE_DROP_TABLE,
        sqlite3.SQLITE_ATTACH,
        sqlite3.SQLITE_PRAGMA,
    ):
        assert denied not in db._ALLOWED_ACTIONS


@pytest.mark.parametrize("statement", WRITE_STATEMENTS)
def test_every_write_statement_fails(database, statement):
    with db.read_only(database) as connection:
        with pytest.raises(sqlite3.DatabaseError):
            connection.execute(statement)


def test_database_is_byte_identical_after_attempted_writes(database):
    """The assertion that actually settles it. v1's DROP reported an error *and*
    destroyed the table; only a hash distinguishes those two outcomes."""
    before = sha256(database)

    for statement in WRITE_STATEMENTS:
        try:
            with db.read_only(database) as connection:
                connection.execute(statement)
                connection.commit()
        except sqlite3.DatabaseError:
            pass

    assert sha256(database) == before


def test_attach_is_rejected(database, tmp_path):
    """ATTACH would open a second, writable database and sidestep mode=ro."""
    other = tmp_path / "scratch.db"
    with db.read_only(database) as connection:
        with pytest.raises(sqlite3.DatabaseError):
            connection.execute(f"ATTACH DATABASE '{other}' AS scratch")


def test_multiple_statements_are_refused(database):
    """Stacked statements are the classic way a single innocuous-looking string
    smuggles a second one past a filter."""
    with pytest.raises(ValueError, match="multiple statements"):
        db.fetch("SELECT 1; DROP TABLE maint", db=database)


def test_rows_are_capped_regardless_of_the_caller(database):
    rows = db.fetch("SELECT machineID FROM machines", db=database, limit=10_000)
    assert len(rows) <= db.MAX_ROWS


def test_missing_database_raises_a_named_error(tmp_path):
    with pytest.raises(db.DatabaseUnavailable, match="not found"):
        with db.read_only(tmp_path / "absent.db"):
            pass


def test_filter_builder_parameterises_every_value():
    """The model supplies values; they must arrive as `?` parameters, never as
    text spliced into the statement."""
    from src.agent.contracts import MachineFilters

    filters = MachineFilters(
        model="model2", min_age=5, max_age=15, error_id="error1", limit=10
    )
    sql, params = db.build_machine_filter(filters)

    assert "model2" not in sql
    assert "error1" not in sql
    assert sql.count("?") == len(params)
    assert "model2" in params and "error1" in params


def test_filter_builder_cannot_be_injected_through_a_value(database):
    """Pydantic constrains `model` to an enum, so the classic payload is
    rejected before SQL is built at all."""
    from pydantic import ValidationError

    from src.agent.contracts import MachineFilters

    with pytest.raises(ValidationError):
        MachineFilters(model="model1'; DROP TABLE maint;--")


def test_error_id_is_free_text_but_still_parameterised(database):
    """`error_id` is not enum-constrained, so it is the one filter field a
    payload could reach. It must still be a bound parameter and must not
    execute."""
    from src.agent.contracts import MachineFilters
    from src.agent.tools import find_machines

    before = sha256(database)
    result = find_machines(
        MachineFilters(error_id="error1'; DROP TABLE maint;--"), database
    )
    assert sha256(database) == before
    # Either no match or an empty list; never an executed statement.
    assert result.status in {"ok", "error"}
