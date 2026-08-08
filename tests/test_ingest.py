"""Properties of the built database: row counts, keys, indexes, determinism."""

from __future__ import annotations

import hashlib
import sqlite3

import pandas as pd
import pytest

from src.data.ingest import (
    DDL,
    EXPECTED_ROW_COUNTS,
    INSERT_COLUMNS,
    SCHEMA_VERSION,
    build,
)
from src.data.schemas import SOURCE_SCHEMAS

EVENT_TABLES = ["telemetry", "errors", "maint", "failures"]


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# Row counts match the source exactly
# --------------------------------------------------------------------------


@pytest.mark.parametrize("table", sorted(SOURCE_SCHEMAS))
def test_row_count_matches_source_csv(conn, source_frames, table):
    (in_db,) = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    assert in_db == len(source_frames[table])
    assert in_db == EXPECTED_ROW_COUNTS[table]


@pytest.mark.parametrize("table", sorted(SOURCE_SCHEMAS))
def test_no_row_duplication(conn, table):
    cols = ", ".join(INSERT_COLUMNS[table])
    (total,) = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    (distinct,) = conn.execute(
        f"SELECT COUNT(*) FROM (SELECT DISTINCT {cols} FROM {table})"
    ).fetchone()
    assert total == distinct


@pytest.mark.parametrize("table", sorted(SOURCE_SCHEMAS))
def test_values_survive_the_round_trip(conn, source_frames, table):
    """Contents, not just counts: a shape-preserving corruption would pass the
    count test."""
    from_db = pd.read_sql_query(
        f"SELECT {', '.join(INSERT_COLUMNS[table])} FROM {table}", conn
    )
    expected = source_frames[table][INSERT_COLUMNS[table]].copy()

    for frame in (from_db, expected):
        if "datetime" in frame.columns:
            frame["datetime"] = pd.to_datetime(frame["datetime"], format="mixed")

    sort_cols = INSERT_COLUMNS[table]
    from_db = from_db.sort_values(sort_cols, ignore_index=True)
    expected = expected.sort_values(sort_cols, ignore_index=True)

    pd.testing.assert_frame_equal(from_db, expected, check_dtype=False)


# --------------------------------------------------------------------------
# Foreign keys
# --------------------------------------------------------------------------


def test_no_foreign_key_violations(conn):
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


@pytest.mark.parametrize("table", EVENT_TABLES)
def test_every_machine_id_resolves(conn, table):
    orphans = conn.execute(
        f"SELECT DISTINCT t.machineID FROM {table} t "
        "LEFT JOIN machines m ON m.machineID = t.machineID "
        "WHERE m.machineID IS NULL"
    ).fetchall()
    assert orphans == []


def test_machines_table_is_the_full_fleet(conn):
    (n,) = conn.execute("SELECT COUNT(DISTINCT machineID) FROM machines").fetchone()
    assert n == EXPECTED_ROW_COUNTS["machines"]


def test_inventory_components_exist_in_maint(conn):
    unknown = conn.execute(
        "SELECT DISTINCT p.component FROM parts_inventory p "
        "LEFT JOIN (SELECT DISTINCT comp FROM maint) m ON m.comp = p.component "
        "WHERE m.comp IS NULL"
    ).fetchall()
    assert unknown == []


def test_foreign_keys_are_declared_and_enforced(tmp_path):
    """The FK constraint is real, not decorative."""
    scratch = sqlite3.connect(tmp_path / "scratch.db")
    scratch.executescript(DDL)
    scratch.execute("PRAGMA foreign_keys = ON")
    scratch.execute("INSERT INTO machines VALUES (1, 'model1', 5)")

    with pytest.raises(sqlite3.IntegrityError):
        scratch.execute(
            "INSERT INTO telemetry VALUES (999, '2015-01-01 06:00:00', 1, 1, 1, 1)"
        )
    scratch.close()


def test_primary_keys_reject_duplicates(tmp_path):
    scratch = sqlite3.connect(tmp_path / "scratch.db")
    scratch.executescript(DDL)
    scratch.execute("INSERT INTO machines VALUES (1, 'model1', 5)")
    row = "(1, '2015-01-01 06:00:00', 1, 1, 1, 1)"
    scratch.execute(f"INSERT INTO telemetry VALUES {row}")

    with pytest.raises(sqlite3.IntegrityError):
        scratch.execute(f"INSERT INTO telemetry VALUES {row}")
    scratch.close()


# --------------------------------------------------------------------------
# Schema shape
# --------------------------------------------------------------------------


def test_schema_version_is_stamped(conn):
    (version,) = conn.execute("PRAGMA user_version").fetchone()
    assert version == SCHEMA_VERSION


@pytest.mark.parametrize("table", EVENT_TABLES)
def test_machine_time_index_exists(conn, table):
    indexes = {
        row[1] for row in conn.execute(f"PRAGMA index_list({table})").fetchall()
    }
    assert f"idx_{table}_machine_time" in indexes

    columns = [
        row[2]
        for row in conn.execute(
            f"PRAGMA index_info(idx_{table}_machine_time)"
        ).fetchall()
    ]
    assert columns == ["machineID", "datetime"]


def test_expected_tables_and_nothing_else(conn):
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    }
    assert tables == set(INSERT_COLUMNS)


def test_datetime_is_stored_sortable(conn):
    """Timestamps are TEXT; lexical order must equal chronological order,
    otherwise every BETWEEN in the project is silently wrong."""
    rows = conn.execute(
        "SELECT datetime FROM telemetry ORDER BY datetime LIMIT 3"
    ).fetchall()
    values = [row[0] for row in rows]
    assert values == sorted(values)
    assert pd.to_datetime(values).is_monotonic_increasing
    assert len(values[0]) == len("YYYY-MM-DD HH:MM:SS")


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_rebuild_is_byte_identical(tmp_path, raw_dir, inventory_csv, built_db):
    """`make data` must be deterministic, not merely repeatable in content."""
    second = tmp_path / "pdm-again.db"
    build(raw=raw_dir, db=second, inventory=inventory_csv, manifest=None, quiet=True)
    assert sha256_file(second) == sha256_file(built_db)


def test_build_returns_stable_content_hashes(conn, built_db, tmp_path, raw_dir, inventory_csv):
    first = build(
        raw=raw_dir,
        db=tmp_path / "a.db",
        inventory=inventory_csv,
        manifest=None,
        quiet=True,
    )
    assert set(first) == set(INSERT_COLUMNS)
    assert all(info["rows"] > 0 for info in first.values())
    assert all(len(info["sha256"]) == 64 for info in first.values())
