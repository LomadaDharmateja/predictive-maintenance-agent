"""Shared fixtures.

The database is built once per test session against the real files in
`data/raw/`. That is slower than a synthetic fixture, but the properties under
test -- exact row counts, total referential integrity, byte-level
reproducibility -- are properties of the real data, and a synthetic stand-in
would let all three pass while the shipped database was wrong.
"""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from src.data.ingest import DEFAULT_RAW, build
from src.data.schemas import SOURCE_SCHEMAS

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / DEFAULT_RAW


def pytest_collection_modifyitems(config, items):
    """Skip the whole suite with one clear message if the raw data is absent."""
    missing = [f for f, _ in SOURCE_SCHEMAS.values() if not (RAW_DIR / f).exists()]
    if not missing:
        return
    skip = pytest.mark.skip(
        reason=f"raw data not present in {RAW_DIR} (missing: {', '.join(missing)}). "
        "See docs/DATA.md section 1."
    )
    for item in items:
        item.add_marker(skip)


@pytest.fixture(scope="session")
def raw_dir() -> Path:
    return RAW_DIR


@pytest.fixture(scope="session")
def inventory_csv(tmp_path_factory, raw_dir) -> Path:
    """Regenerate the inventory rather than trusting the committed copy."""
    import subprocess
    import sys

    out = tmp_path_factory.mktemp("inventory") / "parts_inventory.csv"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "generate_inventory.py"),
            "--raw",
            str(raw_dir),
            "--out",
            str(out),
        ],
        check=True,
        capture_output=True,
        cwd=REPO_ROOT,
    )
    return out


@pytest.fixture(scope="session")
def built_db(tmp_path_factory, raw_dir, inventory_csv) -> Path:
    db = tmp_path_factory.mktemp("db") / "pdm.db"
    build(
        raw=raw_dir,
        db=db,
        inventory=inventory_csv,
        manifest=None,
        quiet=True,
    )
    return db


@pytest.fixture(scope="session")
def conn(built_db) -> sqlite3.Connection:
    connection = sqlite3.connect(built_db)
    connection.execute("PRAGMA foreign_keys = ON")
    yield connection
    connection.close()


@pytest.fixture(scope="session")
def source_frames(raw_dir) -> dict[str, pd.DataFrame]:
    """Raw CSVs read with no validation, for comparison against the database."""
    return {
        table: pd.read_csv(raw_dir / filename)
        for table, (filename, _) in SOURCE_SCHEMAS.items()
    }


@pytest.fixture(scope="session")
def real_store(built_db):
    """FeatureStore over the whole shipped dataset. Built once; ~5 seconds."""
    from src.features.store import FeatureStore

    return FeatureStore.from_database(built_db)


@pytest.fixture
def real_subset_frames(built_db) -> dict[str, pd.DataFrame]:
    """A slice of the real tables: 10 machines over two months.

    Small enough that the leakage tests can rebuild the store per perturbation,
    while keeping the properties a synthetic fixture would not reproduce -- real
    gap distributions, simultaneous events, and maint history predating the
    window. The telemetry slice stays a contiguous balanced panel, which
    `FeatureStore` requires.

    All `maint` rows for those machines are kept, including the pre-2015 records,
    so `hours_since_*` is defined from the first hour of the slice.
    """
    import sqlite3

    machines_wanted = tuple(range(1, 11))
    start, end = "2015-03-01 00:00:00", "2015-05-01 00:00:00"
    placeholders = ", ".join("?" * len(machines_wanted))

    connection = sqlite3.connect(built_db)
    try:
        frames = {
            "telemetry": pd.read_sql_query(
                f"SELECT * FROM telemetry WHERE machineID IN ({placeholders}) "
                "AND datetime BETWEEN ? AND ?",
                connection,
                params=(*machines_wanted, start, end),
            ),
            "errors": pd.read_sql_query(
                f"SELECT * FROM errors WHERE machineID IN ({placeholders})",
                connection,
                params=machines_wanted,
            ),
            "maint": pd.read_sql_query(
                f"SELECT * FROM maint WHERE machineID IN ({placeholders})",
                connection,
                params=machines_wanted,
            ),
            "failures": pd.read_sql_query(
                f"SELECT * FROM failures WHERE machineID IN ({placeholders})",
                connection,
                params=machines_wanted,
            ),
            "machines": pd.read_sql_query(
                f"SELECT * FROM machines WHERE machineID IN ({placeholders})",
                connection,
                params=machines_wanted,
            ),
        }
    finally:
        connection.close()

    # SQLite hands back TEXT timestamps. Parsed here so a test can do timestamp
    # arithmetic on them and so a perturbation appending a Timestamp does not
    # leave the column holding a mixture of types.
    for frame in frames.values():
        if "datetime" in frame.columns:
            frame["datetime"] = pd.to_datetime(
                frame["datetime"], format="mixed"
            ).dt.as_unit("ns")
    return frames


@pytest.fixture
def corrupt_raw(tmp_path, raw_dir) -> Path:
    """A writable copy of the raw directory, for corruption tests.

    Only the small files are copied; telemetry is 80 MB and no corruption test
    needs it, so it is written as a valid one-row stub.
    """
    dest = tmp_path / "raw"
    dest.mkdir()
    for filename in ["PdM_machines.csv", "PdM_errors.csv", "PdM_maint.csv", "PdM_failures.csv"]:
        shutil.copy(raw_dir / filename, dest / filename)
    pd.read_csv(raw_dir / "PdM_telemetry.csv", nrows=1).to_csv(
        dest / "PdM_telemetry.csv", index=False
    )
    return dest
