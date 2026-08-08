"""Schema validation must pass on the real files and fail on corrupted ones.

The second half matters more than the first. A schema that never rejects
anything is indistinguishable from no schema at all, so each failure mode gets
its own test with a named expectation rather than one broad "invalid data
raises" assertion.
"""

from __future__ import annotations

import pandas as pd
import pytest
from pandera.errors import SchemaError, SchemaErrors

from src.data.ingest import IngestError, load_inventory, load_source
from src.data.schemas import (
    EXPECTED_ROW_COUNTS,
    PARTS_INVENTORY_SCHEMA,
    SOURCE_SCHEMAS,
    TELEMETRY_SCHEMA,
)

SCHEMA_FAILURE = (SchemaError, SchemaErrors)


# --------------------------------------------------------------------------
# Real data passes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("table", sorted(SOURCE_SCHEMAS))
def test_real_source_file_validates(raw_dir, table):
    df = load_source(raw_dir, table)
    assert len(df) == EXPECTED_ROW_COUNTS[table]


def test_real_inventory_validates(inventory_csv):
    df = load_inventory(inventory_csv)
    assert len(df) > 0
    assert df["part_id"].is_unique


def test_generated_inventory_covers_every_component_model_pair(
    inventory_csv, raw_dir
):
    inventory = pd.read_csv(inventory_csv)
    components = set(pd.read_csv(raw_dir / "PdM_maint.csv")["comp"].unique())
    models = set(pd.read_csv(raw_dir / "PdM_machines.csv")["model"].unique())

    covered = {
        (component, model)
        for component, model_list in zip(
            inventory["component"], inventory["compatible_models"]
        )
        for model in model_list.split("|")
    }
    assert covered == {(c, m) for c in components for m in models}


# --------------------------------------------------------------------------
# Corrupted data fails, one named failure mode per test
# --------------------------------------------------------------------------


@pytest.fixture
def good_telemetry(raw_dir) -> pd.DataFrame:
    df = pd.read_csv(raw_dir / "PdM_telemetry.csv", nrows=200)
    df["datetime"] = pd.to_datetime(df["datetime"], format="mixed")
    return df


def test_valid_slice_is_actually_valid(good_telemetry):
    """Guards the corruption tests below: if this fails they prove nothing."""
    TELEMETRY_SCHEMA.validate(good_telemetry, lazy=True)


def test_unexpected_column_is_rejected(good_telemetry):
    df = good_telemetry.assign(unexpected_sensor=1.0)
    with pytest.raises(SCHEMA_FAILURE):
        TELEMETRY_SCHEMA.validate(df, lazy=True)


def test_missing_column_is_rejected(good_telemetry):
    df = good_telemetry.drop(columns=["vibration"])
    with pytest.raises(SCHEMA_FAILURE):
        TELEMETRY_SCHEMA.validate(df, lazy=True)


def test_null_is_rejected(good_telemetry):
    df = good_telemetry.copy()
    df.loc[0, "volt"] = None
    with pytest.raises(SCHEMA_FAILURE):
        TELEMETRY_SCHEMA.validate(df, lazy=True)


def test_uncoercible_dtype_is_rejected(good_telemetry):
    df = good_telemetry.copy()
    df["volt"] = df["volt"].astype(str)
    df.loc[0, "volt"] = "not-a-number"
    with pytest.raises(SCHEMA_FAILURE):
        TELEMETRY_SCHEMA.validate(df, lazy=True)


def test_out_of_range_value_is_rejected(good_telemetry):
    df = good_telemetry.copy()
    df.loc[0, "volt"] = -1.0
    with pytest.raises(SCHEMA_FAILURE):
        TELEMETRY_SCHEMA.validate(df, lazy=True)


def test_machine_id_outside_the_fleet_is_rejected(good_telemetry):
    df = good_telemetry.copy()
    df.loc[0, "machineID"] = 9999
    with pytest.raises(SCHEMA_FAILURE):
        TELEMETRY_SCHEMA.validate(df, lazy=True)


def test_timestamp_outside_the_observation_window_is_rejected(good_telemetry):
    df = good_telemetry.copy()
    df.loc[0, "datetime"] = pd.Timestamp("2099-01-01 00:00:00")
    with pytest.raises(SCHEMA_FAILURE):
        TELEMETRY_SCHEMA.validate(df, lazy=True)


def test_duplicate_natural_key_is_rejected(good_telemetry):
    df = pd.concat([good_telemetry, good_telemetry.head(1)], ignore_index=True)
    with pytest.raises(SCHEMA_FAILURE):
        TELEMETRY_SCHEMA.validate(df, lazy=True)


@pytest.mark.parametrize(
    "table, column, bad_value",
    [
        ("maint", "comp", "comp5"),
        ("failures", "failure", "comp0"),
        ("errors", "errorID", "error9"),
        ("machines", "model", "model7"),
    ],
)
def test_value_outside_the_declared_domain_is_rejected(
    raw_dir, table, column, bad_value
):
    filename, schema = SOURCE_SCHEMAS[table]
    df = pd.read_csv(raw_dir / filename, nrows=50)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], format="mixed")
    df.loc[0, column] = bad_value
    with pytest.raises(SCHEMA_FAILURE):
        schema.validate(df, lazy=True)


@pytest.mark.parametrize(
    "column, bad_value",
    [
        ("compatible_models", "model5"),
        ("compatible_models", "widget"),
        ("supplier_id", "SUPPLIER-1"),
        ("stock_quantity", -1),
        ("unit_cost", 0.0),
        ("lead_time_days", 0),
        ("component", "comp9"),
    ],
)
def test_corrupt_inventory_is_rejected(inventory_csv, column, bad_value):
    df = pd.read_csv(inventory_csv)
    df.loc[0, column] = bad_value
    with pytest.raises(SCHEMA_FAILURE):
        PARTS_INVENTORY_SCHEMA.validate(df, lazy=True)


# --------------------------------------------------------------------------
# Corruption on disk, through the real loader
# --------------------------------------------------------------------------


def test_corrupt_file_on_disk_is_rejected_by_the_loader(corrupt_raw):
    path = corrupt_raw / "PdM_machines.csv"
    df = pd.read_csv(path)
    df.loc[0, "model"] = "model99"
    df.to_csv(path, index=False)

    with pytest.raises(SCHEMA_FAILURE):
        load_source(corrupt_raw, "machines")


def test_missing_source_file_raises_a_named_error(tmp_path):
    with pytest.raises(IngestError, match="source file missing"):
        load_source(tmp_path, "machines")


def test_missing_inventory_raises_a_named_error(tmp_path):
    with pytest.raises(IngestError, match="inventory missing"):
        load_inventory(tmp_path / "nope.csv")
