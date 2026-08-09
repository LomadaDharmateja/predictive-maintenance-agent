"""A small, fully controlled dataset for the leakage tests.

Real data cannot answer "does inserting a record at t + 1s change this feature",
because you cannot insert into it. These frames have the same shape as the source
tables -- a balanced hourly telemetry panel, event logs keyed on machineID, maint
history predating the grid -- at a size where every record is accounted for.

Values are deterministic functions of (machine, hour). Nothing here is random, so
there is no seed to fix.
"""

from __future__ import annotations

import pandas as pd

from src.features.config import COMPONENTS, ERROR_IDS, SENSORS

GRID_START = pd.Timestamp("2015-01-01 06:00:00")
N_HOURS = 400
MACHINE_IDS = [1, 2, 3]
MODELS = {1: "model1", 2: "model2", 3: "model3"}
AGES = {1: 5, 2: 12, 3: 0}

#: Every machine gets a replacement of every component before the grid begins,
#: mirroring the real data, where maint starts seven months before telemetry.
#: Without it, `hours_since_*` would be NaN at the start of the series.
PRE_GRID_MAINT = GRID_START - pd.Timedelta(days=30)


def grid(n_hours: int = N_HOURS) -> pd.DatetimeIndex:
    return pd.date_range(GRID_START, periods=n_hours, freq="h")


def make_frames(n_hours: int = N_HOURS) -> dict[str, pd.DataFrame]:
    """The five source tables, as `FeatureStore.from_frames` expects them."""
    times = grid(n_hours)

    telemetry_rows = []
    for hour, when in enumerate(times):
        for machine in MACHINE_IDS:
            row = {"machineID": machine, "datetime": when}
            for offset, sensor in enumerate(SENSORS):
                # Distinct per sensor and per machine, and varying with time, so
                # a rolling mean or std that reads the wrong row produces a
                # different number rather than coincidentally the same one.
                row[sensor] = 100.0 + 10 * offset + machine + (hour % 17) * 1.5
            telemetry_rows.append(row)

    maint_rows = [
        {"machineID": machine, "datetime": PRE_GRID_MAINT, "comp": component}
        for machine in MACHINE_IDS
        for component in COMPONENTS
    ]
    maint_rows += [
        {"machineID": 1, "datetime": times[50], "comp": "comp1"},
        {"machineID": 2, "datetime": times[120], "comp": "comp3"},
    ]

    error_rows = [
        {"machineID": 1, "datetime": times[30], "errorID": "error1"},
        {"machineID": 1, "datetime": times[31], "errorID": "error2"},
        {"machineID": 2, "datetime": times[200], "errorID": "error5"},
    ]

    failure_rows = [
        {"machineID": 1, "datetime": times[60], "failure": "comp2"},
        {"machineID": 3, "datetime": times[300], "failure": "comp4"},
    ]

    return {
        "telemetry": pd.DataFrame(telemetry_rows),
        "errors": pd.DataFrame(error_rows, columns=["machineID", "datetime", "errorID"]),
        "maint": pd.DataFrame(maint_rows, columns=["machineID", "datetime", "comp"]),
        "failures": pd.DataFrame(
            failure_rows, columns=["machineID", "datetime", "failure"]
        ),
        "machines": pd.DataFrame(
            [
                {"machineID": machine, "model": MODELS[machine], "age": AGES[machine]}
                for machine in MACHINE_IDS
            ]
        ),
    }


# ----------------------------------------------------------------------
# Perturbations. Each returns a new dict; the input is never mutated.
# ----------------------------------------------------------------------


def _copy(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {name: frame.copy(deep=True) for name, frame in frames.items()}


def _append(
    frames: dict[str, pd.DataFrame], table: str, row: dict
) -> dict[str, pd.DataFrame]:
    out = _copy(frames)
    out[table] = pd.concat(
        [out[table], pd.DataFrame([row])], ignore_index=True
    )
    return out


def add_error(
    frames: dict[str, pd.DataFrame],
    machine: int,
    when: pd.Timestamp,
    error_id: str = "error1",
) -> dict[str, pd.DataFrame]:
    assert error_id in ERROR_IDS
    return _append(
        frames, "errors", {"machineID": machine, "datetime": when, "errorID": error_id}
    )


def add_maint(
    frames: dict[str, pd.DataFrame],
    machine: int,
    when: pd.Timestamp,
    component: str = "comp1",
) -> dict[str, pd.DataFrame]:
    assert component in COMPONENTS
    return _append(
        frames, "maint", {"machineID": machine, "datetime": when, "comp": component}
    )


def add_failure(
    frames: dict[str, pd.DataFrame],
    machine: int,
    when: pd.Timestamp,
    component: str = "comp1",
) -> dict[str, pd.DataFrame]:
    assert component in COMPONENTS
    return _append(
        frames,
        "failures",
        {"machineID": machine, "datetime": when, "failure": component},
    )


def bump_telemetry(
    frames: dict[str, pd.DataFrame],
    machine: int,
    when: pd.Timestamp,
    delta: float = 50.0,
) -> dict[str, pd.DataFrame]:
    """Change the sensor readings at one (machine, time).

    Telemetry is perturbed rather than appended to: the panel is balanced by
    definition, so an extra row at an existing timestamp is a duplicate and an
    extra row at a new timestamp breaks the grid. Changing the value asks the
    same question -- does this reading reach a feature it should not.
    """
    out = _copy(frames)
    telemetry = out["telemetry"]
    mask = (telemetry["machineID"] == machine) & (telemetry["datetime"] == when)
    if not mask.any():
        raise AssertionError(f"no telemetry row for machine {machine} at {when}")
    for sensor in SENSORS:
        telemetry.loc[mask, sensor] = telemetry.loc[mask, sensor] + delta
    return out
