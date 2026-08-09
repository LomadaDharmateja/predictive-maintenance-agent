"""Grid-aligned arrays that `compute_features` reads.

Why a store rather than raw DataFrames: features are computed one `as_of` at a
time, ~8,700 times over a year, and re-filtering 876,100 telemetry rows on every
call is not viable. The store reshapes each source table once into arrays indexed
by (grid position, machine) so that a single `as_of` is a slice.

The reshaping is part of the one feature implementation, not a shortcut around
it. `tests/test_no_future_leakage.py` perturbs the *source frames* and rebuilds
the store, so the invariance tests exercise this file as well as `compute.py`.

Leakage safety of the precomputation
------------------------------------
Two structures here are cumulative, which is the only part that deserves
scrutiny:

- `error_cumsum[e][x]` is the number of `e` events that became visible strictly
  before grid position `x`. A window count is a difference of two entries at
  positions <= the query index, so it can never read an event that arrives later.
- `maint_last_ns[c][i]` is the timestamp of the most recent `c` replacement at or
  before `times[i]`, built with `side="right"` so a record stamped exactly
  `times[i]` is included and anything after it is not.

Both are asserted directly in the leakage tests rather than argued for here.

Event visibility
----------------
Each event is assigned a grid position by exact arithmetic on the hourly grid:

    arrival(tau) = ceil((tau - times[0]) / 1 hour)

An event with arrival `a` falls inside the window `(times[j] - k hours, times[j]]`
for exactly `j` in `[a, a + k - 1]`, which is what turns a window count into a
difference of two prefix sums.

Two consequences worth stating, because both are easy to get wrong:

- A record at `times[j] + 1s` gets arrival `j + 1`. It is invisible to a window
  ending at `times[j]` and visible at `times[j + 1]`. That is the boundary
  docs/DATA.md section 5.1 specifies, and milestone section 4.2 tests it.
- `arrival` is allowed to be negative. An event shortly *before* the first grid
  point genuinely belongs in the early windows, so the count arrays carry
  `GRID_PAD` rows of history before position zero. Deriving arrival from
  `searchsorted` instead would collapse every pre-grid event onto position 0 and
  count a replacement from six months earlier as if it had just happened. The
  Azure PdM error and failure logs happen to start exactly at the grid origin so
  this never bites on the shipped data -- which is precisely why it needs to be
  correct by construction rather than by luck.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.config import (
    COMPONENTS,
    ERROR_IDS,
    ERROR_WINDOWS,
    LABEL_HORIZON,
    MACHINE_MODELS,
    SENSORS,
    TELEMETRY_WINDOWS,
)

NS_PER_HOUR = 3_600_000_000_000

#: Rows of history kept before grid position zero, and of headroom after the last
#: grid point. Sized from the widest window any consumer asks for, so no caller
#: can request a range the prefix arrays do not cover.
GRID_PAD = max(
    int(window.value // NS_PER_HOUR)
    for window in (*TELEMETRY_WINDOWS.values(), *ERROR_WINDOWS.values(), LABEL_HORIZON)
)


class StoreError(ValueError):
    """Raised when the source frames cannot form a valid store."""


@dataclass(frozen=True)
class FeatureStore:
    """Immutable, grid-aligned view of the source tables."""

    times: pd.DatetimeIndex
    times_ns: np.ndarray  # (T,) int64, cached to avoid repeated conversion
    machine_ids: np.ndarray  # (M,) sorted

    sensors: dict[str, np.ndarray]  # sensor -> (T, M) float64
    error_cumsum: dict[str, np.ndarray]  # errorID -> (T + 1, M) int64
    maint_last_ns: dict[str, np.ndarray]  # comp -> (T, M) int64, NaT as iinfo.min
    failure_cumsum: dict[str, np.ndarray]  # comp -> (T + 1, M) int64

    age: np.ndarray  # (M,) float64
    model_indicator: dict[str, np.ndarray]  # model -> (M,) float64

    @property
    def n_times(self) -> int:
        return len(self.times)

    @property
    def n_machines(self) -> int:
        return len(self.machine_ids)

    def index_of(self, as_of: pd.Timestamp) -> int:
        """Grid position of `as_of`. Raises if it is not a grid point.

        Off-grid prediction times are rejected rather than snapped to the nearest
        grid point: `maint_last_ns` is indexed by grid position, so snapping would
        silently answer a slightly different question than the one asked.
        """
        position = int(np.searchsorted(self.times_ns, as_of.value, side="left"))
        if position >= len(self.times_ns) or self.times_ns[position] != as_of.value:
            raise StoreError(
                f"as_of {as_of} is not on the telemetry grid "
                f"({self.times[0]} to {self.times[-1]}, hourly)"
            )
        return position

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_frames(
        cls,
        telemetry: pd.DataFrame,
        errors: pd.DataFrame,
        maint: pd.DataFrame,
        failures: pd.DataFrame,
        machines: pd.DataFrame,
    ) -> "FeatureStore":
        telemetry = _with_datetime(telemetry, "telemetry")
        errors = _with_datetime(errors, "errors")
        maint = _with_datetime(maint, "maint")
        failures = _with_datetime(failures, "failures")

        machine_ids = np.sort(machines["machineID"].unique())
        times = pd.DatetimeIndex(np.sort(telemetry["datetime"].unique())).as_unit("ns")
        _assert_hourly_grid(times)
        _assert_balanced_panel(telemetry, times, machine_ids)

        times_ns = times.asi8.copy()
        n_times, n_machines = len(times), len(machine_ids)
        column_of = pd.Series(np.arange(n_machines), index=machine_ids)

        sensors = _pivot_sensors(telemetry, times, machine_ids)

        origin_ns = int(times_ns[0])

        error_cumsum = {
            error_id: _arrival_cumsum(
                errors.loc[errors["errorID"] == error_id],
                origin_ns,
                column_of,
                n_times,
                n_machines,
            )
            for error_id in ERROR_IDS
        }

        failure_cumsum = {
            component: _arrival_cumsum(
                failures.loc[failures["failure"] == component],
                origin_ns,
                column_of,
                n_times,
                n_machines,
            )
            for component in COMPONENTS
        }

        maint_last_ns = {
            component: _last_event_ns(
                maint.loc[maint["comp"] == component],
                times_ns,
                column_of,
                n_times,
                n_machines,
            )
            for component in COMPONENTS
        }

        machines_sorted = machines.set_index("machineID").loc[machine_ids]
        age = machines_sorted["age"].to_numpy(dtype="float64")
        model_indicator = {
            model: (machines_sorted["model"].to_numpy() == model).astype("float64")
            for model in MACHINE_MODELS
        }

        return cls(
            times=times,
            times_ns=times_ns,
            machine_ids=machine_ids,
            sensors=sensors,
            error_cumsum=error_cumsum,
            maint_last_ns=maint_last_ns,
            failure_cumsum=failure_cumsum,
            age=age,
            model_indicator=model_indicator,
        )

    @classmethod
    def from_database(cls, db: Path) -> "FeatureStore":
        connection = sqlite3.connect(db)
        try:
            frames = {
                table: pd.read_sql_query(f"SELECT * FROM {table}", connection)
                for table in ["telemetry", "errors", "maint", "failures", "machines"]
            }
        finally:
            connection.close()
        return cls.from_frames(**frames)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _with_datetime(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if "datetime" not in frame.columns:
        raise StoreError(f"{name}: no datetime column")
    out = frame.copy()
    # Pinned to nanosecond resolution. pandas 3 preserves whatever unit it infers
    # -- microseconds, for these files -- and `Timestamp.value` is always in
    # nanoseconds, so mixing the two silently rescales every duration by 1000.
    out["datetime"] = pd.to_datetime(out["datetime"], format="mixed").dt.as_unit("ns")
    return out


def _assert_hourly_grid(times: pd.DatetimeIndex) -> None:
    if len(times) < 2:
        raise StoreError("telemetry has fewer than two distinct timestamps")
    gaps = np.diff(times.asi8)
    if not np.all(gaps == NS_PER_HOUR):
        distinct = np.unique(gaps)
        raise StoreError(
            "telemetry timestamps are not a contiguous hourly grid; "
            f"observed gaps (ns): {distinct[:5]}"
        )


def _assert_balanced_panel(
    telemetry: pd.DataFrame, times: pd.DatetimeIndex, machine_ids: np.ndarray
) -> None:
    """Every machine must have exactly one reading at every grid time.

    The sensor arrays are dense (T, M) matrices, which is only meaningful if the
    panel is complete. docs/DATA.md section 3 records that it is; this turns that
    observation into a precondition.
    """
    expected = len(times) * len(machine_ids)
    if len(telemetry) != expected:
        raise StoreError(
            f"telemetry is not a balanced panel: {len(telemetry):,} rows, expected "
            f"{len(times):,} timestamps x {len(machine_ids)} machines = {expected:,}"
        )
    if telemetry.duplicated(["machineID", "datetime"]).any():
        raise StoreError("telemetry contains duplicate (machineID, datetime) rows")
    observed = np.sort(telemetry["machineID"].unique())
    if not np.array_equal(observed, machine_ids):
        raise StoreError(
            "telemetry machine set differs from the machines table: "
            f"{len(observed)} vs {len(machine_ids)} distinct IDs"
        )


def _pivot_sensors(
    telemetry: pd.DataFrame, times: pd.DatetimeIndex, machine_ids: np.ndarray
) -> dict[str, np.ndarray]:
    ordered = telemetry.sort_values(["datetime", "machineID"], kind="stable")
    out = {}
    for sensor in SENSORS:
        matrix = ordered[sensor].to_numpy(dtype="float64").reshape(
            len(times), len(machine_ids)
        )
        out[sensor] = matrix
    return out


def _arrival_positions(
    frame: pd.DataFrame, origin_ns: int, column_of: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    """(grid arrival position, machine column) for each event row.

    Arrival is `ceil((tau - origin) / 1 hour)`, in integers so it is exact, and
    it may be negative. `-((-x) // h)` is ceiling division for both signs;
    numpy's `//` floors, which would be wrong for pre-grid events.
    """
    event_ns = frame["datetime"].to_numpy(dtype="datetime64[ns]").astype("int64")
    arrival = -np.floor_divide(-(event_ns - origin_ns), NS_PER_HOUR)
    columns = column_of.reindex(frame["machineID"]).to_numpy()
    if np.isnan(columns.astype("float64")).any():
        raise StoreError("event table references a machineID absent from machines")
    return arrival, columns.astype("int64")


def _arrival_cumsum(
    frame: pd.DataFrame,
    origin_ns: int,
    column_of: pd.Series,
    n_times: int,
    n_machines: int,
) -> np.ndarray:
    """Exclusive prefix sums of event arrivals, shape (T + 2*GRID_PAD + 1, M).

    Row `x` counts events with arrival position `< x - GRID_PAD`. A window is the
    difference of two rows at positions no later than the window's own end, so it
    can never read an event that arrives after it closes.

    The padding is what lets an event before the first grid point be counted in
    the early windows at its true distance rather than being flattened onto
    position zero.
    """
    n_rows = n_times + 2 * GRID_PAD
    counts = np.zeros((n_rows, n_machines), dtype="int64")
    if len(frame):
        arrival, columns = _arrival_positions(frame, origin_ns, column_of)
        rows = arrival + GRID_PAD
        # Anything outside the padded span is further away than the widest window
        # any caller can ask for, so dropping it changes no result.
        keep = (rows >= 0) & (rows < n_rows)
        np.add.at(counts, (rows[keep], columns[keep]), 1)
    prefix = np.zeros((n_rows + 1, n_machines), dtype="int64")
    np.cumsum(counts, axis=0, out=prefix[1:])
    return prefix


def _last_event_ns(
    frame: pd.DataFrame,
    times_ns: np.ndarray,
    column_of: pd.Series,
    n_times: int,
    n_machines: int,
) -> np.ndarray:
    """Timestamp of the latest event at or before each grid point, shape (T, M).

    `side="right"` is the whole point: an event stamped exactly `times[i]` is
    included at `i`. docs/DATA.md section 5.1 -- the boundary is closed at `t`.
    """
    out = np.full((n_times, n_machines), np.iinfo("int64").min, dtype="int64")
    if not len(frame):
        return out

    ordered = frame.sort_values("datetime", kind="stable")
    columns = column_of.reindex(ordered["machineID"]).to_numpy().astype("int64")
    event_ns = ordered["datetime"].to_numpy(dtype="datetime64[ns]").astype("int64")

    for machine_column in np.unique(columns):
        machine_events = event_ns[columns == machine_column]
        position = np.searchsorted(machine_events, times_ns, side="right") - 1
        has_prior = position >= 0
        out[has_prior, machine_column] = machine_events[position[has_prior]]
    return out
