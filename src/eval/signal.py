"""Characterise the fault signature: how much warning does it actually give?

`docs/MILESTONE_3B.md` section 2. Milestone 3 established that the matched error
code fires in the 24 hours before 100% of failures. That says nothing about how
much *earlier* it fires, and the whole horizon question turns on that number.

Training data only. Nothing in this module touches validation or test.

Two signals per component:

- **Error onset.** The first occurrence of the matched error code in the 30 days
  before a failure. Lead time is the gap from that first occurrence to the
  failure.
- **Sensor onset.** The first hour in that window where the channel's 24-hour
  rolling mean sits more than two standard deviations from the machine's own
  baseline. Per-machine baselines, not fleet-wide: machines differ, and a
  fleet-wide threshold would measure machine identity rather than deterioration.

Both are reported with a false-positive rate: how often the signal fires with no
failure following inside the window. A signal with long lead time and a 90%
false-positive rate is not a long-horizon signal, it is noise with good timing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.eval.baselines import MATCHED_ERROR
from src.features.config import COMPONENTS

#: The window searched backwards from each failure for a first occurrence.
LOOKBACK = pd.Timedelta(days=30)

#: Deviation from the machine's own baseline that counts as sensor onset.
SENSOR_SIGMA = 2.0

#: Rolling window for the sensor channel, matching the 24h feature.
SENSOR_WINDOW = pd.Timedelta(hours=24)

#: Horizons the false-positive rate is reported at.
FP_HORIZONS = [
    pd.Timedelta(hours=24),
    pd.Timedelta(hours=72),
    pd.Timedelta(days=7),
    pd.Timedelta(days=14),
    pd.Timedelta(days=30),
]

#: Which sensor channel belongs to which component. Measured in Milestone 3 by
#: permutation importance, not assumed from the naming.
MATCHED_SENSOR = {
    "comp1": "volt",
    "comp2": "rotate",
    "comp3": "pressure",
    "comp4": "vibration",
}


@dataclass
class LeadTimes:
    component: str
    signal: str
    n_failures: int
    n_with_signal: int
    hours: np.ndarray = field(repr=False)

    @property
    def coverage(self) -> float:
        """Share of failures preceded by the signal inside the lookback."""
        return self.n_with_signal / self.n_failures if self.n_failures else float("nan")

    def percentile(self, q: float) -> float:
        return float(np.percentile(self.hours, q)) if len(self.hours) else float("nan")

    @property
    def median_hours(self) -> float:
        return self.percentile(50)


def error_lead_times(
    failures: pd.DataFrame, errors: pd.DataFrame, component: str
) -> LeadTimes:
    """Gap between the first matched error in the lookback and the failure."""
    code = MATCHED_ERROR[component]
    component_failures = failures[failures["failure"] == component]
    relevant = errors[errors["errorID"] == code]
    by_machine = {
        machine: np.sort(group["datetime"].to_numpy())
        for machine, group in relevant.groupby("machineID")
    }

    gaps = []
    for machine, when in zip(
        component_failures["machineID"], component_failures["datetime"]
    ):
        events = by_machine.get(machine)
        if events is None:
            continue
        window = events[(events > when - LOOKBACK) & (events <= when)]
        if len(window):
            gaps.append((when - window[0]) / np.timedelta64(1, "h"))

    return LeadTimes(
        component=component,
        signal=f"error code {code}",
        n_failures=len(component_failures),
        n_with_signal=len(gaps),
        hours=np.asarray(gaps, dtype=float),
    )


def sensor_onsets(
    telemetry: pd.DataFrame, machine: int, channel: str
) -> tuple[np.ndarray, np.ndarray]:
    """(timestamps, deviation-in-sigma) for one machine's rolling channel mean.

    The baseline is that machine's own mean and standard deviation over the whole
    training period. Using a trailing baseline instead would be more faithful to
    what an operator sees live, and is left as a limitation rather than silently
    swapped in: this measurement is about whether the signal exists at all, not
    about how to detect it online.
    """
    rows = telemetry[telemetry["machineID"] == machine].sort_values("datetime")
    values = rows[channel].to_numpy(dtype=float)
    when = rows["datetime"].to_numpy()

    rolling = (
        pd.Series(values, index=pd.DatetimeIndex(when))
        .rolling(SENSOR_WINDOW, min_periods=1)
        .mean()
        .to_numpy()
    )
    centre, spread = float(np.mean(rolling)), float(np.std(rolling))
    if spread == 0:
        return when, np.zeros_like(rolling)
    return when, (rolling - centre) / spread


def sensor_lead_times(
    failures: pd.DataFrame, telemetry: pd.DataFrame, component: str
) -> LeadTimes:
    channel = MATCHED_SENSOR[component]
    component_failures = failures[failures["failure"] == component]

    cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    gaps = []
    for machine, when in zip(
        component_failures["machineID"], component_failures["datetime"]
    ):
        if machine not in cache:
            cache[machine] = sensor_onsets(telemetry, machine, channel)
        times, sigma = cache[machine]
        inside = (times > when - LOOKBACK) & (times <= when)
        crossed = inside & (np.abs(sigma) >= SENSOR_SIGMA)
        if crossed.any():
            first = times[crossed][0]
            gaps.append((when - first) / np.timedelta64(1, "h"))

    return LeadTimes(
        component=component,
        signal=f"{channel} 24h mean beyond {SENSOR_SIGMA:g} sigma",
        n_failures=len(component_failures),
        n_with_signal=len(gaps),
        hours=np.asarray(gaps, dtype=float),
    )


def error_false_positive_rates(
    failures: pd.DataFrame, errors: pd.DataFrame, component: str
) -> dict[str, dict[str, float]]:
    """How often the matched code fires with no failure inside the horizon.

    Counted per error occurrence, not per hour: the question an operator asks is
    "this code just fired -- how often does that mean anything".
    """
    code = MATCHED_ERROR[component]
    relevant = errors[errors["errorID"] == code]
    component_failures = failures[failures["failure"] == component]
    failures_by_machine = {
        machine: np.sort(group["datetime"].to_numpy())
        for machine, group in component_failures.groupby("machineID")
    }

    out: dict[str, dict[str, float]] = {}
    for horizon in FP_HORIZONS:
        followed = 0
        total = 0
        for machine, when in zip(relevant["machineID"], relevant["datetime"]):
            total += 1
            events = failures_by_machine.get(machine)
            if events is None:
                continue
            if ((events > when) & (events <= when + horizon)).any():
                followed += 1
        label = _horizon_label(horizon)
        out[label] = {
            "occurrences": total,
            "followed_by_failure": followed,
            "false_positive_rate": 1 - followed / total if total else float("nan"),
        }
    return out


def _horizon_label(horizon: pd.Timedelta) -> str:
    hours = int(horizon.total_seconds() // 3600)
    return f"{hours}h" if hours < 168 else f"{hours // 24}d"


def analyse(
    failures: pd.DataFrame, errors: pd.DataFrame, telemetry: pd.DataFrame
) -> dict:
    results: dict = {"components": {}}
    for component in COMPONENTS:
        error_lead = error_lead_times(failures, errors, component)
        sensor_lead = sensor_lead_times(failures, telemetry, component)
        results["components"][component] = {
            "matched_error": MATCHED_ERROR[component],
            "matched_sensor": MATCHED_SENSOR[component],
            "n_failures": error_lead.n_failures,
            "error_lead": {
                "coverage": error_lead.coverage,
                "n_with_signal": error_lead.n_with_signal,
                "p10": error_lead.percentile(10),
                "median": error_lead.median_hours,
                "p90": error_lead.percentile(90),
                "max": error_lead.percentile(100),
            },
            "sensor_lead": {
                "coverage": sensor_lead.coverage,
                "n_with_signal": sensor_lead.n_with_signal,
                "p10": sensor_lead.percentile(10),
                "median": sensor_lead.median_hours,
                "p90": sensor_lead.percentile(90),
                "max": sensor_lead.percentile(100),
            },
            "error_false_positives": error_false_positive_rates(
                failures, errors, component
            ),
        }
    return results
