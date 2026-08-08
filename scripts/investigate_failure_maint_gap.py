"""Investigate failure records that have no exact match in the maintenance log.

docs/DATA.md section 5.1 reports that 743 of 761 failure records match a
`(machineID, datetime, component)` triple in `maint`, leaving 18 unexplained.

This script tests one hypothesis: that the 18 are timestamp-rounding artifacts,
i.e. a maint record for the same machine and component exists within +/- 1 hour
of the failure. It widens the window progressively so the result is a curve
rather than a single yes/no, and it prints the unmatched records in full so the
conclusion can be checked by hand.

Run:  python scripts/investigate_failure_maint_gap.py [data/raw]

Read-only. Writes nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

RAW = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw")

# Windows tested, in hours. 0 reproduces the exact-match figure in DATA.md.
WINDOWS_HOURS = [0, 1, 2, 3, 6, 12, 24, 24 * 7, 24 * 30]


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    failures = pd.read_csv(RAW / "PdM_failures.csv", parse_dates=["datetime"])
    maint = pd.read_csv(RAW / "PdM_maint.csv", parse_dates=["datetime"])
    return failures, maint


def match_within(
    failures: pd.DataFrame, maint: pd.DataFrame, hours: int
) -> pd.Series:
    """Boolean mask: does each failure have a maint record for the same machine
    and component within +/- `hours`?"""
    tol = pd.Timedelta(hours=hours)
    matched = []
    # Index maint by (machineID, comp) once; the fleet is 100 machines x 4
    # components, so this is 400 small groups.
    groups = {key: g["datetime"].to_numpy() for key, g in maint.groupby(["machineID", "comp"])}
    for machine_id, ts, comp in zip(
        failures["machineID"], failures["datetime"], failures["failure"]
    ):
        times = groups.get((machine_id, comp))
        if times is None:
            matched.append(False)
            continue
        delta = pd.to_timedelta(pd.Series(times) - ts).abs()
        matched.append(bool((delta <= tol).any()))
    return pd.Series(matched, index=failures.index)


def main() -> None:
    failures, maint = load()

    print("=" * 72)
    print("FAILURE -> MAINT MATCHING, BY TOLERANCE WINDOW")
    print("=" * 72)
    print(f"failure records: {len(failures)}")
    print(f"maint records:   {len(maint)}\n")
    print(f"{'window':>12}  {'matched':>8}  {'unmatched':>10}  {'%':>7}")

    masks = {}
    for hours in WINDOWS_HOURS:
        mask = match_within(failures, maint, hours)
        masks[hours] = mask
        label = f"+/- {hours}h" if hours else "exact"
        print(
            f"{label:>12}  {int(mask.sum()):>8}  {int((~mask).sum()):>10}  "
            f"{100 * mask.mean():>6.1f}%"
        )

    unmatched_exact = failures[~masks[0]]
    print("\n" + "=" * 72)
    print(f"THE {len(unmatched_exact)} RECORDS WITH NO EXACT MAINT MATCH")
    print("=" * 72)

    # For each, report the nearest maint record for the same machine+component.
    rows = []
    for idx, row in unmatched_exact.iterrows():
        same = maint[
            (maint["machineID"] == row["machineID"])
            & (maint["comp"] == row["failure"])
        ]
        if same.empty:
            rows.append(
                {
                    "machineID": row["machineID"],
                    "comp": row["failure"],
                    "failure_datetime": row["datetime"],
                    "nearest_maint_datetime": pd.NaT,
                    "delta": pd.NaT,
                    "n_maint_for_pair": 0,
                }
            )
            continue
        delta = (same["datetime"] - row["datetime"]).abs()
        nearest = same.loc[delta.idxmin()]
        rows.append(
            {
                "machineID": row["machineID"],
                "comp": row["failure"],
                "failure_datetime": row["datetime"],
                "nearest_maint_datetime": nearest["datetime"],
                "delta": nearest["datetime"] - row["datetime"],
                "n_maint_for_pair": len(same),
            }
        )

    detail = pd.DataFrame(rows).sort_values(["machineID", "failure_datetime"])
    with pd.option_context("display.width", 160, "display.max_rows", None):
        print(detail.to_string(index=False))

    print("\n" + "-" * 72)
    print("Distribution of the signed gap (nearest maint minus failure time):")
    print(detail["delta"].describe().to_string())

    # Does the same machine+component pair appear in maint at all?
    never = int((detail["n_maint_for_pair"] == 0).sum())
    print(f"\nfailures whose (machine, component) pair never appears in maint: {never}")

    # Are these failures concentrated in time or on particular machines?
    print("\nunmatched failures by component:")
    print(detail["comp"].value_counts().to_string())
    print("\nunmatched failures by machine (machines with >1):")
    vc = detail["machineID"].value_counts()
    print(vc[vc > 1].to_string() if (vc > 1).any() else "  none")
    print("\nunmatched failure timestamps, sorted:")
    print(detail["failure_datetime"].sort_values().dt.strftime("%Y-%m-%d %H:%M").to_string(index=False))

    # Control: how do the timestamps distribute overall? If all data sits on the
    # hour, rounding cannot be the explanation.
    print("\n" + "-" * 72)
    print("CONTROL — are timestamps ever off the hour?")
    for name, df in (("failures", failures), ("maint", maint)):
        minutes = df["datetime"].dt.minute
        seconds = df["datetime"].dt.second
        print(
            f"  {name:<9} distinct minute values: {sorted(minutes.unique())}  "
            f"distinct second values: {sorted(seconds.unique())}"
        )
    print(
        "  If both are [0], timestamps are exactly hourly and there is no\n"
        "  sub-hour rounding for a +/- 1h window to recover."
    )


if __name__ == "__main__":
    main()
