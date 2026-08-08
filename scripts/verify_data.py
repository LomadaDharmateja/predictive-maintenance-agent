"""
Verify the Azure PdM dataset before designing anything on top of it.

Run:  python scripts/verify_data.py data/raw

Read the output yourself. Do not skim it. Every number here is one you
will have to defend in an interview, and every assumption you do not
check now becomes a false claim in the README later.
"""

import sys
from pathlib import Path

import pandas as pd

RAW = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw")

FILES = {
    "telemetry": "PdM_telemetry.csv",
    "errors": "PdM_errors.csv",
    "maint": "PdM_maint.csv",
    "failures": "PdM_failures.csv",
    "machines": "PdM_machines.csv",
}


def main() -> None:
    frames = {}

    print("=" * 70)
    print("1. SHAPE AND COLUMNS")
    print("=" * 70)
    for name, filename in FILES.items():
        path = RAW / filename
        if not path.exists():
            print(f"MISSING: {path}")
            continue
        df = pd.read_csv(path)
        frames[name] = df
        print(f"\n{name}: {df.shape[0]:,} rows x {df.shape[1]} cols")
        print(f"  columns: {list(df.columns)}")
        print(f"  nulls:   {df.isnull().sum().to_dict()}")

    print("\n" + "=" * 70)
    print("2. DO THE TABLES ACTUALLY JOIN?")
    print("=" * 70)
    if "machines" in frames:
        master = set(frames["machines"]["machineID"])
        print(f"\nmachines table: {len(master)} unique machineID")
        for name, df in frames.items():
            if name == "machines" or "machineID" not in df.columns:
                continue
            ids = set(df["machineID"])
            orphans = ids - master
            print(
                f"  {name:<10} {len(ids):>4} distinct IDs | "
                f"orphans (not in machines): {len(orphans)}"
            )
            # An orphan here means the key does not really join.

    print("\n" + "=" * 70)
    print("3. TIME COVERAGE — is it really a time series?")
    print("=" * 70)
    for name in ("telemetry", "errors", "maint", "failures"):
        if name not in frames:
            continue
        df = frames[name]
        dt = pd.to_datetime(df["datetime"])
        print(f"\n{name}: {dt.min()}  ->  {dt.max()}")
        if name == "telemetry":
            per_machine = df.groupby("machineID").size()
            print(f"  readings per machine: min {per_machine.min():,} "
                  f"median {int(per_machine.median()):,} max {per_machine.max():,}")
            # Diff within each machine. Diffing the pooled column instead
            # interleaves 100 machines and reports 0 days for nearly every row.
            gaps = (
                df.assign(datetime=dt)
                .sort_values(["machineID", "datetime"])
                .groupby("machineID")["datetime"]
                .diff()
                .value_counts()
                .head(3)
            )
            print(f"  most common interval between readings:\n{gaps.to_string()}")

    print("\n" + "=" * 70)
    print("4. THE TARGET — how rare is failure?")
    print("=" * 70)
    if "failures" in frames:
        f = frames["failures"]
        print(f"\ntotal failure events: {len(f):,}")
        print(f"by component:\n{f['failure'].value_counts().to_string()}")
        print(f"\nmachines that ever failed: {f['machineID'].nunique()} "
              f"of {len(master) if 'machines' in frames else '?'}")

        if "telemetry" in frames:
            n_hours = len(frames["telemetry"])
            rate = 100 * len(f) / n_hours
            print(f"\nfailure events as % of telemetry hours: {rate:.4f}%")
            print(f"  -> a model predicting 'no failure' always scores "
                  f"{100 - rate:.4f}% accuracy.")
            print("  -> ACCURACY IS USELESS HERE. Report precision, recall, PR-AUC.")

    print("\n" + "=" * 70)
    print("5. MAINTENANCE vs FAILURES — is failures a subset?")
    print("=" * 70)
    if "maint" in frames and "failures" in frames:
        m = frames["maint"].copy()
        f = frames["failures"].copy()
        m["datetime"] = pd.to_datetime(m["datetime"])
        f["datetime"] = pd.to_datetime(f["datetime"])
        m_keys = set(zip(m["machineID"], m["datetime"], m["comp"]))
        f_keys = set(zip(f["machineID"], f["datetime"], f["failure"]))
        overlap = len(f_keys & m_keys)
        print(f"\nmaint records:    {len(m_keys):,}")
        print(f"failure records:  {len(f_keys):,}")
        print(f"failures also in maint: {overlap:,} ({100*overlap/len(f_keys):.1f}%)")
        print("\nIf this is ~100%, then maint contains the answer.")
        print("Using maint as a feature = LEAKAGE. Decide now how you handle it,")
        print("and write the decision down in docs/DATA.md.")

    print("\n" + "=" * 70)
    print("6. COMPONENTS AND MODELS — the keys your inventory must match")
    print("=" * 70)
    if "maint" in frames:
        print(f"\ncomponents in maint: "
              f"{sorted(frames['maint']['comp'].unique())}")
    if "errors" in frames:
        print(f"error types:         {sorted(frames['errors']['errorID'].unique())}")
    if "machines" in frames:
        print(f"machine models:      "
              f"{sorted(frames['machines']['model'].unique())}")
        print(f"age range:           {frames['machines']['age'].min()} - "
              f"{frames['machines']['age'].max()}")

    print("\n" + "=" * 70)
    print("Now write docs/DATA.md from these numbers. Nothing you did not")
    print("see printed above belongs in that file.")
    print("=" * 70)


if __name__ == "__main__":
    main()