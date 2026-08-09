"""Build the train / validation / test feature matrices.

Calls `compute_features` and `compute_labels` once per prediction hour and stacks
the results. There is no batch shortcut: every row written here went through the
same function that inference will call.

Run:  python -m src.features.build [--db data/pdm.db] [--out data/generated]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.compute import compute_features, compute_labels
from src.features.config import (
    COMPONENTS,
    EMBARGO,
    FEATURE_COLUMNS,
    INDEX_COLUMNS,
    LABEL_COLUMNS,
    LABEL_HORIZON,
    MAX_PREDICTION_TIME,
    SPLIT_ORDER,
    SPLITS,
    label_column,
)
from src.features.store import FeatureStore

DEFAULT_DB = Path("data/pdm.db")
DEFAULT_OUT = Path("data/generated")
MANIFEST = DEFAULT_OUT / "build_manifest.json"


class BuildError(RuntimeError):
    pass


def prediction_times(store: FeatureStore, split: str) -> pd.DatetimeIndex:
    """Grid points that are valid prediction times for `split`.

    Two independent trims, both stated as derived quantities:

    - **Embargo.** Rows within `EMBARGO` of the split's end are dropped, so no
      row's label window reaches into the next split. `EMBARGO` is the label
      horizon; it is not a separately chosen number.
    - **Label observability.** Rows whose label window extends past the last
      observed failure are dropped globally. They cannot be confirmed negative,
      only unobserved. This binds on the test split only.
    """
    start, end = SPLITS[split]
    latest = min(end - EMBARGO, MAX_PREDICTION_TIME)
    mask = (store.times >= start) & (store.times <= latest)
    return store.times[mask]


def build_split(store: FeatureStore, split: str) -> pd.DataFrame:
    times = prediction_times(store, split)
    if len(times) == 0:
        raise BuildError(f"{split}: no prediction times in range")

    feature_blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    for as_of in times:
        feature_blocks.append(compute_features(store, as_of).to_numpy())
        label_blocks.append(compute_labels(store, as_of).to_numpy())

    n_machines = store.n_machines
    frame = pd.DataFrame(
        np.vstack(feature_blocks), columns=FEATURE_COLUMNS
    )
    for position, column in enumerate(LABEL_COLUMNS):
        frame[column] = np.vstack(label_blocks)[:, position]

    # Index columns are written as data, not as a pandas index: parquet round
    # trips them identically that way, and the consumer chooses its own index.
    frame.insert(0, "datetime", np.repeat(times.to_numpy(), n_machines))
    frame.insert(0, "machineID", np.tile(store.machine_ids, len(times)))

    # Row order is (datetime, machineID) by construction. Sorting again makes
    # that a guarantee rather than a consequence of the loop order.
    frame = frame.sort_values(["datetime", "machineID"], kind="stable").reset_index(
        drop=True
    )

    _validate(frame, split)
    return frame


def _validate(frame: pd.DataFrame, split: str) -> None:
    expected = INDEX_COLUMNS + FEATURE_COLUMNS + LABEL_COLUMNS
    if list(frame.columns) != expected:
        raise BuildError(f"{split}: column mismatch\n  got {list(frame.columns)}")

    nulls = frame[FEATURE_COLUMNS].isna().sum()
    if nulls.any():
        offenders = nulls[nulls > 0].to_dict()
        raise BuildError(
            f"{split}: null feature values: {offenders}. `hours_since_*` is NaN when "
            "a machine has no prior replacement of that component; docs/DATA.md "
            "section 3 records that every machine has one before the first "
            "telemetry hour, so this means the source data changed."
        )

    start, end = SPLITS[split]
    latest_label_end = frame["datetime"].max() + LABEL_HORIZON
    if latest_label_end > end:
        raise BuildError(
            f"{split}: a label window ends at {latest_label_end}, past the split "
            f"boundary {end}"
        )
    if frame["datetime"].min() < start:
        raise BuildError(f"{split}: rows before the split start")


def content_hash(frame: pd.DataFrame) -> str:
    """SHA-256 over the column bytes in declared order.

    Hashes content rather than the parquet file: parquet embeds a writer version
    string, so file bytes would change on a pyarrow upgrade even though nothing
    about the data had.
    """
    digest = hashlib.sha256()
    for column in frame.columns:
        digest.update(column.encode("utf-8"))
        values = frame[column].to_numpy()
        if values.dtype.kind == "M":
            values = values.astype("datetime64[ns]").astype("int64")
        digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def summarise(frame: pd.DataFrame, split: str) -> dict[str, object]:
    rows = len(frame)
    positives = {
        component: int(frame[label_column(component)].sum())
        for component in COMPONENTS
    }
    any_positive = int(
        (frame[LABEL_COLUMNS].to_numpy().sum(axis=1) > 0).sum()
    )
    return {
        "split": split,
        "rows": rows,
        "machines": int(frame["machineID"].nunique()),
        "prediction_times": int(frame["datetime"].nunique()),
        "first": str(frame["datetime"].min()),
        "last": str(frame["datetime"].max()),
        "positives": positives,
        "positive_rate": {
            component: positives[component] / rows for component in COMPONENTS
        },
        "any_component_positive": any_positive,
        "any_component_positive_rate": any_positive / rows,
        "sha256": content_hash(frame),
    }


def write_manifest(manifest: Path, entries: dict[str, dict[str, object]]) -> None:
    """Merge the feature entries into the existing manifest.

    `make data` rewrites this file and drops the `features` key. That is correct:
    rebuilding the database invalidates any feature matrix derived from it.
    """
    payload: dict[str, object] = {}
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["features"] = entries
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_all(
    db: Path = DEFAULT_DB,
    out: Path = DEFAULT_OUT,
    manifest: Path | None = MANIFEST,
    quiet: bool = False,
) -> dict[str, dict[str, object]]:
    def say(message: str) -> None:
        if not quiet:
            print(message)

    if not db.exists():
        raise BuildError(f"database not found: {db}. Run `make data` first.")

    say(f"loading {db}")
    store = FeatureStore.from_database(db)
    say(f"  grid: {store.n_times:,} hourly points x {store.n_machines} machines")

    out.mkdir(parents=True, exist_ok=True)
    entries: dict[str, dict[str, object]] = {}

    for split in SPLIT_ORDER:
        say(f"building {split} ...")
        frame = build_split(store, split)
        path = out / f"features_{split}.parquet"
        frame.to_parquet(path, index=False, compression="snappy")
        entry = summarise(frame, split)
        entry["path"] = path.as_posix()
        entries[split] = entry
        say(
            f"  {split}: {entry['rows']:,} rows, "
            f"{entry['prediction_times']:,} prediction times, "
            f"{entry['first']} -> {entry['last']}"
        )

    if manifest is not None:
        write_manifest(manifest, entries)
        say(f"wrote {manifest}")

    return entries


def report(entries: dict[str, dict[str, object]]) -> None:
    print("\n" + "=" * 78)
    print("SPLITS")
    print("=" * 78)
    print(f"{'split':<7}{'rows':>12}{'hours':>9}  {'first':<20}{'last':<20}")
    for split in SPLIT_ORDER:
        e = entries[split]
        print(
            f"{split:<7}{e['rows']:>12,}{e['prediction_times']:>9,}  "
            f"{e['first']:<20}{e['last']:<20}"
        )

    print("\n" + "=" * 78)
    print("POSITIVES PER COMPONENT (count and rate)")
    print("=" * 78)
    header = f"{'split':<7}" + "".join(f"{c:>20}" for c in COMPONENTS) + f"{'any':>20}"
    print(header)
    for split in SPLIT_ORDER:
        e = entries[split]
        cells = "".join(
            f"{e['positives'][c]:>10,} {100 * e['positive_rate'][c]:>7.3f}%"
            for c in COMPONENTS
        )
        cells += (
            f"{e['any_component_positive']:>10,} "
            f"{100 * e['any_component_positive_rate']:>7.3f}%"
        )
        print(f"{split:<7}{cells}")

    test_positive = entries["test"]["any_component_positive"]
    print(
        "\nThe test split contains "
        f"{test_positive:,} positive rows across "
        f"{sum(entries['test']['positives'].values()):,} component-level positives, "
        "derived from roughly 127 distinct failure events. Recall estimated on a "
        "sample this small carries wide uncertainty and must be reported with an "
        "interval, not as a point estimate."
    )

    print("\n" + "=" * 78)
    print("CONTENT HASHES")
    print("=" * 78)
    for split in SPLIT_ORDER:
        print(f"  {split:<7}{entries[split]['sha256']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the feature matrices.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()

    entries = build_all(args.db, args.out, args.manifest)
    report(entries)


if __name__ == "__main__":
    main()
