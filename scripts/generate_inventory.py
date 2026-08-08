"""Generate a synthetic parts inventory keyed on the real dataset values.

The Azure PdM dataset has no parts, suppliers, stock levels or lead times. This
project needs them in order to reason about whether a predicted failure can
actually be acted on. Rather than inventing values at query time, they are
generated once, from a fixed seed, and committed alongside the generator.

Nothing here is measured. Every number is invented. Any downstream result that
depends on these values is a result about simulated logistics and must be
labelled as such. See docs/DATA.md section 6.

Design constraints:

- Keys are the values that actually exist in the source data: `comp1`-`comp4`
  from PdM_maint.csv and `model1`-`model4` from PdM_machines.csv. The generator
  reads those files and asserts the values it finds, so the inventory cannot
  drift away from the fleet it describes.
- For each component, the four machine models are partitioned into disjoint
  groups and one part is emitted per group. This guarantees that every
  (component, model) pair resolves to exactly one part, so a join is total and
  unambiguous by construction rather than by luck.
- Randomness uses `random.Random`, whose Mersenne Twister stream CPython
  documents as reproducible across versions. NumPy's Generator does not carry
  the same guarantee, so it is not used here.

Run:  python scripts/generate_inventory.py [--raw data/raw] [--out data/generated/parts_inventory.csv]
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

SEED = 20240601

DEFAULT_RAW = Path("data/raw")
DEFAULT_OUT = Path("data/generated/parts_inventory.csv")

N_SUPPLIERS = 5

# Per-component cost band, in arbitrary currency units. Ordered so the four
# components are distinguishable downstream; the ordering carries no meaning.
COST_BANDS = {
    "comp1": (120.0, 480.0),
    "comp2": (60.0, 240.0),
    "comp3": (300.0, 1_200.0),
    "comp4": (45.0, 180.0),
}

STOCK_RANGE = (0, 40)
LEAD_TIME_RANGE = (2, 45)


def read_domain_values(raw: Path) -> tuple[list[str], list[str]]:
    """Read the component and model identifiers from the source data itself."""
    maint = pd.read_csv(raw / "PdM_maint.csv", usecols=["comp"])
    machines = pd.read_csv(raw / "PdM_machines.csv", usecols=["model"])

    components = sorted(maint["comp"].unique().tolist())
    models = sorted(machines["model"].unique().tolist())

    if not components or not models:
        raise ValueError(f"No component or model values found under {raw}")
    return components, models


def partition(items: list[str], rng: random.Random) -> list[list[str]]:
    """Split `items` into a random number of disjoint, non-empty groups."""
    shuffled = items[:]
    rng.shuffle(shuffled)
    n_groups = rng.randint(1, len(shuffled))
    groups: list[list[str]] = [[] for _ in range(n_groups)]
    # Seed each group with one item so none is empty, then scatter the rest.
    for i in range(n_groups):
        groups[i].append(shuffled[i])
    for item in shuffled[n_groups:]:
        groups[rng.randrange(n_groups)].append(item)
    return [sorted(g) for g in groups]


def build_inventory(components: list[str], models: list[str]) -> pd.DataFrame:
    rng = random.Random(SEED)
    suppliers = [f"SUP-{i:03d}" for i in range(1, N_SUPPLIERS + 1)]
    # Each supplier has a characteristic lead time; per-part lead times vary
    # around it. Without this, lead time carries no supplier signal at all.
    supplier_base_lead = {
        s: rng.randint(*LEAD_TIME_RANGE) for s in sorted(suppliers)
    }

    rows = []
    for component in components:
        low, high = COST_BANDS.get(component, (50.0, 500.0))
        for seq, group in enumerate(partition(models, rng), start=1):
            supplier = rng.choice(suppliers)
            base = supplier_base_lead[supplier]
            rows.append(
                {
                    "part_id": f"PN-{component.upper()}-{seq:03d}",
                    "component": component,
                    "compatible_models": "|".join(group),
                    "stock_quantity": rng.randint(*STOCK_RANGE),
                    "unit_cost": round(rng.uniform(low, high), 2),
                    "supplier_id": supplier,
                    "lead_time_days": max(1, base + rng.randint(-3, 7)),
                }
            )

    df = pd.DataFrame(rows)
    # Deterministic row order, independent of dict or grouping order.
    df = df.sort_values("part_id", ignore_index=True)
    _assert_total_coverage(df, components, models)
    return df


def _assert_total_coverage(
    df: pd.DataFrame, components: list[str], models: list[str]
) -> None:
    """Every (component, model) pair must resolve to exactly one part."""
    seen: dict[tuple[str, str], int] = {}
    for component, model_list in zip(df["component"], df["compatible_models"]):
        for model in model_list.split("|"):
            seen[(component, model)] = seen.get((component, model), 0) + 1

    missing = [
        (c, m) for c in components for m in models if (c, m) not in seen
    ]
    duplicated = [pair for pair, n in seen.items() if n > 1]
    if missing:
        raise AssertionError(f"inventory does not cover: {missing}")
    if duplicated:
        raise AssertionError(f"inventory covers these pairs more than once: {duplicated}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    components, models = read_domain_values(args.raw)
    df = build_inventory(components, models)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # newline="" and lineterminator pinned so the file is byte-identical on
    # Windows and POSIX; otherwise the "reproducible" output differs by CRLF.
    df.to_csv(args.out, index=False, lineterminator="\n")

    print(f"seed: {SEED}")
    print(f"components: {components}")
    print(f"models:     {models}")
    print(f"wrote {len(df)} parts to {args.out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
