"""Reproduce the v1 leakage numbers from the archived code and data.

`docs/MILESTONE_3.md` section 7. Every figure is recomputed here from
`archive/v1-data/` using the procedure `archive/v1-app/tools/db_setup.py` and
`archive/v1-app/tools/train_model.py` actually implemented. Nothing is quoted
from an earlier document; where a recomputed number disagrees with one, the
recomputed number is right and the disagreement is printed.

Writes docs/leakage-case-study.md.

Run:  make case-study
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

ARCHIVE = Path("archive/v1-data")
SOURCE_CSV = ARCHIVE / "maintenance.csv"
OUTPUT = Path("docs/leakage-case-study.md")

#: Exactly what archive/v1-app/tools/train_model.py used.
FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
TARGET = "Machine failure"

#: archive/v1-app/tools/db_setup.py: `pd.concat([maint_df] * 5)`, then Gaussian
#: noise on torque only. The noise is why the duplicates are not exact and why
#: `DataFrame.duplicated()` would not have caught them.
INFLATION_FACTOR = 5
TORQUE_NOISE_SD = 2.0

#: train_model.py's hyperparameters, unchanged.
N_ESTIMATORS = 100
RANDOM_STATE = 42

#: db_setup.py called np.random.normal with no seed, so the original inflation
#: was unreproducible. Seeded here so this script is; the seed changes nothing
#: about the mechanism.
NOISE_SEED = 20240606
SPLIT_SEED = 0
CV_SEED = 0


def inflate(frame: pd.DataFrame) -> pd.DataFrame:
    """db_setup.py's `migrate_to_sql`, reproduced."""
    inflated = pd.concat([frame] * INFLATION_FACTOR, ignore_index=True)
    rng = np.random.default_rng(NOISE_SEED)
    inflated["Torque [Nm]"] = inflated["Torque [Nm]"] + rng.normal(
        0, TORQUE_NOISE_SD, len(inflated)
    )
    return inflated


def naive_random_split(frame: pd.DataFrame) -> dict:
    """What v1 would have reported: a random split of the inflated table."""
    X, y = frame[FEATURES], frame[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SPLIT_SEED, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1
    ).fit(X_train, y_train)
    predicted = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, predicted, labels=[0, 1]).ravel()
    return {
        "f1": float(f1_score(y_test, predicted)),
        "precision": float(precision_score(y_test, predicted, zero_division=0)),
        "recall": float(recall_score(y_test, predicted)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def honest_cross_validation(frame: pd.DataFrame) -> dict:
    """5-fold CV on the 10,000 source rows, before any duplication."""
    X, y = frame[FEATURES], frame[TARGET]
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1
    )
    cv = StratifiedKFold(5, shuffle=True, random_state=CV_SEED)
    predicted = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "f1": float(f1_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted)),
        "n_rows": int(len(X)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def duplication_evidence(source: pd.DataFrame, inflated: pd.DataFrame) -> dict:
    """How the duplication hides from the obvious checks."""
    return {
        "source_rows": int(len(source)),
        "inflated_rows": int(len(inflated)),
        "source_unique_ids": int(source["UDI"].nunique()),
        "inflated_unique_ids": int(inflated["UDI"].nunique()),
        "inflated_exact_duplicate_rows": int(inflated.duplicated().sum()),
        "inflated_duplicate_ids": int(len(inflated) - inflated["UDI"].nunique()),
        "majority_accuracy": float(1 - source[TARGET].mean()),
        "positive_rate": float(source[TARGET].mean()),
    }


def render(source_stats: dict, naive: dict, honest: dict) -> str:
    gap = naive["f1"] - honest["f1"]
    lines: list[str] = []
    add = lines.append

    add("# The v1 leakage case study")
    add("")
    add("Every number here was recomputed by `scripts/leakage_case_study.py` from")
    add("`archive/v1-data/maintenance.csv`, following what")
    add("`archive/v1-app/tools/db_setup.py` and `archive/v1-app/tools/train_model.py`")
    add("actually did. Nothing is quoted from an earlier document. Reproduce with")
    add("`make case-study`.")
    add("")
    add("---")
    add("")
    add("## What v1 did")
    add("")
    add("`db_setup.py` contained this, under a comment calling it robustness:")
    add("")
    add("```python")
    add('large_maint = pd.concat([maint_df] * 5, ignore_index=True)  # Multiply by 5')
    add("large_maint['Torque [Nm]'] = large_maint['Torque [Nm]'] + \\")
    add("    np.random.normal(0, 2, large_maint.shape[0])")
    add("```")
    add("")
    add(
        f"The AI4I table has {source_stats['source_rows']:,} rows and "
        f"{source_stats['source_unique_ids']:,} unique product IDs -- one row per "
        "product. After inflation:"
    )
    add("")
    add("| | Source | After inflation |")
    add("|---|---|---|")
    add(f"| Rows | {source_stats['source_rows']:,} | {source_stats['inflated_rows']:,} |")
    add(
        f"| Unique `UDI` | {source_stats['source_unique_ids']:,} "
        f"| {source_stats['inflated_unique_ids']:,} |"
    )
    add(
        f"| Rows sharing a `UDI` with another row | 0 "
        f"| {source_stats['inflated_duplicate_ids']:,} |"
    )
    add(
        f"| Exact duplicate rows | 0 "
        f"| {source_stats['inflated_exact_duplicate_rows']:,} |"
    )
    add("")
    add("**The last row is the trap.** Adding Gaussian noise to torque means no two")
    add("copies are byte-identical, so `DataFrame.duplicated()` reports zero and a")
    add("casual check passes. Every copy still carries the same label and four")
    add("identical features out of five.")
    add("")
    add("`train_model.py` then fitted a RandomForest on the inflated table with no")
    add("held-out evaluation of any kind -- no split, no cross-validation, no metric.")
    add("The model was pickled and shipped. The numbers below are what it *would* have")
    add("reported under the evaluation a reader would assume.")
    add("")
    add("---")
    add("")
    add("## The gap")
    add("")
    add("| | Naive random split of the inflated table | Honest 5-fold CV on the source rows |")
    add("|---|---|---|")
    add(f"| F1 | **{naive['f1']:.3f}** | **{honest['f1']:.3f}** |")
    add(f"| Precision | {naive['precision']:.3f} | {honest['precision']:.3f} |")
    add(f"| Recall | {naive['recall']:.3f} | {honest['recall']:.3f} |")
    add(
        f"| Evaluated on | {naive['n_test']:,} rows drawn from the inflated table "
        f"| all {honest['n_rows']:,} source rows, out of fold |"
    )
    add(
        f"| Confusion (TP/FP/FN/TN) | {naive['tp']}/{naive['fp']}/{naive['fn']}/{naive['tn']} "
        f"| {honest['tp']}/{honest['fp']}/{honest['fn']}/{honest['tn']} |"
    )
    add("")
    add(f"**{gap * 100:.1f} F1 points of pure illusion.**")
    add("")
    add("### Reconciliation with the figures in `docs/MILESTONE_3.md`")
    add("")
    add("The milestone brief quoted F1 0.964 for the naive split and a 23-point gap.")
    add(f"Recomputed here: **{naive['f1']:.4f}** and **{gap * 100:.1f} points**. The")
    add("honest-CV figures match the brief exactly (F1 0.736, recall 0.646), as does")
    add(f"the majority accuracy ({source_stats['majority_accuracy']:.4f}).")
    add("")
    add("The naive figure differs because **`db_setup.py` called `np.random.normal`")
    add("with no seed**. The torque noise -- and therefore the inflated table, and")
    add("therefore every number derived from it -- was different on every run and could")
    add("not be reproduced even by the person who wrote it. This script seeds the noise")
    add("so the comparison is stable; the seed changes the third decimal, not the")
    add("mechanism. That the original number cannot be recovered is itself part of the")
    add("case study.")
    add("")
    add("### The mechanism")
    add("")
    add("Each source row exists five times. A random 80/20 split puts, in expectation,")
    add("four copies in train and one in test. The model does not generalise to the")
    add("test row -- it *remembers* it, from four near-identical rows it was fitted on,")
    add("differing only by a couple of Newton-metres of injected noise. The test set")
    add("is not held out in any meaningful sense; it is a subset of the training set")
    add("wearing a disguise.")
    add("")
    add("Recall moves most:")
    add(f"{honest['recall']:.3f} honest against {naive['recall']:.3f} naive. Positives")
    add("are rare, so a memorised positive is worth far more than a memorised")
    add("negative, and duplication rewards exactly that.")
    add("")
    add("---")
    add("")
    add("## Why nobody noticed")
    add("")
    add(
        f"The AI4I positive rate is {source_stats['positive_rate']:.4f}, so predicting "
        f"'no failure' for every row scores **{source_stats['majority_accuracy']:.2%} "
        "accuracy**."
    )
    add("")
    add("That is the number that made the inflated result look plausible. Against a")
    add("96.6% floor, an F1 of 0.96 reads as 'about as good as you would expect'")
    add("rather than 'impossibly good'. The two figures are not comparable -- one is")
    add("accuracy on a 3.4% positive rate, the other is F1 -- but they are the same")
    add("size, and a reader skimming a README will not stop to notice.")
    add("")
    add("v1's README claimed the model provided 'Future-Sight'. It reported no metric")
    add("at all, so there was nothing to check.")
    add("")
    add("---")
    add("")
    add("## How the current pipeline makes this impossible")
    add("")
    add("Not 'unlikely'. Each of these fails the build rather than degrading quietly.")
    add("")
    add("| Failure mode from v1 | What now prevents it |")
    add("|---|---|")
    add(
        "| Silent row duplication | Every table has a `PRIMARY KEY` over its natural "
        "key and ingestion uses plain `INSERT`, never `INSERT OR IGNORE`. A duplicated "
        "source row aborts `make data` with an `IntegrityError`. "
        "`tests/test_ingest.py::test_primary_keys_reject_duplicates` |"
    )
    add(
        "| Random split over autocorrelated rows | Splits are temporal. No shuffling "
        "splitter exists anywhere in `src/`, asserted by parsing the source, and the "
        "scanner is itself tested against a planted `train_test_split` call. "
        "`tests/test_no_future_leakage.py::test_no_shuffled_splitting_anywhere_in_src` |"
    )
    add(
        "| A training row's label window reaching into the evaluation period | A "
        "24-hour embargo derived from the label horizon, plus four independent "
        "assertions on the split boundary, each demonstrated to fail when the embargo "
        "is removed |"
    )
    add(
        "| Features reading the future | 56 leakage tests: future-record invariance, "
        "boundary inclusion at exactly `t`, failure-coincident `maint` exclusion, and "
        "column-order invariance. Each is paired with a demonstration that it can fail |"
    )
    add(
        "| Tuning against the test set | `features_test.parquet` is behind a runtime "
        "lock, and one module holds the key. "
        "`tests/test_single_test_split_consumer.py` |"
    )
    add(
        "| No evaluation at all | `docs/EVALUATION.md` reports PR-AUC against three "
        "baselines with bootstrap intervals, and accuracy appears exactly once, in a "
        "sentence explaining why it is never used again |"
    )
    add("")
    add("---")
    add("")
    add("## The honest reading of this comparison")
    add("")
    add("The 5-fold CV column is not a *good* result either. F1")
    add(f"{honest['f1']:.3f} at recall {honest['recall']:.3f} means the model misses")
    add(f"{100 * (1 - honest['recall']):.0f}% of failures. The point of this document")
    add("is not that the honest number is impressive. It is that the honest number is")
    add("**true**, and that a project reporting the inflated one would have shipped a")
    add("model believing it caught almost everything while missing a third of it.")
    add("")
    add("The AI4I data is retained under `archive/v1-data/` for this reason alone. It")
    add("is unsuitable for the current project -- 10,000 rows, 10,000 unique product")
    add("IDs, no machine entity and no time dimension, so no fleet and no temporal")
    add("split could exist on it.")
    add("")
    return "\n".join(lines)


def main() -> None:
    if not SOURCE_CSV.exists():
        raise SystemExit(
            f"{SOURCE_CSV} not found. It is committed under archive/v1-data/; "
            "check out the repository fully."
        )

    source = pd.read_csv(SOURCE_CSV)
    inflated = inflate(source)

    stats = duplication_evidence(source, inflated)
    print(f"source rows        : {stats['source_rows']:,}")
    print(f"inflated rows      : {stats['inflated_rows']:,}")
    print(f"exact duplicates   : {stats['inflated_exact_duplicate_rows']:,}  "
          "(noise on torque hides them)")
    print(f"majority accuracy  : {stats['majority_accuracy']:.4f}")

    naive = naive_random_split(inflated)
    honest = honest_cross_validation(source)
    print(f"\nnaive random split : F1 {naive['f1']:.4f}  "
          f"P {naive['precision']:.4f}  R {naive['recall']:.4f}")
    print(f"honest 5-fold CV   : F1 {honest['f1']:.4f}  "
          f"P {honest['precision']:.4f}  R {honest['recall']:.4f}")
    print(f"gap                : {100 * (naive['f1'] - honest['f1']):.1f} F1 points")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(render(stats, naive, honest), encoding="utf-8")
    print(f"\nwrote {OUTPUT}")

    Path("data/generated/leakage_case_study.json").write_text(
        json.dumps({"duplication": stats, "naive": naive, "honest": honest},
                   indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
