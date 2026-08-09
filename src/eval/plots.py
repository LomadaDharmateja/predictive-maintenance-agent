"""PR and reliability curves, written as committed PNGs.

Deliberately plain: no styling library, no colour cycling beyond what is needed
to tell four lines apart, and every axis labelled with the units it carries. The
no-skill line on each PR curve is the positive rate, which is the whole point of
using PR rather than ROC on a 0.4% base rate -- it makes the floor visible
instead of leaving the reader to assume 0.5.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display in CI or on a headless machine

import matplotlib.pyplot as plt
import numpy as np

from src.eval.metrics import pr_curve

IMAGES_DIR = Path("docs/images")

# Colour-blind safe, and distinguishable in greyscale by line style as well.
SERIES_STYLE = {
    "lgbm": ("#0072B2", "-"),
    "logreg": ("#D55E00", "--"),
    "any_error_24h": ("#009E73", "-."),
    "error_count_24h": ("#56B4E9", ":"),
    "majority": ("#999999", (0, (1, 3))),
}


def _style(name: str):
    return SERIES_STYLE.get(name, ("#333333", "-"))


def pr_curves(
    series: dict[str, tuple[np.ndarray, np.ndarray]],
    positive_rate: float,
    component: str,
    directory: Path = IMAGES_DIR,
    suffix: str = "val",
) -> Path:
    """One panel per component, every model and baseline on the same axes."""
    directory.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.0, 4.5), dpi=140)

    for name, (y_true, y_score) in series.items():
        precision, recall, _ = pr_curve(y_true, y_score)
        colour, dash = _style(name)
        axis.step(
            recall, precision, where="post", label=name, color=colour, linestyle=dash,
            linewidth=1.6,
        )

    axis.axhline(
        positive_rate,
        color="#666666",
        linewidth=1.0,
        linestyle=(0, (4, 4)),
        label=f"no skill = positive rate ({positive_rate:.4%})",
    )

    axis.set_xlabel("recall")
    axis.set_ylabel("precision")
    axis.set_xlim(-0.02, 1.02)
    axis.set_ylim(-0.02, 1.02)
    axis.set_title(f"Precision-recall, {component}, {suffix}")
    axis.legend(loc="lower left", fontsize=7, framealpha=0.9)
    axis.grid(alpha=0.25, linewidth=0.5)
    figure.tight_layout()

    path = directory / f"pr_{component}_{suffix}.png"
    figure.savefig(path)
    plt.close(figure)
    return path


def reliability_curves(
    reports: list,
    component: str,
    directory: Path = IMAGES_DIR,
    suffix: str = "val",
) -> Path:
    """Predicted probability against observed frequency, before and after
    calibration, on a log x-axis because everything lives below 0.05."""
    directory.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.0, 4.5), dpi=140)

    limits = [1.0, 0.0]
    for report in reports:
        if len(report.bin_centres) == 0:
            continue
        axis.plot(
            report.bin_centres,
            report.bin_observed,
            marker="o",
            markersize=3.5,
            linewidth=1.4,
            label=f"{report.method} (Brier {report.brier:.5f})",
        )
        limits[0] = min(limits[0], report.bin_centres.min(), report.bin_observed.min())
        limits[1] = max(limits[1], report.bin_centres.max(), report.bin_observed.max())

    low = max(min(limits[0], 1e-6), 1e-6)
    high = max(limits[1], low * 10)
    axis.plot([low, high], [low, high], color="#666666", linewidth=1.0,
              linestyle=(0, (4, 4)), label="perfect calibration")

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("mean predicted probability (bin)")
    axis.set_ylabel("observed frequency (bin)")
    axis.set_title(f"Reliability, {component}, {suffix}")
    axis.legend(loc="upper left", fontsize=7, framealpha=0.9)
    axis.grid(alpha=0.25, linewidth=0.5, which="both")
    figure.tight_layout()

    path = directory / f"reliability_{component}_{suffix}.png"
    figure.savefig(path)
    plt.close(figure)
    return path


def cost_curves(
    curves: dict[float, tuple[np.ndarray, np.ndarray]],
    chosen: dict[float, float],
    component: str,
    directory: Path = IMAGES_DIR,
) -> Path:
    """Expected cost against threshold, one line per cost ratio."""
    directory.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.0, 4.5), dpi=140)

    for ratio, (thresholds, costs) in sorted(curves.items()):
        line, = axis.plot(
            thresholds, costs, linewidth=1.5, label=f"{ratio:.0f}:1"
        )
        mark = chosen.get(ratio)
        if mark is not None:
            index = int(np.argmin(np.abs(thresholds - mark)))
            axis.plot(
                thresholds[index], costs[index], marker="v", markersize=7,
                color=line.get_color(),
            )

    axis.set_xscale("log")
    axis.set_xlabel("threshold")
    axis.set_ylabel("expected cost (units of one false alarm)")
    axis.set_title(f"Cost against threshold, {component}, validation")
    axis.legend(title="miss : false alarm", loc="upper left", fontsize=7)
    axis.grid(alpha=0.25, linewidth=0.5, which="both")
    figure.tight_layout()

    path = directory / f"cost_{component}_val.png"
    figure.savefig(path)
    plt.close(figure)
    return path
