"""Paper-oriented two-panel plot from saved Experiment 4 trajectory data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np


BLUE = "#365F83"
ORANGE = "#C8793F"
INK = "#26323C"
GRID = "#E5E9EC"


def _series(rows, key):
    times = np.array([row["t"] for row in rows], dtype=float)
    values = np.array([row[key] for row in rows], dtype=float)
    if values[0] <= 0 or not np.isfinite(values).all():
        raise ValueError(f"Invalid positive trajectory series: {key}")
    ratio = values / values[0]
    ratio[ratio <= 0] = np.nan
    return times, ratio, values[0], values[-1]


def _scientific(value):
    mantissa, exponent = f"{value:.2e}".split("e")
    return mantissa + r"\times10^{" + str(int(exponent)) + "}"


def _panel(ax, times, ratio, first, last, *, color, title, ylabel, symbol):
    ax.plot(times, ratio, color=color, linewidth=2.0, solid_capstyle="round")
    finite = np.isfinite(ratio)
    ax.plot(times[finite][-1], ratio[finite][-1], "o", color=color,
            markersize=3.5, zorder=3)
    ax.set_yscale("log")
    ax.set_xlim(left=0)
    ax.set_xlabel(r"Gradient-flow time $t$")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="semibold", pad=9)
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(axis="y", color=GRID, linewidth=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="both", colors=INK, labelcolor=INK,
                   length=3)
    label = "$" + symbol + "_0=" + _scientific(first) + "$\n"
    label += "$" + symbol + "_T=" + _scientific(last) + "$"
    ax.text(
        0.98, 0.95, label,
        transform=ax.transAxes, ha="right", va="top", fontsize=8.2,
        color=INK,
        bbox={"boxstyle": "round,pad=0.26", "facecolor": "white",
              "edgecolor": "none", "alpha": 0.92},
    )


def make_figure(report, pdf_path):
    pdf_path = Path(pdf_path)
    column = report["branches"]["column"]
    row = report["branches"]["row"]
    if (column.get("reference")
            and column["reference"]["usable_for_distance_plot"]):
        left_key = "distance_to_reference"
        left_ylabel = r"$\|\theta(t)-\theta_*\|_2/\|\theta_0-\theta_*\|_2$"
        left_symbol = "d"
        left_description = (
            r"normalized parameter distance to a separately refined local "
            r"critical point $\theta_*$"
        )
    else:
        left_key = "gradient_norm"
        left_ylabel = r"$\|\nabla J(\theta(t))\|_2/\|\nabla J(\theta_0)\|_2$"
        left_symbol = "g"
        left_description = (
            "normalized gradient norm because no ball-contained reference "
            "critical point met the refinement tolerance"
        )
    lt, ly, l0, lT = _series(column["sampled"], left_key)
    rt, ry, r0, rT = _series(row["sampled"], "residual_norm")

    with plt.rc_context({
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
    }):
        fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.72))
        _panel(
            axes[0], lt, ly, l0, lT, color=BLUE,
            title="(a) Full column: distance to critical point",
            ylabel=left_ylabel, symbol=left_symbol,
        )
        _panel(
            axes[1], rt, ry, r0, rT, color=ORANGE,
            title="(b) Full row: residual decay",
            ylabel=r"$\|R(\theta(t))\|_2/\|R(\theta_0)\|_2$",
            symbol="r",
        )
        fig.subplots_adjust(left=0.105, right=0.985, bottom=0.21, top=0.88,
                            wspace=0.34)
        fig.savefig(pdf_path, bbox_inches="tight")
        fig.savefig(pdf_path.with_suffix(".png"), dpi=240, bbox_inches="tight")
        plt.close(fig)

    caption = (
        "Local gradient-flow behavior after unfreezing from the same trained "
        "random-feature warm start. (a) Full-column branch: "
        f"{left_description}. "
        "(b) Full-row branch: normalized empirical residual norm. "
        "Both branches use the same cubic PDE, six weighted residual "
        "components, and a fixed radius rho = 1e-3. The small labels report "
        "the unnormalized initial and final quantities; full trajectories "
        "and stopping reasons are in results.json. Observed decay does not "
        "certify the theorem's uniform-neighborhood assumptions.\n"
    )
    pdf_path.with_name(pdf_path.stem + "_caption.txt").write_text(
        caption, encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Redraw Experiment 4 from an existing results.json."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".png").exists():
        raise FileExistsError("Requested plot output already exists.")
    report = json.loads(args.input.read_text(encoding="utf-8"))
    if report.get("status") != "completed" or "branches" not in report:
        raise ValueError("Input has no completed trajectories to plot.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    make_figure(report, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
