"""Rebuild paper-facing Experiment 2 materials from frozen formal analysis.

This post-processing step never changes the campaign data or the original
``derived`` directory. It reads saved JSON only, so no GPU is required.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np


PLAN_SHA256 = "44f4e9e03940d4bf6af342f190c49fc567e96cf0db97989782e891bc9b4bc700"
STEPS = (0, 50, 200, 1000, 5000, 10000, 20000, 50000)
SEEDS = tuple(range(10))


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(root):
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    analysis_path = root / "derived" / "analysis.json"
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    frozen_plan = root / "EXPERIMENT2_PLAN_frozen.md"
    if not (summary["mode"] == analysis["mode"] == "formal"
            and summary["expected_runs"] == len(summary["records"]) == 73
            and analysis["run_count"] == 73
            and not summary["skipped"] and not summary["interrupted"]
            and summary["plan_sha256"] == _hash(frozen_plan) == PLAN_SHA256
            and analysis["analysis_source_sha256"]
            == summary["source_sha256"]["analysis.py"]):
        raise ValueError("Formal analysis provenance or run count does not match.")
    data = {(row["basis"], row["branch"], row["seed"], row["initialization"]): row
            for row in analysis["details"]}
    if len(data) != 73:
        raise ValueError("Missing or duplicated trajectory records.")
    return analysis_path, analysis, data


def _positions(steps):
    return np.log10(np.asarray(steps, dtype=float) + 1.0)


def _qr_traces(data):
    """Return only the saved, independently evaluated QR checkpoints."""
    traces = []
    for seed in SEEDS:
        expected = data[("qr", "p4", seed, "plus")]
        steps = np.asarray(sorted(map(int, expected["checkpoints"])), dtype=int)
        rho = np.asarray([expected["checkpoints"][str(step)]["rho"]
                          for step in steps], dtype=float)
        if (steps[0] != 0 or 50 not in steps
                or steps[-1] != expected["accepted_steps"]
                or expected["stop_reason"] != "step_underflow"
                or not np.isfinite(rho).all() or np.any(rho <= 0)):
            raise ValueError(f"QR run {seed} has invalid checkpoints or stop reason.")
        traces.append((steps, rho))
    return traces


def _figure(data, destination, style, qr_traces=None):
    blue, orange, teal, green, ink = (
        "#315F7C", "#B66C43", "#2B7780", "#4E8276", "#35424A")
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["STIX Two Text", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 8.5,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 0.75, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), constrained_layout=True)
    ticks = (0, 50, 200, 1000, 10000, 50000)
    for ax in axes:
        ax.tick_params(axis="both", which="major", length=3, width=0.7,
                       colors=ink, labelsize=7.6)
        ax.grid(axis="y", color="#DCE4E8", linewidth=0.5, alpha=0.8)
        ax.set_axisbelow(True)
    axes[0].set_xlim(-0.06, float(_positions((50000,))[0]) + 0.08)
    axes[0].set_xticks(_positions(ticks), ("0", "50", "200", "1k", "10k", "50k"))
    axes[0].set_xlabel(r"Accepted steps $k$", fontsize=8.3)

    cubic_steps = (0, 50, 200, 5000, 50000)
    cubic = [data[("random", "cubic", seed, "zero")] for seed in SEEDS]
    cubic_ratios = []
    for row in cubic:
        checkpoints = {point["step"]: point for point in row["checkpoints"]}
        cubic_ratios.append([checkpoints[step]["ratio"] for step in cubic_steps])
    ratios = np.asarray(cubic_ratios, dtype=float)
    if not np.isfinite(ratios).all() or np.any(ratios > 1):
        raise ValueError("Cubic residual ratios are unavailable or exceed the bound.")
    cubic_x = _positions(cubic_steps)
    axes[0].fill_between(cubic_x, ratios.min(axis=0), ratios.max(axis=0),
                         color=teal, alpha=0.17, linewidth=0)
    axes[0].plot(cubic_x, np.median(ratios, axis=0), color=teal, linewidth=1.8,
                 marker="o", markersize=3.2)
    axes[0].axhline(1.0, color="#73818B", linewidth=0.95,
                    linestyle=(0, (4, 2)),
                    zorder=1)
    axes[0].text(float(_positions((50000,))[0]), 1.015, "bound",
                 ha="right", va="bottom", fontsize=7.1, color="#73818B")
    axes[0].set_ylim(0, 1.08)
    axes[0].set_yticks((0, 0.25, 0.5, 0.75, 1.0))
    axes[0].set_ylabel(r"Residual ratio $E(a)/B(a)$", fontsize=8.3)
    axes[0].text(0.0, 1.045, "(a) Cubic error-to-bound ratio", transform=axes[0].transAxes,
                 fontsize=8.7, fontweight="bold",
                 ha="left", va="bottom", color=ink)

    random = [data[("random", "p4", seed, "plus")] for seed in SEEDS]
    qr = [data[("qr", "p4", seed, "plus")] for seed in SEEDS]
    good = data[("control", "p4", 30001, "plus")]
    if any(row["accepted_steps"] != 50000 for row in random) or (
            good["accepted_steps"] != 50000):
        raise ValueError("The original-coordinate curves lack a planned checkpoint.")

    if style in ("trajectories", "qr_inset"):
        axes[1].set_xlim(-0.06, float(_positions((50000,))[0]) + 0.08)
        axes[1].set_xticks(_positions(ticks), ("0", "50", "200", "1k", "10k", "50k"))
        axes[1].set_xlabel(r"Accepted steps $k$", fontsize=8.3)
        positions = _positions(STEPS)
        random_values = np.asarray([
            [row["checkpoints"][str(step)]["rho"] for step in STEPS]
            for row in random], dtype=float)
        if not np.isfinite(random_values).all() or np.any(random_values <= 0):
            raise ValueError("Random Ritz stationarity values must be positive and finite.")
        random_median = np.median(random_values, axis=0)
        q1, q3 = np.percentile(random_values, (25, 75), axis=0)
        good_rho = [good["checkpoints"][str(step)]["rho"] for step in STEPS]
        axes[1].fill_between(positions, q1, q3, color=blue, alpha=0.13, linewidth=0)
        axes[1].plot(positions, random_median, color=blue, marker="o",
                     markersize=3.2, linewidth=1.8,
                     label="Random features: median (IQR)")
        axes[1].plot(positions, good_rho, color=orange, marker="^",
                     markersize=3.4, linewidth=1.6,
                     label="Constructed basis: one run")
        if style == "qr_inset":
            if qr_traces is None or len(qr_traces) != len(SEEDS):
                raise ValueError("The QR inset requires all ten saved trajectories.")
            common = np.asarray([[rho[np.where(steps == step)[0][0]]
                                  for step in (0, 50)]
                                 for steps, rho in qr_traces], dtype=float)
            axes[1].plot(_positions((0, 50)), np.median(common, axis=0),
                         color=green, linewidth=1.65, marker="s",
                         markersize=3.0, label="QR coordinates: early median")
            inset = axes[1].inset_axes((0.59, 0.37, 0.37, 0.32), zorder=5)
            inset.set_facecolor("white")
            for steps, rho in qr_traces:
                inset.plot(_positions(steps), rho, color=green,
                           alpha=0.57, linewidth=0.75)
            inset.set_xlim(-0.03, float(_positions((560,))[0]) + 0.03)
            inset.set_ylim(6e-9, 35)
            inset.set_yscale("log")
            inset.set_xticks(_positions((0, 50, 500)), ("0", "50", "500"))
            inset.set_yticks((1e1, 1e-3, 1e-7))
            inset.tick_params(axis="both", labelsize=6.0, length=2, width=0.6,
                              colors=ink, pad=1.5)
            for spine in inset.spines.values():
                spine.set_visible(True)
                spine.set_color("#84929B")
                spine.set_linewidth(0.65)
            inset.grid(axis="y", color="#E2E8EA", linewidth=0.4)
            inset.text(0.96, 0.93, "QR (10 runs)", transform=inset.transAxes,
                       ha="right", va="top", fontsize=6.7, color=ink,
                       bbox=dict(facecolor="white", edgecolor="none",
                                 alpha=0.9, pad=0.8), zorder=6)
            connector = ConnectionPatch(
                xyA=(_positions((50,))[0], float(np.median(common[:, -1]))),
                xyB=(0, 0.65), coordsA="data", coordsB="axes fraction",
                axesA=axes[1], axesB=inset, arrowstyle="-|>",
                color="#788690", linewidth=0.8, mutation_scale=7,
                connectionstyle="arc3,rad=-0.15", zorder=7)
            fig.add_artist(connector)
        axes[1].text(0.0, 1.045, r"(b) $p=4$ Deep Ritz stationarity",
                     transform=axes[1].transAxes, fontsize=8.7,
                     fontweight="bold",
                     ha="left", va="bottom", color=ink)
        if style == "qr_inset":
            axes[1].legend(frameon=False, loc="lower left", fontsize=6.7,
                           handlelength=1.6, borderaxespad=0.3)
        else:
            axes[1].legend(frameon=False, loc="center right", fontsize=7.0,
                           handlelength=1.7, borderaxespad=0.25)
    elif style == "endpoints":
        if any(row["stop_reason"] != "step_underflow" for row in qr):
            raise ValueError("QR stopping reasons changed; revise the caption.")
        qr_steps = np.asarray([row["accepted_steps"] for row in qr], dtype=int)
        if qr_steps.min() != 65 or qr_steps.max() != 543:
            raise ValueError("QR step range changed; revise the category label.")
        axes[1].set_xlim(-0.5, 2.5)
        axes[1].set_xticks(
            (0, 1, 2),
            ("Random\n50k steps", "Constructed\n50k steps", "QR\n65-543 steps"))
        groups = (
            (0, np.asarray([row["final_rho"] for row in random], dtype=float),
             blue, "o"),
            (1, np.asarray([good["final_rho"]], dtype=float), orange, "^"),
            (2, np.asarray([row["final_rho"] for row in qr], dtype=float),
             green, "D"),
        )
        for x, values, color, marker in groups:
            if not np.isfinite(values).all() or np.any(values <= 0):
                raise ValueError("Final Ritz residuals must be positive and finite.")
            median = float(np.median(values))
            if len(values) > 1:
                axes[1].errorbar(
                    x, median,
                    yerr=np.array([[median - values.min()],
                                   [values.max() - median]]),
                    fmt=marker, color=color, ecolor=color, markersize=6,
                    markeredgecolor="white", markeredgewidth=0.7,
                    elinewidth=1.3, capsize=4, capthick=1.3, zorder=3)
            else:
                axes[1].plot(x, median, marker=marker, linestyle="none",
                             markersize=7, color=color, zorder=3)
        axes[1].text(0.0, 1.045, r"(b) Final Ritz residual ($p=4$)",
                     transform=axes[1].transAxes, fontsize=8.7,
                     ha="left", va="bottom", color=ink)
        axes[1].text(0.98, 0.96, "10-seed groups: median and range",
                     transform=axes[1].transAxes, fontsize=7.0,
                     ha="right", va="top", color="#62717A")
    else:
        raise ValueError(f"Unknown figure style: {style}")
    axes[1].set_yscale("log")
    axes[1].set_ylim(6e-9, 35)
    axes[1].set_ylabel(r"Stationarity residual $\rho_4$", fontsize=8.3)
    fig.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(destination.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)

    caption = (
        "Figure 3. Horizontal positions use $\\log_{10}(k+1)$, with ticks "
        "labeled by actual accepted Armijo steps $k$. (a) Independent-quadrature "
        "solution-error ratio $E(a)/B(a)$ for the cubic semilinear problem: "
        "line, median; band, full range over 10 fixed random-feature seeds; "
        "dashed line, theoretical upper bound 1. "
    )
    if style in ("trajectories", "qr_inset"):
        caption += (
            "(b) Independent 256-point Ritz stationarity residual $\\rho_4$: "
            "blue line and band, median and IQR over 10 unscreened random-feature "
            "trajectories in original coordinates; orange line, one fixed "
            "constructed-basis trajectory. Both are shown at their actual accepted "
            "steps through 50,000. "
        )
        if style == "qr_inset":
            caption += (
                "The green main-panel line connects the QR-coordinate medians at "
                "the two common early checkpoints ($k=0,50$). The arrow leads to "
                "an inset connecting each of the 10 QR runs' saved independent-"
                "evaluation checkpoints through its actual stop (65-543 steps); "
                "no post-stop values are drawn. All QR runs stopped by numerical "
                "step underflow, not by the gradient threshold. QR changes the "
                "gradient dynamics, so this is not a same-dynamics rate comparison."
            )
        else:
            caption += (
                "Matched QR-coordinate runs change the gradient dynamics and stop "
                "much earlier; their endpoint and stopping-reason statistics are "
                "reported separately in the appendix."
            )
    else:
        caption += (
            "(b) Final independent 256-point Ritz stationarity residual $\\rho_4$. "
            "Random original-coordinate and matched QR-coordinate groups each have "
            "10 fixed feature seeds; symbols mark the medians and whiskers their "
            "full ranges. The constructed basis has one prespecified run. Both "
            "original-coordinate groups took 50,000 accepted steps; QR runs stopped "
            "after 65-543 steps, all by numerical step underflow rather than the "
            "gradient threshold. QR changes the gradient dynamics, so these are "
            "finite-computation endpoint diagnostics, not a same-trajectory rate "
            "comparison or proof of asymptotic convergence."
        )
    destination.with_name(destination.name + "_caption.txt").write_text(
        caption + "\n", encoding="utf-8")


def _table(analysis, data, destination):
    rows = {row["branch"]: row for row in analysis["table"]}
    cubic, p4, ac = rows["cubic"], rows["p4"], rows["allen_cahn"]
    p4_good_d = analysis["p4_gaps"]["good_p4_coefficient_gap"]
    ac_worse = sum(data[("random", "ac", seed, "plus")]["final_rho"]
                   > data[("random", "ac", seed, "plus")]["checkpoints"]["0"]["rho"]
                   for seed in SEEDS)
    if ac_worse != 8:
        raise ValueError("Allen--Cahn initial/final comparison changed.")

    def scientific(value):
        coefficient, exponent = f"{value:.2e}".split("e")
        return rf"{coefficient}\times10^{{{int(exponent)}}}"

    content = r"""%% Derived from output/experiment2/formal_20260917_v1/derived/analysis.json.
%% The source data and formal plan remain unchanged. Requires only standard LaTeX.
\begin{table*}[t]
\centering
\small
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.15}
\begin{tabular}{@{}p{0.13\textwidth}p{0.12\textwidth}p{0.36\textwidth}p{0.32\textwidth}@{}}
\hline
Problem & Rank check & Unscreened random features & Constructed basis \\
\hline
Cubic PINN & $Q,P$: %s & $\max E/B=%.3f$; $\min(B_{256}-E_{256})=%s$ & not applicable \\
$p=4$ Ritz & $D$: %s & $\max\Delta_{\rm fun}=%s$; $\max D_{\rm end}=%.3f$; $\max\rho_4=%.2f$ & $\Delta_{\rm fun}=%s$; $D_{\rm end}=%s$; $\max\rho_4=%s$ \\
Allen--Cahn & $V$: %s & $\max\rho_{\rm AC}=%.3f$; $%d/10$ final residuals exceed their initial values & $\rho_{\rm AC}=%s$ (numerical stop) \\
\hline
\end{tabular}
\caption{Experiment 2, independent-quadrature diagnostics. Rank counts use 10 fixed feature seeds. All unscreened random-feature and constructed $p=4$ trajectories reached the 50,000-step budget. Constructed Allen--Cahn stopped at step 9,579 due to numerical step underflow, not the gradient threshold. Stationarity residuals use an independent 256-point rule. The constructed basis is illustrative, not representative of random features.}
\label{tab:experiment2}
\end{table*}
""" % (
        cubic["Q_rank"] if cubic["Q_rank"] == cubic["P_rank"] else
        cubic["Q_rank"] + ", " + cubic["P_rank"],
        cubic["general_max_E_over_B"], scientific(cubic["general_min_bound_margin_256"]),
        p4["D_rank"], scientific(p4["general_max_delta_fun"]),
        p4["general_max_D_end"], p4["general_max_rho"],
        scientific(p4["good_delta_fun"]), scientific(p4_good_d),
        scientific(p4["good_max_rho"]), ac["V_rank"],
        ac["general_max_rho"], ac_worse, scientific(ac["good_rho"]),
    )
    destination.write_text(content, encoding="utf-8")


def build(root, output_name="derived_paper_20260917_v16", style="qr_inset"):
    root = Path(root)
    analysis_path, analysis, data = _load(root)
    if not output_name.startswith("derived_paper_") or Path(output_name).name != output_name:
        raise ValueError("Output name must be a single derived_paper_* directory name.")
    destination = root / output_name
    if destination.exists():
        raise FileExistsError(f"Paper-facing output already exists: {destination}")
    destination.mkdir()
    qr_traces = _qr_traces(data) if style == "qr_inset" else None
    _figure(data, destination / "figure3_experiment2", style, qr_traces)
    _table(analysis, data, destination / "table3_experiment2.tex")
    manifest = {"source_analysis": str(analysis_path.relative_to(root)),
                "source_analysis_sha256": _hash(analysis_path),
                "paper_script_sha256": _hash(Path(__file__)),
                "formal_plan_sha256": PLAN_SHA256,
                "run_count": len(data),
                "figure_style": style,
                "note": "Post-processing only; original formal outputs are unchanged."}
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-name", default="derived_paper_20260917_v16")
    parser.add_argument("--style", choices=("trajectories", "endpoints", "qr_inset"),
                        default="qr_inset")
    args = parser.parse_args()
    print(build(args.root, args.output_name, args.style))
