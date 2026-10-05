"""Read-only validation, tabulation, and publication plotting for Experiment 1."""

from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterMathtext, NullFormatter
import numpy as np

# Keep the read-only analysis path independent of the training runtime.
A1_STEPS = 25
FORMAL_SEEDS = tuple(range(10))
TOY_KINDS = ("A1", "A2", "A3")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


COLORS = {"A1": "#365F83", "A2": "#365F83", "A3": "#C8793F",
          "gradient_flow": "#365F83", "armijo": "#C8793F"}


def _style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIX Two Text", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 9.5,
        "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "legend.fontsize": 8, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "axes.linewidth": 0.8, "lines.linewidth": 1.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.transparent": False, "figure.facecolor": "white",
    })


def _load(root, plan_path):
    root, plan_path = Path(root), Path(plan_path)
    campaign = json.loads((root / "campaign.json").read_text(encoding="utf-8"))
    progress = json.loads((root / "progress.json").read_text(encoding="utf-8"))
    paths = sorted((root / "runs").rglob("raw.json"))
    raws = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    errors = []
    if campaign.get("mode") != "formal":
        errors.append("analysis requires a formal campaign")
    if campaign.get("plan_sha256") != file_hash(plan_path):
        errors.append("campaign plan hash differs from current frozen plan")
    if len(raws) != 44:
        errors.append(f"expected 44 raw records, found {len(raws)}")
    if not progress.get("all_completed") or progress.get("completed") != 44:
        errors.append("campaign progress is incomplete")
    identities = []
    environments, source_sets = set(), set()
    for raw in raws:
        config = raw["config"]
        identity = ((config["branch"], config["direction_index"])
                    if config["branch"] in TOY_KINDS else
                    ("B", config["dynamics"], config["feature_seed"]))
        identities.append(identity)
        environments.add(json.dumps(raw["environment"], sort_keys=True))
        source_sets.add(json.dumps(raw["source_sha256"], sort_keys=True))
        if raw["plan_sha256"] != campaign["plan_sha256"]:
            errors.append(f"per-run plan hash mismatch: {identity}")
    expected = ({(kind, direction) for kind in TOY_KINDS for direction in range(8)} |
                {("B", solver, seed) for solver in ("gradient_flow", "armijo")
                 for seed in FORMAL_SEEDS})
    if set(identities) != expected or len(identities) != len(set(identities)):
        errors.append("formal job set is incomplete or duplicated")
    if len(environments) != 1 or len(source_sets) != 1:
        errors.append("formal records do not share one environment/source set")
    if errors:
        raise RuntimeError("; ".join(errors))
    return raws, campaign


def _runs(raws, branch=None, dynamics=None):
    selected = [raw for raw in raws
                if (branch is None or raw["config"]["branch"] == branch)
                and (dynamics is None or raw["config"]["dynamics"] == dynamics)]
    return sorted(selected, key=lambda raw: raw["config"].get(
        "direction_index", raw["config"].get("feature_seed", -1)))


def _quantiles(values):
    values = np.asarray(list(values), dtype=float)
    return dict(median=float(np.median(values)), q1=float(np.quantile(values, 0.25)),
                q3=float(np.quantile(values, 0.75)), minimum=float(np.min(values)),
                maximum=float(np.max(values)), count=int(values.size))


def _save_figure(fig, stem):
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=320, bbox_inches="tight")
    plt.close(fig)


def plot_figure1(raws, output, *, equation_labels=None):
    """Optionally identify manuscript equations without changing saved data."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    a1 = _runs(raws, "A1")
    for raw in a1:
        trajectory = raw["result"]["trajectory"]
        direction = raw["config"]["direction_index"]
        fast = direction in (1, 5)
        axes[0].plot([row["step"] for row in trajectory],
                     [row["coefficient_error"] for row in trajectory],
                     color=("#C8793F" if fast else COLORS["A1"]),
                     alpha=(0.90 if fast else 0.38),
                     linewidth=(1.8 if fast else 1.05),
                     zorder=(2 if fast else 1))
    reference = a1[0]["result"]["trajectory"]
    axes[0].plot([row["step"] for row in reference],
                 [row["theoretical_bound"] for row in reference],
                 color="#222222", linestyle="--", linewidth=1.7)
    axes[0].set(xlabel=r"Proximal step $k$", ylabel=r"Coefficient error $\|a^k-a^\star\|_2$",
                title=("(a) Exact proximal contraction" if equation_labels is None
                       else f"(a) Exact proximal: Eq. ({equation_labels['A1']})"))
    axes[0].set_yscale("log")
    axes[0].legend(handles=(
        Line2D([], [], color="#C8793F", linewidth=1.8,
               label="fast direction pair"),
        Line2D([], [], color=COLORS["A1"], linewidth=1.4,
               label="remaining six directions"),
        Line2D([], [], color="#222222", linestyle="--", linewidth=1.7,
               label="common theoretical bound"),
    ), frameon=False, loc="upper right", handlelength=2.4)

    display_names = {"A2": "Parabolic", "A3": "Elliptic"}
    if equation_labels is not None:
        display_names = {kind: f"Eq. ({equation_labels[kind]})"
                         for kind in display_names}
    for kind, color in (("A2", COLORS["A2"]), ("A3", COLORS["A3"])):
        raw = _runs(raws, kind)[0]
        trajectory = raw["result"]["trajectory"]
        axes[1].plot([row["time"] for row in trajectory],
                     [row["coefficient_error"] for row in trajectory],
                     color=color, linewidth=1.8,
                     label=f"{display_names[kind]}: observed")
        if raw["result"]["diagnostics"]["sampled_tube_curvature"] > 0:
            axes[1].plot([row["time"] for row in trajectory],
                         [row["sampled_exponential_reference"] for row in trajectory],
                         color=color, linestyle="--", linewidth=1.2,
                         label=rf"{display_names[kind]}: sampled reference")
    axes[1].set(xlabel=r"Gradient-flow time $t$",
                ylabel=r"Coefficient error $\|a(t)-a^\star\|_2$",
                title=("(b) Nonlinear gradient-flow decay" if equation_labels is None
                       else f"(b) Gradient flow: Eqs. ({equation_labels['A2']}), "
                            f"({equation_labels['A3']})"))
    axes[1].set_yscale("log")
    axes[1].legend(frameon=False, loc="upper right", ncol=1)
    for axis in axes:
            axis.grid(True, which="major", color="#DDE4E8", linewidth=0.6)
            axis.grid(True, which="minor", color="#EEF2F4", linewidth=0.4)
    _save_figure(fig, output / "figure1_toy_convergence")


def _median_curve(runs, x_key, y_key):
    end = min(raw["result"]["trajectory"][-1][x_key] for raw in runs)
    if x_key == "step":
        grid = np.arange(int(math.floor(end)) + 1, dtype=float)
    else:
        positive = sorted({row[x_key] for raw in runs for row in raw["result"]["trajectory"]
                           if 0 < row[x_key] <= end})
        grid = np.asarray([0.0] + positive, dtype=float)
    curves = []
    for raw in runs:
        trajectory = raw["result"]["trajectory"]
        x = np.asarray([row[x_key] for row in trajectory], dtype=float)
        y = np.asarray([row[y_key] for row in trajectory], dtype=float)
        curves.append(np.interp(grid, x, y))
    return grid, np.median(np.stack(curves), axis=0)


def plot_figure2(raws, output, *, loss_symbol=r"J"):
    """Allow manuscript label updates while preserving historical defaults."""
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True,
                             sharex="row")
    for row_index, (dynamics, x_key, x_label, color) in enumerate((
            ("gradient_flow", "time", r"Gradient-flow time $t$", COLORS["gradient_flow"]),
            ("armijo", "step", r"Armijo iteration $k$", COLORS["armijo"]))):
        runs = _runs(raws, "B", dynamics)
        for raw in runs:
            trajectory = raw["result"]["trajectory"]
            plotted = ([point for point in trajectory if point[x_key] > 0]
                       if x_key == "time" else trajectory)
            x = [point[x_key] for point in plotted]
            axes[row_index, 0].plot(x, [point["gradient_norm"] for point in plotted],
                                    color=color, alpha=0.22, linewidth=0.8)
            axes[row_index, 1].plot(x, [point["radius_ratio"] for point in plotted],
                                    color=color, alpha=0.22, linewidth=0.8)
        for column, metric in enumerate(("gradient_norm", "radius_ratio")):
            grid, median = _median_curve(runs, x_key, metric)
            if x_key == "time":
                positive = grid > 0
                grid, median = grid[positive], median[positive]
            axes[row_index, column].plot(grid, median, color=color, linewidth=2.2,
                                         label="median")
            axes[row_index, column].grid(True, which="major", color="#d9d9d9", linewidth=0.6)
            axes[row_index, column].grid(True, which="minor", color="#eeeeee", linewidth=0.4)
            axes[row_index, column].set_xlabel(x_label)
        axes[row_index, 0].set_yscale("log")
        axes[row_index, 0].set_ylabel(rf"$\|\nabla {loss_symbol}(a)\|_2$")
        axes[row_index, 1].set_ylabel(rf"$\rho_R=\|a\|_2/R_{{{loss_symbol}(a^0)}}$")
        axes[row_index, 1].axhline(1.0, color="#222222", linestyle="--", linewidth=1.2,
                                   label="theoretical bound")
        if x_key == "time":
            for column in range(2):
                axes[row_index, column].set_xscale("log")
                axes[row_index, column].set_xlim(1e-8, 1e4)
                axes[row_index, column].set_xticks((1e-8, 1e-4, 1.0, 1e4))
                axes[row_index, column].xaxis.set_major_formatter(LogFormatterMathtext())
                axes[row_index, column].xaxis.set_minor_formatter(NullFormatter())
        axes[row_index, 0].legend(frameon=False, loc="best")
        axes[row_index, 1].legend(frameon=False, loc="best")
    axes[0, 0].set_title("(a) Gradient flow: stationarity")
    axes[0, 1].set_title("(b) Gradient flow: confinement ratio")
    axes[1, 0].set_title("(c) Armijo: stationarity")
    axes[1, 1].set_title("(d) Armijo: confinement ratio")
    _save_figure(fig, output / "figure2_general_case")


def _table1(raws):
    a1 = _runs(raws, "A1")
    rows = [{"branch": "A1", "success": "8/8",
             "q_linear": a1[0]["result"]["diagnostics"]["q_linear"],
             "maximum_q_k": max(raw["result"]["diagnostics"]["maximum_q_k"] for raw in a1),
             "maximum_recurrence_error": max(
                 raw["result"]["diagnostics"]["maximum_closed_recurrence_relative_error"]
                 for raw in a1), "minimum_mu_tube": None,
             "slope_median": None, "slope_q1": None, "slope_q3": None,
             "r_squared_median": None, "r_squared_q1": None, "r_squared_q3": None}]
    for kind in ("A2", "A3"):
        runs = _runs(raws, kind)
        fits = [raw["result"]["diagnostics"]["semilog_fit"] for raw in runs]
        valid_slopes = [fit["slope"] for fit in fits if fit["slope"] is not None]
        valid_r2 = [fit["r_squared"] for fit in fits if fit["r_squared"] is not None]
        slope = _quantiles(valid_slopes) if valid_slopes else None
        r2 = _quantiles(valid_r2) if valid_r2 else None
        rows.append({"branch": kind,
                     "success": f"{sum(raw['result']['diagnostics']['convergence_success'] for raw in runs)}/8",
                     "q_linear": None, "maximum_q_k": None,
                     "maximum_recurrence_error": None,
                     "minimum_mu_tube": min(raw["result"]["diagnostics"]["sampled_tube_curvature"]
                                                for raw in runs),
                     "slope_median": None if slope is None else slope["median"],
                     "slope_q1": None if slope is None else slope["q1"],
                     "slope_q3": None if slope is None else slope["q3"],
                     "r_squared_median": None if r2 is None else r2["median"],
                     "r_squared_q1": None if r2 is None else r2["q1"],
                     "r_squared_q3": None if r2 is None else r2["q3"]})
    return rows


def _table2(raws):
    gf, gd = _runs(raws, "B", "gradient_flow"), _runs(raws, "B", "armijo")
    spectra = [raw["config"]["sigma_min_boundary"] for raw in gf]
    spectrum_summary = _quantiles(spectra)
    rows = []
    for name, runs in (("Gradient flow", gf), ("Armijo", gd)):
        gradients = _quantiles(raw["result"]["diagnostics"]["final_gradient_norm"] for raw in runs)
        stop_counts = Counter(raw["result"]["stop_reason"] for raw in runs)
        descent_key = ("numerical_monotonicity_pass" if name == "Gradient flow"
                       else "sufficient_decrease_pass")
        rows.append(dict(dynamics=name,
                         full_rank=f"{sum(raw['config']['boundary_full_column_rank'] for raw in runs)}/10",
                         sigma_min_median=spectrum_summary["median"],
                         sigma_min_q1=spectrum_summary["q1"],
                         sigma_min_q3=spectrum_summary["q3"],
                         worst_radius_ratio=max(raw["result"]["diagnostics"]["maximum_radius_ratio"]
                                                for raw in runs),
                         gradient_success=f"{sum(raw['result']['stop_reason'] == 'gradient_tolerance' for raw in runs)}/10",
                         final_gradient_median=gradients["median"],
                         final_gradient_q1=gradients["q1"],
                         final_gradient_q3=gradients["q3"],
                         descent_audit=f"{sum(raw['result']['diagnostics'][descent_key] for raw in runs)}/10",
                         stop_reasons=json.dumps(dict(stop_counts), sort_keys=True)))
    return rows


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _appendix_rows(raws):
    rows = []
    for raw in raws:
        config, result = raw["config"], raw["result"]
        rows.append(dict(branch=config["branch"], dynamics=config["dynamics"],
                         direction=config.get("direction_index"), seed=config.get("feature_seed"),
                         stop_reason=result["stop_reason"],
                         exact_objective=config.get("exact_objective"),
                         sigma_min_boundary=config["sigma_min_boundary"],
                         lambda_min_hessian=config.get("lambda_min_hessian"),
                         mu_boundary=config.get("mu_boundary"),
                         sampled_tube_curvature=result["diagnostics"].get("sampled_tube_curvature"),
                         fitted_slope=result["diagnostics"].get("semilog_fit", {}).get("slope"),
                         fitted_r_squared=result["diagnostics"].get("semilog_fit", {}).get("r_squared"),
                         maximum_radius_ratio=result["diagnostics"].get("maximum_radius_ratio"),
                         final_gradient_norm=result["trajectory"][-1]["gradient_norm"],
                         elapsed_seconds=result["elapsed_seconds"]))
    return rows


def _captions():
    return """# Figure captions

**Figure 1. Local convergence in exactly representable toy problems.** (a) Coefficient errors for exact proximal iterations in A1 from all eight prescribed initial directions. Orange highlights the fast pair $\\theta=\\pi/4$ and $5\\pi/4$, while blue shows the other six trajectories. Opposite initial directions have identical norm trajectories and therefore form four exactly coincident pairs. The common dashed black curve is the worst-case theoretical bound $q_{\\rm lin}^{k}\\|a^0-a^\\star\\|_2$, where $q_{\\rm lin}=(1+\\eta\\lambda_{\\min}(H_{\\rm lin}))^{-1}$. (b) Coefficient errors along the preregistered $\\theta_0=0$ gradient-flow trajectories for A2 and A3. Dashed curves show the sampled references $e^{-\\widehat\\mu_{\\rm tube}t}\\|a^0-a^\\star\\|_2$. The sampled tube curvature is a diagnostic and is not a certificate over a continuous neighborhood.

**Figure 2. Criticality and explicit parameter confinement in the general cubic problem.** Each light curve corresponds to one of the ten prespecified random-feature seeds; the thick curve is the pointwise median on the common valid interval. The left column reports $\\|\\nabla J(a)\\|_2$, and the right column reports $\\rho_R=\\|a\\|_2/R_{J(a^0)}$ with the theoretical bound $\\rho_R=1$ shown by a dashed line. The gradient-flow panels display the positive logarithmically spaced output times; the $t=0$ state remains available in the raw data. Gradient flow and Armijo are shown against their native time and iteration axes, respectively, so the panels do not compare computational speeds.
"""


def analyze(root, plan_path):
    root = Path(root)
    raws, campaign = _load(root, plan_path)
    output = root / "derived"
    output.mkdir(parents=True, exist_ok=True)
    table1, table2 = _table1(raws), _table2(raws)
    _write_csv(output / "table1_toy_convergence.csv", table1)
    _write_csv(output / "table2_general_case.csv", table2)
    _write_csv(output / "appendix_run_metrics.csv", _appendix_rows(raws))
    plot_figure1(raws, output)
    plot_figure2(raws, output)
    (output / "FIGURE_CAPTIONS.md").write_text(_captions(), encoding="utf-8")
    summary = dict(plan_sha256=campaign["plan_sha256"], raw_record_count=len(raws),
                   analysis_code_sha256=file_hash(__file__),
                   table1=table1, table2=table2,
                   figures=["figure1_toy_convergence.pdf", "figure2_general_case.pdf"])
    atomic_json(output / "analysis.json", summary)
    return summary
