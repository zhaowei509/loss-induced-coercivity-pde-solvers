"""Independent-quadrature analysis and reproducible Experiment 2 figures."""

from collections import Counter
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .experiment import CONTROL_SEED, default_plan, qr_factor
from .features import DTYPE, RandomFeatures, qr_dual_residual, spectrum
from .io import atomic_json, file_hash, load_run
from .objectives import AllenCahnRitz, P4Ritz, cubic_solution, cubic_source, p4_source
from .sampling import CALIBRATION_SEEDS, FORMAL_SEEDS, gauss_legendre


CUBIC_CHECKPOINTS = (0, 50, 200, 5000)
RITZ_CHECKPOINTS = (0, 50, 200, 1000, 5000, 10000, 20000, 50000)


def _tensor(value, device="cpu"):
    return torch.as_tensor(value, dtype=DTYPE, device=device)


def _features(config, device="cpu"):
    return RandomFeatures(_tensor(config["feature_weights"], device),
                          _tensor(config["feature_biases"], device))


def _coefficient(config, stored):
    value = _tensor(stored)
    if config["coordinate"] == "b=Ra":
        factor = _tensor(config["qr_r_factor"])
        return torch.linalg.solve_triangular(factor, value[:, None], upper=True)[:, 0]
    return value


def _cubic_quadrature(features, coefficients, order, *, device, weighted_features=False):
    quadrature = gauss_legendre(order, 2, device=device)
    error_squared = torch.zeros((), dtype=DTYPE, device=device)
    residual_squared = torch.zeros_like(error_squared)
    q_rows = []
    for start in range(0, quadrature.points.shape[0], 4096):
        end = min(start + 4096, quadrature.points.shape[0])
        x, weights = quadrature.points[start:end], quadrature.weights[start:end]
        jet = features.jet(x)
        u = jet.values @ coefficients
        residual = -jet.laplacian @ coefficients + u**3 - cubic_source(x)
        error_squared += torch.dot(weights, (u - cubic_solution(x)).square())
        residual_squared += torch.dot(weights, residual.square())
        if weighted_features:
            q_rows.append((weights.sqrt()[:, None] * jet.values).detach().cpu())
    error = math.sqrt(float(error_squared))
    bound = math.sqrt(float(residual_squared)) / (2 * math.pi**2)
    return error, bound, torch.cat(q_rows) if weighted_features else None


def _cubic_record(meta, trajectory):
    config, result = meta["config"], meta["result"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    features = _features(config, device)
    points = _tensor(config["training_points"], device)
    p_spec = spectrum(features.jet(points).values)
    final_step = result["accepted_steps"]
    selected = sorted({k for k in CUBIC_CHECKPOINTS if k <= final_step} | {final_step})
    checkpoints, q_spec = [], None
    for step in selected:
        coefficients = _tensor(trajectory["parameters"][step], device)
        error, bound, q = _cubic_quadrature(
            features, coefficients, 128, device=device, weighted_features=q_spec is None)
        if q is not None:
            q_spec = spectrum(q)
        checkpoints.append(dict(step=step, E=error, B=bound,
                                ratio=error / bound if bound >= 1e-12 else None))
    error256, bound256, _ = _cubic_quadrature(
        features, _tensor(trajectory["parameters"][-1], device), 256, device=device)
    final = checkpoints[-1]
    return dict(basis="random", branch="cubic", seed=config["feature_seed"],
                initialization=config["initialization"],
                accepted_steps=final_step, stop_reason=result["stop_reason"],
                Q_full_rank=q_spec.full_column_rank, cV=q_spec.sigma_min,
                Q_rank_threshold=q_spec.threshold,
                P_full_rank=p_spec.full_column_rank, sigma_P=p_spec.sigma_min,
                P_rank_threshold=p_spec.threshold,
                c_lead_LB=(p_spec.sigma_min**6 / (4 * points.shape[0]**3)
                           if p_spec.full_column_rank else None),
                checkpoints=checkpoints, final_E256=error256, final_B256=bound256,
                final_bound_margin_256=bound256 - error256,
                quadrature_difference=(abs(error256 - final["E"])
                                       + abs(bound256 - final["B"])),
                final_gradient=float(trajectory["gradient_norm"][-1]))


def _ritz_matrices(config, order):
    quadrature = gauss_legendre(order, 1)
    jet = _features(config).jet(quadrature.points)
    root = quadrature.weights.sqrt()[:, None]
    derivative = jet.gradient[:, :, 0]
    kind = config["branch"]
    feature = root * (derivative if kind == "p4" else jet.values)
    metric = (feature if kind == "p4" else
              torch.cat((feature, 0.1 * root * derivative), dim=0))
    objective = (P4Ritz(jet.values, derivative, p4_source(quadrature.points),
                        quadrature.weights) if kind == "p4" else
                 AllenCahnRitz(jet.values, derivative, quadrature.weights, epsilon=0.1))
    return quadrature, jet, feature, metric, objective


def _ritz_record(meta, trajectory):
    config, result = meta["config"], meta["result"]
    _, _, feature128, metric128, _ = _ritz_matrices(config, 128)
    feature_spec = spectrum(feature128)
    metric128_spec = spectrum(metric128)
    _, _, _, metric256, objective256 = _ritz_matrices(config, 256)
    factor256 = qr_factor(metric256)
    final_step = result["accepted_steps"]
    selected = sorted({min(k, final_step) for k in RITZ_CHECKPOINTS} | {final_step})
    checkpoints = {}
    for step in selected:
        a = _coefficient(config, trajectory["parameters"][step])
        energy, gradient = objective256.analytic_value_gradient(a)
        rho = qr_dual_residual(gradient, factor256) if factor256 is not None else None
        checkpoints[str(step)] = dict(step=step, energy_256=energy, rho=rho,
                                      coefficients=a.tolist())
    final = checkpoints[str(final_step)]
    energy512 = rho512 = None
    if config["basis"] in ("control", "qr"):
        _, _, _, metric512, objective512 = _ritz_matrices(config, 512)
        factor512 = qr_factor(metric512)
        energy512, gradient512 = objective512.analytic_value_gradient(
            _tensor(final["coefficients"]))
        rho512 = qr_dual_residual(gradient512, factor512) if factor512 is not None else None
    singular = torch.linalg.svdvals(metric128)
    feature_singular = torch.linalg.svdvals(feature128)
    return dict(basis=config["basis"], branch=config["branch"],
                seed=config["feature_seed"], initialization=config["initialization"],
                accepted_steps=final_step, stop_reason=result["stop_reason"],
                feature_full_rank=feature_spec.full_column_rank,
                sigma_feature=feature_spec.sigma_min,
                feature_rank_threshold=feature_spec.threshold,
                metric_full_rank=metric128_spec.full_column_rank,
                metric_condition_number=float(singular[0] / singular[-1]),
                feature_condition_number=float(feature_singular[0] / feature_singular[-1]),
                final_energy_128=float(trajectory["objective"][-1]),
                final_energy_256=final["energy_256"], final_energy_512=energy512,
                energy_256_512_difference=(abs(energy512 - final["energy_256"])
                                           if energy512 is not None else None),
                energy_quadrature_difference=abs(
                    final["energy_256"] - float(trajectory["objective"][-1])),
                final_rho=final["rho"], final_rho_512=rho512,
                rho_256_512_difference=(abs(rho512 - final["rho"])
                                        if rho512 is not None and final["rho"] is not None
                                        else None),
                final_coefficient_norm=float(np.linalg.norm(final["coefficients"])),
                final_coordinate_gradient=float(trajectory["gradient_norm"][-1]),
                checkpoints=checkpoints)


def _p4_gap(entries, step):
    config = entries[0][1]
    if any(entry[1]["feature_weights"] != config["feature_weights"] or
           entry[1]["feature_biases"] != config["feature_biases"] for entry in entries):
        raise ValueError("Function gap requires the same fixed features.")
    quadrature, jet, _, _, _ = _ritz_matrices(config, 256)
    derivative = jet.gradient[:, :, 0]
    coefficients = [_tensor(row["checkpoints"][str(min(step, row["accepted_steps"]))]["coefficients"])
                    for row, _ in entries]
    functions = [derivative @ a for a in coefficients]

    def l4(vector):
        return float(torch.dot(quadrature.weights, vector.abs().pow(4)).pow(0.25))

    norms = [l4(function) for function in functions]
    gap = max(l4(functions[left] - functions[right])
              / max(1.0, norms[left], norms[right])
              for left in range(len(entries)) for right in range(left + 1, len(entries)))
    coefficient_gap = max(
        float(torch.linalg.vector_norm(coefficients[left] - coefficients[right]))
        / max(1.0, float(torch.linalg.vector_norm(coefficients[left])),
              float(torch.linalg.vector_norm(coefficients[right])))
        for left in range(len(entries)) for right in range(left + 1, len(entries)))
    return gap, coefficient_gap


def _median_curve(entries, steps):
    values = []
    for step in steps:
        available = [row["checkpoints"][str(min(step, row["accepted_steps"]))]["rho"]
                     for row, _ in entries]
        values.append(float(np.median(available)) if all(v is not None for v in available)
                      else None)
    return values


def _summary(data, seeds):
    cubic = [data[("random", "cubic", seed, "zero")][0] for seed in seeds]
    random_p4 = [data[("random", "p4", seed, label)][0]
                 for seed in seeds for label in ("zero", "plus", "minus")]
    random_ac = [data[("random", "ac", seed, "plus")][0] for seed in seeds]
    good_p4 = [data[("control", "p4", CONTROL_SEED, label)][0]
               for label in ("plus", "minus")]
    good_ac = data[("control", "ac", CONTROL_SEED, "plus")][0]
    ratios = [checkpoint["ratio"] for row in cubic for checkpoint in row["checkpoints"]
              if checkpoint["ratio"] is not None]
    p4_function_gaps, p4_coefficient_gaps = [], []
    for seed in seeds:
        entries = [data[("random", "p4", seed, label)]
                   for label in ("zero", "plus", "minus")]
        gap, coefficient_gap = _p4_gap(entries, max(r[0]["accepted_steps"] for r in entries))
        p4_function_gaps.append(gap)
        p4_coefficient_gaps.append(coefficient_gap)
    good_gap, good_coefficient_gap = _p4_gap(
        [data[("control", "p4", CONTROL_SEED, label)] for label in ("plus", "minus")],
        max(row["accepted_steps"] for row in good_p4))

    def maximum_rho(rows):
        values = [row["final_rho"] for row in rows]
        return max(values) if all(value is not None for value in values) else None

    table = [
        dict(branch="cubic",
             Q_rank=f"{sum(row['Q_full_rank'] for row in cubic)}/{len(cubic)}",
             P_rank=f"{sum(row['P_full_rank'] for row in cubic)}/{len(cubic)}",
             general_max_E_over_B=max(ratios) if ratios else None,
             general_min_bound_margin_256=min(row["final_bound_margin_256"] for row in cubic)),
        dict(branch="p4",
             D_rank=f"{sum(data[('random', 'p4', seed, 'plus')][0]['feature_full_rank'] for seed in seeds)}/{len(seeds)}",
             general_max_delta_fun=max(p4_function_gaps),
             general_max_rho=maximum_rho(random_p4),
             general_max_D_end=max(p4_coefficient_gaps),
             good_delta_fun=good_gap, good_max_rho=maximum_rho(good_p4)),
        dict(branch="allen_cahn",
             V_rank=f"{sum(row['feature_full_rank'] for row in random_ac)}/{len(seeds)}",
             general_max_rho=maximum_rho(random_ac),
             general_stop_reasons=dict(Counter(row["stop_reason"] for row in random_ac)),
             good_rho=good_ac["final_rho"], good_stop_reason=good_ac["stop_reason"])]
    return table, dict(general_p4_function_gaps=p4_function_gaps,
                       general_p4_coefficient_gaps=p4_coefficient_gaps,
                       good_p4_coefficient_gap=good_coefficient_gap)


def _condition_table(data, seeds):
    rows = []
    for kind in ("p4", "ac"):
        general = [data[("random", kind, seed, "plus")][0] for seed in seeds]
        qr = [data[("qr", kind, seed, "plus")][0]
              for seed in seeds if ("qr", kind, seed, "plus") in data]
        good = data[("control", kind, CONTROL_SEED, "plus")][0]

        def stats(values):
            return (float(np.median(values)), max(values)) if values else (None, None)

        raw_med, raw_max = stats([r["final_rho"] for r in general
                                  if r["final_rho"] is not None])
        qr_med, qr_max = stats([r["final_rho"] for r in qr
                                if r["final_rho"] is not None])
        rows.append(dict(branch=kind,
                         good_feature_condition_number=good["feature_condition_number"],
                         good_metric_condition_number=good["metric_condition_number"],
                         good_rho=good["final_rho"], good_stop=good["stop_reason"],
                         random_condition_median=float(np.median(
                             [r["metric_condition_number"] for r in general])),
                         random_rho_median=raw_med, random_rho_max=raw_max,
                         qr_rho_median=qr_med, qr_rho_max=qr_max,
                         qr_runs=f"{len(qr)}/{len(seeds)}",
                         qr_stop_reasons=dict(Counter(r["stop_reason"] for r in qr))))
    return rows


def _figure(data, seeds, destination, mode, max_steps):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), constrained_layout=True)
    cubic = [data[("random", "cubic", seed, "zero")][0] for seed in seeds]
    points = [(checkpoint["B"], checkpoint["E"]) for row in cubic
              for checkpoint in row["checkpoints"]
              if checkpoint["B"] > 0 and checkpoint["E"] > 0]
    x, y = np.asarray(points).T
    axes[0].scatter(x, y, s=18, color="#0072B2", alpha=0.8, edgecolors="none",
                    label="Random features", zorder=3)
    low, high = min(x.min(), y.min()), max(x.max(), y.max())
    axes[0].plot([low, high], [low, high], "--", color="#333333",
                 linewidth=1.1, label=r"$E=B$")
    axes[0].set(xscale="log", yscale="log", xlabel=r"Residual bound $B(a)$",
                ylabel=r"Solution error $E(a)$",
                title="(a) Continuous residual stability")
    axes[0].legend(frameon=False, loc="best", fontsize=8)
    axes[0].grid(alpha=0.16, linewidth=0.6)

    random = [data[("random", "p4", seed, "plus")] for seed in seeds]
    qr = [data[("qr", "p4", seed, "plus")] for seed in seeds
          if ("qr", "p4", seed, "plus") in data]
    good = [data[("control", "p4", CONTROL_SEED, "plus")]]
    display_steps = tuple(k for k in RITZ_CHECKPOINTS if k <= max_steps)
    if max_steps not in display_steps:
        display_steps += (max_steps,)
    positions = np.log10(np.asarray(display_steps) + 1)
    series = [
        (random, "#0072B2", "o", "Random, original (median)"),
        (qr, "#009E73", "s", "Same random features, QR (median)"),
        (good, "#D55E00", "^", "Well-conditioned (one trajectory)"),
    ]
    for entries, color, marker, label in series:
        if not entries:
            continue
        values = _median_curve(entries, display_steps)
        plotted = np.asarray([v if v is not None and v > 0 else np.nan
                              for v in values], dtype=float)
        axes[1].plot(positions, plotted, color=color, marker=marker,
                     markersize=4, linewidth=1.5, label=label)
    axes[1].set(yscale="log", xlabel=r"Accepted step $k$ (position $\log_{10}(k+1)$)",
                ylabel=r"Stationarity residual $\rho_4$",
                title="(b) Conditioning and stationarity")
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels([str(step) if step < 1000 else f"{step//1000}k"
                             for step in display_steps], rotation=35)
    axes[1].grid(alpha=0.16, linewidth=0.6)
    axes[1].legend(frameon=False, loc="best", fontsize=7.3)
    if mode != "formal":
        fig.suptitle("Short calibration — not for the paper", fontsize=9)
    fig.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(destination.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)
    early_qr = [row for row, _ in qr if row["accepted_steps"] < max_steps]
    if early_qr:
        reasons = dict(Counter(row["stop_reason"] for row in early_qr))
        steps = [row["accepted_steps"] for row in early_qr]
        reason_text = ", ".join(f"{count} {reason.replace('_', ' ')}"
                                for reason, count in sorted(reasons.items()))
        early_note = (f"For the QR curve, {len(early_qr)}/{len(qr)} trajectories "
                      f"stopped at steps {min(steps)}–{max(steps)} "
                      f"({reason_text}); exact per-seed stops are in the supplementary table.")
    else:
        early_note = "The QR trajectories did not stop before the step limit."
    caption = (
        "Figure 3. (a) Independent-quadrature solution error E(a) versus the "
        "continuous-residual bound B(a) for the cubic semilinear problem; "
        "the dashed line is E=B. (b) Independent 256-point stationarity residual "
        "rho_4 for the p=4 Ritz problem. Random original-coordinate and matched "
        "QR-coordinate curves are medians over the preset seeds; the fixed "
        "well-conditioned curve is one prespecified trajectory. Values after "
        "an early numerical stop are held at the attained final state solely "
        "to form the fixed-budget summary, not interpreted as further training. "
        "QR changes the gradient dynamics and is a conditioning diagnostic. "
        + early_note
    )
    destination.with_name(destination.name + "_caption.txt").write_text(
        caption + "\n", encoding="utf-8")


def analyze(root):
    root = Path(root)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    mode = summary["mode"]
    seeds = ((30000,) if mode == "smoke" else
             CALIBRATION_SEEDS if mode in ("short_calibration", "calibration") else
             FORMAL_SEEDS if mode == "formal" else None)
    if seeds is None:
        raise ValueError("Unknown campaign mode.")
    expected = len(seeds) * 7 + 3
    if summary["expected_runs"] != expected or (
            len(summary["records"]) + len(summary["skipped"]) != expected):
        raise ValueError("Incomplete campaign; not generating paper artifacts.")
    if mode == "formal" and summary["plan_sha256"] != file_hash(default_plan()):
        raise ValueError("The formal run does not match the current plan.")
    data, sources = {}, set()
    for item in summary["records"]:
        meta, trajectory = load_run(root / item["run"])
        if meta["plan_sha256"] != summary["plan_sha256"]:
            raise ValueError("Mixed plan hashes in campaign.")
        sources.add(json.dumps(meta["source_sha256"], sort_keys=True))
        config = meta["config"]
        key = (config["basis"], config["branch"],
               config["feature_seed"], config["initialization"])
        if key in data:
            raise ValueError(f"Duplicate trajectory {key}.")
        row = (_cubic_record(meta, trajectory) if config["branch"] == "cubic"
               else _ritz_record(meta, trajectory))
        data[key] = (row, config)
    if len(sources) != 1:
        raise ValueError("Mixed source versions in campaign.")
    required = ({("random", "cubic", seed, "zero") for seed in seeds}
                | {("random", "p4", seed, label) for seed in seeds
                   for label in ("zero", "plus", "minus")}
                | {("random", "ac", seed, "plus") for seed in seeds}
                | {("control", "p4", CONTROL_SEED, label) for label in ("plus", "minus")}
                | {("control", "ac", CONTROL_SEED, "plus")})
    if not required.issubset(data):
        raise ValueError(f"Missing required trajectories: {sorted(required - set(data))}")
    table, p4_gaps = _summary(data, seeds)
    conditioning = _condition_table(data, seeds)
    derived = root / "derived"
    derived.mkdir(exist_ok=True)
    report = dict(mode=mode, run_count=len(data), skipped=summary["skipped"],
                  analysis_source_sha256=file_hash(Path(__file__)),
                  table=table, conditioning=conditioning, p4_gaps=p4_gaps,
                  details=[row for row, _ in data.values()])
    atomic_json(derived / "analysis.json", report)
    fields = sorted(set().union(*(row.keys() for row in table)))
    with (derived / "table3_experiment2.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for row in table:
            writer.writerow({key: json.dumps(value) if isinstance(value, dict) else value
                             for key, value in row.items()})
    condition_fields = sorted(set().union(*(row.keys() for row in conditioning)))
    with (derived / "appendix_conditioning.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, condition_fields)
        writer.writeheader()
        for row in conditioning:
            writer.writerow({key: json.dumps(value) if isinstance(value, dict) else value
                             for key, value in row.items()})
    with (derived / "appendix_stops.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, ["basis", "branch", "seed", "initialization",
                     "accepted_steps", "stop_reason"])
        writer.writeheader()
        for row, _ in data.values():
            writer.writerow({key: row[key] for key in writer.fieldnames})
    _figure(data, seeds, derived / "figure3_experiment2", mode, summary["max_steps"])
    return report
