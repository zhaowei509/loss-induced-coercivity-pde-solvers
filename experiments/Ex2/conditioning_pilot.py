"""Frozen small pilot: a well-conditioned control and QR-coordinate comparisons."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.Ex2.dynamics import armijo
from experiments.Ex2.experiment import ARMIJO, GRADIENT_TOLERANCE, default_plan
from experiments.Ex2.experiment import initializations, make_problem
from experiments.Ex2.features import DTYPE, RandomFeatures, qr_dual_residual, spectrum
from experiments.Ex2.io import atomic_json, file_hash, load_run, save_run
from experiments.Ex2.objectives import AllenCahnRitz, P4Ritz, p4_source
from experiments.Ex2.sampling import gauss_legendre


SEED = 30001
WIDTH = 8
MAX_STEPS = 50000
CASES = (("control", "p4", "plus"), ("control", "p4", "minus"),
         ("control", "ac", "plus"), ("control", "ac", "minus"),
         ("qr_current", "p4", "plus"), ("qr_current", "ac", "plus"))


def control_features():
    """Eight preset tanh transition centers; no outcome-dependent seed selection."""
    centers = torch.linspace(0.1, 0.9, WIDTH, dtype=DTYPE)
    slopes = torch.full((WIDTH, 1), 16.0, dtype=DTYPE)
    return RandomFeatures(slopes, -16.0 * centers)


def make_control(kind):
    features = control_features()
    quadrature = gauss_legendre(128, 1)
    points, weights = quadrature.points, quadrature.weights
    jet = features.jet(points)
    derivative = jet.gradient[:, :, 0]
    function = (P4Ritz(jet.values, derivative, p4_source(points), weights)
                if kind == "p4" else
                AllenCahnRitz(jet.values, derivative, weights, epsilon=0.1))
    return dict(kind=kind, seed=SEED, features=features, points=points,
                weights=weights, jet=jet, function=function)


def metric_matrix(kind, jet, weights):
    root = weights.sqrt()[:, None]
    derivative = jet.gradient[:, :, 0]
    return (root * derivative if kind == "p4" else
            torch.cat((root * jet.values, 0.1 * root * derivative), dim=0))


def qr_factor(matrix):
    if not spectrum(matrix).full_column_rank:
        raise ValueError("The QR pilot requires a full-rank feature metric.")
    r = torch.linalg.qr(matrix, mode="reduced").R
    signs = torch.where(torch.diag(r) < 0, -torch.ones_like(torch.diag(r)),
                        torch.ones_like(torch.diag(r)))
    return signs[:, None] * r


class QRObjective:
    """Same Ritz energy in b=Ra coordinates, with exact transformed gradient."""

    def __init__(self, objective, r_factor):
        self.objective = objective
        self.r_factor = r_factor

    def coefficients(self, b):
        return torch.linalg.solve_triangular(self.r_factor, b[:, None], upper=True)[:, 0]

    def __call__(self, b):
        return self.objective(self.coefficients(b))

    def analytic_value(self, b):
        return self.objective.analytic_value(self.coefficients(b))

    def analytic_value_gradient(self, b):
        value, gradient_a = self.objective.analytic_value_gradient(self.coefficients(b))
        gradient_b = torch.linalg.solve_triangular(
            self.r_factor.T, gradient_a[:, None], upper=False)[:, 0]
        return value, gradient_b


def check():
    torch.set_num_threads(1)
    for kind in ("p4", "ac"):
        for basis in ("control", "qr_current"):
            problem = make_control(kind) if basis == "control" else make_problem(kind, SEED, device="cpu")
            matrix = metric_matrix(kind, problem["jet"], problem["weights"])
            rank = spectrum(matrix)
            if not rank.full_column_rank:
                raise AssertionError(f"Rank failure in {basis}/{kind}.")
            if basis == "control" and float(torch.linalg.cond(matrix)) > 20:
                raise AssertionError(f"The proposed {kind} control is not well-conditioned.")
            r_factor = qr_factor(matrix)
            a0 = dict(initializations(kind, SEED, device="cpu"))["plus"]
            b0 = r_factor @ a0
            wrapped = QRObjective(problem["function"], r_factor)
            if not torch.allclose(wrapped.coefficients(b0), a0, atol=1e-9, rtol=1e-9):
                raise AssertionError("QR coordinates changed the initial function.")
            candidate = b0.detach().clone().requires_grad_(True)
            automatic = torch.autograd.grad(wrapped(candidate), candidate)[0]
            value, analytic = wrapped.analytic_value_gradient(b0)
            if (not torch.allclose(automatic, analytic, atol=1e-7, rtol=1e-8)
                    or abs(value - problem["function"].analytic_value(a0)) > 1e-9):
                raise AssertionError(f"QR value/gradient mismatch in {basis}/{kind}.")
            result = armijo(wrapped, b0, max_steps=3, gradient_tolerance=0.0, **ARMIJO)
            if any(right["objective"] > left["objective"] + 1e-12
                   for left, right in zip(result.trajectory, result.trajectory[1:])):
                raise AssertionError("QR Armijo did not decrease the energy.")
    print("conditioning_pilot_preflight_passed=all")


def campaign(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    plan = default_plan()
    summary = dict(mode="conditioning_pilot", created_utc=datetime.now(timezone.utc).isoformat(),
                   plan_sha256=file_hash(plan), accepted_step_limit=MAX_STEPS,
                   feature_rule="m=8; w_j=16; b_j=-16 c_j; c_j evenly spaced on [0.1,0.9]",
                   cases=CASES, records=[])
    for basis, kind, label in CASES:
        problem = make_control(kind) if basis == "control" else make_problem(kind, SEED, device="cpu")
        a0 = dict(initializations(kind, SEED, device="cpu"))[label]
        r_factor = None
        objective, initial = problem["function"], a0
        if basis == "qr_current":
            r_factor = qr_factor(metric_matrix(kind, problem["jet"], problem["weights"]))
            objective, initial = QRObjective(objective, r_factor), r_factor @ a0
        result = armijo(objective, initial, max_steps=MAX_STEPS,
                        gradient_tolerance=GRADIENT_TOLERANCE[kind], **ARMIJO)
        config = dict(run_kind="calibration", pilot_kind="conditioning_pilot",
                      basis=basis, branch=kind, feature_seed=SEED,
                      initialization=label, coordinate="b=Ra" if r_factor is not None else "a",
                      qr_r_factor=r_factor, device="cpu", dtype="float64", width=WIDTH,
                      accepted_step_limit=MAX_STEPS,
                      gradient_tolerance=GRADIENT_TOLERANCE[kind], armijo=ARMIJO,
                      feature_weights=problem["features"].weights,
                      feature_biases=problem["features"].biases,
                      training_points=problem["points"],
                      training_weights=problem["weights"])
        name = f"{basis}_{kind}_seed{SEED}_{label}"
        metadata = save_run(output / name, config, result, plan)
        summary["records"].append(dict(run=name, metadata=str(metadata),
                                       stop_reason=result.stop_reason,
                                       accepted_steps=result.accepted_steps,
                                       elapsed_seconds=result.elapsed_seconds))
        atomic_json(output / "summary.json", summary)
        print(f"completed {name}: {result.accepted_steps} steps, {result.stop_reason}", flush=True)
    return summary


def evaluate(meta, trajectory, step):
    config = meta["config"]
    device = "cpu"
    features = RandomFeatures(torch.as_tensor(config["feature_weights"], dtype=DTYPE, device=device),
                              torch.as_tensor(config["feature_biases"], dtype=DTYPE, device=device))
    quadrature = gauss_legendre(256, 1)
    jet = features.jet(quadrature.points)
    derivative = jet.gradient[:, :, 0]
    kind = config["branch"]
    objective = (P4Ritz(jet.values, derivative, p4_source(quadrature.points), quadrature.weights)
                 if kind == "p4" else
                 AllenCahnRitz(jet.values, derivative, quadrature.weights, epsilon=0.1))
    coefficients = torch.as_tensor(trajectory["parameters"][step], dtype=DTYPE)
    if config.get("qr_r_factor") is not None:
        r_factor = torch.as_tensor(config["qr_r_factor"], dtype=DTYPE)
        coefficients = torch.linalg.solve_triangular(r_factor, coefficients[:, None],
                                                     upper=True)[:, 0]
    energy, gradient_a = objective.analytic_value_gradient(coefficients)
    metric = metric_matrix(kind, jet, quadrature.weights)
    rho = qr_dual_residual(gradient_a, qr_factor(metric))
    return dict(step=step, energy_256=energy, rho_256=rho,
                raw_gradient_a=float(torch.linalg.vector_norm(gradient_a)),
                coefficients=coefficients.tolist(),
                gradient_training_coordinates=float(trajectory["gradient_norm"][step]))


def pair_gap(left, right, kind, config):
    a = np.asarray(left["coefficients"])
    b = np.asarray(right["coefficients"])
    coefficient_gap = float(np.linalg.norm(a - b) / max(1.0, np.linalg.norm(a), np.linalg.norm(b)))
    if kind != "p4":
        return dict(coefficient_gap=coefficient_gap)
    features = RandomFeatures(torch.as_tensor(config["feature_weights"], dtype=DTYPE),
                              torch.as_tensor(config["feature_biases"], dtype=DTYPE))
    quadrature = gauss_legendre(256, 1)
    derivative = features.jet(quadrature.points).gradient[:, :, 0]
    du_a = derivative @ torch.as_tensor(a, dtype=DTYPE)
    du_b = derivative @ torch.as_tensor(b, dtype=DTYPE)
    def l4(vector):
        return float(torch.dot(quadrature.weights, vector.abs().pow(4)).pow(0.25))
    return dict(coefficient_gap=coefficient_gap,
                function_gap=l4(du_a - du_b) / max(1.0, l4(du_a), l4(du_b)))


def analyze(output, baseline):
    output, baseline = Path(output), Path(baseline)
    rows, metadata = {}, {}
    for basis, kind, label in CASES:
        name = f"{basis}_{kind}_seed{SEED}_{label}"
        meta, trajectory = load_run(output / name)
        final = meta["result"]["accepted_steps"]
        rows[name] = dict(basis=basis, branch=kind, initialization=label,
                          accepted_steps=final, stop_reason=meta["result"]["stop_reason"],
                          initial=evaluate(meta, trajectory, 0),
                          final=evaluate(meta, trajectory, final))
        metadata[name] = meta
    controls = {}
    for kind in ("p4", "ac"):
        left = f"control_{kind}_seed{SEED}_plus"
        right = f"control_{kind}_seed{SEED}_minus"
        controls[kind] = pair_gap(rows[left]["final"], rows[right]["final"],
                                  kind, metadata[left]["config"])
    qr_comparisons = {}
    for kind in ("p4", "ac"):
        name = f"qr_current_{kind}_seed{SEED}_plus"
        old_meta, old_trajectory = load_run(baseline / f"{kind}_seed{SEED}_plus")
        original = evaluate(old_meta, old_trajectory, old_meta["result"]["accepted_steps"])
        old_initial = evaluate(old_meta, old_trajectory, 0)
        initial_a = np.asarray(rows[name]["initial"]["coefficients"])
        if not np.allclose(initial_a, old_initial["coefficients"], atol=1e-9, rtol=1e-9):
            raise ValueError("QR and original trajectories do not share their initial function.")
        qr_comparisons[kind] = dict(original_50k=original, qr_50k=rows[name]["final"])
    report = dict(mode="conditioning_pilot", runs=rows, control_pairs=controls,
                  qr_comparisons=qr_comparisons)
    atomic_json(output / "analysis.json", report)
    return report


def main():
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "campaign", "analyze"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline", type=Path,
                        default=ROOT / "output" / "experiment2" / "calibration_20260916_v2")
    args = parser.parse_args()
    if args.command == "check":
        check()
    elif args.output is None:
        parser.error("--output is required for campaign and analyze")
    elif args.command == "campaign":
        campaign(args.output)
    else:
        report = analyze(args.output, args.baseline)
        print(f"conditioning_pilot_analyzed={len(report['runs'])}")


if __name__ == "__main__":
    main()
