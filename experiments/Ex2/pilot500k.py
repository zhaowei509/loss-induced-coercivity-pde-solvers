"""Exploratory 500k-step pilot on four B/C calibration trajectories."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.Ex2.analysis import _features, _function_gap, _ritz_record, _tensor
from experiments.Ex2.dynamics import armijo
from experiments.Ex2.experiment import ARMIJO, GRADIENT_TOLERANCE, WIDTH, default_plan
from experiments.Ex2.experiment import initializations, make_problem
from experiments.Ex2.features import qr_dual_residual, spectrum
from experiments.Ex2.io import atomic_json, file_hash, load_run, save_run
from experiments.Ex2.sampling import gauss_legendre


PILOT_RUNS = (("p4", 30001, "plus"), ("p4", 30001, "minus"),
              ("ac", 30001, "plus"), ("ac", 30001, "minus"))
MAX_STEPS = 500000


def _training_residual(kind, seed, coefficients):
    problem = make_problem(kind, seed, device="cpu")
    jet, weights = problem["jet"], problem["weights"]
    if kind == "p4":
        metric = weights.sqrt()[:, None] * jet.gradient[:, :, 0]
    else:
        metric = torch.cat((weights.sqrt()[:, None] * jet.values,
                            0.1 * weights.sqrt()[:, None] * jet.gradient[:, :, 0]), dim=0)
    if not spectrum(metric).full_column_rank:
        return None
    energy, gradient = problem["function"].analytic_value_gradient(_tensor(coefficients))
    return dict(energy=energy,
                rho=qr_dual_residual(gradient, torch.linalg.qr(metric, mode="reduced").R))


def campaign(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    plan = default_plan()
    summary = dict(mode="exploratory_500k", created_utc=datetime.now(timezone.utc).isoformat(),
                   plan_sha256=file_hash(plan), accepted_step_limit=MAX_STEPS,
                   calibration_source="calibration_20260916_v2",
                   records=[])
    for kind, seed, label in PILOT_RUNS:
        device = "cpu"
        problem = make_problem(kind, seed, device=device)
        initial = dict(initializations(kind, seed, device=device))[label]
        result = armijo(problem["function"], initial, max_steps=MAX_STEPS,
                        gradient_tolerance=GRADIENT_TOLERANCE[kind], **ARMIJO)
        config = dict(run_kind="calibration", pilot_kind="exploratory_500k",
                      branch=kind, feature_seed=seed, initialization=label,
                      device=device, dtype="float64", width=WIDTH[kind],
                      accepted_step_limit=MAX_STEPS,
                      gradient_tolerance=GRADIENT_TOLERANCE[kind], armijo=ARMIJO,
                      feature_weights=problem["features"].weights,
                      feature_biases=problem["features"].biases,
                      training_points=problem["points"],
                      training_weights=problem["weights"])
        run_name = f"{kind}_seed{seed}_{label}"
        metadata = save_run(output / run_name, config, result, plan)
        summary["records"].append(dict(run=run_name, metadata=str(metadata),
                                       stop_reason=result.stop_reason,
                                       accepted_steps=result.accepted_steps,
                                       final_gradient=result.trajectory[-1]["gradient_norm"],
                                       elapsed_seconds=result.elapsed_seconds))
        atomic_json(output / "summary.json", summary)
        print(f"completed {run_name}: {result.accepted_steps} steps, {result.stop_reason}",
              flush=True)
    return summary


def analyze(output, baseline):
    output, baseline = Path(output), Path(baseline)
    records = {}
    comparisons = []
    for kind, seed, label in PILOT_RUNS:
        name = f"{kind}_seed{seed}_{label}"
        meta, trajectory = load_run(output / name)
        old_meta, old_trajectory = load_run(baseline / name)
        if (meta["config"]["feature_seed"] != old_meta["config"]["feature_seed"]
                or meta["config"]["initialization"] != old_meta["config"]["initialization"]):
            raise ValueError("Baseline trajectory does not match the pilot selection.")
        if meta["result"]["accepted_steps"] < 50000:
            raise ValueError("Pilot stopped before the 50k comparison point.")
        difference = float(np.linalg.norm(trajectory["parameters"][50000]
                                          - old_trajectory["parameters"][-1]))
        if difference > 1e-9:
            raise ValueError(f"The 50k prefix is not reproducible for {name}: {difference}.")
        comparisons.append(dict(run=name, prefix_coefficient_difference=difference,
                                accepted_steps=meta["result"]["accepted_steps"],
                                stop_reason=meta["result"]["stop_reason"],
                                elapsed_seconds=meta["result"]["elapsed_seconds"]))
        record = _ritz_record(meta, trajectory)
        checks = {str(step): dict(energy_256=record["checkpoints"][str(step)]["energy_256"],
                                  rho=record["checkpoints"][str(step)]["rho"],
                                  training_check=_training_residual(
                                      kind, seed, trajectory["parameters"][step]),
                                  raw_gradient=float(trajectory["gradient_norm"][step]))
                  for step in (50000, meta["result"]["accepted_steps"])}
        records[name] = dict(branch=kind, feature_full_rank=record["feature_full_rank"],
                             metric_full_rank=record["metric_full_rank"],
                             checkpoints=checks)
    p4_rows = []
    for label in ("plus", "minus"):
        meta, trajectory = load_run(output / f"p4_seed30001_{label}")
        p4_rows.append(_ritz_record(meta, trajectory))
    features = _features(load_run(output / "p4_seed30001_plus")[0]["config"], "cpu")
    refined = gauss_legendre(256, 1)
    derivative = features.jet(refined.points).gradient[:, :, 0]
    pair_gap = {str(step): _function_gap(p4_rows, step, derivative, refined.weights)
                for step in (50000, MAX_STEPS)}
    pair_coefficients = {}
    for step in (50000, MAX_STEPS):
        left = np.asarray(p4_rows[0]["checkpoints"][str(step)]["parameters"])
        right = np.asarray(p4_rows[1]["checkpoints"][str(step)]["parameters"])
        pair_coefficients[str(step)] = float(np.linalg.norm(left - right)
                                             / max(1.0, np.linalg.norm(left),
                                                   np.linalg.norm(right)))
    report = dict(mode="exploratory_500k", comparisons=comparisons,
                  p4_pair_function_gap=pair_gap,
                  p4_pair_coefficient_gap=pair_coefficients, details=records)
    atomic_json(output / "comparison.json", report)
    return report


def main():
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("campaign", "analyze"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path,
                        default=ROOT / "output" / "experiment2" / "calibration_20260916_v2")
    args = parser.parse_args()
    if args.command == "campaign":
        campaign(args.output)
    else:
        report = analyze(args.output, args.baseline)
        print(f"pilot_500k_analyzed={len(report['comparisons'])}")


if __name__ == "__main__":
    main()
