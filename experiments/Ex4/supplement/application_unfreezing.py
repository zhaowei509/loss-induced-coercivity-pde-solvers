"""S3B: same-model bias unfreezing from S3A central endpoints."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from experiments.Ex2.supplement.application_common import (
    DTYPE, FORMAL_SEEDS, SMOKE_SEED, MasterBank, boundary_metrics,
    gauss_legendre_metrics, make_master_bank, target_source, validation_residual)
from experiments.shared.artifacts import (create_directory, freeze_file, replace_json, sha256_file,
                                          write_json)
from experiments.shared.manifests import build_manifest
from experiments.shared.numerics import spectrum


WIDTH = 128
TRAINING_COUNT = 16384
ACTIVE_BIASES = 8
EVALUATION_CHECKPOINTS = (0, 1, 5, 20, 50, 100)


def hard_basis_bias_derivative(points: torch.Tensor, weights: torch.Tensor,
                               biases: torch.Tensor, active: int = ACTIVE_BIASES
                               ) -> tuple[torch.Tensor, torch.Tensor,
                                          torch.Tensor, torch.Tensor]:
    """Basis, -Laplacian, and their diagonal derivatives in active biases."""
    x, y = points[:, 0], points[:, 1]
    z = 2 * points - 1
    activation = torch.tanh(z @ weights.T + biases)
    first = 1 - activation.square()
    second = -2 * activation * first
    third = (-2 + 6 * activation.square()) * first
    h = x * (1 - x) * y * (1 - y)
    hx = (1 - 2 * x) * y * (1 - y)
    hy = (1 - 2 * y) * x * (1 - x)
    laplacian_h = -2 * (y * (1 - y) + x * (1 - x))
    wx, wy = weights[:, 0], weights[:, 1]
    norm_squared = weights.square().sum(dim=1)
    values = h[:, None] * activation
    laplacian = (laplacian_h[:, None] * activation
                 + 4 * hx[:, None] * first * wx
                 + 4 * hy[:, None] * first * wy
                 + 4 * h[:, None] * second * norm_squared)
    derivative_values = h[:, None] * first[:, :active]
    derivative_laplacian = (
        laplacian_h[:, None] * first[:, :active]
        + 4 * hx[:, None] * second[:, :active] * wx[:active]
        + 4 * hy[:, None] * second[:, :active] * wy[:active]
        + 4 * h[:, None] * third[:, :active] * norm_squared[:active])
    return values, -laplacian, derivative_values, -derivative_laplacian


class JointObjective:
    def __init__(self, points: torch.Tensor, weights: torch.Tensor,
                 frozen_biases: torch.Tensor, frequency: int = 1):
        self.points = points
        self.weights = weights
        self.frozen_biases = frozen_biases
        self.source = target_source(points, frequency)

    def split(self, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if theta.shape != (WIDTH + ACTIVE_BIASES,):
            raise ValueError("Incorrect S3B active parameter dimension.")
        biases = self.frozen_biases.clone()
        biases[:ACTIVE_BIASES] = theta[WIDTH:]
        return theta[:WIDTH], biases

    def residual_jacobian(self, theta: torch.Tensor
                          ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        coefficients, biases = self.split(theta)
        values, negative_laplacian, dvalues, dnegative_laplacian = (
            hard_basis_bias_derivative(self.points, self.weights, biases))
        u = values @ coefficients
        residual = negative_laplacian @ coefficients + u.pow(3) - self.source
        output_jacobian = negative_laplacian + 3 * u.square()[:, None] * values
        hidden_jacobian = coefficients[:ACTIVE_BIASES] * (
            dnegative_laplacian + 3 * u.square()[:, None] * dvalues)
        return residual, output_jacobian, hidden_jacobian

    def value_gradient(self, theta: torch.Tensor) -> tuple[float, torch.Tensor]:
        residual, output, hidden = self.residual_jacobian(theta)
        gradient = torch.cat((output.T @ residual, hidden.T @ residual)) / residual.numel()
        return float(0.5 * residual.square().mean()), gradient


def geometry(objective: JointObjective, theta: torch.Tensor) -> dict[str, Any]:
    residual, output_raw, hidden_raw = objective.residual_jacobian(theta)
    root_n = math.sqrt(residual.numel())
    normalized_residual = residual / root_n
    output = output_raw / root_n
    hidden = hidden_raw / root_n
    output_spectrum = spectrum(output)
    raw_output_gradient = output.T @ normalized_residual
    raw_hidden_gradient = hidden.T @ normalized_residual
    result: dict[str, Any] = {
        "A": output_spectrum.as_dict(),
        "output_gradient_norm": float(torch.linalg.vector_norm(raw_output_gradient)),
        "hidden_gradient_norm": float(torch.linalg.vector_norm(raw_hidden_gradient)),
        "hidden_directional_derivative": -float(torch.dot(raw_hidden_gradient,
                                                           raw_hidden_gradient)),
        "normalized_residual_norm": float(torch.linalg.vector_norm(normalized_residual)),
    }
    if not output_spectrum.full_column_rank:
        result.update({"status": "rank_failure", "C_perp": None, "D": None})
        return result
    q, _ = torch.linalg.qr(output, mode="reduced")
    perpendicular = hidden - q @ (q.T @ hidden)
    full = torch.cat((output, hidden), dim=1)
    result.update({"status": "pass", "C_perp": spectrum(perpendicular).as_dict(),
                   "D": spectrum(full).as_dict(),
                   "transverse_gradient_norm": float(torch.linalg.vector_norm(
                       perpendicular.T @ normalized_residual))})
    return result


def _evaluation(bank: MasterBank, theta: torch.Tensor, scale: float,
                objective: JointObjective, initial: torch.Tensor) -> dict[str, Any]:
    coefficients, biases = objective.split(theta)
    current_bank = replace(bank, biases=biases)
    value, gradient = objective.value_gradient(theta)
    return {
        "training_objective": value,
        "training_rms_residual": math.sqrt(max(0.0, 2 * value)),
        "validation_rms_residual": validation_residual(
            current_bank, coefficients, WIDTH, scale, 1),
        "relative_stationarity": float(torch.linalg.vector_norm(gradient))
        / max(1.0, float(torch.linalg.vector_norm(theta))),
        "gauss_legendre": gauss_legendre_metrics(
            current_bank, coefficients, WIDTH, scale, 1),
        "boundary": boundary_metrics(current_bank, coefficients, WIDTH, scale),
        "relative_displacement": float(torch.linalg.vector_norm(theta - initial))
        / max(1.0, float(torch.linalg.vector_norm(initial))),
    }


def armijo(objective: JointObjective, initial: torch.Tensor, bank: MasterBank,
           scale: float, *, max_accepted_steps: int = 100) -> dict[str, Any]:
    theta = initial.detach().clone()
    value, gradient = objective.value_gradient(theta)
    started = time.perf_counter()
    accepted = 0
    reason = "accepted_step_budget"
    trajectory = [{"accepted_step": 0, "objective": value,
                   "gradient_norm": float(torch.linalg.vector_norm(gradient)),
                   "parameter_norm": float(torch.linalg.vector_norm(theta)),
                   "evaluation": _evaluation(bank, theta, scale, objective, initial)}]
    while accepted < max_accepted_steps:
        relative = float(torch.linalg.vector_norm(gradient)) / max(
            1.0, float(torch.linalg.vector_norm(theta)))
        if relative <= 1e-8:
            reason = "stationarity_pass"
            break
        if not math.isfinite(value) or not bool(torch.isfinite(gradient).all()):
            reason = "nonfinite"
            break
        gradient_squared = float(torch.dot(gradient, gradient))
        step = 1.0
        for backtracks in range(60):
            candidate = theta - step * gradient
            if torch.equal(candidate, theta):
                reason = "armijo_failure"
                break
            candidate_value, _ = objective.value_gradient(candidate)
            if (math.isfinite(candidate_value)
                    and candidate_value <= value - 1e-4 * step * gradient_squared):
                break
            step *= 0.5
        else:
            reason = "armijo_failure"
            break
        if reason == "armijo_failure":
            break
        previous = value
        theta = candidate.detach()
        value, gradient = objective.value_gradient(theta)
        accepted += 1
        row = {"accepted_step": accepted, "objective": value,
               "gradient_norm": float(torch.linalg.vector_norm(gradient)),
               "parameter_norm": float(torch.linalg.vector_norm(theta)),
               "step_size": step, "backtracks": backtracks,
               "armijo_margin": previous - 1e-4 * step * gradient_squared - value}
        if accepted in EVALUATION_CHECKPOINTS:
            row["evaluation"] = _evaluation(bank, theta, scale, objective, initial)
        trajectory.append(row)
    if accepted not in EVALUATION_CHECKPOINTS:
        trajectory[-1]["evaluation"] = _evaluation(
            bank, theta, scale, objective, initial)
    return {"parameters": theta, "stop_reason": reason,
            "accepted_steps": accepted, "elapsed_seconds": time.perf_counter() - started,
            "trajectory": trajectory}


def run_seed(upstream: dict[str, Any], scale: float, *, device: str,
             max_accepted_steps: int = 100) -> dict[str, Any]:
    seed = int(upstream["feature_seed"])
    base = {"schema_version": 1, "supplement": "S3B", "feature_seed": seed,
            "feature_scale": scale, "device": device, "dtype": "float64"}
    if (upstream.get("configuration") != "W128_C"
            or float(upstream.get("feature_scale", float("nan"))) != scale
            or upstream.get("evaluation") is None or upstream.get("training") is None):
        return {**base, "stop_reason": "upstream_failure", "geometry": None,
                "trajectory": None}
    bank = make_master_bank(seed, device=device)
    if bank.hashes() != upstream["bank_hashes"]:
        raise ValueError(f"Master-bank hash mismatch for seed {seed}.")
    coefficients = torch.as_tensor(upstream["training"]["parameters"],
                                   dtype=DTYPE, device=device)
    theta = torch.cat((coefficients, bank.biases[:ACTIVE_BIASES]))
    objective = JointObjective(bank.training_points[:TRAINING_COUNT],
                               scale * bank.base_weights[:WIDTH],
                               bank.biases[:WIDTH])
    initial_geometry = geometry(objective, theta)
    dynamics = armijo(objective, theta, bank, scale,
                      max_accepted_steps=max_accepted_steps)
    return {**base, "stop_reason": dynamics["stop_reason"],
            "upstream_stop_reason": upstream["stop_reason"],
            "upstream_bank_hashes": upstream["bank_hashes"],
            "initial_geometry": initial_geometry, "dynamics": dynamics}


def _summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    available = [row for row in results if row.get("dynamics") is not None]
    return {
        "pre_registered_count": len(results), "available_endpoint_count": len(available),
        "rank_pass_count": sum(row["initial_geometry"]["status"] == "pass"
                               for row in available),
        "nonzero_hidden_gradient_count": sum(
            row["initial_geometry"]["hidden_gradient_norm"] > 0 for row in available),
        "stop_reasons": dict(Counter(row["stop_reason"] for row in results)),
    }


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PLAN_PATH = PROJECT_ROOT / "PAPER_A_SUPPLEMENTARY_EXPERIMENT_PLAN.md"
SOURCE_PATHS = (
    Path(__file__),
    PROJECT_ROOT / "experiments/Ex2/supplement/application_common.py",
    PROJECT_ROOT / "experiments/shared/artifacts.py",
    PROJECT_ROOT / "experiments/shared/numerics.py",
    PROJECT_ROOT / "experiments/shared/manifests.py",
)


def campaign(upstream_root: Path, output: Path, *, kind: str, scale: float,
             device: str, resume: bool = False) -> dict[str, Any]:
    seeds = (SMOKE_SEED,) if kind == "smoke" else FORMAL_SEEDS
    max_accepted_steps = 1 if kind == "smoke" else 100
    upstream_manifest = upstream_root / "manifest.json"
    if not upstream_manifest.is_file():
        raise FileNotFoundError(upstream_manifest)
    upstream_metadata = json.loads(upstream_manifest.read_text(encoding="utf-8"))
    upstream_frozen = upstream_metadata["frozen_config"]
    if (float(upstream_frozen["feature_scale"]) != scale
            or upstream_frozen["device"] != device
            or not set(seeds).issubset(set(upstream_frozen["seeds"]))):
        raise ValueError("The S3A upstream manifest does not match S3B scale/device/seeds.")
    if resume:
        root = output
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if manifest["frozen_config"]["upstream_manifest_sha256"] != sha256_file(upstream_manifest):
            raise ValueError("Upstream S3A manifest changed since this campaign was frozen.")
    else:
        root = create_directory(output)
        freeze_file(PLAN_PATH, root / "plan_frozen.md")
        manifest = build_manifest(
            project_root=PROJECT_ROOT, supplement="S3B", run_kind=kind,
            plan_path=PLAN_PATH, source_paths=SOURCE_PATHS,
            frozen_config={"seeds": seeds, "feature_scale": scale,
                           "upstream_root": str(upstream_root.resolve()),
                           "upstream_manifest_sha256": sha256_file(upstream_manifest),
                           "active_parameters": "128 output coefficients + first 8 biases",
                           "armijo": {"trial_step": 1.0, "shrink": 0.5,
                                      "constant": 1e-4, "max_backtracks": 60,
                                      "max_accepted_steps": max_accepted_steps},
                           "device": device})
        write_json(root / "manifest.json", manifest)
    results = []
    for seed in seeds:
        destination = root / f"seed_{seed}.json"
        if destination.is_file():
            results.append(json.loads(destination.read_text(encoding="utf-8")))
            continue
        source = upstream_root / f"W128_C_seed_{seed}.json"
        if source.is_file():
            upstream = json.loads(source.read_text(encoding="utf-8"))
            result = run_seed(upstream, scale, device=device,
                              max_accepted_steps=max_accepted_steps)
        else:
            result = {"schema_version": 1, "supplement": "S3B",
                      "feature_seed": seed, "feature_scale": scale,
                      "stop_reason": "upstream_failure", "geometry": None,
                      "trajectory": None}
        write_json(destination, result)
        results.append(result)
        replace_json(root / "summary.json", _summary(results))
    summary = _summary(results)
    replace_json(root / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "formal"))
    parser.add_argument("upstream", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(campaign(args.upstream, args.output, kind=args.mode, scale=args.scale,
                   device=args.device, resume=args.resume))


if __name__ == "__main__":
    main()
