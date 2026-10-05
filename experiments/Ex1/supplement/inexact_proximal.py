"""Audited nonlinear inexact proximal iterations for Experiment 1, class B."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math
import time
from typing import Any

import numpy as np
import torch

from experiments.Ex1.diagnostics import Objective, confinement_radius, spectrum
from experiments.Ex1.experiment import B_BOUNDARY_PENALTY, B_WIDTH, make_b_problem
from experiments.Ex1.features import DTYPE, require_double


CHECKPOINTS = (25, 50, 100, 200)
FORMAL_SEEDS = tuple(range(10))
INITIALIZATIONS = ("zero", "pcg64_direction")


@dataclass(frozen=True)
class InexactProximalConfig:
    eta: float = 0.5
    relative_residual: float = 0.1
    gradient_tolerance: float = 1e-8
    max_outer_steps: int = 100
    checkpoints: tuple[int, ...] = CHECKPOINTS
    history_size: int = 20
    radius_tolerance: float = 1e-10

    def validate(self) -> None:
        if self.eta <= 0 or not 0 < self.relative_residual < 1:
            raise ValueError("eta must be positive and relative_residual in (0,1).")
        if self.gradient_tolerance < 0 or self.max_outer_steps < 0:
            raise ValueError("Invalid stopping configuration.")
        if (not self.checkpoints or tuple(sorted(set(self.checkpoints))) != self.checkpoints
                or self.checkpoints[0] <= 0 or self.history_size < 1):
            raise ValueError("Checkpoints must be positive, unique, and increasing.")


def initial_coefficients(seed: int, kind: str, *, device: str = "cpu") -> torch.Tensor:
    if kind == "zero":
        return torch.zeros(B_WIDTH, dtype=DTYPE, device=device)
    if kind != "pcg64_direction":
        raise ValueError(f"Unknown initialization {kind!r}.")
    generator = np.random.Generator(np.random.PCG64(10000 + seed))
    direction = generator.standard_normal(B_WIDTH)
    norm = float(np.linalg.norm(direction))
    if not math.isfinite(norm) or norm == 0:
        raise RuntimeError("PCG64 produced an invalid direction.")
    return torch.as_tensor(0.1 * direction / norm, dtype=DTYPE, device=device)


def _value_gradient(function: Any, x: torch.Tensor) -> tuple[float, torch.Tensor]:
    analytic = getattr(function, "analytic_value_gradient", None)
    if analytic is not None:
        value, gradient = analytic(x.detach())
        return float(value), gradient.detach()
    with torch.enable_grad():
        point = x.detach().clone().requires_grad_(True)
        value = function(point)
        gradient = torch.autograd.grad(value, point)[0]
    return float(value.detach()), gradient.detach()


def _prox_value_gradient(function: Any, candidate: torch.Tensor,
                         anchor: torch.Tensor, eta: float) -> tuple[float, torch.Tensor]:
    value, gradient = _value_gradient(function, candidate)
    displacement = candidate.detach() - anchor
    return (value + 0.5 / eta * float(torch.dot(displacement, displacement)),
            gradient + displacement / eta)


def solve_inner(function: Any, anchor: torch.Tensor, anchor_value: float,
                config: InexactProximalConfig) -> dict[str, Any]:
    """Run one continuous L-BFGS state and inspect cumulative checkpoints."""
    require_double(anchor)
    candidate = anchor.detach().clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [candidate], lr=1.0, max_iter=1, max_eval=25,
        tolerance_grad=0.0, tolerance_change=0.0,
        history_size=config.history_size, line_search_fn="strong_wolfe")
    closure_count = 0
    audits: list[dict[str, Any]] = []

    for iteration in range(1, config.checkpoints[-1] + 1):
        def closure() -> torch.Tensor:
            nonlocal closure_count
            optimizer.zero_grad(set_to_none=True)
            displacement = candidate - anchor
            value = function(candidate) + 0.5 / config.eta * torch.dot(
                displacement, displacement)
            value.backward()
            closure_count += 1
            return value

        try:
            optimizer.step(closure)
        except RuntimeError as error:
            return {
                "status": "nonfinite" if "nonfinite" in str(error).lower() else "optimizer_failure",
                "candidate": candidate.detach(),
                "iteration": iteration,
                "closure_count": closure_count,
                "checkpoints": audits,
                "error": repr(error),
            }
        if not bool(torch.isfinite(candidate).all()):
            return {"status": "nonfinite", "candidate": candidate.detach(),
                    "iteration": iteration, "closure_count": closure_count,
                    "checkpoints": audits}
        if iteration not in config.checkpoints:
            continue

        q_value, q_gradient = _prox_value_gradient(
            function, candidate.detach(), anchor, config.eta)
        displacement = candidate.detach() - anchor
        step_norm = float(torch.linalg.vector_norm(displacement))
        q_gradient_norm = float(torch.linalg.vector_norm(q_gradient))
        descent_margin = anchor_value - q_value
        residual_margin = config.relative_residual * step_norm - q_gradient_norm
        ratio = None if step_norm == 0 else q_gradient_norm / step_norm
        finite = (math.isfinite(q_value) and math.isfinite(q_gradient_norm)
                  and bool(torch.isfinite(q_gradient).all()))
        audit = {
            "checkpoint": iteration,
            "closure_count": closure_count,
            "q_value": q_value,
            "q_minus_anchor": q_value - anchor_value,
            "sufficient_descent_margin": descent_margin,
            "step_norm": step_norm,
            "q_gradient_norm": q_gradient_norm,
            "relative_residual_ratio": ratio,
            "relative_residual_margin": residual_margin,
            "finite": finite,
            "descent_pass": finite and descent_margin >= 0,
            "relative_residual_pass": finite and step_norm > 0 and residual_margin >= 0,
        }
        audits.append(audit)
        if not finite:
            return {"status": "nonfinite", "candidate": candidate.detach(),
                    "iteration": iteration, "closure_count": closure_count,
                    "checkpoints": audits}
        if step_norm == 0:
            _, anchor_gradient = _value_gradient(function, anchor)
            stationary = float(torch.linalg.vector_norm(anchor_gradient)) <= config.gradient_tolerance
            return {"status": "stationary_zero_step" if stationary else "zero_step_failure",
                    "candidate": candidate.detach(), "iteration": iteration,
                    "closure_count": closure_count, "checkpoints": audits}
        if audit["descent_pass"] and audit["relative_residual_pass"]:
            return {"status": "accepted", "candidate": candidate.detach(),
                    "iteration": iteration, "closure_count": closure_count,
                    "checkpoints": audits, "accepted_audit": audit}

    return {"status": "inner_certificate_failure", "candidate": candidate.detach(),
            "iteration": config.checkpoints[-1], "closure_count": closure_count,
            "checkpoints": audits}


def run_trajectory(seed: int, initialization: str, *, device: str = "cpu",
                   config: InexactProximalConfig | None = None,
                   interior_count: int = 2048) -> dict[str, Any]:
    config = config or InexactProximalConfig()
    config.validate()
    problem = make_b_problem(seed, device=device, interior_count=interior_count)
    function = problem["function"]
    objective = Objective(function)
    current = initial_coefficients(seed, initialization, device=device)
    initial = current.detach().clone()
    initial_value, initial_gradient = objective.value_gradient(current)
    radius = confinement_radius(initial_value, problem["boundary_matrix"],
                                B_BOUNDARY_PENALTY)
    boundary_spectrum = spectrum(problem["boundary_matrix"])
    started = time.perf_counter()
    total_closures = 0
    rows: list[dict[str, Any]] = [{
        "outer_step": 0,
        "parameters": current.detach().cpu().tolist(),
        "objective": initial_value,
        "gradient_norm": float(torch.linalg.vector_norm(initial_gradient)),
        "parameter_norm": float(torch.linalg.vector_norm(current)),
        "radius_ratio": (None if radius is None else
                         float(torch.linalg.vector_norm(current)) / radius),
        "elapsed_seconds": 0.0,
    }]
    reason = "outer_budget"
    accepted_steps = 0

    for outer_step in range(1, config.max_outer_steps + 1):
        value, gradient = objective.value_gradient(current)
        gradient_norm = float(torch.linalg.vector_norm(gradient))
        if not math.isfinite(value) or not bool(torch.isfinite(gradient).all()):
            reason = "nonfinite"
            break
        if gradient_norm <= config.gradient_tolerance:
            reason = "gradient_tolerance"
            break
        inner = solve_inner(function, current, value, config)
        total_closures += int(inner["closure_count"])
        if inner["status"] == "stationary_zero_step":
            reason = "gradient_tolerance"
            break
        if inner["status"] != "accepted":
            reason = ("inner_certificate_failure" if inner["status"] in
                      {"zero_step_failure", "optimizer_failure"} else inner["status"])
            rows.append({
                "outer_step": outer_step,
                "accepted": False,
                "inner_status": inner["status"],
                "inner_iterations": inner["iteration"],
                "inner_closure_count": inner["closure_count"],
                "checkpoint_audits": inner["checkpoints"],
                "error": inner.get("error"),
                "elapsed_seconds": time.perf_counter() - started,
            })
            break

        next_point = inner["candidate"]
        next_value, next_gradient = objective.value_gradient(next_point)
        parameter_norm = float(torch.linalg.vector_norm(next_point))
        ratio = None if radius is None else parameter_norm / radius
        accepted = inner["accepted_audit"]
        rows.append({
            "outer_step": outer_step,
            "accepted": True,
            "parameters": next_point.detach().cpu().tolist(),
            "objective": next_value,
            "gradient_norm": float(torch.linalg.vector_norm(next_gradient)),
            "parameter_norm": parameter_norm,
            "radius_ratio": ratio,
            "q_minus_anchor": accepted["q_minus_anchor"],
            "sufficient_descent_margin": accepted["sufficient_descent_margin"],
            "step_norm": accepted["step_norm"],
            "q_gradient_norm": accepted["q_gradient_norm"],
            "relative_residual_ratio": accepted["relative_residual_ratio"],
            "relative_residual_margin": accepted["relative_residual_margin"],
            "first_passing_checkpoint": accepted["checkpoint"],
            "inner_iterations": inner["iteration"],
            "inner_closure_count": inner["closure_count"],
            "checkpoint_audits": inner["checkpoints"],
            "elapsed_seconds": time.perf_counter() - started,
        })
        current = next_point
        accepted_steps += 1
        if ratio is not None and ratio > 1 + config.radius_tolerance:
            reason = "confinement_violation"
            break
    else:
        reason = "outer_budget"

    final_value, final_gradient = objective.value_gradient(current)
    accepted_rows = [row for row in rows if row.get("accepted")]
    return {
        "schema_version": 1,
        "supplement": "S1",
        "seed": seed,
        "initialization": initialization,
        "device": device,
        "dtype": "float64",
        "interior_count": interior_count,
        "boundary_count": int(problem["boundary_matrix"].shape[0]),
        "width": B_WIDTH,
        "boundary_penalty": B_BOUNDARY_PENALTY,
        "feature_weights": problem["features"].weights,
        "feature_biases": problem["features"].biases,
        "initial_parameters": initial,
        "config": asdict(config),
        "boundary_spectrum": {
            "singular_values": boundary_spectrum.singular_values,
            "numerical_rank": boundary_spectrum.numerical_rank,
            "threshold": boundary_spectrum.threshold,
            "sigma_min": boundary_spectrum.sigma_min,
            "full_column_rank": boundary_spectrum.full_column_rank,
        },
        "confinement_radius": radius,
        "stop_reason": reason,
        "accepted_steps": accepted_steps,
        "total_inner_closure_count": total_closures,
        "final_objective": final_value,
        "final_gradient_norm": float(torch.linalg.vector_norm(final_gradient)),
        "elapsed_seconds": time.perf_counter() - started,
        "minimum_descent_margin": (min(row["sufficient_descent_margin"]
                                       for row in accepted_rows)
                                   if accepted_rows else None),
        "minimum_relative_residual_margin": (
            min(row["relative_residual_margin"] for row in accepted_rows)
            if accepted_rows else None),
        "maximum_relative_residual_ratio": (
            max(row["relative_residual_ratio"] for row in accepted_rows)
            if accepted_rows else None),
        "maximum_radius_ratio": (
            max(row["radius_ratio"] for row in rows
                if row.get("radius_ratio") is not None)
            if radius is not None else None),
        "checkpoint_counts": dict(Counter(
            row["first_passing_checkpoint"] for row in accepted_rows)),
        "trajectory": rows,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [row for result in results for row in result["trajectory"]
                if row.get("accepted")]
    ratios = [row["relative_residual_ratio"] for row in accepted]
    margins = [row["sufficient_descent_margin"] for row in accepted]
    final_gradients = np.asarray([result["final_gradient_norm"] for result in results])
    return {
        "trajectory_count": len(results),
        "completed_without_certificate_failure": sum(
            result["stop_reason"] != "inner_certificate_failure" for result in results),
        "certificate_failure_count": sum(
            result["stop_reason"] == "inner_certificate_failure" for result in results),
        "stop_reasons": dict(Counter(result["stop_reason"] for result in results)),
        "accepted_step_count": len(accepted),
        "checkpoint_counts": dict(Counter(
            row["first_passing_checkpoint"] for row in accepted)),
        "minimum_sufficient_descent_margin": min(margins) if margins else None,
        "maximum_relative_residual_ratio": max(ratios) if ratios else None,
        "final_gradient_median": float(np.median(final_gradients)),
        "final_gradient_q1": float(np.quantile(final_gradients, 0.25)),
        "final_gradient_q3": float(np.quantile(final_gradients, 0.75)),
        "maximum_radius_ratio": max(
            (result["maximum_radius_ratio"] for result in results
             if result["maximum_radius_ratio"] is not None), default=None),
    }

