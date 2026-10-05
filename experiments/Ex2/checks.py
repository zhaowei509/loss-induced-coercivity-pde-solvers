"""Deterministic preflight for copied kernels and new Experiment 2 objectives."""

import math
from pathlib import Path
import tempfile

import torch

from .dynamics import armijo
from .experiment import (QRObjective, default_plan, initializations, make_problem,
                         qr_factor, training_metric)
from .features import DTYPE, RandomFeatures, qr_dual_residual, spectrum
from .io import load_run, save_run
from .objectives import (AllenCahnRitz, P4Ritz, cubic_solution, cubic_source,
                         p4_source)
from .sampling import gauss_legendre, sobol_points


def _close(actual, expected, *, atol=1e-10, rtol=1e-9):
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        difference = float((actual - expected).abs().max())
        raise AssertionError(f"Numerical check failed; max difference={difference:.3e}")


def _feature_check(device):
    features = RandomFeatures.draw(2, 4, 30000, device=device)
    x = torch.tensor([[0.23, 0.41], [0.58, 0.67]], dtype=DTYPE,
                     device=device, requires_grad=True)
    jet = features.jet(x)
    for j in range(4):
        first = torch.autograd.grad(jet.values[:, j].sum(), x, create_graph=True)[0]
        second = sum(torch.autograd.grad(first[:, d].sum(), x,
                                         retain_graph=True)[0][:, d]
                     for d in range(2))
        _close(first, jet.gradient[:, j, :])
        _close(second, jet.laplacian[:, j])
    boundary = torch.tensor([[0.0, 0.4], [1.0, 0.3], [0.2, 0.0], [0.8, 1.0]],
                            dtype=DTYPE, device=device)
    _close(features.jet(boundary).values, torch.zeros(4, 4, dtype=DTYPE, device=device))


def _objective_check(kind, device):
    problem = make_problem(kind, 30000, device=device)
    function = problem["function"]
    label, initial = initializations(kind, 30000, device=device)[-1]
    a = initial.detach().clone().requires_grad_(True)
    autograd = torch.autograd.grad(function(a), a)[0]
    value, analytic = function.analytic_value_gradient(a.detach())
    if not math.isfinite(value):
        raise AssertionError(f"Nonfinite {kind} objective")
    _close(analytic, autograd, atol=1e-9, rtol=1e-9)
    if kind in ("p4", "ac"):
        refined = gauss_legendre(256, 1, device=device)
        jet = problem["features"].jet(refined.points)
        derivative = jet.gradient[:, :, 0]
        if kind == "p4":
            refined_objective = P4Ritz(jet.values, derivative,
                                       p4_source(refined.points), refined.weights)
        else:
            refined_objective = AllenCahnRitz(jet.values, derivative,
                                              refined.weights, epsilon=0.1)
        refined_value = refined_objective.analytic_value(initial)
        if abs(value - refined_value) > 1e-8 * max(1.0, abs(refined_value)):
            raise AssertionError(f"128/256 quadrature disagreement for {kind}.")
        coarse_jet = problem["features"].jet(problem["points"])
        if kind == "p4":
            coarse_metric = problem["weights"].sqrt()[:, None] * coarse_jet.gradient[:, :, 0]
            refined_metric = refined.weights.sqrt()[:, None] * derivative
        else:
            coarse_metric = torch.cat((problem["weights"].sqrt()[:, None] * coarse_jet.values,
                                       0.1 * problem["weights"].sqrt()[:, None]
                                       * coarse_jet.gradient[:, :, 0]), dim=0)
            refined_metric = torch.cat((refined.weights.sqrt()[:, None] * jet.values,
                                        0.1 * refined.weights.sqrt()[:, None] * derivative), dim=0)
        if not spectrum(coarse_metric).full_column_rank or not spectrum(refined_metric).full_column_rank:
            raise AssertionError(f"Expected full-rank test metrics for {kind}.")
        coarse_rho = qr_dual_residual(analytic, torch.linalg.qr(coarse_metric, mode="reduced").R)
        refined_gradient = refined_objective.analytic_value_gradient(initial)[1]
        refined_rho = qr_dual_residual(
            refined_gradient, torch.linalg.qr(refined_metric, mode="reduced").R)
        if abs(coarse_rho - refined_rho) > 1e-6 * max(1.0, refined_rho):
            raise AssertionError(f"128/256 QR residual disagreement for {kind}.")
    if kind == "ac":
        zero = torch.zeros_like(initial)
        _, gradient = function.analytic_value_gradient(zero)
        _close(gradient, zero, atol=1e-13, rtol=0)
        if abs(function.analytic_value(zero) - 0.25) > 1e-13:
            raise AssertionError("Allen-Cahn zero energy must be 1/4.")
    result = armijo(function, initial, max_steps=3,
                    gradient_tolerance=0.0)
    try:
        armijo(function, initial, max_steps=3, stop_requested=lambda: True)
    except InterruptedError:
        pass
    else:
        raise AssertionError("External stop request was not honored.")
    values = [row["objective"] for row in result.trajectory]
    if any(right > left + 1e-12 for left, right in zip(values, values[1:])):
        raise AssertionError(f"Armijo did not decrease {kind} objective.")
    if result.accepted_steps != len(result.trajectory) - 1:
        raise AssertionError("Armijo trajectory length mismatch.")
    with tempfile.TemporaryDirectory(prefix="ex2-preflight-") as temporary:
        config = dict(run_kind="unit_smoke", branch=kind, feature_seed=30000,
                      initialization=label, width=initial.numel())
        save_run(Path(temporary) / "run", config, result, default_plan())
        metadata, arrays = load_run(Path(temporary) / "run")
        if metadata["result"]["accepted_steps"] != result.accepted_steps:
            raise AssertionError("Saved step count mismatch.")
        _close(torch.as_tensor(arrays["parameters"][-1], dtype=DTYPE),
               result.parameters.cpu())
        if arrays["parameters"].dtype.name != "float64":
            raise AssertionError("Lossless float64 coefficients required.")


def _new_protocol_check():
    """Check the fixed control, QR coordinate map, and Allen--Cahn symmetry."""
    for kind in ("p4", "ac"):
        for basis in ("random", "control"):
            problem = make_problem(kind, 30001, device="cpu", basis=basis)
            metric = training_metric(problem)
            factor = qr_factor(metric)
            if factor is None:
                raise AssertionError(f"{basis}/{kind} is not numerically full rank.")
            if basis == "control" and float(torch.linalg.cond(metric)) > 20:
                raise AssertionError(f"{kind} control is not well conditioned.")
            initial = dict(initializations(kind, 30001, device="cpu"))["plus"]
            transformed = factor @ initial
            objective = QRObjective(problem["function"], factor)
            _close(objective.coefficients(transformed), initial, atol=1e-9, rtol=1e-9)
            candidate = transformed.detach().clone().requires_grad_(True)
            automatic = torch.autograd.grad(objective(candidate), candidate)[0]
            value, analytic = objective.analytic_value_gradient(transformed)
            _close(automatic, analytic, atol=1e-7, rtol=1e-8)
            if abs(value - problem["function"].analytic_value(initial)) > 1e-9:
                raise AssertionError("QR changed the initial energy.")
            result = armijo(objective, transformed, max_steps=3,
                            gradient_tolerance=0.0)
            values = [row["objective"] for row in result.trajectory]
            if any(right > left + 1e-12 for left, right in zip(values, values[1:])):
                raise AssertionError("QR Armijo did not decrease the energy.")
            if kind == "ac":
                energy_plus, gradient_plus = problem["function"].analytic_value_gradient(initial)
                energy_minus, gradient_minus = problem["function"].analytic_value_gradient(-initial)
                if abs(energy_plus - energy_minus) > 1e-12:
                    raise AssertionError("Allen--Cahn energy is not even.")
                _close(gradient_minus, -gradient_plus, atol=1e-12, rtol=1e-10)


def run_checks():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the two-dimensional preflight.")
    torch.set_num_threads(1)
    quadrature = gauss_legendre(16, 2)
    if abs(float(quadrature.weights.sum()) - 1.0) > 1e-13:
        raise AssertionError("Quadrature weights do not sum to 1.")
    points = sobol_points(4096, 2, 30000)
    if points.shape != (4096, 2):
        raise AssertionError("Sobol shape mismatch.")
    manufactured = points[:8].detach().clone().requires_grad_(True)
    u = cubic_solution(manufactured)
    first = torch.autograd.grad(u.sum(), manufactured, create_graph=True)[0]
    laplacian = sum(torch.autograd.grad(first[:, d].sum(), manufactured,
                                         retain_graph=True)[0][:, d]
                    for d in range(2))
    _close(-laplacian + u**3, cubic_source(manufactured))
    x = torch.tensor([[0.17], [0.39], [0.73]], dtype=DTYPE,
                     requires_grad=True)
    p4_solution = torch.sin(torch.pi * x[:, 0])
    derivative = torch.autograd.grad(p4_solution.sum(), x,
                                     create_graph=True)[0][:, 0]
    flux_derivative = torch.autograd.grad(derivative.pow(3).sum(), x)[0][:, 0]
    _close(-flux_derivative, p4_source(x))
    _feature_check("cpu")
    _feature_check("cuda")
    for kind in ("cubic", "p4", "ac"):
        _objective_check(kind, "cuda" if kind == "cubic" else "cpu")
    matrix = torch.eye(8, dtype=DTYPE)
    if not spectrum(matrix).full_column_rank:
        raise AssertionError("Identity matrix should have full column rank.")
    metric = torch.diag(torch.tensor([1e-3, 2.0], dtype=DTYPE))
    gradient = torch.tensor([1e-3, 2.0], dtype=DTYPE)
    residual = qr_dual_residual(gradient, torch.linalg.qr(metric, mode="reduced").R)
    if abs(residual - math.sqrt(2.0)) > 1e-11:
        raise AssertionError("QR dual residual did not remove feature scaling.")
    _new_protocol_check()
    print("experiment2_preflight_passed=all")
