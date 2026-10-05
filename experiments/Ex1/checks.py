"""Deterministic preflight checks required by EXPERIMENT1_PLAN.md."""

from datetime import datetime, timezone
import math
from pathlib import Path

import torch

from .diagnostics import Objective
from .dynamics import armijo, gradient_flow
from .experiment import make_b_problem, make_toy_problem, run_a1
from .features import DTYPE
from .io import atomic_json, environment_snapshot, file_hash
from .problems import LeastSquares, toy_coefficients, toy_features


def _relative(numerator, denominator):
    return float(torch.linalg.vector_norm(numerator) /
                 max(float(torch.linalg.vector_norm(denominator)), 1e-30))


def _direct_operator(point, kind):
    point = point.detach().clone().requires_grad_(True)
    features = toy_features()
    exact = toy_coefficients()

    def solution(z):
        return torch.tanh(z @ features.weights.T + features.biases) @ exact

    value = solution(point)
    gradient = torch.autograd.grad(value, point, create_graph=True)[0]
    second = []
    for coordinate in range(2):
        second.append(torch.autograd.grad(gradient[coordinate], point,
                                          retain_graph=True)[0][coordinate])
    if kind == "A1":
        return -sum(second) + value
    if kind == "A2":
        return gradient[1] - 0.2**2 * second[0] + value**3 - value
    if kind == "A3":
        return -sum(second) + value**3
    raise ValueError(kind)


def _manufactured_source_check(kind):
    problem = make_toy_problem(kind, order=5)
    indices = torch.linspace(0, problem["interior_points"].shape[0] - 1,
                             9, dtype=torch.int64)
    independent = torch.stack([
        _direct_operator(problem["interior_points"][index], kind)
        for index in indices])
    recorded = problem["source"][indices]
    return dict(maximum_absolute_error=float(torch.max(torch.abs(independent - recorded))),
                relative_error=_relative(independent - recorded, recorded))


def _loss_derivative_check(kind):
    function = make_toy_problem(kind, order=7)["function"]
    point = torch.tensor([0.91, -0.63], dtype=DTYPE)
    objective = Objective(function)
    _, gradient = objective.value_gradient(point)
    hessian = objective.hessian(point)
    step = 2e-5
    basis = torch.eye(2, dtype=DTYPE)
    finite_gradient = torch.empty_like(point)
    finite_hessian = torch.empty_like(hessian)
    for column in range(2):
        plus = point + step * basis[column]
        minus = point - step * basis[column]
        finite_gradient[column] = (Objective(function).value(plus)
                                   - Objective(function).value(minus)) / (2 * step)
        plus_gradient = Objective(function).value_gradient(plus)[1]
        minus_gradient = Objective(function).value_gradient(minus)[1]
        finite_hessian[:, column] = (plus_gradient - minus_gradient) / (2 * step)
    return dict(gradient_relative_error=_relative(finite_gradient - gradient, gradient),
                hessian_relative_error=_relative(finite_hessian - hessian, hessian),
                hessian_symmetry_error=float(torch.max(torch.abs(hessian - hessian.T))))


def _event_checks():
    quadratic = lambda x: 0.5 * torch.dot(x - 1.0, x - 1.0)
    target = gradient_flow(
        Objective(quadratic), torch.zeros(1, dtype=DTYPE), time_limit=30,
        rtol=1e-10, atol=1e-12,
        target=lambda x: torch.linalg.vector_norm(x - 1.0),
        target_tolerance=1e-8)
    gradient = gradient_flow(
        Objective(quadratic), torch.ones(1, dtype=DTYPE), time_limit=1,
        rtol=1e-10, atol=1e-12, gradient_tolerance=1e-12)
    outward = lambda x: -x[0]
    radius = gradient_flow(
        Objective(outward), torch.zeros(1, dtype=DTYPE), time_limit=1,
        rtol=1e-10, atol=1e-12, center=torch.zeros(1, dtype=DTYPE), radius=0.1)
    budget = gradient_flow(
        Objective(quadratic), torch.zeros(1, dtype=DTYPE), time_limit=1e-5,
        rtol=1e-10, atol=1e-12)
    nonfinite = gradient_flow(
        Objective(lambda x: x[0] * torch.tensor(float("nan"), dtype=DTYPE)),
        torch.zeros(1, dtype=DTYPE), time_limit=1,
        rtol=1e-10, atol=1e-12)
    armijo_gradient = armijo(Objective(quadratic), torch.ones(1, dtype=DTYPE))
    armijo_budget = armijo(Objective(quadratic), torch.zeros(1, dtype=DTYPE),
                            max_steps=0)
    quartic = lambda x: x[0]**4
    armijo_failure = armijo(Objective(quartic), torch.ones(1, dtype=DTYPE),
                             trial_step=1000.0, max_backtracks=1)
    armijo_radius = armijo(Objective(outward), torch.zeros(1, dtype=DTYPE),
                            max_steps=2, confinement_radius=0.1)
    observed = dict(flow_target=target.stop_reason,
                    flow_gradient=gradient.stop_reason,
                    flow_radius=radius.stop_reason,
                    flow_budget=budget.stop_reason,
                    flow_nonfinite=nonfinite.stop_reason,
                    armijo_gradient=armijo_gradient.stop_reason,
                    armijo_budget=armijo_budget.stop_reason,
                    armijo_failure=armijo_failure.stop_reason,
                    armijo_radius=armijo_radius.stop_reason)
    expected = dict(flow_target="target_tolerance",
                    flow_gradient="gradient_tolerance",
                    flow_radius="radius_audit", flow_budget="time_budget",
                    flow_nonfinite="nonfinite_objective_or_gradient",
                    armijo_gradient="gradient_tolerance",
                    armijo_budget="iteration_budget",
                    armijo_failure="line_search_failure",
                    armijo_radius="radius_audit")
    return dict(observed=observed, expected=expected, passed=observed == expected)


def _b_analytic_gradient_check():
    function = make_b_problem(30000, interior_count=128)["function"]
    errors = []
    for point in (torch.zeros(8, dtype=DTYPE),
                  torch.linspace(-0.3, 0.4, 8, dtype=DTYPE)):
        analytic_value, analytic_gradient = function.analytic_value_gradient(point)
        variable = point.detach().clone().requires_grad_(True)
        autodiff_value = function(variable)
        autodiff_gradient = torch.autograd.grad(autodiff_value, variable)[0]
        errors.append(dict(
            value_absolute_error=abs(float(analytic_value - autodiff_value.detach())),
            gradient_relative_error=_relative(
                analytic_gradient - autodiff_gradient.detach(), autodiff_gradient.detach())))
    return dict(cases=errors,
                maximum_value_absolute_error=max(row["value_absolute_error"] for row in errors),
                maximum_gradient_relative_error=max(row["gradient_relative_error"] for row in errors))


def run_checks(output, plan_path):
    output, plan_path = Path(output), Path(plan_path)
    output.mkdir(parents=True, exist_ok=False)
    sources = {kind: _manufactured_source_check(kind) for kind in ("A1", "A2", "A3")}
    derivatives = {kind: _loss_derivative_check(kind) for kind in ("A1", "A2", "A3")}
    toy = {}
    for kind in ("A1", "A2", "A3"):
        problem = make_toy_problem(kind)
        objective = Objective(problem["function"])
        exact_value = objective.value(problem["exact"])
        singular_values = torch.linalg.svdvals(problem["boundary_matrix"])
        toy[kind] = dict(exact_objective=exact_value,
                         boundary_rank=int(torch.linalg.matrix_rank(problem["boundary_matrix"])),
                         sigma_min_boundary=float(singular_values[-1]))
    _, a1 = run_a1(0, "unit_smoke", max_steps=25)
    events = _event_checks()
    b_analytic = _b_analytic_gradient_check()
    tolerances = dict(source_absolute=2e-12, derivative_relative=2e-7,
                      exact_objective=1e-20, recurrence_relative=1e-12)
    passed = (all(row["maximum_absolute_error"] <= tolerances["source_absolute"]
                  for row in sources.values())
              and all(row["gradient_relative_error"] <= tolerances["derivative_relative"]
                      and row["hessian_relative_error"] <= tolerances["derivative_relative"]
                      for row in derivatives.values())
              and all(row["exact_objective"] <= tolerances["exact_objective"]
                      and row["boundary_rank"] == 2 for row in toy.values())
              and a1.diagnostics["recurrence_pass"]
              and a1.diagnostics["contraction_pass"] and events["passed"]
              and b_analytic["maximum_value_absolute_error"] <= 1e-12
              and b_analytic["maximum_gradient_relative_error"] <= 2e-12)
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), passed=passed,
                  plan_sha256=file_hash(plan_path), environment=environment_snapshot("cpu"),
                  tolerances=tolerances, manufactured_sources=sources,
                  loss_derivatives=derivatives, toy_prechecks=toy,
                  a1_identity=a1.diagnostics, stopping_events=events)
    report["b_analytic_gradient"] = b_analytic
    atomic_json(output / "check_report.json", report)
    if not passed:
        raise RuntimeError(f"Experiment 1 preflight failed; see {output / 'check_report.json'}")
    return report
