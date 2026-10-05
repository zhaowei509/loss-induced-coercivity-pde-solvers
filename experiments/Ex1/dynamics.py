"""Exact quadratic proximal, Armijo descent, and DOP853 gradient flow."""

from dataclasses import dataclass, field
import math
import time

import numpy as np
from scipy.integrate import solve_ivp
import torch

from .diagnostics import ScipyAdapter
from .features import require_double


@dataclass
class SolverResult:
    parameters: torch.Tensor
    stop_reason: str
    trajectory: list
    counts: dict
    accepted_steps: int = 0
    elapsed_seconds: float = 0.0
    diagnostics: dict = field(default_factory=dict)


def _row(step, x, value, gradient, started, **extra):
    return dict(step=step, parameters=x.detach().cpu().tolist(), objective=value,
                gradient_norm=float(torch.linalg.vector_norm(gradient)),
                parameter_norm=float(torch.linalg.vector_norm(x)),
                elapsed_seconds=time.perf_counter() - started, **extra)


def _finite(value, gradient):
    return math.isfinite(value) and bool(torch.isfinite(gradient).all())


def exact_quadratic_step(hessian, linear_term, current, *, eta=0.5):
    """Exact prox of J(a)=a^T H a/2-c^T a+constant via a linear solve."""
    require_double(hessian, linear_term, current)
    if (current.ndim != 1 or linear_term.shape != current.shape
            or hessian.shape != (current.numel(), current.numel()) or eta <= 0
            or not torch.allclose(hessian, hessian.T, atol=1e-14, rtol=1e-12)):
        raise ValueError("Invalid quadratic problem.")
    system = torch.eye(current.numel(), dtype=current.dtype, device=current.device) + eta * hessian
    _, info = torch.linalg.cholesky_ex(system)
    if int(info) != 0:
        raise ValueError("The proximal subproblem is not strictly convex.")
    return torch.linalg.solve(system, current + eta * linear_term).detach()


def armijo(objective, initial, *, max_steps=5000, gradient_tolerance=1e-8,
           trial_step=1.0, shrink=0.5, sufficient_decrease=1e-4,
           max_backtracks=60, confinement_radius=None, radius_tolerance=1e-10):
    """Full-batch Armijo descent with the frozen Experiment 1 defaults."""
    require_double(initial)
    if (max_steps < 0 or gradient_tolerance < 0 or trial_step <= 0
            or not 0 < shrink < 1 or not 0 < sufficient_decrease < 1
            or max_backtracks < 1 or radius_tolerance < 0
            or (confinement_radius is not None and confinement_radius <= 0)):
        raise ValueError("Invalid Armijo configuration.")
    started = time.perf_counter()
    x = initial.detach().clone()
    value, gradient = objective.value_gradient(x)
    rows = [_row(0, x, value, gradient, started)]
    reason, accepted = "iteration_budget", 0
    while True:
        if not _finite(value, gradient):
            reason = "nonfinite_objective_or_gradient"
            break
        if float(torch.linalg.vector_norm(gradient)) <= gradient_tolerance:
            reason = "gradient_tolerance"
            break
        if accepted == max_steps:
            break
        step = trial_step
        gradient_squared = float(torch.dot(gradient, gradient))
        for backtracks in range(max_backtracks):
            candidate = x - step * gradient
            if torch.equal(candidate, x):
                reason = "step_underflow"
                break
            candidate_value = objective.value(candidate)
            if (math.isfinite(candidate_value)
                    and candidate_value <= value - sufficient_decrease * step * gradient_squared):
                break
            step *= shrink
        else:
            reason = "line_search_failure"
            break
        if reason == "step_underflow":
            break
        previous_value = value
        x = candidate.detach()
        value, gradient = objective.value_gradient(x)
        accepted += 1
        rows.append(_row(accepted, x, value, gradient, started, step_size=step,
                         backtracks=backtracks,
                         armijo_margin=previous_value
                         - sufficient_decrease * step * gradient_squared - value))
        if (confinement_radius is not None
                and float(torch.linalg.vector_norm(x))
                > (1 + radius_tolerance) * confinement_radius):
            reason = "radius_audit"
            break
    return SolverResult(x, reason, rows, objective.snapshot(), accepted,
                        time.perf_counter() - started)


def gradient_flow(objective, initial, *, time_limit, rtol, atol,
                  target=None, target_tolerance=None, gradient_tolerance=None,
                  center=None, radius=None, radius_tolerance=1e-10,
                  output_times=None, output_count=200):
    """Integrate a'=-grad J by DOP853 with registered stopping events."""
    require_double(initial)
    if center is not None:
        require_double(center)
    if (initial.ndim != 1 or time_limit <= 0 or rtol <= 0 or atol <= 0
            or output_count < 2 or radius_tolerance < 0
            or (target is None) != (target_tolerance is None)
            or (target_tolerance is not None and target_tolerance < 0)
            or (gradient_tolerance is not None and gradient_tolerance < 0)
            or (center is None) != (radius is None)
            or (center is not None and (center.shape != initial.shape or radius <= 0))):
        raise ValueError("Invalid gradient-flow configuration.")
    started = time.perf_counter()
    bridge = ScipyAdapter(objective, initial)
    initial = initial.detach().clone()
    center = None if center is None else center.detach().clone()
    value, gradient = objective.value_gradient(initial)
    first = _row(0, initial, value, gradient, started, time=0.0)
    if not _finite(value, gradient):
        return SolverResult(initial, "nonfinite_objective_or_gradient", [first],
                            objective.snapshot(), elapsed_seconds=time.perf_counter() - started)

    roundoff = (0.0 if center is None else 16 * torch.finfo(initial.dtype).eps * max(
        radius, float(torch.linalg.vector_norm(initial - center))))
    if center is not None and float(torch.linalg.vector_norm(initial - center)) > radius + roundoff:
        return SolverResult(initial, "radius_audit", [first], objective.snapshot(),
                            elapsed_seconds=time.perf_counter() - started)
    if target is not None and float(target(initial)) <= target_tolerance:
        return SolverResult(initial, "target_tolerance", [first], objective.snapshot(),
                            elapsed_seconds=time.perf_counter() - started)
    if (gradient_tolerance is not None
            and float(torch.linalg.vector_norm(gradient)) <= gradient_tolerance):
        return SolverResult(initial, "gradient_tolerance", [first], objective.snapshot(),
                            elapsed_seconds=time.perf_counter() - started)

    def rhs(_, state):
        current_value, current_gradient = objective.value_gradient(bridge.tensor(state))
        if not _finite(current_value, current_gradient):
            raise FloatingPointError("Nonfinite gradient-flow RHS.")
        return -current_gradient.cpu().numpy()

    events, event_names = [], []
    if target is not None:
        def target_event(_, state):
            return float(target(bridge.tensor(state))) - target_tolerance
        target_event.terminal, target_event.direction = True, -1
        events.append(target_event)
        event_names.append("target_tolerance")
    if gradient_tolerance is not None:
        def gradient_event(_, state):
            _, current_gradient = objective.value_gradient(bridge.tensor(state))
            return float(torch.linalg.vector_norm(current_gradient)) - gradient_tolerance
        gradient_event.terminal, gradient_event.direction = True, -1
        events.append(gradient_event)
        event_names.append("gradient_tolerance")
    if center is not None:
        def radius_event(_, state):
            current = bridge.tensor(state)
            return float(torch.linalg.vector_norm(current - center)) - radius - roundoff
        radius_event.terminal, radius_event.direction = True, 1
        events.append(radius_event)
        event_names.append("radius_audit")

    try:
        solution = solve_ivp(rhs, (0.0, time_limit), initial.cpu().numpy(), method="DOP853",
                             rtol=rtol, atol=atol, dense_output=True,
                             events=tuple(events) if events else None)
    except (FloatingPointError, RuntimeError, ValueError):
        return SolverResult(initial, "nonfinite_objective_or_gradient", [first],
                            objective.snapshot(), elapsed_seconds=time.perf_counter() - started)
    event_records = solution.t_events if events else ()
    triggered = [name for name, values in zip(event_names, event_records) if values.size]
    if triggered:
        reason = triggered[0]
    elif not solution.success:
        reason = "integration_failure"
    else:
        reason = "time_budget"

    final_time = float(solution.t[-1])
    if output_times is None:
        times = np.linspace(0.0, final_time, output_count)
    else:
        requested = np.asarray(output_times, dtype=float)
        if requested.ndim != 1 or requested.size == 0 or requested[0] != 0 or np.any(np.diff(requested) <= 0):
            raise ValueError("Output times must be a strictly increasing vector starting at zero.")
        times = requested[requested <= final_time]
        if times.size == 0 or times[-1] < final_time:
            times = np.append(times, final_time)
    states = solution.sol(times).T
    rows = []
    for step, (current_time, state) in enumerate(zip(times, states)):
        current = bridge.tensor(state)
        current_value, current_gradient = objective.value_gradient(current)
        rows.append(_row(step, current, current_value, current_gradient, started,
                         time=float(current_time)))
    counts = objective.snapshot()
    counts.update(ode_rhs_evaluations=solution.nfev,
                  ode_accepted_steps=len(solution.t) - 1)
    return SolverResult(bridge.tensor(states[-1]), reason, rows, counts,
                        elapsed_seconds=time.perf_counter() - started)
