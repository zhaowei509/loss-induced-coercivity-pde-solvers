"""Full-batch Armijo descent copied and narrowed from Experiment 1."""

from dataclasses import dataclass
import math
import time

import torch

from .features import require_double


@dataclass
class SolverResult:
    parameters: torch.Tensor
    stop_reason: str
    trajectory: list
    accepted_steps: int = 0
    elapsed_seconds: float = 0.0


def _row(step, x, value, gradient, started, **extra):
    return dict(step=step, parameters=x.detach().cpu().tolist(), objective=value,
                gradient_norm=float(torch.linalg.vector_norm(gradient)),
                parameter_norm=float(torch.linalg.vector_norm(x)),
                elapsed_seconds=time.perf_counter() - started, **extra)


def _finite(value, gradient):
    return math.isfinite(value) and bool(torch.isfinite(gradient).all())


def armijo(objective, initial, *, max_steps=50000, gradient_tolerance=1e-8,
           trial_step=1.0, shrink=0.5, sufficient_decrease=1e-4,
           max_backtracks=60, stop_requested=None):
    require_double(initial)
    if (max_steps < 0 or gradient_tolerance < 0 or trial_step <= 0
            or not 0 < shrink < 1 or not 0 < sufficient_decrease < 1
            or max_backtracks < 1):
        raise ValueError("Invalid Armijo configuration.")
    started = time.perf_counter()
    x = initial.detach().clone()
    value, gradient = objective.analytic_value_gradient(x)
    rows = [_row(0, x, value, gradient, started)]
    reason, accepted = "iteration_budget", 0
    while True:
        if (stop_requested is not None and accepted % 200 == 0
                and stop_requested()):
            raise InterruptedError("Campaign stop requested; completed trajectories remain saved.")
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
            candidate_value = objective.analytic_value(candidate)
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
        value, gradient = objective.analytic_value_gradient(x)
        accepted += 1
        rows.append(_row(accepted, x, value, gradient, started,
                         step_size=step, backtracks=backtracks,
                         armijo_margin=previous_value
                         - sufficient_decrease * step * gradient_squared - value))
    return SolverResult(x, reason, rows, accepted,
                        time.perf_counter() - started)
