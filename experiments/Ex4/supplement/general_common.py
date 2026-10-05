"""Shared S2B/S2C soft-boundary model, geometry, and certificate funnel."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.stats import qmc

from experiments.shared.numerics import spectrum

from .certificate_common import Jet, certificate


RADII = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
GRADIENT_TOLERANCE = 1e-12


@dataclass(frozen=True)
class SoftConfiguration:
    name: str
    width: int
    interior_count: int
    boundary_count: int
    boundary_penalty: float
    column_bias_count: int
    equation: str


def points(configuration: SoftConfiguration) -> tuple[np.ndarray, np.ndarray]:
    interior = qmc.Sobol(d=2, scramble=True, seed=1000).random_base2(
        configuration.interior_count.bit_length() - 1)
    per_edge = configuration.boundary_count // 4
    mid = (np.arange(per_edge, dtype=np.float64) + 0.5) / per_edge
    boundary = np.vstack((
        np.column_stack((mid, np.zeros(per_edge))),
        np.column_stack((np.ones(per_edge), mid)),
        np.column_stack((1 - mid, np.ones(per_edge))),
        np.column_stack((np.zeros(per_edge), 1 - mid)),
    ))
    return interior, boundary


def features(seed: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.Generator(np.random.PCG64(seed))
    return (generator.standard_normal((width, 2)),
            generator.standard_normal(width))


def basis(locations: np.ndarray, weights: np.ndarray, biases: np.ndarray
          ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    activation = np.tanh(locations @ weights.T + biases)
    first = 1 - activation**2
    second = -2 * activation * first
    third = (-2 + 6 * activation**2) * first
    return activation, first, second, third


def residual_and_blocks(coefficients: np.ndarray, weights: np.ndarray,
                        biases: np.ndarray, interior: np.ndarray,
                        boundary: np.ndarray, source: np.ndarray,
                        configuration: SoftConfiguration
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ti, d1i, d2i, d3i = basis(interior, weights, biases)
    tb, d1b, _, _ = basis(boundary, weights, biases)
    norm_squared = np.sum(weights**2, axis=1)
    u = ti @ coefficients
    interior_scale = 1 / math.sqrt(configuration.interior_count)
    boundary_scale = math.sqrt(
        2 * configuration.boundary_penalty / configuration.boundary_count)
    ri = (-((d2i * norm_squared) @ coefficients) + u**3 - source) * interior_scale
    rb = boundary_scale * (tb @ coefficients)
    residual = np.concatenate((ri, rb))
    output_i = (-d2i * norm_squared + 3 * u[:, None]**2 * ti) * interior_scale
    output_b = boundary_scale * tb
    output = np.vstack((output_i, output_b))
    common = -(d3i * norm_squared) + 3 * u[:, None]**2 * d1i
    bias_i = coefficients * common * interior_scale
    bias_b = boundary_scale * coefficients * d1b
    bias = np.vstack((bias_i, bias_b))
    weight_i = np.empty((configuration.interior_count, configuration.width, 2))
    weight_b = np.empty((configuration.boundary_count, configuration.width, 2))
    for axis in range(2):
        weight_i[:, :, axis] = coefficients * (
            common * interior[:, [axis]] - 2 * d2i * weights[:, axis]) * interior_scale
        weight_b[:, :, axis] = boundary_scale * coefficients * d1b * boundary[:, [axis]]
    weight = np.vstack((weight_i.reshape(configuration.interior_count, -1),
                        weight_b.reshape(configuration.boundary_count, -1)))
    return residual, output, bias, weight


def output_objective(coefficients: np.ndarray, weights: np.ndarray,
                     biases: np.ndarray, interior: np.ndarray,
                     boundary: np.ndarray, source: np.ndarray,
                     configuration: SoftConfiguration
                     ) -> tuple[float, np.ndarray, np.ndarray]:
    residual, output, _, _ = residual_and_blocks(
        coefficients, weights, biases, interior, boundary, source, configuration)
    ti = basis(interior, weights, biases)[0]
    u = ti @ coefficients
    ri = residual[:configuration.interior_count]
    hessian = output.T @ output
    hessian += ti.T @ (((6 * u * ri) / math.sqrt(configuration.interior_count))[:, None]
                       * ti)
    return 0.5 * float(residual @ residual), output.T @ residual, hessian


def solve_output(weights: np.ndarray, biases: np.ndarray, interior: np.ndarray,
                 boundary: np.ndarray, source: np.ndarray,
                 configuration: SoftConfiguration) -> tuple[np.ndarray, dict[str, Any]]:
    evaluate = lambda value: output_objective(
        value, weights, biases, interior, boundary, source, configuration)
    result = minimize(lambda value: evaluate(value)[0],
                      np.zeros(configuration.width), method="trust-exact",
                      jac=lambda value: evaluate(value)[1],
                      hess=lambda value: evaluate(value)[2],
                      options={"gtol": GRADIENT_TOLERANCE, "maxiter": 100})
    point = result.x.copy()
    polish = []
    for _ in range(3):
        old_value, gradient, hessian = evaluate(point)
        old_norm = float(np.linalg.norm(gradient))
        if old_norm <= GRADIENT_TOLERANCE or np.linalg.eigvalsh(hessian)[0] <= 0:
            break
        try:
            candidate = point - np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            break
        new_value, new_gradient, _ = evaluate(candidate)
        new_norm = float(np.linalg.norm(new_gradient))
        accepted = bool(np.all(np.isfinite(candidate)) and new_norm <= 0.5 * old_norm
                        and new_value <= old_value + 32 * np.finfo(float).eps
                        * max(1.0, abs(old_value)))
        polish.append({"gradient_before": old_norm, "gradient_after": new_norm,
                       "accepted": accepted})
        if not accepted:
            break
        point = candidate
    value, gradient, _ = evaluate(point)
    return point, {"objective": value, "gradient_norm": float(np.linalg.norm(gradient)),
                   "stationary": bool(np.linalg.norm(gradient) <= GRADIENT_TOLERANCE),
                   "iterations": int(result.nit), "solver_status": int(result.status),
                   "solver_message": str(result.message), "polish": polish}


def active_vector(coefficients: np.ndarray, weights: np.ndarray,
                  biases: np.ndarray, branch: str,
                  configuration: SoftConfiguration) -> np.ndarray:
    if branch == "column":
        return np.concatenate((coefficients, biases[:configuration.column_bias_count]))
    if branch == "row":
        return np.concatenate((coefficients, weights.reshape(-1), biases))
    raise ValueError(branch)


def unpack_active(theta: np.ndarray, fixed_weights: np.ndarray,
                  fixed_biases: np.ndarray, branch: str,
                  configuration: SoftConfiguration
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    width = configuration.width
    coefficients = theta[:width]
    if branch == "column":
        weights = fixed_weights
        biases = fixed_biases.copy()
        biases[:configuration.column_bias_count] = theta[width:]
        return coefficients, weights, biases
    weights = theta[width:width + 2 * width].reshape(width, 2)
    biases = theta[width + 2 * width:]
    return coefficients, weights, biases


def residual_jacobian(theta: np.ndarray, fixed_weights: np.ndarray,
                      fixed_biases: np.ndarray, branch: str,
                      interior: np.ndarray, boundary: np.ndarray,
                      source: np.ndarray, configuration: SoftConfiguration
                      ) -> tuple[np.ndarray, np.ndarray]:
    coefficients, weights, biases = unpack_active(
        theta, fixed_weights, fixed_biases, branch, configuration)
    residual, output, bias, weight = residual_and_blocks(
        coefficients, weights, biases, interior, boundary, source, configuration)
    hidden = (bias[:, :configuration.column_bias_count] if branch == "column"
              else np.column_stack((weight, bias)))
    return residual, np.column_stack((output, hidden))


def derivative_check(theta: np.ndarray, fixed_weights: np.ndarray,
                     fixed_biases: np.ndarray, branch: str,
                     interior: np.ndarray, boundary: np.ndarray,
                     source: np.ndarray, configuration: SoftConfiguration) -> dict[str, Any]:
    _, analytic = residual_jacobian(
        theta, fixed_weights, fixed_biases, branch, interior, boundary,
        source, configuration)
    numeric = np.empty_like(analytic)
    base_relative_step = 1e-6
    for index in range(theta.size):
        estimates = []
        for factor in (1.0, 0.5):
            step = base_relative_step * factor * max(1.0, abs(theta[index]))
            plus, minus = theta.copy(), theta.copy()
            plus[index] += step
            minus[index] -= step
            estimates.append((
                residual_jacobian(plus, fixed_weights, fixed_biases, branch,
                                  interior, boundary, source, configuration)[0]
                - residual_jacobian(minus, fixed_weights, fixed_biases, branch,
                                    interior, boundary, source, configuration)[0]
            ) / (2 * step))
        numeric[:, index] = estimates[1] + (estimates[1] - estimates[0]) / 3
    difference = analytic - numeric
    relative = float(np.linalg.norm(difference) / max(1.0, np.linalg.norm(numeric)))
    return {
        "maximum_absolute_error": float(np.max(np.abs(difference))),
        "relative_frobenius_error": relative,
        "tolerance": 1e-7,
        "pass": relative <= 1e-7,
        "scheme": "two-level centered difference with Richardson extrapolation",
        "base_relative_step": base_relative_step,
        "analytic_jacobian_bytes": int(analytic.nbytes),
        "numeric_jacobian_bytes": int(numeric.nbytes),
    }


def interval_builder(coefficients: np.ndarray, weights: np.ndarray,
                     biases: np.ndarray, branch: str, interior: np.ndarray,
                     boundary: np.ndarray, source: np.ndarray,
                     configuration: SoftConfiguration):
    def build(active: list[Jet]) -> list[Jet]:
        dimension = len(active)
        width = configuration.width
        a = active[:width]
        if branch == "column":
            w = [[Jet.constant(weights[j, axis], dimension) for axis in range(2)]
                 for j in range(width)]
            b = [active[width + j] if j < configuration.column_bias_count
                 else Jet.constant(biases[j], dimension) for j in range(width)]
        else:
            offset = width
            w = [[active[offset + 2 * j + axis] for axis in range(2)]
                 for j in range(width)]
            offset += 2 * width
            b = active[offset:offset + width]

        def field(point):
            u = Jet.constant(0.0, dimension)
            laplacian = Jet.constant(0.0, dimension)
            for j in range(width):
                q = point[0] * w[j][0] + point[1] * w[j][1] + b[j]
                activation = q.tanh()
                norm_squared = w[j][0].square() + w[j][1].square()
                second = -2 * (activation - activation.cube())
                u = u + a[j] * activation
                laplacian = laplacian + a[j] * second * norm_squared
            return u, laplacian

        rows = []
        for index, point in enumerate(interior):
            u, laplacian = field(point)
            rows.append((-laplacian + u.cube() - source[index])
                        / math.sqrt(configuration.interior_count))
        boundary_scale = math.sqrt(
            2 * configuration.boundary_penalty / configuration.boundary_count)
        for point in boundary:
            u, _ = field(point)
            rows.append(boundary_scale * u)
        return rows

    return build


def geometry(coefficients: np.ndarray, weights: np.ndarray, biases: np.ndarray,
             interior: np.ndarray, boundary: np.ndarray, source: np.ndarray,
             configuration: SoftConfiguration) -> dict[str, Any]:
    residual, output, bias, weight = residual_and_blocks(
        coefficients, weights, biases, interior, boundary, source, configuration)
    output_spectrum = spectrum(output)
    q = None
    if output_spectrum.full_column_rank:
        q, _ = np.linalg.qr(output, mode="reduced")
    branches = {}
    for branch in ("column", "row"):
        hidden = (bias[:, :configuration.column_bias_count] if branch == "column"
                  else np.column_stack((weight, bias)))
        joint = np.column_stack((output, hidden))
        hidden_gradient = hidden.T @ residual
        joint_spectrum = spectrum(joint)
        transverse = None if q is None else hidden - q @ (q.T @ hidden)
        branches[branch] = {
            "D": joint_spectrum.as_dict(),
            "C_perp": None if transverse is None else spectrum(transverse).as_dict(),
            "hidden_gradient_norm": float(np.linalg.norm(hidden_gradient)),
            "hidden_directional_derivative": -float(hidden_gradient @ hidden_gradient),
            "output_gradient_norm": float(np.linalg.norm(output.T @ residual)),
        }
        if branch == "row":
            lower_squared = (joint_spectrum.sigma_min**2 * (residual @ residual)
                             - np.linalg.norm(output.T @ residual)**2)
            branches[branch]["descent_lower_bound"] = math.sqrt(max(0.0, lower_squared))
            branches[branch]["strict_descent_margin"] = (
                joint_spectrum.sigma_min * np.linalg.norm(residual)
                - np.linalg.norm(output.T @ residual))
    return {"residual_norm": float(np.linalg.norm(residual)),
            "output": output_spectrum.as_dict(), "branches": branches}


def gradient_flow(start: np.ndarray, fixed_weights: np.ndarray,
                  fixed_biases: np.ndarray, branch: str, radius: float,
                  interior: np.ndarray, boundary: np.ndarray, source: np.ndarray,
                  configuration: SoftConfiguration) -> dict[str, Any]:
    def rhs(_, theta):
        residual, jacobian = residual_jacobian(
            theta, fixed_weights, fixed_biases, branch, interior, boundary,
            source, configuration)
        return -(jacobian.T @ residual)

    def exit_event(_, theta):
        return radius - np.linalg.norm(theta - start)

    exit_event.terminal = True
    exit_event.direction = -1
    started = time.perf_counter()
    solution = solve_ivp(rhs, (0.0, 2000.0), start, method="Radau",
                         rtol=1e-9, atol=1e-11, events=exit_event,
                         dense_output=False)
    endpoint = solution.y[:, -1]
    residual, jacobian = residual_jacobian(
        endpoint, fixed_weights, fixed_biases, branch, interior, boundary,
        source, configuration)
    gradient_norm = float(np.linalg.norm(jacobian.T @ residual))
    return {"status": ("trajectory_exit" if solution.t_events[0].size
                       else "gradient_tolerance" if gradient_norm <= 1e-10
                       else "time_budget"),
            "solver_success": bool(solution.success), "solver_message": solution.message,
            "function_evaluations": int(solution.nfev), "t_end": float(solution.t[-1]),
            "endpoint": endpoint.tolist(), "endpoint_gradient_norm": gradient_norm,
            "endpoint_displacement": float(np.linalg.norm(endpoint - start)),
            "raw_solver": {"time": solution.t.tolist(),
                           "state": solution.y.T.tolist()},
            "elapsed_seconds": time.perf_counter() - started}


def run_seed(seed: int, configuration: SoftConfiguration,
             source_function, *, certify: bool = True,
             subdivision_depth: int = 0) -> dict[str, Any]:
    interior, boundary = points(configuration)
    source = np.asarray(source_function(interior), dtype=np.float64)
    weights, biases = features(seed, configuration.width)
    coefficients, optimization = solve_output(
        weights, biases, interior, boundary, source, configuration)
    diagnostic = geometry(coefficients, weights, biases, interior, boundary,
                          source, configuration)
    checks = {}
    for branch in ("column", "row"):
        start = active_vector(coefficients, weights, biases, branch, configuration)
        checks[branch] = derivative_check(
            start, weights, biases, branch, interior, boundary, source,
            configuration)
    branches = {}
    for branch in ("column", "row"):
        block = diagnostic["branches"][branch]
        d = block["D"]
        transverse = block["C_perp"]
        if branch == "column":
            f2 = (diagnostic["output"]["full_column_rank"]
                  and d["full_column_rank"]
                  and transverse is not None
                  and transverse["full_column_rank"])
        else:
            expected_transverse_rank = (
                configuration.interior_count + configuration.boundary_count
                - configuration.width)
            f2 = (diagnostic["output"]["full_column_rank"]
                  and d["full_row_rank"] and transverse is not None
                  and transverse["numerical_rank"] == expected_transverse_rank)
        f3 = (block["strict_descent_margin"] > 0 if branch == "row"
              else block["hidden_gradient_norm"] > 0)
        radius_rows = []
        start = active_vector(coefficients, weights, biases, branch, configuration)
        builder = interval_builder(coefficients, weights, biases, branch,
                                   interior, boundary, source, configuration)
        for radius in RADII:
            if branch == "row":
                necessary_margin = d["sigma_min"] * radius - diagnostic["residual_norm"]
            else:
                necessary_margin = (d["sigma_min"]**2 * radius / 2
                                    - math.sqrt(block["output_gradient_norm"]**2
                                                + block["hidden_gradient_norm"]**2))
            row = {"radius": radius, "necessary_margin_float64": necessary_margin,
                   "F4_pass": necessary_margin > 0, "certificate": None,
                   "F5_pass": False}
            if certify and optimization["stationary"] and f2 and necessary_margin > 0:
                validated = certificate(builder, start, radius, branch,
                                        subdivision_depth=subdivision_depth)
                row["certificate"] = validated
                row["F5_pass"] = validated["status"] == "certified"
            radius_rows.append(row)
        certified_radii = [row["radius"] for row in radius_rows if row["F5_pass"]]
        dynamics = None
        if certified_radii:
            chosen = min(certified_radii)
            dynamics = gradient_flow(start, weights, biases, branch, chosen,
                                     interior, boundary, source, configuration)
        branches[branch] = {
            **block, "F0_pass": True, "F1_pass": optimization["stationary"],
            "F2_pass": f2, "F3_pass": f3, "radii": radius_rows,
            "first_failure": ("F1" if not optimization["stationary"] else
                              "F2" if not f2 else "F3" if not f3 else
                              "F4" if not any(row["F4_pass"] for row in radius_rows) else
                              "F5" if not certified_radii else
                              "F6" if dynamics and dynamics["status"] not in
                              {"gradient_tolerance", "time_budget"} else None),
            "dynamics": dynamics,
        }
    return {"schema_version": 1, "seed": seed,
            "configuration": configuration.__dict__,
            "features": {"weights": weights, "biases": biases},
            "points": {"interior": interior, "boundary": boundary},
            "source": source, "coefficients": coefficients,
            "optimization": optimization, "geometry": diagnostic,
            "derivative_checks": checks,
            "working_set_estimate_bytes": int(
                (configuration.interior_count + configuration.boundary_count)
                * (configuration.width + 3 * configuration.width) * 8),
            "branches": branches}
