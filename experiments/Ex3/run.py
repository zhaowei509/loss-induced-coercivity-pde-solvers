"""Run the predetermined Experiment 3 campaign; importing has no side effects."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys

import numpy as np
import scipy
from scipy.linalg import solve_triangular
from scipy.optimize import minimize

from .core import (B_1D, DELTAS_1D, EPSILON_2D, RADII,
                   SEEDS_2D, STAR_1D, W_1D, boundary_2d,
                   one_dimensional, quadrature, random_features, two_dimensional,
                   two_dimensional_preconditioner, two_dimensional_raw)


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "EXPERIMENT3_PLAN.md"
MAX_POLISH_STEPS = 3
POLISH_GRADIENT_FACTOR = 0.5
POLISH_OBJECTIVE_ULPS = 32


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path, data):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def _newton_polish(objective, start, tolerance):
    """Apply the same gradient-checked Newton correction to every eligible case."""
    point = np.asarray(start, dtype=np.float64).copy()
    gradient_norm = float(np.linalg.norm(objective.jac(point)))
    attempts = []
    if gradient_norm <= tolerance:
        return point, dict(stop_reason="not_needed", attempts=attempts)
    stop_reason = "step_limit"
    for _ in range(MAX_POLISH_STEPS):
        hessian = objective.hess(point)
        lambda_min = float(np.linalg.eigvalsh(hessian)[0])
        if not np.isfinite(lambda_min) or lambda_min <= 0:
            stop_reason = "hessian_not_positive_definite"
            break
        try:
            correction = np.linalg.solve(hessian, objective.jac(point))
        except np.linalg.LinAlgError:
            stop_reason = "linear_solve_failed"
            break
        candidate = point - correction
        old_value = objective.fun(point)
        new_value = objective.fun(candidate)
        new_gradient_norm = float(np.linalg.norm(objective.jac(candidate)))
        objective_allowance = (POLISH_OBJECTIVE_ULPS * np.finfo(np.float64).eps
                               * max(1.0, abs(old_value)))
        accepted = bool(np.all(np.isfinite(candidate))
                        and np.isfinite(new_value)
                        and np.isfinite(new_gradient_norm)
                        and new_gradient_norm <= POLISH_GRADIENT_FACTOR * gradient_norm
                        and new_value <= old_value + objective_allowance)
        attempts.append(dict(gradient_before=gradient_norm,
                             gradient_after=new_gradient_norm,
                             objective_before=old_value,
                             objective_after=new_value,
                             step_norm=float(np.linalg.norm(correction)),
                             lambda_min_hessian=lambda_min,
                             accepted=accepted))
        if not accepted:
            stop_reason = "acceptance_failed"
            break
        point, gradient_norm = candidate, new_gradient_norm
        if gradient_norm <= tolerance:
            stop_reason = "gradient_tolerance_reached"
            break
    return point, dict(stop_reason=stop_reason, attempts=attempts)


def _solve(objective, start, max_iterations, tolerance):
    result = minimize(objective.fun, np.asarray(start, dtype=np.float64),
                      method="trust-exact", jac=objective.jac,
                      hess=objective.hess,
                      options=dict(gtol=tolerance, maxiter=max_iterations))
    point, polish = _newton_polish(objective, result.x, tolerance)
    final = objective.diagnostic(point)
    return dict(parameters=point.tolist(), final=final, polish=polish,
                reached_gradient_tolerance=bool(final["gradient_norm"] <= tolerance),
                solver=dict(success=bool(result.success), status=int(result.status),
                            message=str(result.message), iterations=int(result.nit),
                            function_evaluations=int(result.nfev),
                            gradient_evaluations=int(result.njev),
                            hessian_evaluations=int(result.nhev)))


def _refinement(row, refined):
    row["refined"] = refined.diagnostic(row["parameters"])
    row["quadrature_delta_curv_difference"] = (
        row["refined"]["delta_curv"] - row["final"]["delta_curv"])
    row["quadrature_sign_stable"] = bool(
        (row["final"]["delta_curv"] > 0) == (row["refined"]["delta_curv"] > 0))
    return row


def _numerical_conditions(row, tolerance, *, require_nonzero_residual=False):
    """Report finite-precision diagnostics, not a proof at an exact critical point."""
    final = row["final"]
    scale = max(1.0, abs(final["mu_GN"]), abs(final["mu_B"]),
                abs(final["delta_curv"]), abs(final["lambda_min_hessian"]))
    slack = 1e-10 * scale
    checks = dict(
        stationary=bool(final["gradient_norm"] <= tolerance),
        boundary_full_column_rank=bool(final["boundary_full_column_rank"]),
        gauss_newton_above_boundary_floor=bool(
            final["mu_GN"] >= final["mu_B"] - slack),
        hessian_above_curvature_gap=bool(
            final["lambda_min_hessian"] >= final["delta_curv"] - slack),
        positive_curvature_gap=bool(final["delta_curv"] > 0),
        refinement_sign_stable=bool(row["quadrature_sign_stable"]),
    )
    if require_nonzero_residual:
        checks["nonzero_residual"] = bool(final["residual_norm"] > 1e-10)
    row["numerical_conditions"] = checks
    row["numerical_conditions_met"] = bool(all(checks.values()))
    return row


def _one_dimensional(output):
    train_points, train_weights = quadrature(256, 1)
    refined_points, refined_weights = quadrature(512, 1)
    np.savez_compressed(output / "one_dimensional_inputs.npz",
                        w=W_1D, b=B_1D,
                        boundary_points=np.array([[0.0], [1.0]]),
                        train_points=train_points, train_weights=train_weights,
                        refined_points=refined_points,
                        refined_weights=refined_weights)
    # The zero-residual inequality uses all 32 predetermined points.
    zero = one_dimensional(0.0, 256)
    zero_refined = one_dimensional(0.0, 512)
    angles = 2 * np.pi * np.arange(8) / 8
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    points = np.vstack([STAR_1D + radius * directions for radius in RADII])
    errors = np.linalg.norm(points - STAR_1D, axis=1)
    mu_b = zero.diagnostic(STAR_1D)["mu_B"]
    bounds = np.array([np.sqrt(2 * zero.fun(a) / mu_b) for a in points])
    refined_bounds = np.array([np.sqrt(2 * zero_refined.fun(a) / mu_b) for a in points])
    np.savez_compressed(output / "one_dimensional_bound.npz", points=points,
                        errors=errors, bounds=bounds, refined_bounds=refined_bounds,
                        radii=np.asarray(RADII), directions=directions)
    _json(output / "one_dimensional_bound.json",
          dict(exact=STAR_1D.tolist(), mu_B=mu_b,
               exact_point=zero.diagnostic(STAR_1D),
               max_error_to_bound=float(np.max(errors / bounds)),
               max_error_to_refined_bound=float(np.max(errors / refined_bounds)),
               tested_points=len(points)))
    for delta in DELTAS_1D:
        coarse = one_dimensional(delta, 256)
        refined = one_dimensional(delta, 512)
        if delta == 0:
            row = dict(parameters=STAR_1D.tolist(), final=coarse.diagnostic(STAR_1D),
                       reached_gradient_tolerance=True,
                       solver=dict(message="analytic zero-residual point", iterations=0))
        else:
            row = _solve(coarse, STAR_1D, 100, 1e-10)
        row.update(branch="one_dimensional", delta=delta, train_order=256,
                   refinement_order=512)
        row = _refinement(row, refined)
        row = _numerical_conditions(row, 1e-10,
                                    require_nonzero_residual=delta > 0)
        _json(output / f"one_dimensional_delta_{delta:g}.json", row)


def _two_dimensional(output):
    for seed in SEEDS_2D:
        transform, rank_checks = two_dimensional_preconditioner(seed)
        w, b = random_features(seed)
        train_points, train_weights = quadrature(24, 2)
        refined_points, refined_weights = quadrature(48, 2)
        train_h = np.sin(np.pi * train_points[:, 0]) * np.sin(np.pi * train_points[:, 1])
        refined_h = np.sin(np.pi * refined_points[:, 0]) * np.sin(np.pi * refined_points[:, 1])
        np.savez_compressed(output / f"two_dimensional_seed_{seed}_inputs.npz",
                            w=w, b=b, transform=(transform if transform is not None
                                                else np.empty((0, 0))),
                            boundary_points=boundary_2d(),
                            train_points=train_points, train_weights=train_weights,
                            train_source=2 * np.pi**2 * train_h + EPSILON_2D * train_h**2,
                            refined_points=refined_points,
                            refined_weights=refined_weights,
                            refined_source=2 * np.pi**2 * refined_h
                                           + EPSILON_2D * refined_h**2)
        if transform is None:
            row = dict(branch="two_dimensional", seed=seed, epsilon=EPSILON_2D,
                       status="numerical_rank_failed", rank_checks=rank_checks,
                       reached_gradient_tolerance=False)
        else:
            coarse = two_dimensional(seed, 24, transform)
            refined = two_dimensional(seed, 48, transform)
            row = _solve(coarse, np.zeros(6), 200, 1e-9)
            row.update(branch="two_dimensional", seed=seed, epsilon=EPSILON_2D,
                       rank_checks=rank_checks, train_order=24,
                       refinement_order=48)
            row = _refinement(row, refined)
            row = _numerical_conditions(row, 1e-9,
                                        require_nonzero_residual=True)
            c = np.asarray(row["parameters"], dtype=np.float64)
            a = solve_triangular(transform, c, lower=False)
            raw = two_dimensional_raw(seed, 24)
            raw_refined = two_dimensional_raw(seed, 48)
            raw_residual = raw.residual(a)
            normalized_residual = coarse.residual(c)
            raw_gradient = raw.jac(a)
            normalized_gradient = coarse.jac(c)
            raw_hessian = raw.hess(a)
            normalized_hessian = coarse.hess(c)
            residual_disagreement = float(np.max(np.abs(raw_residual - normalized_residual)))
            objective_disagreement = abs(raw.fun(a) - coarse.fun(c))
            gradient_disagreement = float(np.max(np.abs(
                solve_triangular(transform.T, raw_gradient, lower=True)
                - normalized_gradient)))
            hessian_disagreement = float(np.max(np.abs(
                transform.T @ normalized_hessian @ transform - raw_hessian)))
            if (residual_disagreement > 1e-10 or objective_disagreement > 1e-10
                    or gradient_disagreement > 1e-10
                    or hessian_disagreement > 1e-9):
                raise AssertionError("Raw/normalized coordinate identity failed.")
            raw_final = raw.diagnostic(a)
            raw_refined_diagnostic = raw_refined.diagnostic(a)
            row["raw_coordinates"] = dict(
                parameters=a.tolist(), final=raw_final,
                refined=raw_refined_diagnostic,
                quadrature_sign_stable=bool(
                    (raw_final["delta_curv"] > 0)
                    == (raw_refined_diagnostic["delta_curv"] > 0)),
                residual_disagreement=residual_disagreement,
                objective_disagreement=objective_disagreement,
                gradient_transform_disagreement=gradient_disagreement,
                hessian_transform_disagreement=hessian_disagreement)
            _numerical_conditions(row["raw_coordinates"], 1e-9,
                                  require_nonzero_residual=True)
        _json(output / f"two_dimensional_seed_{seed}.json", row)


def campaign(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(PLAN, output / "EXPERIMENT3_PLAN_frozen.md")
    source = Path(__file__).resolve().parent
    frozen_source = output / "source_frozen"
    frozen_source.mkdir()
    for path in sorted(source.glob("*.py")):
        shutil.copy2(path, frozen_source / path.name)
    manifest = dict(mode="formal", created_utc=datetime.now(timezone.utc).isoformat(),
                    solver_protocol="trust-exact with uniform Newton polishing",
                    max_polish_steps=MAX_POLISH_STEPS,
                    polish_gradient_factor=POLISH_GRADIENT_FACTOR,
                    polish_objective_ulps=POLISH_OBJECTIVE_ULPS,
                    plan_sha256=_hash(PLAN),
                    source_sha256={p.name: _hash(p) for p in sorted(source.glob("*.py"))},
                    environment=dict(python=platform.python_version(),
                                     numpy=np.__version__, scipy=scipy.__version__,
                                     platform=platform.platform()),
                    expected_1d_deltas=list(DELTAS_1D),
                    expected_2d_seeds=list(SEEDS_2D),
                    expected_2d_epsilon=EPSILON_2D)
    _json(output / "manifest.json", manifest)
    _one_dimensional(output)
    _two_dimensional(output)
    _json(output / "COMPLETE.json", dict(completed_utc=datetime.now(timezone.utc).isoformat(),
                                          mode="formal", status="complete"))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check")
    run = sub.add_parser("run")
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "check":
        from .checks import run_checks
        print(json.dumps(run_checks(), indent=2))
    else:
        print(campaign(args.output))


if __name__ == "__main__":
    main(sys.argv[1:])
