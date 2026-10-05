"""Run the controlled main case of Experiment 4; no work occurs on import."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import torch

from .core import (
    A_REFERENCE,
    BOUNDARY,
    DELTA,
    INTERIOR,
    RHO,
    ControlledProblem,
    rank_diagnostic,
)
from .plot import make_figure


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "EXPERIMENT4_PLAN.md"
GRAD_TOL = 1e-12
COL_STOP = 1e-10
ROW_STOP = 1e-9
FORMAL_T_MAX = 2000.0


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def optimize_output(problem):
    start = A_REFERENCE.copy()
    result = minimize(
        lambda z: problem.value(z, "output"),
        start,
        method="trust-exact",
        jac=lambda z: problem.gradient(z, "output"),
        hess=lambda z: problem.hessian(z, "output"),
        options={"gtol": GRAD_TOL, "maxiter": 100},
    )
    point = result.x.copy()
    polish = []
    for _ in range(3):
        gradient = problem.gradient(point, "output")
        old_norm = float(np.linalg.norm(gradient))
        if old_norm <= GRAD_TOL:
            break
        hessian = problem.hessian(point, "output")
        if np.linalg.eigvalsh(hessian)[0] <= 0:
            break
        try:
            candidate = point - np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            break
        new_norm = float(np.linalg.norm(problem.gradient(candidate, "output")))
        old_value = problem.value(point, "output")
        new_value = problem.value(candidate, "output")
        accepted = bool(
            np.all(np.isfinite(candidate))
            and new_norm <= 0.5 * old_norm
            and new_value <= old_value + 32 * np.finfo(float).eps * max(1.0, abs(old_value))
        )
        polish.append({
            "gradient_before": old_norm,
            "gradient_after": new_norm,
            "accepted": accepted,
        })
        if not accepted:
            break
        point = candidate
    gradient_norm = float(np.linalg.norm(problem.gradient(point, "output")))
    return point, {
        "a": point.tolist(),
        "objective": problem.value(point, "output"),
        "residual_norm": float(np.linalg.norm(problem.residual(point, "output"))),
        "gradient_norm": gradient_norm,
        "stationary": bool(gradient_norm <= GRAD_TOL),
        "solver": {
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "iterations": int(result.nit),
        },
        "polish": polish,
    }


def geometry(problem, a):
    output_start = problem.starting_vector(a, "output")
    column_start = problem.starting_vector(a, "column")
    row_start = problem.starting_vector(a, "row")
    r = problem.residual(output_start, "output")
    a_block = problem.jacobian(output_start, "output")
    q, _ = np.linalg.qr(a_block, mode="reduced")
    branches = {}
    for branch, start in (("column", column_start), ("row", row_start)):
        joint = problem.jacobian(start, branch)
        c = joint[:, 2:]
        transverse = c - q @ (q.T @ c)
        transverse_spectrum = rank_diagnostic(transverse)
        joint_spectrum = rank_diagnostic(joint)
        branches[branch] = {
            "active_start": start.tolist(),
            "hidden_parameter_count": int(c.shape[1]),
            "transverse": transverse_spectrum,
            "joint": joint_spectrum,
            "rank_identity_holds": bool(
                joint_spectrum["rank"]
                == rank_diagnostic(a_block)["rank"] + transverse_spectrum["rank"]
            ),
        }
    row_joint = problem.jacobian(row_start, "row")
    e = a_block.T @ r
    h = row_joint[:, 2:].T @ r
    delta_row = branches["row"]["joint"]["sigma_min"]
    lower_bound = np.sqrt(
        max(0.0, delta_row**2 * np.dot(r, r) - np.dot(e, e))
    )
    branches["row"]["descent"] = {
        "output_gradient_norm": float(np.linalg.norm(e)),
        "hidden_gradient_norm": float(np.linalg.norm(h)),
        "lower_bound": float(lower_bound),
        "bound_holds": bool(np.linalg.norm(h) + 1e-12 >= lower_bound),
        "strict_margin": float(delta_row * np.linalg.norm(r) - np.linalg.norm(e)),
        "directional_derivative": float(-np.dot(h, h)),
    }
    return {
        "boundary_features": rank_diagnostic(problem.boundary_matrix()),
        "output_jacobian": rank_diagnostic(a_block),
        "initial_residual_norm": float(np.linalg.norm(r)),
        "column": branches["column"],
        "row": branches["row"],
    }


def derivative_check(problem, vector, branch):
    analytic = problem.jacobian(vector, branch)
    numeric = np.empty_like(analytic)
    for k in range(vector.size):
        step = 1e-6 * max(1.0, abs(vector[k]))
        plus, minus = vector.copy(), vector.copy()
        plus[k] += step
        minus[k] -= step
        numeric[:, k] = (
            problem.residual(plus, branch) - problem.residual(minus, branch)
        ) / (2 * step)
    grad = problem.gradient(vector, branch)
    grad_from_jacobian = analytic.T @ problem.residual(vector, branch)
    hessian = problem.hessian(vector, branch)
    numeric_hessian = np.empty_like(hessian)
    for k in range(vector.size):
        step = 1e-6 * max(1.0, abs(vector[k]))
        plus, minus = vector.copy(), vector.copy()
        plus[k] += step
        minus[k] -= step
        numeric_hessian[:, k] = (
            problem.gradient(plus, branch) - problem.gradient(minus, branch)
        ) / (2 * step)
    return {
        "jacobian_relative_error": float(
            np.linalg.norm(analytic - numeric) / max(1.0, np.linalg.norm(numeric))
        ),
        "gradient_identity_error": float(np.linalg.norm(grad - grad_from_jacobian)),
        "hessian_relative_error": float(
            np.linalg.norm(hessian - numeric_hessian)
            / max(1.0, np.linalg.norm(numeric_hessian))
        ),
    }


def local_diagnostic(problem, vector, branch, start):
    residual = problem.residual(vector, branch)
    gradient = problem.gradient(vector, branch)
    jacobian = problem.jacobian(vector, branch)
    row = {
        "objective": 0.5 * float(np.dot(residual, residual)),
        "residual_norm": float(np.linalg.norm(residual)),
        "gradient_norm": float(np.linalg.norm(gradient)),
        "displacement": float(np.linalg.norm(vector - start)),
        "jacobian_sigma_min": rank_diagnostic(jacobian)["sigma_min"],
    }
    if branch == "column":
        row["hessian_lambda_min"] = float(
            np.linalg.eigvalsh(problem.hessian(vector, branch))[0]
        )
    return row


def sample_trajectory(problem, solution, branch, start, count):
    end = float(solution.t[-1])
    if end <= 0:
        times = np.array([0.0])
    else:
        linear = np.linspace(0, end, count)
        logarithmic = np.geomspace(max(end * 1e-6, 1e-12), end, count)
        times = np.unique(np.r_[linear, logarithmic])
    states = solution.sol(times).T
    rows = []
    for t, vector in zip(times, states):
        rows.append({
            "t": float(t),
            "state": vector.tolist(),
            **local_diagnostic(problem, vector, branch, start),
        })
    return rows


def refine_column_reference(problem, endpoint, start):
    result = minimize(
        lambda z: problem.value(z, "column"),
        endpoint,
        method="trust-exact",
        jac=lambda z: problem.gradient(z, "column"),
        hess=lambda z: problem.hessian(z, "column"),
        options={"gtol": GRAD_TOL, "maxiter": 100},
    )
    point = result.x.copy()
    for _ in range(3):
        gradient = problem.gradient(point, "column")
        if np.linalg.norm(gradient) <= GRAD_TOL:
            break
        hessian = problem.hessian(point, "column")
        if np.linalg.eigvalsh(hessian)[0] <= 0:
            break
        correction = np.linalg.solve(hessian, gradient)
        candidate = point - correction
        if (np.linalg.norm(problem.gradient(candidate, "column"))
                > 0.5 * np.linalg.norm(gradient)):
            break
        if problem.value(candidate, "column") > problem.value(point, "column") + 1e-14:
            break
        point = candidate
    gradient_norm = float(np.linalg.norm(problem.gradient(point, "column")))
    displacement = float(np.linalg.norm(point - start))
    return {
        "theta_star": point.tolist(),
        "gradient_norm": gradient_norm,
        "displacement_from_start": displacement,
        "hessian_lambda_min": float(np.linalg.eigvalsh(
            problem.hessian(point, "column")
        )[0]),
        "usable_for_distance_plot": bool(
            gradient_norm <= GRAD_TOL and displacement < RHO
        ),
        "solver_status": int(result.status),
        "solver_message": str(result.message),
    }


def integrate_branch(problem, branch, start, t_max, mode):
    target = COL_STOP if branch == "column" else ROW_STOP

    def target_event(t, z):
        if branch == "column":
            return np.linalg.norm(problem.gradient(z, branch)) - target
        return np.linalg.norm(problem.residual(z, branch)) - target

    def ball_event(t, z):
        return RHO - np.linalg.norm(z - start)

    target_event.terminal = True
    target_event.direction = -1
    ball_event.terminal = True
    ball_event.direction = -1

    initial_target = target_event(0.0, start)
    if initial_target <= 0:
        return {
            "stop_reason": "initial_target_met",
            "solver_success": True,
            "solver_message": "Initial state already meets the stopping threshold.",
            "raw_solver": {"t": [0.0], "state": [start.tolist()]},
            "sampled": [{
                "t": 0.0,
                "state": start.tolist(),
                **local_diagnostic(problem, start, branch, start),
            }],
            "endpoint": start.tolist(),
            "endpoint_recheck": None,
        }
    solution = solve_ivp(
        lambda t, z: -problem.gradient(z, branch),
        (0.0, t_max),
        start,
        method="Radau",
        jac=lambda t, z: -problem.hessian(z, branch),
        events=(target_event, ball_event),
        rtol=1e-9,
        atol=1e-11,
        max_step=10.0,
        dense_output=True,
    )
    reason = (
        "target_met" if solution.t_events[0].size else
        "left_ball" if solution.t_events[1].size else
        "time_limit" if solution.success else "solver_failure"
    )
    count = 21 if mode == "smoke" else 81
    sampled = sample_trajectory(problem, solution, branch, start, count)
    endpoint = solution.y[:, -1]
    recheck = None
    if solution.success and solution.t[-1] > 0:
        stricter = solve_ivp(
            lambda t, z: -problem.gradient(z, branch),
            (0.0, float(solution.t[-1])),
            start,
            method="Radau",
            jac=lambda t, z: -problem.hessian(z, branch),
            rtol=1e-10,
            atol=1e-12,
            max_step=10.0,
        )
        recheck = {
            "solver_success": bool(stricter.success),
            "endpoint_state_difference": float(
                np.linalg.norm(stricter.y[:, -1] - endpoint)
            ),
            "endpoint_residual_norm": float(np.linalg.norm(
                problem.residual(stricter.y[:, -1], branch)
            )),
            "endpoint_gradient_norm": float(np.linalg.norm(
                problem.gradient(stricter.y[:, -1], branch)
            )),
        }
    result = {
        "stop_reason": reason,
        "solver_success": bool(solution.success),
        "solver_message": str(solution.message),
        "function_evaluations": int(solution.nfev),
        "jacobian_evaluations": int(solution.njev),
        "t_end": float(solution.t[-1]),
        "raw_solver": {
            "t": solution.t.tolist(),
            "state": solution.y.T.tolist(),
        },
        "sampled": sampled,
        "endpoint": endpoint.tolist(),
        "endpoint_recheck": recheck,
    }
    if branch == "column":
        reference = refine_column_reference(problem, endpoint, start)
        result["reference"] = reference
        if reference["usable_for_distance_plot"]:
            star = np.array(reference["theta_star"])
            for item in sampled:
                item["distance_to_reference"] = float(
                    np.linalg.norm(np.array(item["state"]) - star)
                )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--t-max", type=float, default=0.2,
                        help="Smoke-only integration horizon.")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.t_max <= 0:
        raise ValueError("--t-max must be positive")
    t_max = FORMAL_T_MAX if args.mode == "formal" else args.t_max
    began = time.perf_counter()
    problem = ControlledProblem()
    a, warm_start = optimize_output(problem)
    report = {
        "status": "initializing",
        "mode": args.mode,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": digest(PLAN),
        "source_sha256": {
            name: digest(Path(__file__).with_name(name))
            for name in ("core.py", "run.py", "plot.py")
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
        },
        "configuration": {
            "feature_seed": 9,
            "w": problem.w.tolist(),
            "b": problem.b.tolist(),
            "a_reference": A_REFERENCE.tolist(),
            "interior": INTERIOR.tolist(),
            "boundary": BOUNDARY.tolist(),
            "f_delta": problem.f.numpy().tolist(),
            "g": problem.g.numpy().tolist(),
            "delta": DELTA,
            "rho": RHO,
            "t_max": t_max,
            "rtol": 1e-9,
            "atol": 1e-11,
            "max_step": 10.0,
        },
        "warm_start": warm_start,
    }
    args.output.mkdir(parents=True)
    if not warm_start["stationary"]:
        report["status"] = "warm_start_not_stationary"
        write_json(args.output / "results.json", report)
        print(json.dumps({"status": report["status"], "output": str(args.output)}))
        return
    geometry_report = geometry(problem, a)
    report["geometry"] = geometry_report
    report["derivative_checks"] = {
        branch: derivative_check(
            problem,
            problem.starting_vector(a, branch),
            branch,
        ) for branch in ("output", "column", "row")
    }
    valid = (
        geometry_report["boundary_features"]["rank"] == 2
        and geometry_report["output_jacobian"]["rank"] == 2
        and geometry_report["column"]["transverse"]["rank"] == 1
        and geometry_report["column"]["joint"]["rank"] == 3
        and geometry_report["row"]["transverse"]["rank"] == 4
        and geometry_report["row"]["joint"]["rank"] == 6
        and geometry_report["column"]["rank_identity_holds"]
        and geometry_report["row"]["rank_identity_holds"]
        and geometry_report["row"]["descent"]["strict_margin"] > 0
        and geometry_report["row"]["descent"]["bound_holds"]
        and all(
            value["jacobian_relative_error"] < 1e-7
            and value["hessian_relative_error"] < 1e-7
            for value in report["derivative_checks"].values()
        )
    )
    report["preconditions_passed"] = bool(valid)
    if not valid:
        report["status"] = "geometry_or_derivative_check_failed"
        write_json(args.output / "results.json", report)
        print(json.dumps({"status": report["status"], "output": str(args.output)}))
        return
    report["branches"] = {}
    report["status"] = "geometry_checked"
    write_json(args.output / "results.json", report)
    for branch in ("column", "row"):
        start = problem.starting_vector(a, branch)
        report["branches"][branch] = integrate_branch(
            problem, branch, start, t_max, args.mode
        )
        report["status"] = branch + "_completed"
        report["elapsed_seconds"] = float(time.perf_counter() - began)
        write_json(args.output / "results.json", report)
        print(json.dumps({
            "checkpoint": report["status"],
            "stop_reason": report["branches"][branch]["stop_reason"],
            "t_end": report["branches"][branch].get("t_end", 0.0),
            "elapsed_seconds": report["elapsed_seconds"],
        }), flush=True)
    report["status"] = "completed"
    report["elapsed_seconds"] = float(time.perf_counter() - began)
    try:
        make_figure(report, args.output / "figure4_local_unfreezing.pdf")
    except Exception as exc:
        report["status"] = "plot_failed"
        report["plot_error"] = repr(exc)
        write_json(args.output / "results.json", report)
        raise
    write_json(args.output / "results.json", report)
    print(json.dumps({
        "status": report["status"],
        "mode": args.mode,
        "output": str(args.output),
        "warm_start_gradient": warm_start["gradient_norm"],
        "geometry": {
            "column_rank": geometry_report["column"]["joint"]["rank"],
            "row_rank": geometry_report["row"]["joint"]["rank"],
            "descent_margin": geometry_report["row"]["descent"]["strict_margin"],
        },
        "stops": {
            name: row["stop_reason"] for name, row in report["branches"].items()
        },
        "elapsed_seconds": report["elapsed_seconds"],
    }))


if __name__ == "__main__":
    main()
