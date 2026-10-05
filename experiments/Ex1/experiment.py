"""Frozen construction and execution of the 44 Experiment 1 trajectories."""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from .diagnostics import Objective, confinement_radius, spectrum
from .dynamics import SolverResult, armijo, exact_quadratic_step, gradient_flow
from .features import DTYPE, RandomFeatures
from .io import atomic_json, environment_snapshot, file_hash, json_safe, save_run
from .problems import (cubic_objective, cubic_source, initial_lateral_points,
                       toy_coefficients, toy_features, toy_objective, toy_operator)
from .sampling import (CALIBRATION_SEEDS, FORMAL_SEEDS, circle_directions,
                       gauss_legendre, perimeter_midpoints, sobol_points)


PLAN_NAME = "EXPERIMENT1_PLAN.md"
TOY_KINDS = ("A1", "A2", "A3")
TOY_ORDER = 24
TOY_BOUNDARY_COUNT = 32
TOY_RADIUS = 0.4
TOY_INITIAL_RADIUS = 0.2
TOY_BOUNDARY_PENALTY = 1.0
A1_ETA = 0.5
A1_STEPS = 25
A_FLOW_CONFIG = dict(rtol=1e-10, atol=1e-12, target_tolerance=1e-10,
                     gradient_tolerance=1e-12, output_count=200)
B_WIDTH = 8
B_INTERIOR_COUNT = 2048
B_BOUNDARY_COUNT = 32
B_BOUNDARY_PENALTY = 10.0
B_FLOW_CONFIG = dict(time_limit=1e4, rtol=1e-9, atol=1e-11,
                     gradient_tolerance=1e-8)
B_ARMIJO_CONFIG = dict(max_steps=5000, gradient_tolerance=1e-8,
                       trial_step=1.0, shrink=0.5,
                       sufficient_decrease=1e-4, max_backtracks=60)
FORMAL_WORKERS = 4


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def _project_root():
    return Path(__file__).resolve().parents[2]


def default_plan_path():
    return _project_root() / PLAN_NAME


def _boundary_points(kind, device="cpu"):
    if kind == "A2":
        return initial_lateral_points(8, 16, device=device)
    return perimeter_midpoints(TOY_BOUNDARY_COUNT, device=device)


def make_toy_problem(kind, *, device="cpu", order=TOY_ORDER):
    if kind not in TOY_KINDS:
        raise ValueError(f"Unknown toy problem {kind!r}.")
    features = toy_features(device=device)
    exact = toy_coefficients(device=device)
    quadrature = gauss_legendre(order, 2, device=device)
    interior_jet = features.jet(quadrature.points)
    boundary_points = _boundary_points(kind, device=device)
    boundary_matrix = features.jet(boundary_points).values
    boundary_target = boundary_matrix @ exact
    source = toy_operator(interior_jet, exact, kind)
    function = toy_objective(kind, interior_jet, quadrature.weights,
                             boundary_matrix, boundary_target, source,
                             boundary_penalty=TOY_BOUNDARY_PENALTY)
    return dict(kind=kind, features=features, exact=exact,
                interior_points=quadrature.points,
                interior_weights=quadrature.weights,
                interior_jet=interior_jet, boundary_points=boundary_points,
                boundary_matrix=boundary_matrix,
                boundary_target=boundary_target, source=source,
                function=function)


def make_b_problem(seed, *, device="cpu", interior_count=B_INTERIOR_COUNT):
    features = RandomFeatures.draw(2, B_WIDTH, seed, device=device)
    interior_points = sobol_points(interior_count, 2, seed, device=device)
    interior_weights = torch.full((interior_count,), 1 / interior_count,
                                  dtype=DTYPE, device=device)
    boundary_points = perimeter_midpoints(B_BOUNDARY_COUNT, device=device)
    interior_jet = features.jet(interior_points)
    boundary_matrix = features.jet(boundary_points).values
    source = cubic_source(interior_points)
    function = cubic_objective(interior_jet, source, interior_weights,
                               boundary_matrix,
                               boundary_penalty=B_BOUNDARY_PENALTY)
    return dict(seed=seed, features=features, interior_points=interior_points,
                interior_weights=interior_weights, interior_jet=interior_jet,
                boundary_points=boundary_points,
                boundary_matrix=boundary_matrix, source=source,
                function=function)


def _toy_static_diagnostics(problem):
    exact = problem["exact"]
    objective = Objective(problem["function"])
    value = objective.value(exact)
    hessian = objective.hessian(exact)
    eigenvalues = torch.linalg.eigvalsh(hessian)
    boundary = spectrum(problem["boundary_matrix"])
    mu_boundary = (2 * TOY_BOUNDARY_PENALTY
                   / problem["boundary_matrix"].shape[0]
                   * boundary.sigma_min**2)
    return dict(exact_objective=value, hessian=hessian,
                hessian_eigenvalues=eigenvalues,
                lambda_min_hessian=float(eigenvalues[0]),
                boundary_spectrum=boundary,
                mu_boundary=mu_boundary)


def _base_toy_config(problem, static, mode, direction):
    return dict(run_kind=mode, experiment="1", branch=problem["kind"],
                direction_index=direction,
                direction_angle=2 * math.pi * direction / 8,
                dtype="float64", device="cpu", dimension=2, width=2,
                feature_weights=problem["features"].weights,
                feature_biases=problem["features"].biases,
                exact_coefficients=problem["exact"],
                interior_rule=f"tensor Gauss--Legendre order {TOY_ORDER}",
                interior_points=problem["interior_points"],
                interior_weights=problem["interior_weights"],
                boundary_points=problem["boundary_points"],
                boundary_matrix=problem["boundary_matrix"],
                boundary_target=problem["boundary_target"],
                boundary_penalty=TOY_BOUNDARY_PENALTY,
                source=problem["source"],
                exact_objective=static["exact_objective"],
                hessian_at_exact=static["hessian"],
                hessian_eigenvalues_at_exact=static["hessian_eigenvalues"],
                lambda_min_hessian=static["lambda_min_hessian"],
                boundary_singular_values=static["boundary_spectrum"].singular_values,
                boundary_numerical_rank=static["boundary_spectrum"].numerical_rank,
                sigma_min_boundary=static["boundary_spectrum"].sigma_min,
                mu_boundary=static["mu_boundary"])


def _trajectory_row(step, coefficients, objective, started, **extra):
    value, gradient = objective.value_gradient(coefficients)
    return dict(step=step, parameters=coefficients.detach().cpu().tolist(),
                objective=value,
                gradient_norm=float(torch.linalg.vector_norm(gradient)),
                parameter_norm=float(torch.linalg.vector_norm(coefficients)),
                elapsed_seconds=time.perf_counter() - started, **extra)


def run_a1(direction, mode="formal", *, max_steps=None):
    problem = make_toy_problem("A1")
    static = _toy_static_diagnostics(problem)
    steps = A1_STEPS if max_steps is None else max_steps
    exact, hessian = problem["exact"], static["hessian"]
    direction_vector = circle_directions(8)[direction]
    initial = exact + TOY_INITIAL_RADIUS * direction_vector
    objective = Objective(problem["function"])
    contraction = torch.linalg.inv(
        torch.eye(2, dtype=DTYPE) + A1_ETA * hessian)
    q_linear = 1 / (1 + A1_ETA * static["lambda_min_hessian"])
    started = time.perf_counter()
    error_vector = initial - exact
    current = exact + error_vector
    trajectory = []
    maximum_identity_error = 0.0
    previous_error = None
    for step in range(steps + 1):
        error = float(torch.linalg.vector_norm(error_vector))
        predicted_error = torch.linalg.matrix_power(contraction, step) @ (initial - exact)
        denominator = max(float(torch.linalg.vector_norm(predicted_error)),
                          torch.finfo(DTYPE).tiny)
        identity_error = float(torch.linalg.vector_norm(error_vector - predicted_error)) / denominator
        maximum_identity_error = max(maximum_identity_error, identity_error)
        q_k = None if previous_error is None else error / previous_error
        trajectory.append(_trajectory_row(
            step, current, objective, started, coefficient_error=error,
            normalized_coefficient_error=error / TOY_INITIAL_RADIUS,
            theoretical_bound=q_linear**step * TOY_INITIAL_RADIUS,
            q_k=q_k, closed_recurrence_relative_error=identity_error))
        if step < steps:
            previous_error = error
            error_vector = contraction @ error_vector
            current = exact + error_vector
    result = SolverResult(current, "fixed_steps", trajectory,
                          objective.snapshot(), steps,
                          time.perf_counter() - started,
                          diagnostics=dict(q_linear=q_linear,
                                           maximum_q_k=max(row["q_k"] for row in trajectory[1:]),
                                           maximum_closed_recurrence_relative_error=maximum_identity_error,
                                           recurrence_pass=maximum_identity_error <= 1e-12,
                                           contraction_pass=all(row["q_k"] <= q_linear + 5e-15
                                                                for row in trajectory[1:])))
    config = _base_toy_config(problem, static, mode, direction)
    config.update(dynamics="exact_proximal", eta=A1_ETA,
                  prescribed_steps=steps, initial_coefficients=initial,
                  q_linear=q_linear)
    return config, result


def _slope_fit(trajectory, initial_error):
    times, logs = [], []
    for row in trajectory:
        ratio = row["coefficient_error"] / initial_error
        if 1e-8 <= ratio <= 0.5:
            times.append(row["time"])
            logs.append(math.log(row["coefficient_error"]))
    if len(times) < 20:
        return dict(valid_point_count=len(times), slope=None, r_squared=None)
    slope, intercept = np.polyfit(np.asarray(times), np.asarray(logs), 1)
    fitted = slope * np.asarray(times) + intercept
    residual = float(np.sum((np.asarray(logs) - fitted)**2))
    total = float(np.sum((np.asarray(logs) - np.mean(logs))**2))
    return dict(valid_point_count=len(times), slope=float(slope),
                r_squared=(1.0 if total == 0 else 1 - residual / total))


def _tube_curvature(function, exact, trajectory):
    diagnostic = Objective(function)
    minimum = math.inf
    for row in trajectory:
        current = torch.tensor(row["parameters"], dtype=DTYPE)
        for fraction in np.linspace(0.0, 1.0, 11):
            point = exact + float(fraction) * (current - exact)
            eigenvalue = float(torch.linalg.eigvalsh(diagnostic.hessian(point))[0])
            minimum = min(minimum, eigenvalue)
    return minimum, diagnostic.snapshot()


def run_nonlinear_toy(kind, direction, mode="formal", *, time_limit=None,
                      output_count=None, compute_tube=True):
    if kind not in ("A2", "A3"):
        raise ValueError("Nonlinear toy must be A2 or A3.")
    problem = make_toy_problem(kind)
    static = _toy_static_diagnostics(problem)
    exact = problem["exact"]
    initial = exact + TOY_INITIAL_RADIUS * circle_directions(8)[direction]
    prescribed_limit = min(30 / static["lambda_min_hessian"], 1e4) \
        if static["lambda_min_hessian"] > 0 else 0.0
    actual_limit = prescribed_limit if time_limit is None else min(time_limit, prescribed_limit)
    count = A_FLOW_CONFIG["output_count"] if output_count is None else output_count
    objective = Objective(problem["function"])
    if static["lambda_min_hessian"] <= 0:
        row = _trajectory_row(0, initial, objective, time.perf_counter(), time=0.0,
                              coefficient_error=TOY_INITIAL_RADIUS,
                              normalized_coefficient_error=1.0)
        result = SolverResult(initial, "curvature_precheck_failure", [row],
                              objective.snapshot())
    else:
        result = gradient_flow(
            objective, initial, time_limit=actual_limit,
            rtol=A_FLOW_CONFIG["rtol"], atol=A_FLOW_CONFIG["atol"],
            target=lambda a: torch.linalg.vector_norm(a - exact),
            target_tolerance=A_FLOW_CONFIG["target_tolerance"],
            gradient_tolerance=A_FLOW_CONFIG["gradient_tolerance"],
            center=exact, radius=TOY_RADIUS, output_count=count)
        for row in result.trajectory:
            error = float(torch.linalg.vector_norm(
                torch.tensor(row["parameters"], dtype=DTYPE) - exact))
            row["coefficient_error"] = error
            row["normalized_coefficient_error"] = error / TOY_INITIAL_RADIUS
    tube, tube_counts = (None, {})
    if compute_tube and static["lambda_min_hessian"] > 0:
        tube, tube_counts = _tube_curvature(problem["function"], exact,
                                            result.trajectory)
    if tube is not None and tube > 0:
        for row in result.trajectory:
            row["sampled_exponential_reference"] = (
                TOY_INITIAL_RADIUS * math.exp(-tube * row["time"]))
    fit = _slope_fit(result.trajectory, TOY_INITIAL_RADIUS)
    result.diagnostics.update(
        sampled_tube_curvature=tube, tube_hessian_counts=tube_counts,
        semilog_fit=fit,
        convergence_success=result.stop_reason in
        ("target_tolerance", "gradient_tolerance"))
    config = _base_toy_config(problem, static, mode, direction)
    config.update(dynamics="gradient_flow", initial_coefficients=initial,
                  initial_radius=TOY_INITIAL_RADIUS,
                  audit_radius=TOY_RADIUS,
                  prescribed_time_limit=prescribed_limit,
                  actual_time_limit=actual_limit,
                  solver_config=dict(A_FLOW_CONFIG, output_count=count))
    return config, result


def _b_config(problem, static, mode, solver_name, radius, actual_budget):
    return dict(run_kind=mode, experiment="1", branch="B",
                dynamics=solver_name, feature_seed=problem["seed"],
                sobol_seed=1000 + problem["seed"], dtype="float64",
                device="cpu", dimension=2, width=B_WIDTH,
                feature_distribution="independent standard Gaussian weights and biases",
                feature_normalization="none",
                feature_weights=problem["features"].weights,
                feature_biases=problem["features"].biases,
                interior_count=problem["interior_points"].shape[0],
                interior_points=problem["interior_points"],
                interior_weights=problem["interior_weights"],
                boundary_count=B_BOUNDARY_COUNT,
                boundary_points=problem["boundary_points"],
                boundary_matrix=problem["boundary_matrix"],
                boundary_target=torch.zeros(B_BOUNDARY_COUNT, dtype=DTYPE),
                boundary_penalty=B_BOUNDARY_PENALTY,
                initial_coefficients=torch.zeros(B_WIDTH, dtype=DTYPE),
                initial_objective=static["initial_objective"],
                boundary_singular_values=static["spectrum"].singular_values,
                numerical_rank_threshold=static["spectrum"].threshold,
                boundary_numerical_rank=static["spectrum"].numerical_rank,
                boundary_full_column_rank=static["spectrum"].full_column_rank,
                sigma_min_boundary=static["spectrum"].sigma_min,
                explicit_confinement_radius=radius,
                actual_budget=actual_budget)


def run_b(seed, solver_name, mode="formal", *, time_limit=None, max_steps=None,
          interior_count=B_INTERIOR_COUNT):
    if solver_name not in ("gradient_flow", "armijo"):
        raise ValueError("Unknown B dynamics.")
    problem = make_b_problem(seed, interior_count=interior_count)
    initial = torch.zeros(B_WIDTH, dtype=DTYPE)
    initial_objective = Objective(problem["function"]).value(initial)
    boundary_spectrum = spectrum(problem["boundary_matrix"])
    radius = (confinement_radius(initial_objective, problem["boundary_matrix"],
                                 B_BOUNDARY_PENALTY)
              if boundary_spectrum.full_column_rank else None)
    static = dict(initial_objective=initial_objective, spectrum=boundary_spectrum)
    objective = Objective(problem["function"])
    if solver_name == "gradient_flow":
        budget = B_FLOW_CONFIG["time_limit"] if time_limit is None else time_limit
        output_times = np.concatenate(([0.0], np.geomspace(1e-8, B_FLOW_CONFIG["time_limit"], 200)))
        result = gradient_flow(
            objective, initial, time_limit=budget,
            rtol=B_FLOW_CONFIG["rtol"], atol=B_FLOW_CONFIG["atol"],
            gradient_tolerance=B_FLOW_CONFIG["gradient_tolerance"],
            center=(initial if radius is not None else None), radius=radius,
            output_times=output_times)
        tolerance = 1e-10 * max(1.0, initial_objective)
        monotone = all(right["objective"] <= left["objective"] + tolerance
                       for left, right in zip(result.trajectory, result.trajectory[1:]))
        result.diagnostics.update(numerical_monotonicity_pass=monotone,
                                  monotonicity_tolerance=tolerance)
        config = _b_config(problem, static, mode, solver_name, radius, budget)
        config["solver_config"] = dict(B_FLOW_CONFIG,
                                       output_times="0 plus 200 log-spaced times on [1e-8,1e4]")
    else:
        budget = B_ARMIJO_CONFIG["max_steps"] if max_steps is None else max_steps
        settings = dict(B_ARMIJO_CONFIG, max_steps=budget,
                        confinement_radius=radius)
        result = armijo(objective, initial, **settings)
        margins = [row["armijo_margin"] for row in result.trajectory[1:]]
        result.diagnostics.update(
            sufficient_decrease_pass=all(margin >= -1e-14 for margin in margins),
            minimum_armijo_margin=min(margins) if margins else None)
        config = _b_config(problem, static, mode, solver_name, radius, budget)
        config["solver_config"] = B_ARMIJO_CONFIG
    for row in result.trajectory:
        row["radius_ratio"] = (None if radius is None else
                               row["parameter_norm"] / radius)
    result.diagnostics.update(
        maximum_radius_ratio=(None if radius is None else
                              max(row["radius_ratio"] for row in result.trajectory)),
        final_gradient_norm=result.trajectory[-1]["gradient_norm"],
        final_objective=result.trajectory[-1]["objective"])
    return config, result


def _run_directory(root, config):
    if config["branch"] in TOY_KINDS:
        return root / "runs" / config["branch"] / f"direction_{config['direction_index']}"
    return root / "runs" / "B" / config["dynamics"] / f"seed_{config['feature_seed']}"


def _identity(config):
    if config["branch"] in TOY_KINDS:
        return (config["branch"], config["direction_index"])
    return (config["branch"], config["dynamics"], config["feature_seed"])


def _validate_existing(path, expected_config, plan_hash):
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("plan_sha256") != plan_hash:
        raise RuntimeError(f"Plan hash mismatch in {path}.")
    if _identity(raw["config"]) != _identity(expected_config):
        raise RuntimeError(f"Run identity mismatch in {path}.")
    return raw


def _jobs(mode):
    if mode == "formal":
        return ([("toy", kind, direction) for kind in TOY_KINDS for direction in range(8)]
                + [("B", solver, seed) for solver in ("gradient_flow", "armijo")
                   for seed in FORMAL_SEEDS])
    return (["A1", "A2", "A3"]
            + [("B", solver, seed) for solver in ("gradient_flow", "armijo")
               for seed in CALIBRATION_SEEDS])


def _execute_job(job, mode):
    if mode == "calibration" and isinstance(job, str):
        if job == "A1":
            return run_a1(0, mode, max_steps=3)
        return run_nonlinear_toy(job, 0, mode, time_limit=1.0,
                                 output_count=25, compute_tube=False)
    kind, first, second = job
    if kind == "toy":
        return (run_a1(second, mode) if first == "A1" else
                run_nonlinear_toy(first, second, mode))
    if first == "gradient_flow":
        return run_b(second, first, mode, time_limit=(5.0 if mode == "calibration" else None))
    return run_b(second, first, mode, max_steps=(50 if mode == "calibration" else None))


def _execute_job_worker(job, mode):
    torch.set_num_threads(1)
    return _execute_job(job, mode)


def _campaign_constants(mode):
    return dict(mode=mode, toy_order=TOY_ORDER, toy_boundary_count=TOY_BOUNDARY_COUNT,
                toy_initial_radius=TOY_INITIAL_RADIUS, toy_audit_radius=TOY_RADIUS,
                a1_eta=A1_ETA, a1_steps=A1_STEPS, a_flow=A_FLOW_CONFIG,
                b_width=B_WIDTH, b_interior_count=B_INTERIOR_COUNT,
                b_boundary_count=B_BOUNDARY_COUNT,
                b_boundary_penalty=B_BOUNDARY_PENALTY,
                b_flow=B_FLOW_CONFIG, b_armijo=B_ARMIJO_CONFIG,
                formal_parallel_workers=FORMAL_WORKERS,
                torch_intraop_threads=1)


def run_campaign(root, mode, plan_path=None):
    if mode not in ("calibration", "formal"):
        raise ValueError("Mode must be calibration or formal.")
    root = Path(root)
    plan_path = default_plan_path() if plan_path is None else Path(plan_path)
    plan_hash = file_hash(plan_path)
    root.mkdir(parents=True, exist_ok=True)
    campaign_path = root / "campaign.json"
    intended = dict(schema_version=1, experiment="1", mode=mode,
                    created_utc=utc_now(), plan_path=str(plan_path),
                    plan_sha256=plan_hash, environment=environment_snapshot("cpu"),
                    frozen_constants=_campaign_constants(mode))
    if campaign_path.exists():
        existing = json.loads(campaign_path.read_text(encoding="utf-8"))
        for key in ("experiment", "mode", "plan_sha256", "frozen_constants"):
            if json_safe(existing[key]) != json_safe(intended[key]):
                raise RuntimeError(f"Cannot resume campaign: mismatch in {key}.")
    else:
        atomic_json(campaign_path, intended)
    jobs = _jobs(mode)
    completed = 0
    def record(config, result):
        nonlocal completed
        directory = _run_directory(root, config)
        raw_path = directory / "raw.json"
        if raw_path.exists():
            raw = _validate_existing(raw_path, config, plan_hash)
            stop_reason = raw["result"]["stop_reason"]
        else:
            save_run(directory, config, result, plan_path)
            stop_reason = result.stop_reason
        completed += 1
        progress = dict(updated_utc=utc_now(), completed=completed,
                        total=len(jobs), all_completed=completed == len(jobs),
                        last_identity=_identity(config),
                        last_stop_reason=stop_reason)
        atomic_json(root / "progress.json", progress)
        print(json.dumps(json_safe(progress)), flush=True)
    if mode == "formal":
        with ProcessPoolExecutor(max_workers=FORMAL_WORKERS) as executor:
            futures = [executor.submit(_execute_job_worker, job, mode) for job in jobs]
            for future in as_completed(futures):
                record(*future.result())
    else:
        for job in jobs:
            record(*_execute_job(job, mode))
    return aggregate_raw(root)


def _quantiles(values):
    array = np.asarray([value for value in values if value is not None], dtype=float)
    return None if not array.size else dict(median=float(np.median(array)),
                                             q1=float(np.quantile(array, 0.25)),
                                             q3=float(np.quantile(array, 0.75)),
                                             minimum=float(np.min(array)),
                                             maximum=float(np.max(array)),
                                             count=int(array.size))


def aggregate_raw(root):
    root = Path(root)
    raws = [json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((root / "runs").rglob("raw.json"))]
    rows = []
    for raw in raws:
        config, result = raw["config"], raw["result"]
        row = dict(identity=_identity(config), branch=config["branch"],
                   dynamics=config["dynamics"], stop_reason=result["stop_reason"],
                   elapsed_seconds=result["elapsed_seconds"],
                   final_objective=result["trajectory"][-1]["objective"],
                   final_gradient_norm=result["trajectory"][-1]["gradient_norm"])
        row.update(result.get("diagnostics", {}))
        rows.append(row)
    aggregate = dict(created_utc=utc_now(), raw_record_count=len(raws), runs=rows,
                     stop_reason_counts=dict(Counter(row["stop_reason"] for row in rows)),
                     elapsed_seconds=_quantiles(row["elapsed_seconds"] for row in rows))
    atomic_json(root / "aggregate.json", aggregate)
    return aggregate
