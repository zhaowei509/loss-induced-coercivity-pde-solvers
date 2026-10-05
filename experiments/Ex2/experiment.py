"""Frozen Experiment 2 construction and per-trajectory execution."""

from datetime import datetime, timezone
import json
import math
from pathlib import Path

import torch

from .dynamics import armijo
from .features import DTYPE, RandomFeatures, spectrum
from .io import atomic_json, file_hash, load_run, save_run
from .objectives import AllenCahnRitz, CubicResidual, P4Ritz, cubic_source, p4_source
from .sampling import CALIBRATION_SEEDS, FORMAL_SEEDS, gauss_legendre, sobol_points, unit_direction


PLAN_NAME = "EXPERIMENT2_PLAN.md"
WIDTH = {"cubic": 32, "p4": 8, "ac": 8}
CONTROL_SEED = 30001
GRADIENT_TOLERANCE = {"cubic": 1e-8, "p4": 1e-10, "ac": 1e-10}
ARMIJO = dict(trial_step=1.0, shrink=0.5, sufficient_decrease=1e-4,
              max_backtracks=60)


def default_plan():
    return Path(__file__).resolve().parents[2] / PLAN_NAME


def control_features(*, device="cpu"):
    centers = torch.linspace(0.1, 0.9, 8, dtype=DTYPE)
    weights = torch.full((8, 1), 16.0, dtype=DTYPE)
    return RandomFeatures(weights.to(device), (-16.0 * centers).to(device))


def make_problem(kind, seed, *, device, basis="random"):
    if kind not in WIDTH:
        raise ValueError(f"Unknown experiment branch: {kind}")
    if basis not in ("random", "control") or (basis == "control" and kind == "cubic"):
        raise ValueError("Control features are defined only for the Ritz branches.")
    dimension = 2 if kind == "cubic" else 1
    features = (control_features(device=device) if basis == "control" else
                RandomFeatures.draw(dimension, WIDTH[kind], seed, device=device))
    if kind == "cubic":
        points = sobol_points(4096, 2, seed, device=device)
        jet = features.jet(points)
        function = CubicResidual(jet.values, jet.laplacian, cubic_source(points))
        weights = None
    else:
        quadrature = gauss_legendre(128, 1, device=device)
        points, weights = quadrature.points, quadrature.weights
        jet = features.jet(points)
        derivative = jet.gradient[:, :, 0]
        if kind == "p4":
            function = P4Ritz(jet.values, derivative, p4_source(points), weights)
        else:
            function = AllenCahnRitz(jet.values, derivative, weights, epsilon=0.1)
    return dict(kind=kind, seed=seed, basis=basis, features=features, points=points,
                weights=weights, jet=jet, function=function)


def training_metric(problem):
    if problem["kind"] == "cubic":
        raise ValueError("QR training metric is defined only for Ritz problems.")
    root = problem["weights"].sqrt()[:, None]
    derivative = problem["jet"].gradient[:, :, 0]
    return (root * derivative if problem["kind"] == "p4" else
            torch.cat((root * problem["jet"].values, 0.1 * root * derivative), dim=0))


def qr_factor(matrix):
    if not spectrum(matrix).full_column_rank:
        return None
    factor = torch.linalg.qr(matrix, mode="reduced").R
    signs = torch.where(torch.diag(factor) < 0, -torch.ones_like(torch.diag(factor)),
                        torch.ones_like(torch.diag(factor)))
    return signs[:, None] * factor


class QRObjective:
    """The same energy in b=Ra coordinates, with its exact transformed gradient."""

    def __init__(self, objective, factor):
        self.objective, self.factor = objective, factor

    def coefficients(self, b):
        return torch.linalg.solve_triangular(self.factor, b[:, None], upper=True)[:, 0]

    def __call__(self, b):
        return self.objective(self.coefficients(b))

    def analytic_value(self, b):
        return self.objective.analytic_value(self.coefficients(b))

    def analytic_value_gradient(self, b):
        value, gradient = self.objective.analytic_value_gradient(self.coefficients(b))
        transformed = torch.linalg.solve_triangular(
            self.factor.T, gradient[:, None], upper=False)[:, 0]
        return value, transformed


def initializations(kind, seed, *, device):
    width = WIDTH[kind]
    zero = torch.zeros(width, dtype=DTYPE, device=device)
    if kind == "cubic":
        return (("zero", zero),)
    direction = unit_direction(width, seed, device=device)
    if kind == "p4":
        return (("zero", zero), ("plus", direction), ("minus", -direction))
    return (("plus", direction), ("minus", -direction))


def _source_hashes():
    module_root = Path(__file__).resolve().parent
    return {path.name: file_hash(path) for path in sorted(module_root.glob("*.py"))}


def _planned_names(seeds):
    names = []
    for kind in ("cubic", "p4", "ac"):
        labels = ("zero",) if kind == "cubic" else (
            ("zero", "plus", "minus") if kind == "p4" else ("plus",))
        names.extend(f"random_{kind}_seed{seed}_{label}"
                     for seed in seeds for label in labels)
    names.extend(("control_p4_seed30001_plus", "control_p4_seed30001_minus",
                  "control_ac_seed30001_plus"))
    names.extend(f"qr_{kind}_seed{seed}_plus"
                 for kind in ("p4", "ac") for seed in seeds)
    return names


def run_campaign(output, mode, *, plan=None, smoke_steps=5, resume=False):
    if mode not in ("smoke", "short_calibration", "calibration", "formal"):
        raise ValueError("Unknown campaign mode.")
    if smoke_steps < 1 or smoke_steps > 500:
        raise ValueError("Smoke-run budget must be in [1,500].")
    plan = default_plan() if plan is None else Path(plan)
    if not plan.is_file():
        raise FileNotFoundError(plan)
    if not torch.cuda.is_available():
        raise RuntimeError("Experiment 2 requires CUDA for the cubic branch.")
    output = Path(output)
    seeds = ((30000,) if mode == "smoke"
             else CALIBRATION_SEEDS if mode in ("short_calibration", "calibration")
             else FORMAL_SEEDS)
    max_steps = smoke_steps if mode in ("smoke", "short_calibration") else 50000
    run_kind = ("unit_smoke" if mode == "smoke" else
                "calibration" if mode == "short_calibration" else mode)
    planned = _planned_names(seeds)
    expected = len(seeds) * 7 + 3
    if len(planned) != expected or len(set(planned)) != expected:
        raise AssertionError("Invalid experiment manifest.")
    plan_sha256, source_sha256 = file_hash(plan), _source_hashes()
    summary_path = output / "summary.json"
    stop_file = output / "STOP_REQUESTED"
    if stop_file.exists():
        raise InterruptedError(f"Campaign stop marker is present: {stop_file}")
    if resume:
        if not summary_path.is_file():
            raise FileNotFoundError("Cannot resume without an existing summary.json.")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for key, value in (("mode", mode), ("max_steps", max_steps),
                           ("expected_runs", expected), ("plan_sha256", plan_sha256),
                           ("source_sha256", source_sha256), ("planned_runs", planned)):
            if summary.get(key) != value:
                raise ValueError(f"Cannot resume: campaign {key} differs.")
    else:
        output.mkdir(parents=True, exist_ok=False)
        summary = dict(mode=mode, created_utc=datetime.now(timezone.utc).isoformat(),
                       plan_sha256=plan_sha256, source_sha256=source_sha256,
                       max_steps=max_steps, expected_runs=expected,
                       planned_runs=planned, records=[], skipped=[], interrupted=[])
        atomic_json(summary_path, summary)
    recorded = {item["run"]: item for item in summary["records"]}
    if len(recorded) != len(summary["records"]) or not set(recorded).issubset(planned):
        raise ValueError("Duplicate or unexpected completed trajectories.")
    if len({(item["branch"], item["seed"]) for item in summary["skipped"]}) != len(
            summary["skipped"]):
        raise ValueError("Duplicate QR rank-failure records.")

    def validate_saved(run_name, *, kind, seed, label, basis):
        metadata, arrays = load_run(output / run_name)
        config = metadata["config"]
        if (metadata["plan_sha256"] != plan_sha256
                or metadata["source_sha256"] != source_sha256
                or config["run_kind"] != run_kind
                or config["accepted_step_limit"] != max_steps
                or (config["branch"], config["feature_seed"],
                    config["initialization"], config["basis"])
                != (kind, seed, label, basis)):
            raise ValueError(f"Saved trajectory does not match campaign: {run_name}")
        return metadata, arrays

    def record_saved(run_name, metadata, arrays):
        result = metadata["result"]
        item = dict(run=run_name, metadata=str(output / run_name / "metadata.json"),
                    stop_reason=result["stop_reason"],
                    accepted_steps=result["accepted_steps"],
                    final_gradient=float(arrays["gradient_norm"][-1]),
                    elapsed_seconds=result["elapsed_seconds"])
        summary["records"].append(item)
        recorded[run_name] = item
        atomic_json(summary_path, summary)

    def quarantine_incomplete(run_name):
        directory = output / run_name
        if not directory.exists():
            return
        if (directory / "metadata.json").is_file() and (
                directory / "trajectory.npz").is_file():
            return
        archive = output / "interrupted"
        archive.mkdir(exist_ok=True)
        number = 1
        while (archive / f"{run_name}_{number}").exists():
            number += 1
        destination = archive / f"{run_name}_{number}"
        directory.rename(destination)
        summary["interrupted"].append(dict(run=run_name, saved_as=str(destination)))
        atomic_json(summary_path, summary)
        print(f"preserved incomplete {run_name} as {destination}", flush=True)

    def execute(problem, label, initial, *, basis, factor=None):
        if stop_file.exists():
            raise InterruptedError(f"Campaign stop marker is present: {stop_file}")
        kind, seed = problem["kind"], problem["seed"]
        run_name = f"{basis}_{kind}_seed{seed}_{label}"
        if run_name in recorded:
            validate_saved(run_name, kind=kind, seed=seed, label=label, basis=basis)
            print(f"already completed {run_name}", flush=True)
            return
        quarantine_incomplete(run_name)
        if (output / run_name).exists():
            metadata, arrays = validate_saved(
                run_name, kind=kind, seed=seed, label=label, basis=basis)
            record_saved(run_name, metadata, arrays)
            print(f"recovered completed {run_name}", flush=True)
            return
        objective = QRObjective(problem["function"], factor) if factor is not None else problem["function"]
        start = factor @ initial if factor is not None else initial
        result = armijo(objective, start, max_steps=max_steps,
                        gradient_tolerance=GRADIENT_TOLERANCE[kind],
                        stop_requested=stop_file.exists, **ARMIJO)
        config = dict(run_kind=run_kind, branch=kind, feature_seed=seed,
                      initialization=label, basis=basis,
                      coordinate="b=Ra" if factor is not None else "a",
                      qr_r_factor=factor, device=str(start.device), dtype="float64",
                      width=WIDTH[kind], accepted_step_limit=max_steps,
                      gradient_tolerance=GRADIENT_TOLERANCE[kind], armijo=ARMIJO,
                      feature_weights=problem["features"].weights,
                      feature_biases=problem["features"].biases,
                      training_points=problem["points"],
                      training_weights=problem["weights"])
        path = save_run(output / run_name, config, result, plan)
        summary["records"].append(dict(run=run_name, metadata=str(path),
                                       stop_reason=result.stop_reason,
                                       accepted_steps=result.accepted_steps,
                                       final_gradient=result.trajectory[-1]["gradient_norm"],
                                       elapsed_seconds=result.elapsed_seconds))
        recorded[run_name] = summary["records"][-1]
        atomic_json(summary_path, summary)
        print(f"completed {run_name}: {result.accepted_steps} steps, {result.stop_reason}", flush=True)
        values = [row["objective"] for row in result.trajectory]
        if (result.stop_reason in ("nonfinite_objective_or_gradient", "line_search_failure")
                or any(not math.isfinite(row["gradient_norm"])
                       for row in result.trajectory)
                or any(right > left + 1e-10 * max(1.0, abs(left))
                       for left, right in zip(values, values[1:]))):
            raise RuntimeError(f"Critical numerical anomaly in {run_name}; campaign stopped.")

    for kind in ("cubic", "p4", "ac"):
        device = "cuda" if kind == "cubic" else "cpu"
        for seed in seeds:
            problem = make_problem(kind, seed, device=device)
            for label, initial in initializations(kind, seed, device=device):
                if kind == "ac" and label == "minus":
                    continue  # Exact sign-mirror trajectory of the even Allen--Cahn energy.
                execute(problem, label, initial, basis="random")

    for kind in ("p4", "ac"):
        problem = make_problem(kind, CONTROL_SEED, device="cpu", basis="control")
        for label, initial in initializations(kind, CONTROL_SEED, device="cpu"):
            if (kind == "p4" and label == "zero") or (kind == "ac" and label == "minus"):
                continue
            execute(problem, label, initial, basis="control")

    for kind in ("p4", "ac"):
        for seed in seeds:
            problem = make_problem(kind, seed, device="cpu")
            factor = qr_factor(training_metric(problem))
            if factor is None:
                if (kind, seed) not in {
                        (item["branch"], item["seed"]) for item in summary["skipped"]}:
                    summary["skipped"].append(dict(branch=kind, seed=seed, basis="qr",
                                                   reason="numerical_rank_failure"))
                    atomic_json(summary_path, summary)
                continue
            if (kind, seed) in {
                    (item["branch"], item["seed"]) for item in summary["skipped"]}:
                raise ValueError("Previously skipped QR matrix changed rank on resume.")
            initial = dict(initializations(kind, seed, device="cpu"))["plus"]
            execute(problem, "plus", initial, basis="qr", factor=factor)
    if len(summary["records"]) + len(summary["skipped"]) != expected:
        raise RuntimeError("Campaign finished with an incomplete manifest.")
    return summary
