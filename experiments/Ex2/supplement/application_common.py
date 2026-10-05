"""Shared frozen objects and diagnostics for supplements S3A--S3C."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from typing import Any, Callable

import numpy as np
from scipy.special import roots_legendre
from scipy.stats import qmc
import torch

from experiments.shared.artifacts import sha256_array
from experiments.shared.numerics import spectrum


DTYPE = torch.float64
MASTER_WIDTH = 256
TRAIN_MASTER_COUNT = 32768
VALIDATION_COUNT = 65536
RANK_MASTER_COUNT = 1024
BOUNDARY_COUNT = 4096
FORMAL_SEEDS = tuple(range(20, 30))
CALIBRATION_SEEDS = (10000, 10001, 10002)
SMOKE_SEED = 10003
SCALE_CANDIDATES = (0.5, 1.0, 2.0, 4.0)
CONFIGURATIONS = {
    "W128_C": (128, 16384),
    "W64": (64, 16384),
    "W256": (256, 16384),
    "N4096": (128, 4096),
    "N32768": (128, 32768),
}


@dataclass(frozen=True)
class OptimizerConfig:
    max_iterations: int = 500
    max_closures: int = 2000
    history_size: int = 50
    relative_gradient_tolerance: float = 1e-8

    def validate(self) -> None:
        if (self.max_iterations < 0 or self.max_closures < 1
                or self.history_size < 1 or self.relative_gradient_tolerance < 0):
            raise ValueError("Invalid L-BFGS configuration.")


@dataclass
class MasterBank:
    seed: int
    base_weights: torch.Tensor
    biases: torch.Tensor
    training_points: torch.Tensor
    validation_points: torch.Tensor
    rank_points: torch.Tensor
    boundary_points: torch.Tensor

    def hashes(self) -> dict[str, str]:
        return {
            "base_weights": sha256_array(self.base_weights),
            "biases": sha256_array(self.biases),
            "training_points": sha256_array(self.training_points),
            "validation_points": sha256_array(self.validation_points),
            "rank_points": sha256_array(self.rank_points),
            "boundary_points": sha256_array(self.boundary_points),
        }


def _sobol(count: int, seed: int) -> np.ndarray:
    if count < 1 or count & (count - 1):
        raise ValueError("Sobol count must be a positive power of two.")
    return qmc.Sobol(d=2, scramble=True, seed=seed).random_base2(
        count.bit_length() - 1).copy()


def perimeter_midpoints(count: int) -> np.ndarray:
    s = (np.arange(count, dtype=np.float64) + 0.5) / count
    coordinate = 4 * s
    side = np.floor(coordinate).astype(np.int64)
    offset = coordinate - side
    points = np.empty((count, 2), dtype=np.float64)
    choices = ((offset, np.zeros(count)), (np.ones(count), offset),
               (1 - offset, np.ones(count)), (np.zeros(count), 1 - offset))
    for index, (x, y) in enumerate(choices):
        mask = side == index
        points[mask] = np.stack((x[mask], y[mask]), axis=1)
    return points


def make_master_bank(seed: int, *, device: str = "cpu",
                     train_count: int = TRAIN_MASTER_COUNT,
                     validation_count: int = VALIDATION_COUNT,
                     rank_count: int = RANK_MASTER_COUNT,
                     boundary_count: int = BOUNDARY_COUNT) -> MasterBank:
    if train_count > TRAIN_MASTER_COUNT or validation_count > VALIDATION_COUNT:
        raise ValueError("Requested bank exceeds the frozen master-bank size.")
    generator = np.random.Generator(np.random.PCG64(seed))
    weights = generator.standard_normal((MASTER_WIDTH, 2))
    biases = generator.standard_normal(MASTER_WIDTH)
    return MasterBank(
        seed=seed,
        base_weights=torch.as_tensor(weights, dtype=DTYPE, device=device),
        biases=torch.as_tensor(biases, dtype=DTYPE, device=device),
        training_points=torch.as_tensor(_sobol(train_count, 1000 + seed),
                                        dtype=DTYPE, device=device),
        validation_points=torch.as_tensor(_sobol(validation_count, 2000 + seed),
                                          dtype=DTYPE, device=device),
        rank_points=torch.as_tensor(_sobol(rank_count, 3000 + seed),
                                    dtype=DTYPE, device=device),
        boundary_points=torch.as_tensor(perimeter_midpoints(boundary_count),
                                        dtype=DTYPE, device=device),
    )


def target_solution(points: torch.Tensor, frequency: int) -> torch.Tensor:
    if frequency < 1:
        raise ValueError("A positive target frequency is required.")
    return (torch.sin(frequency * torch.pi * points[:, 0])
            * torch.sin(frequency * torch.pi * points[:, 1]))


def target_source(points: torch.Tensor, frequency: int) -> torch.Tensor:
    solution = target_solution(points, frequency)
    return 2 * frequency**2 * torch.pi**2 * solution + solution.pow(3)


def hard_basis(points: torch.Tensor, weights: torch.Tensor,
               biases: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return psi and -Delta(psi) in physical x coordinates."""
    if (points.dtype != DTYPE or weights.dtype != DTYPE or biases.dtype != DTYPE
            or points.ndim != 2 or points.shape[1] != 2
            or weights.ndim != 2 or weights.shape[1] != 2
            or biases.shape != (weights.shape[0],)):
        raise ValueError("Invalid hard-boundary basis inputs.")
    x, y = points[:, 0], points[:, 1]
    z = 2 * points - 1
    activation = torch.tanh(z @ weights.T + biases)
    first_activation = 1 - activation.square()
    second_activation = -2 * activation * first_activation
    h = x * (1 - x) * y * (1 - y)
    hx = (1 - 2 * x) * y * (1 - y)
    hy = (1 - 2 * y) * x * (1 - x)
    laplacian_h = -2 * (y * (1 - y) + x * (1 - x))
    tx = 2 * first_activation * weights[:, 0]
    ty = 2 * first_activation * weights[:, 1]
    laplacian_t = 4 * second_activation * weights.square().sum(dim=1)
    values = h[:, None] * activation
    laplacian = (laplacian_h[:, None] * activation
                 + 2 * hx[:, None] * tx + 2 * hy[:, None] * ty
                 + h[:, None] * laplacian_t)
    return values, -laplacian


class CubicObjective:
    def __init__(self, values: torch.Tensor, negative_laplacian: torch.Tensor,
                 source: torch.Tensor):
        if (values.shape != negative_laplacian.shape or values.ndim != 2
                or source.shape != (values.shape[0],)):
            raise ValueError("Incompatible cubic objective arrays.")
        self.values = values
        self.negative_laplacian = negative_laplacian
        self.source = source

    def tensor_value_gradient(self, coefficients: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        u = self.values @ coefficients
        residual = self.negative_laplacian @ coefficients + u.pow(3) - self.source
        jacobian = self.negative_laplacian + 3 * u.square()[:, None] * self.values
        return 0.5 * residual.square().mean(), jacobian.T @ residual / residual.numel()

    def value_gradient(self, coefficients: torch.Tensor) -> tuple[float, torch.Tensor]:
        value, gradient = self.tensor_value_gradient(coefficients)
        return float(value), gradient.detach()

    def value(self, coefficients: torch.Tensor) -> float:
        u = self.values @ coefficients
        residual = self.negative_laplacian @ coefficients + u.pow(3) - self.source
        return float(0.5 * residual.square().mean())

    def residual(self, coefficients: torch.Tensor) -> torch.Tensor:
        u = self.values @ coefficients
        return self.negative_laplacian @ coefficients + u.pow(3) - self.source


class ClosureBudgetReached(RuntimeError):
    pass


def train_lbfgs(objective: CubicObjective, width: int, *, device: str,
                config: OptimizerConfig | None = None,
                progress: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    config = config or OptimizerConfig()
    config.validate()
    coefficients = torch.zeros(width, dtype=DTYPE, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [coefficients], lr=1.0, max_iter=1, max_eval=25,
        tolerance_grad=0.0, tolerance_change=0.0,
        history_size=config.history_size, line_search_fn="strong_wolfe")
    if coefficients.is_cuda:
        torch.cuda.reset_peak_memory_stats(coefficients.device)
    closure_count = 0
    iterations = 0
    reason = "optimizer_budget"
    started = time.perf_counter()
    trajectory: list[dict[str, Any]] = []
    initial_value, initial_gradient = objective.value_gradient(coefficients.detach())
    trajectory.append({"iteration": 0, "closure_count": 0,
                       "objective": initial_value,
                       "relative_gradient": float(torch.linalg.vector_norm(initial_gradient)),
                       "parameter_norm": 0.0})

    while iterations < config.max_iterations and closure_count < config.max_closures:
        safe = coefficients.detach().clone()

        def closure() -> torch.Tensor:
            nonlocal closure_count
            if closure_count >= config.max_closures:
                raise ClosureBudgetReached
            optimizer.zero_grad(set_to_none=True)
            value, gradient = objective.tensor_value_gradient(coefficients)
            coefficients.grad = gradient.detach().clone()
            closure_count += 1
            return value.detach()

        try:
            optimizer.step(closure)
        except ClosureBudgetReached:
            with torch.no_grad():
                coefficients.copy_(safe)
            reason = "optimizer_budget"
            break
        except torch.cuda.OutOfMemoryError:
            with torch.no_grad():
                coefficients.copy_(safe)
            reason = "memory_failure"
            break
        except RuntimeError as error:
            with torch.no_grad():
                coefficients.copy_(safe)
            reason = "optimizer_failure"
            return _training_result(coefficients, objective, config, iterations,
                                    closure_count, reason, trajectory, started,
                                    error=repr(error))
        iterations += 1
        value, gradient = objective.value_gradient(coefficients.detach())
        parameter_norm = float(torch.linalg.vector_norm(coefficients.detach()))
        relative_gradient = float(torch.linalg.vector_norm(gradient)) / max(1.0, parameter_norm)
        row = {"iteration": iterations, "closure_count": closure_count,
               "objective": value, "relative_gradient": relative_gradient,
               "parameter_norm": parameter_norm}
        trajectory.append(row)
        if progress is not None:
            progress(row)
        if (not math.isfinite(value) or not math.isfinite(relative_gradient)
                or not bool(torch.isfinite(coefficients).all())):
            reason = "nonfinite"
            break
        if relative_gradient <= config.relative_gradient_tolerance:
            reason = "stationarity_pass"
            break
    return _training_result(coefficients, objective, config, iterations,
                            closure_count, reason, trajectory, started)


def _training_result(coefficients: torch.Tensor, objective: CubicObjective,
                     config: OptimizerConfig, iterations: int, closure_count: int,
                     reason: str, trajectory: list[dict[str, Any]], started: float,
                     *, error: str | None = None) -> dict[str, Any]:
    value, gradient = objective.value_gradient(coefficients.detach())
    parameter_norm = float(torch.linalg.vector_norm(coefficients.detach()))
    relative_gradient = float(torch.linalg.vector_norm(gradient)) / max(1.0, parameter_norm)
    peak = (int(torch.cuda.max_memory_allocated(coefficients.device))
            if coefficients.is_cuda else None)
    return {
        "parameters": coefficients.detach(),
        "stop_reason": reason,
        "iterations": iterations,
        "closure_count": closure_count,
        "objective": value,
        "gradient_norm": float(torch.linalg.vector_norm(gradient)),
        "relative_gradient": relative_gradient,
        "stationarity_pass": reason == "stationarity_pass",
        "elapsed_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": peak,
        "optimizer_config": asdict(config),
        "error": error,
        "trajectory": trajectory,
    }


def build_training_objective(bank: MasterBank, width: int, count: int,
                             scale: float, frequency: int) -> CubicObjective:
    points = bank.training_points[:count]
    weights = scale * bank.base_weights[:width]
    biases = bank.biases[:width]
    values, negative_laplacian = hard_basis(points, weights, biases)
    return CubicObjective(values, negative_laplacian, target_source(points, frequency))


def rank_diagnostic(bank: MasterBank, width: int, scale: float) -> dict[str, Any]:
    points = bank.rank_points[:4 * width]
    values, _ = hard_basis(points, scale * bank.base_weights[:width],
                           bank.biases[:width])
    diagnostic = spectrum(values)
    result = diagnostic.as_dict()
    result["rank_point_count"] = int(points.shape[0])
    result["condition_number"] = (
        diagnostic.sigma_max / diagnostic.sigma_min
        if diagnostic.sigma_min > 0 else None)
    result["spectral_margin"] = diagnostic.sigma_min - diagnostic.threshold
    result["pass"] = diagnostic.full_column_rank and result["spectral_margin"] > 0
    return result


def validation_residual(bank: MasterBank, coefficients: torch.Tensor, width: int,
                        scale: float, frequency: int, *, chunk_size: int = 4096) -> float:
    square_sum = 0.0
    count = int(bank.validation_points.shape[0])
    weights = scale * bank.base_weights[:width]
    biases = bank.biases[:width]
    for start in range(0, count, chunk_size):
        points = bank.validation_points[start:start + chunk_size]
        values, negative_laplacian = hard_basis(points, weights, biases)
        u = values @ coefficients
        residual = negative_laplacian @ coefficients + u.pow(3) - target_source(points, frequency)
        square_sum += float(torch.dot(residual, residual))
    return math.sqrt(square_sum / count)


def gauss_legendre_metrics(bank: MasterBank, coefficients: torch.Tensor, width: int,
                           scale: float, frequency: int, *, order: int = 256,
                           chunk_size: int = 4096) -> dict[str, Any]:
    nodes, one_weights = roots_legendre(order)
    nodes, one_weights = (nodes + 1) / 2, one_weights / 2
    xx, yy = np.meshgrid(nodes, nodes, indexing="ij")
    wx, wy = np.meshgrid(one_weights, one_weights, indexing="ij")
    point_array = np.stack((xx.reshape(-1), yy.reshape(-1)), axis=1)
    quadrature_weights = (wx * wy).reshape(-1)
    error_squared = 0.0
    residual_squared = 0.0
    feature_weights = scale * bank.base_weights[:width]
    biases = bank.biases[:width]
    for start in range(0, point_array.shape[0], chunk_size):
        stop = start + chunk_size
        points = torch.as_tensor(point_array[start:stop], dtype=DTYPE,
                                 device=coefficients.device)
        weights = torch.as_tensor(quadrature_weights[start:stop], dtype=DTYPE,
                                  device=coefficients.device)
        values, negative_laplacian = hard_basis(points, feature_weights, biases)
        u = values @ coefficients
        exact = target_solution(points, frequency)
        residual = negative_laplacian @ coefficients + u.pow(3) - target_source(points, frequency)
        error_squared += float(torch.dot(weights, (u - exact).square()))
        residual_squared += float(torch.dot(weights, residual.square()))
    absolute_error = math.sqrt(max(0.0, error_squared))
    residual_norm = math.sqrt(max(0.0, residual_squared))
    bound = residual_norm / (2 * math.pi**2)
    return {
        "quadrature_order": order,
        "absolute_l2_error": absolute_error,
        "relative_l2_error": absolute_error / 0.5,
        "residual_l2": residual_norm,
        "proposition_bound": bound,
        "error_over_bound": absolute_error / bound if bound >= 1e-12 else None,
        "bound_margin": bound - absolute_error,
    }


def boundary_metrics(bank: MasterBank, coefficients: torch.Tensor, width: int,
                     scale: float) -> dict[str, float]:
    values, _ = hard_basis(bank.boundary_points,
                           scale * bank.base_weights[:width], bank.biases[:width])
    boundary = values @ coefficients
    return {"maximum_absolute_error": float(boundary.abs().max()),
            "rms_error": float(boundary.square().mean().sqrt())}


def evaluate_run(bank: MasterBank, coefficients: torch.Tensor, objective: CubicObjective,
                 width: int, scale: float, frequency: int) -> dict[str, Any]:
    train_value = objective.value(coefficients)
    train_rms = math.sqrt(max(0.0, 2 * train_value))
    validation_rms = validation_residual(bank, coefficients, width, scale, frequency)
    return {
        "training_objective": train_value,
        "training_rms_residual": train_rms,
        "validation_rms_residual": validation_rms,
        "validation_minus_training": validation_rms - train_rms,
        "validation_over_training": (
            validation_rms / train_rms if train_rms >= 1e-12 else None),
        "gauss_legendre": gauss_legendre_metrics(
            bank, coefficients, width, scale, frequency),
        "boundary": boundary_metrics(bank, coefficients, width, scale),
    }


def run_configuration(seed: int, label: str, scale: float, frequency: int,
                      *, device: str, optimizer_config: OptimizerConfig) -> dict[str, Any]:
    if label not in CONFIGURATIONS:
        raise ValueError(f"Unknown S3 configuration {label!r}.")
    width, count = CONFIGURATIONS[label]
    bank = make_master_bank(seed, device=device)
    rank = rank_diagnostic(bank, width, scale)
    base = {
        "schema_version": 1, "feature_seed": seed, "configuration": label,
        "width": width, "training_count": count, "feature_scale": scale,
        "frequency": frequency, "device": device, "dtype": "float64",
        "bank_hashes": bank.hashes(), "rank": rank,
        "resource_limit_bytes": resource_limit_bytes(device),
    }
    if not rank["pass"]:
        return {**base, "stop_reason": "conditioning_failure", "training": None,
                "evaluation": None}
    try:
        objective = build_training_objective(bank, width, count, scale, frequency)
        training = train_lbfgs(objective, width, device=device, config=optimizer_config)
        limit = resource_limit_bytes(device)
        if (limit is not None and training["peak_gpu_memory_bytes"] is not None
                and training["peak_gpu_memory_bytes"] > limit):
            training["resource_gate_pass"] = False
            return {**base, "stop_reason": "memory_failure", "training": training,
                    "evaluation": None}
        training["resource_gate_pass"] = True
        if training["stop_reason"] in {"memory_failure", "nonfinite", "optimizer_failure"}:
            return {**base, "stop_reason": training["stop_reason"],
                    "training": training, "evaluation": None}
        evaluation = evaluate_run(bank, training["parameters"], objective,
                                  width, scale, frequency)
        return {**base, "stop_reason": training["stop_reason"],
                "training": training, "evaluation": evaluation}
    except torch.cuda.OutOfMemoryError as error:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {**base, "stop_reason": "memory_failure", "training": None,
                "evaluation": None, "error": repr(error)}


def resource_limit_bytes(device: str) -> int | None:
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return None
    total = torch.cuda.get_device_properties(torch.device(device)).total_memory
    return int(min(5.5 * 1024**3, 0.7 * total))


def run_calibration(seed: int, scale: float, *, device: str) -> dict[str, Any]:
    width, count = CONFIGURATIONS["W128_C"]
    bank = make_master_bank(seed, device=device)
    rank = rank_diagnostic(bank, width, scale)
    base = {"feature_seed": seed, "feature_scale": scale, "rank": rank,
            "bank_hashes": bank.hashes(), "device": device, "dtype": "float64"}
    if not rank["pass"]:
        return {**base, "stop_reason": "conditioning_failure", "training": None,
                "validation_rms_residual": None}
    objective = build_training_objective(bank, width, count, scale, 1)
    training = train_lbfgs(
        objective, width, device=device,
        config=OptimizerConfig(max_iterations=500, max_closures=100,
                               history_size=50, relative_gradient_tolerance=0.0))
    validation = None
    if training["stop_reason"] not in {"memory_failure", "nonfinite", "optimizer_failure"}:
        validation = validation_residual(bank, training["parameters"], width, scale, 1)
    return {**base, "stop_reason": training["stop_reason"], "training": training,
            "validation_rms_residual": validation}


def choose_scale(results: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[float, list[dict[str, Any]]] = {scale: [] for scale in SCALE_CANDIDATES}
    for result in results:
        grouped[float(result["feature_scale"])].append(result)
    candidates = []
    for scale in SCALE_CANDIDATES:
        rows = grouped[scale]
        eligible = (len(rows) == len(CALIBRATION_SEEDS)
                    and all(row["rank"]["pass"]
                            and row["validation_rms_residual"] is not None for row in rows))
        median = (float(np.median([row["validation_rms_residual"] for row in rows]))
                  if eligible else None)
        candidates.append({"feature_scale": scale, "eligible": eligible,
                           "median_validation_rms_residual": median})
    eligible = [row for row in candidates if row["eligible"]]
    if not eligible:
        return {"status": "conditioning_failure", "selected_scale": None,
                "candidates": candidates}
    best = min(row["median_validation_rms_residual"] for row in eligible)
    tie_tolerance = 1e-6 * max(1.0, best)
    selected = min(row["feature_scale"] for row in eligible
                   if row["median_validation_rms_residual"] <= best + tie_tolerance)
    return {"status": "selected", "selected_scale": selected,
            "best_median": best, "tie_tolerance": tie_tolerance,
            "candidates": candidates}
