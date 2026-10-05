"""Automatic differentiation, numerical rank, and confinement diagnostics."""

from dataclasses import asdict, dataclass
import math

import numpy as np
import torch

from .features import require_double


@dataclass
class EvaluationCounts:
    loss_evaluations: int = 0
    gradient_evaluations: int = 0
    hessian_evaluations: int = 0
    residual_evaluations: int = 0


class Objective:
    """Float64 objective wrapper with explicit evaluation accounting."""

    def __init__(self, function):
        self.function = function
        self.residual_function = getattr(function, "residual", None)
        self.counts = EvaluationCounts()

    def graph_value(self, x):
        require_double(x)
        if x.ndim != 1:
            raise ValueError("Parameter vectors must be one-dimensional.")
        self.counts.loss_evaluations += 1
        if self.residual_function is not None:
            self.counts.residual_evaluations += 1
        value = self.function(x)
        require_double(x, value)
        if value.ndim != 0:
            raise ValueError("Objective must return a scalar.")
        return value

    def value(self, x):
        analytic = getattr(self.function, "analytic_value", None)
        if analytic is not None:
            require_double(x)
            self.counts.loss_evaluations += 1
            self.counts.residual_evaluations += 1
            return float(analytic(x.detach()))
        with torch.enable_grad():
            return float(self.graph_value(x.detach()).detach())

    def value_gradient(self, x):
        analytic = getattr(self.function, "analytic_value_gradient", None)
        if analytic is not None:
            require_double(x)
            self.counts.loss_evaluations += 1
            self.counts.gradient_evaluations += 1
            self.counts.residual_evaluations += 1
            value, gradient = analytic(x.detach())
            require_double(x, gradient)
            return float(value), gradient.detach()
        with torch.enable_grad():
            z = x.detach().clone().requires_grad_(True)
            value = self.graph_value(z)
            self.counts.gradient_evaluations += 1
            gradient = torch.autograd.grad(value, z, allow_unused=True)[0]
            if gradient is None:
                gradient = torch.zeros_like(z)
            return float(value.detach()), gradient.detach()

    def hessian(self, x):
        require_double(x)
        self.counts.hessian_evaluations += 1
        with torch.enable_grad():
            return torch.autograd.functional.hessian(self.graph_value, x.detach()).detach()

    def snapshot(self):
        return asdict(self.counts)


class ScipyAdapter:
    """CPU/NumPy bridge used by the DOP853 integrator."""

    def __init__(self, objective, reference):
        self.objective = objective
        self.reference = reference

    def tensor(self, value):
        array = np.asarray(value).copy()
        return torch.as_tensor(array, dtype=torch.float64, device=self.reference.device)


@dataclass
class Spectrum:
    singular_values: torch.Tensor
    numerical_rank: int
    threshold: float
    sigma_min: float
    full_column_rank: bool


def spectrum(matrix):
    """SVD rank using the frozen machine-precision threshold."""
    require_double(matrix)
    if matrix.ndim != 2 or not bool(torch.isfinite(matrix).all()):
        raise ValueError("A finite matrix is required.")
    values = torch.linalg.svdvals(matrix.detach().cpu())
    if values.numel() == 0:
        return Spectrum(values, 0, 0.0, 0.0, matrix.shape[1] == 0)
    threshold = max(matrix.shape) * torch.finfo(matrix.dtype).eps * float(values[0])
    rank = int((values > threshold).sum())
    sigma_min = float(values[-1])
    return Spectrum(values, rank, threshold, sigma_min, rank == matrix.shape[1])


def confinement_radius(initial_objective, boundary_matrix, boundary_penalty, *,
                       boundary_target_norm=0.0, interior_lower_bound=0.0):
    """Theorem 3.2 radius for a numerically full-column-rank boundary matrix."""
    diagnostic = spectrum(boundary_matrix)
    if not diagnostic.full_column_rank:
        return None
    if (initial_objective < 0 or boundary_penalty <= 0 or boundary_target_norm < 0
            or interior_lower_bound < 0):
        raise ValueError("Invalid confinement-radius inputs.")
    sigma_squared = diagnostic.sigma_min**2
    radius_squared = (2 * boundary_matrix.shape[0]
                      * (initial_objective + interior_lower_bound)
                      / (boundary_penalty * sigma_squared)
                      + 2 * boundary_target_norm**2 / sigma_squared)
    return math.sqrt(radius_squared)
