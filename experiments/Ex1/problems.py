"""Frozen least-squares problems for Experiment 1."""

from dataclasses import dataclass
from typing import Callable

import torch

from .features import DTYPE, FeatureJet, RandomFeatures, require_double


TOY_COEFFICIENTS = torch.tensor([1.0, -0.75], dtype=DTYPE)
TOY_WEIGHTS = torch.tensor([[2.0, 1.0], [-1.0, 2.0]], dtype=DTYPE)
TOY_BIASES = torch.tensor([-0.4, 0.3], dtype=DTYPE)


def checked_weights(weights, count):
    require_double(weights)
    if (weights.shape != (count,) or not bool(torch.isfinite(weights).all())
            or bool((weights <= 0).any())):
        raise ValueError("Positive finite weights of shape (n,) required.")
    return weights


def weighted_residual(*blocks):
    """Stack residual blocks after multiplying each entry by sqrt(weight)."""
    scaled = []
    for residual, weights in blocks:
        require_double(residual, weights)
        if residual.ndim != 1:
            raise ValueError("Residuals must be vectors.")
        scaled.append(checked_weights(weights, residual.numel()).sqrt() * residual)
    if not scaled:
        raise ValueError("At least one residual block required.")
    return torch.cat(scaled)


def toy_features(*, device="cpu"):
    return RandomFeatures(TOY_WEIGHTS.to(device), TOY_BIASES.to(device))


def toy_coefficients(*, device="cpu"):
    return TOY_COEFFICIENTS.to(device)


def initial_lateral_points(initial_count=8, lateral_count=16, *, device="cpu"):
    """Midpoints on t=0 and equally split x=0,1 lateral boundaries."""
    if initial_count < 1 or lateral_count < 2 or lateral_count % 2:
        raise ValueError("Positive initial count and even lateral count required.")
    x = (torch.arange(initial_count, dtype=DTYPE, device=device) + 0.5) / initial_count
    initial = torch.stack((x, torch.zeros_like(x)), dim=1)
    half = lateral_count // 2
    t = (torch.arange(half, dtype=DTYPE, device=device) + 0.5) / half
    left = torch.stack((torch.zeros_like(t), t), dim=1)
    right = torch.stack((torch.ones_like(t), t), dim=1)
    return torch.cat((initial, left, right), dim=0)


def toy_operator(jet, coefficients, kind, *, epsilon=0.2):
    """Evaluate the A1--A3 differential operator at supplied coefficients."""
    require_double(jet.values, jet.gradient, jet.second_diagonal, coefficients)
    u = jet.values @ coefficients
    if kind == "A1":
        return -jet.laplacian @ coefficients + u
    if kind == "A2":
        u_t = jet.gradient[:, :, 1] @ coefficients
        u_xx = jet.second_diagonal[:, :, 0] @ coefficients
        return u_t - epsilon**2 * u_xx + u**3 - u
    if kind == "A3":
        return -jet.laplacian @ coefficients + u**3
    raise ValueError(f"Unknown toy problem {kind!r}.")


def toy_objective(kind, interior_jet, interior_weights, boundary_matrix,
                  boundary_target, source, *, boundary_penalty=1.0,
                  epsilon=0.2):
    """Construct an A-class objective with manufactured source and data."""
    require_double(interior_jet.values, interior_weights, boundary_matrix,
                   boundary_target, source)
    n, m = interior_jet.values.shape
    if (source.shape != (n,) or boundary_matrix.ndim != 2
            or boundary_matrix.shape[1] != m
            or boundary_target.shape != (boundary_matrix.shape[0],)
            or boundary_penalty <= 0):
        raise ValueError("Incompatible toy-problem data.")
    interior_scale = checked_weights(interior_weights, n).sqrt()
    boundary_scale = (2 * boundary_penalty / boundary_matrix.shape[0])**0.5

    def residual(a):
        require_double(a, interior_jet.values)
        if a.shape != (m,):
            raise ValueError("Incorrect coefficient vector shape.")
        interior = interior_scale * (
            toy_operator(interior_jet, a, kind, epsilon=epsilon) - source)
        boundary = boundary_scale * (boundary_matrix @ a - boundary_target)
        return torch.cat((interior, boundary))

    return LeastSquares(residual)


@dataclass
class LeastSquares:
    residual: Callable[[torch.Tensor], torch.Tensor]

    def __call__(self, parameters):
        r = self.residual(parameters)
        require_double(parameters, r)
        if r.ndim != 1:
            raise ValueError("Least-squares residual must be a vector.")
        return 0.5 * torch.dot(r, r)


def cubic_solution(x):
    require_double(x)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("The cubic problem requires points of shape (n,2).")
    return torch.sin(torch.pi * x[:, 0]) * torch.sin(torch.pi * x[:, 1])


def cubic_source(x):
    u = cubic_solution(x)
    return 2 * torch.pi**2 * u + u**3


@dataclass
class CubicLeastSquares:
    """Cubic residual with an algebraically exact, allocation-light gradient."""

    values: torch.Tensor
    negative_laplacian: torch.Tensor
    source: torch.Tensor
    interior_weights: torch.Tensor
    boundary_matrix: torch.Tensor
    boundary_penalty: float

    def raw_blocks(self, a):
        u = self.values @ a
        interior = self.negative_laplacian @ a + u**3 - self.source
        boundary = self.boundary_matrix @ a
        return u, interior, boundary

    def residual(self, a):
        require_double(a, self.values)
        _, interior, boundary = self.raw_blocks(a)
        boundary_scale = (2 * self.boundary_penalty
                          / self.boundary_matrix.shape[0])**0.5
        return torch.cat((self.interior_weights.sqrt() * interior,
                          boundary_scale * boundary))

    def __call__(self, a):
        residual = self.residual(a)
        return 0.5 * torch.dot(residual, residual)

    def analytic_value(self, a):
        _, interior, boundary = self.raw_blocks(a)
        return (0.5 * torch.dot(self.interior_weights, interior.square())
                + self.boundary_penalty / self.boundary_matrix.shape[0]
                * torch.dot(boundary, boundary))

    def analytic_value_gradient(self, a):
        u, interior, boundary = self.raw_blocks(a)
        jacobian = self.negative_laplacian + 3 * u.square()[:, None] * self.values
        gradient = jacobian.T @ (self.interior_weights * interior)
        gradient = gradient + (2 * self.boundary_penalty
                               / self.boundary_matrix.shape[0]
                               * (self.boundary_matrix.T @ boundary))
        return self.analytic_value(a), gradient


def cubic_objective(jet, source, interior_weights, boundary_matrix, *,
                    boundary_penalty=10.0):
    """Return the frozen B-class objective from EXPERIMENT1_PLAN.md."""
    require_double(jet.values, jet.laplacian, source, interior_weights, boundary_matrix)
    n, m = jet.values.shape
    if (source.shape != (n,) or jet.laplacian.shape != (n, m)
            or boundary_matrix.ndim != 2 or boundary_matrix.shape[1] != m
            or boundary_matrix.shape[0] == 0 or boundary_penalty <= 0):
        raise ValueError("Incompatible cubic-problem data.")
    checked_weights(interior_weights, n)
    return CubicLeastSquares(jet.values, -jet.laplacian, source,
                             interior_weights, boundary_matrix,
                             boundary_penalty)
