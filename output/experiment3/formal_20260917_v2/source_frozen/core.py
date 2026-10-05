"""Frozen features, quadrature, and analytic coefficient derivatives."""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import roots_legendre


DT = np.float64
STAR_1D = np.array([1.0, 1.0], dtype=DT)
W_1D = np.array([[2.0], [3.0]], dtype=DT)
B_1D = np.array([1.0, -1.0], dtype=DT)
DELTAS_1D = (0.0, 0.05, 0.2, 0.5)
RADII = (0.05, 0.1, 0.2, 0.4)
SEEDS_2D = tuple(range(5))
EPSILON_2D = 0.1


def quadrature(order, dimension):
    if order < 1 or dimension not in (1, 2):
        raise ValueError("Expected positive order and dimension 1 or 2.")
    nodes, weights = roots_legendre(order)
    nodes, weights = (nodes + 1) / 2, weights / 2
    grids = np.meshgrid(*([nodes] * dimension), indexing="ij")
    weight_grids = np.meshgrid(*([weights] * dimension), indexing="ij")
    points = np.stack([grid.ravel() for grid in grids], axis=1)
    product = np.prod(np.stack(weight_grids), axis=0).ravel()
    return points.astype(DT), product.astype(DT)


def boundary_2d():
    mid = (np.arange(8, dtype=DT) + 0.5) / 8
    zeros, ones = np.zeros(8), np.ones(8)
    return np.vstack((np.column_stack((mid, zeros)),
                      np.column_stack((ones, mid)),
                      np.column_stack((1 - mid, ones)),
                      np.column_stack((zeros, 1 - mid))))


def feature_matrices(points, w, b):
    """Return tanh values and spatial Laplacians, both shape (n, m)."""
    z = np.asarray(points, dtype=DT) @ w.T + b
    phi = np.tanh(z)
    laplacian = -2 * phi * (1 - phi**2) * np.sum(w**2, axis=1)
    return phi, laplacian


def random_features(seed):
    rng = np.random.Generator(np.random.PCG64(seed))
    return rng.normal(size=(6, 2)).astype(DT), rng.normal(size=6).astype(DT)


@dataclass
class LeastSquares:
    phi: np.ndarray
    operator_phi: np.ndarray
    source: np.ndarray
    weights: np.ndarray
    boundary: np.ndarray
    boundary_target: np.ndarray
    boundary_weight: float
    nonlinear_coefficient: float = 1.0

    def __post_init__(self):
        n, m = self.phi.shape
        if (self.operator_phi.shape != (n, m) or self.source.shape != (n,)
                or self.weights.shape != (n,) or self.boundary.shape[1] != m
                or self.boundary_target.shape != (self.boundary.shape[0],)
                or not np.all(self.weights > 0) or self.boundary_weight <= 0
                or self.nonlinear_coefficient <= 0):
            raise ValueError("Inconsistent least-squares dimensions or weights.")

    def residual_parts(self, a):
        a = np.asarray(a, dtype=DT)
        u = self.phi @ a
        r = self.operator_phi @ a + self.nonlinear_coefficient * u**2 - self.source
        dr = self.operator_phi + (2 * self.nonlinear_coefficient * u)[:, None] * self.phi
        boundary_residual = self.boundary @ a - self.boundary_target
        return r, dr, boundary_residual

    def residual(self, a):
        r, _, br = self.residual_parts(a)
        return np.concatenate((np.sqrt(self.weights) * r,
                               np.sqrt(2 * self.boundary_weight) * br))

    def value_gradient_hessian(self, a):
        r, dr, br = self.residual_parts(a)
        wd = self.weights[:, None] * dr
        value = (0.5 * np.dot(self.weights, r**2)
                 + self.boundary_weight * np.dot(br, br))
        grad = (dr.T @ (self.weights * r)
                + 2 * self.boundary_weight * self.boundary.T @ br)
        gn = dr.T @ wd + 2 * self.boundary_weight * self.boundary.T @ self.boundary
        remainder = (2 * self.nonlinear_coefficient
                     * self.phi.T @ ((self.weights * r)[:, None] * self.phi))
        hessian = gn + remainder
        return float(value), grad, hessian, gn, remainder

    def fun(self, a):
        return self.value_gradient_hessian(a)[0]

    def jac(self, a):
        return self.value_gradient_hessian(a)[1]

    def hess(self, a):
        return self.value_gradient_hessian(a)[2]

    def diagnostic(self, a):
        value, grad, h, gn, rem = self.value_gradient_hessian(a)
        gn_min = float(np.linalg.eigvalsh(gn)[0])
        rem_min = float(np.linalg.eigvalsh(rem)[0])
        kappa = max(0.0, -rem_min)
        singular = np.linalg.svd(self.boundary, compute_uv=False)
        sigma_min, sigma_max = float(singular[-1]), float(singular[0])
        rank_threshold = max(self.boundary.shape) * np.finfo(DT).eps * sigma_max
        return dict(objective=value, residual_norm=float(np.linalg.norm(self.residual(a))),
                    gradient_norm=float(np.linalg.norm(grad)),
                    sigma_min_boundary=sigma_min, rank_threshold=rank_threshold,
                    boundary_full_column_rank=bool(sigma_min > rank_threshold),
                    mu_B=float(2 * self.boundary_weight * sigma_min**2),
                    mu_GN=gn_min, kappa_rem=kappa,
                    delta_curv=gn_min - kappa,
                    lambda_min_hessian=float(np.linalg.eigvalsh(h)[0]))


def one_dimensional(delta, order):
    points, weights = quadrature(order, 1)
    phi, lap = feature_matrices(points, W_1D, B_1D)
    star_u = phi @ STAR_1D
    source = lap @ STAR_1D + star_u**2 + delta * np.sin(5 * np.pi * points[:, 0])
    boundary = feature_matrices(np.array([[0.0], [1.0]]), W_1D, B_1D)[0]
    return LeastSquares(phi, lap, source, weights, boundary,
                        boundary @ STAR_1D, 1 / 2)


def two_dimensional_preconditioner(seed):
    points, weights = quadrature(24, 2)
    w, b = random_features(seed)
    _, lap = feature_matrices(points, w, b)
    boundary = feature_matrices(boundary_2d(), w, b)[0]
    linearized = np.vstack((np.sqrt(weights)[:, None] * (-lap),
                            np.sqrt(2 / 32) * boundary))
    spectra = {}
    for name, matrix in (("raw_boundary", boundary), ("linearized", linearized)):
        values = np.linalg.svd(matrix, compute_uv=False)
        threshold = max(matrix.shape) * np.finfo(DT).eps * values[0]
        spectra[name] = dict(sigma_min=float(values[-1]),
                             sigma_max=float(values[0]),
                             rank_threshold=float(threshold),
                             full_column_rank=bool(values[-1] > threshold))
    if not all(row["full_column_rank"] for row in spectra.values()):
        return None, spectra
    _, transform = np.linalg.qr(linearized, mode="reduced")
    signs = np.where(np.diag(transform) < 0, -1.0, 1.0)
    transform = signs[:, None] * transform
    transformed_boundary = solve_triangular(transform.T, boundary.T, lower=True).T
    values = np.linalg.svd(transformed_boundary, compute_uv=False)
    threshold = max(transformed_boundary.shape) * np.finfo(DT).eps * values[0]
    spectra["normalized_boundary"] = dict(
        sigma_min=float(values[-1]), sigma_max=float(values[0]),
        rank_threshold=float(threshold),
        full_column_rank=bool(values[-1] > threshold))
    if not spectra["normalized_boundary"]["full_column_rank"]:
        return None, spectra
    return transform, spectra


def two_dimensional_raw(seed, order):
    points, weights = quadrature(order, 2)
    w, b = random_features(seed)
    phi, lap = feature_matrices(points, w, b)
    h = np.sin(np.pi * points[:, 0]) * np.sin(np.pi * points[:, 1])
    source = 2 * np.pi**2 * h + EPSILON_2D * h**2
    boundary = feature_matrices(boundary_2d(), w, b)[0]
    return LeastSquares(phi, -lap,
                        source, weights, boundary,
                        np.zeros(32, dtype=DT), 1 / 32, EPSILON_2D)


def two_dimensional(seed, order, transform):
    if transform is None or transform.shape != (6, 6):
        raise ValueError("A valid frozen 6-by-6 preconditioner is required.")
    raw = two_dimensional_raw(seed, order)
    normalized_phi = solve_triangular(transform.T, raw.phi.T, lower=True).T
    normalized_operator = solve_triangular(transform.T, raw.operator_phi.T,
                                           lower=True).T
    normalized_boundary = solve_triangular(transform.T, raw.boundary.T,
                                           lower=True).T
    return LeastSquares(normalized_phi, normalized_operator, raw.source,
                        raw.weights, normalized_boundary,
                        raw.boundary_target, raw.boundary_weight,
                        raw.nonlinear_coefficient)
