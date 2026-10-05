"""Fast, non-campaign checks for the Experiment 3 implementation."""

import numpy as np

from .core import (EPSILON_2D, STAR_1D, W_1D, B_1D, boundary_2d,
                   feature_matrices, one_dimensional, quadrature,
                   random_features, two_dimensional, two_dimensional_preconditioner,
                   two_dimensional_raw)


def _relative(error, scale):
    return float(np.max(np.abs(error)) / max(1.0, float(np.max(np.abs(scale)))))


def _check_derivatives(objective, a):
    eps = 2e-6
    _, grad, hess, gn, remainder = objective.value_gradient_hessian(a)
    gradient_fd = np.empty_like(grad)
    hessian_fd = np.empty_like(hess)
    for j in range(len(a)):
        step = np.zeros_like(a)
        step[j] = eps
        gradient_fd[j] = (objective.fun(a + step) - objective.fun(a - step)) / (2 * eps)
        hessian_fd[:, j] = (objective.jac(a + step) - objective.jac(a - step)) / (2 * eps)
    grad_error = _relative(gradient_fd - grad, grad)
    hessian_error = _relative(hessian_fd - hess, hess)
    identity_error = _relative(hess - gn - remainder, hess)
    if grad_error > 1e-7 or hessian_error > 1e-7 or identity_error > 1e-12:
        raise AssertionError((grad_error, hessian_error, identity_error))
    if abs(objective.fun(a) - 0.5 * np.linalg.norm(objective.residual(a))**2) > 1e-10:
        raise AssertionError("Residual stacking and objective disagree.")
    return dict(gradient_relative_error=grad_error,
                hessian_relative_error=hessian_error,
                identity_relative_error=identity_error)


def _check_laplacian(w, b, x):
    eps = 1e-4
    _, analytic = feature_matrices(x, w, b)
    center = feature_matrices(x, w, b)[0]
    finite = np.zeros_like(center)
    for d in range(x.shape[1]):
        step = np.zeros_like(x)
        step[:, d] = eps
        finite += (feature_matrices(x + step, w, b)[0] - 2 * center
                   + feature_matrices(x - step, w, b)[0]) / eps**2
    error = _relative(finite - analytic, analytic)
    if error > 2e-7:
        raise AssertionError(f"Spatial Laplacian relative error: {error}")
    return error


def run_checks():
    one = one_dimensional(0.2, 256)
    one_refined = one_dimensional(0.2, 512)
    zero = one_dimensional(0.0, 256)
    if np.linalg.norm(zero.residual(STAR_1D)) > 1e-12:
        raise AssertionError("Manufactured one-dimensional point is not zero residual.")
    expected_boundary = np.array([[np.tanh(1), -np.tanh(1)],
                                  [np.tanh(3), np.tanh(2)]])
    if not np.allclose(one.boundary, expected_boundary, atol=1e-15):
        raise AssertionError("One-dimensional boundary matrix is incorrect.")
    if np.linalg.det(one.boundary) <= 0:
        raise AssertionError("One-dimensional boundary matrix is singular.")
    one_result = _check_derivatives(one, np.array([0.8, 1.15]))
    if abs(one.fun(STAR_1D) - one_refined.fun(STAR_1D)) > 1e-12:
        raise AssertionError("One-dimensional quadrature changed the manufactured residual.")
    points = boundary_2d()
    if len(points) != 32 or len(np.unique(points, axis=0)) != 32:
        raise AssertionError("Expected 32 distinct boundary midpoints.")
    if not np.all(np.any(np.isclose(points, 0) | np.isclose(points, 1), axis=1)):
        raise AssertionError("A boundary point lies in the interior.")
    check_seed = 0
    transform, rank_checks = two_dimensional_preconditioner(check_seed)
    if transform is None or not all(row["full_column_rank"] for row in rank_checks.values()):
        raise AssertionError("Check-seed feature matrices must be numerically full rank.")
    two = two_dimensional(check_seed, 24, transform)
    two_refined = two_dimensional(check_seed, 48, transform)
    _, _, _, gn_zero, _ = two.value_gradient_hessian(np.zeros(6))
    if not np.allclose(gn_zero, np.eye(6), atol=1e-10, rtol=1e-10):
        raise AssertionError("Training GN matrix at c=0 must be the identity.")
    if not np.array_equal(two.boundary, two_refined.boundary):
        raise AssertionError("Refinement must reuse the training preconditioner.")
    quad_points, _ = quadrature(24, 2)
    h = np.sin(np.pi * quad_points[:, 0]) * np.sin(np.pi * quad_points[:, 1])
    expected_source = 2 * np.pi**2 * h + EPSILON_2D * h**2
    if not np.allclose(two.source, expected_source, atol=1e-14):
        raise AssertionError("Nonlinear manufactured source is incorrect.")
    two_result = _check_derivatives(two, np.array([0.2, -0.1, 0.3, 0.0, 0.1, -0.2]))
    raw = two_dimensional_raw(check_seed, 24)
    w, b = random_features(check_seed)
    physical_phi = feature_matrices(quad_points, w, b)[0]
    if not np.allclose(raw.phi, physical_phi, atol=0, rtol=0):
        raise AssertionError("Raw coefficients must multiply the actual trial features.")
    if raw.nonlinear_coefficient != EPSILON_2D:
        raise AssertionError("The two-dimensional nonlinearity coefficient changed.")
    c = np.array([0.2, -0.1, 0.3, 0.0, 0.1, -0.2])
    a = np.linalg.solve(transform, c)
    if not np.allclose(raw.residual(a), two.residual(c), atol=1e-11, rtol=1e-11):
        raise AssertionError("Raw and normalized residuals disagree.")
    if not np.allclose(raw.jac(a), transform.T @ two.jac(c), atol=1e-10):
        raise AssertionError("Raw and normalized gradients disagree.")
    if not np.allclose(raw.hess(a), transform.T @ two.hess(c) @ transform,
                       atol=1e-9, rtol=1e-9):
        raise AssertionError("Raw and normalized Hessians disagree.")
    return dict(one_dimensional=one_result, two_dimensional=two_result,
                raw_coordinate_derivatives=_check_derivatives(raw, a),
                zero_state_gauss_newton_identity_error=float(
                    np.max(np.abs(gn_zero - np.eye(6)))),
                one_dimensional_laplacian_error=_check_laplacian(
                    W_1D, B_1D, np.array([[0.27], [0.62]])),
                two_dimensional_laplacian_error=_check_laplacian(
                    w, b, np.array([[0.23, 0.41], [0.66, 0.38]])),
                zero_residual_norm=float(np.linalg.norm(zero.residual(STAR_1D))),
                boundary_count=len(points))
