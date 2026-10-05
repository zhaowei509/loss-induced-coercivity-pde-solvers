"""Controlled cubic PDE and exact autodiff derivatives for Experiment 4."""

from __future__ import annotations

import numpy as np
import torch


DTYPE = torch.float64
RHO = 1e-3
A_REFERENCE = np.array([1.0, -0.8], dtype=np.float64)
DELTA = 1e-5
INTERIOR = np.array([[0.3, 0.35], [0.7, 0.65]], dtype=np.float64)
BOUNDARY = np.array(
    [[0.5, 0.0], [1.0, 0.5], [0.5, 1.0], [0.0, 0.5]],
    dtype=np.float64,
)


def fixed_features():
    rng = np.random.Generator(np.random.PCG64(9))
    return 1.5 * rng.normal(size=(2, 2)), rng.normal(size=2)


def rank_diagnostic(matrix):
    singular = np.linalg.svd(matrix, compute_uv=False)
    threshold = max(matrix.shape) * np.finfo(np.float64).eps * singular[0]
    return {
        "shape": list(matrix.shape),
        "rank": int(np.sum(singular > threshold)),
        "sigma_min": float(singular[-1]),
        "sigma_max": float(singular[0]),
        "rank_threshold": float(threshold),
    }


class ControlledProblem:
    def __init__(self):
        w, b = fixed_features()
        self.w = w
        self.b = b
        self.x = torch.tensor(INTERIOR, dtype=DTYPE)
        self.y = torch.tensor(BOUNDARY, dtype=DTYPE)
        self.w_fixed = torch.tensor(w.reshape(-1), dtype=DTYPE)
        self.b2_fixed = torch.tensor([b[1]], dtype=DTYPE)
        reference = torch.tensor(
            np.concatenate((A_REFERENCE, w.reshape(-1), b)), dtype=DTYPE
        )
        ref_u, ref_lap = self._field(reference, self.x)
        self.f = (
            -ref_lap
            + ref_u**3
            + DELTA * torch.sin(torch.pi * self.x[:, 0])
            * torch.sin(torch.pi * self.x[:, 1])
        ).detach()
        self.g = self._field(reference, self.y)[0].detach()

    @staticmethod
    def _field(theta, locations):
        a = theta[:2]
        w = theta[2:6].reshape(2, 2)
        b = theta[6:8]
        tanh = torch.tanh(locations @ w.T + b)
        lap_basis = -2 * tanh * (1 - tanh**2) * torch.sum(w**2, dim=1)
        return tanh @ a, lap_basis @ a

    def full_theta(self, active, branch):
        if branch == "output":
            return torch.cat((active, self.w_fixed, torch.tensor(self.b, dtype=DTYPE)))
        if branch == "column":
            return torch.cat((active[:2], self.w_fixed, active[2:3], self.b2_fixed))
        if branch == "row":
            return active
        raise ValueError(f"Unknown branch: {branch}")

    def starting_vector(self, a, branch):
        if branch == "output":
            return np.asarray(a, dtype=np.float64).copy()
        if branch == "column":
            return np.concatenate((a, self.b[:1]))
        if branch == "row":
            return np.concatenate((a, self.w.reshape(-1), self.b))
        raise ValueError(branch)

    def residual_tensor(self, active, branch):
        theta = self.full_theta(active, branch)
        u, lap = self._field(theta, self.x)
        boundary_u = self._field(theta, self.y)[0]
        return torch.cat((
            (-lap + u**3 - self.f) / np.sqrt(2.0),
            np.sqrt(2.0 / 4.0) * (boundary_u - self.g),
        ))

    def residual(self, vector, branch):
        active = torch.tensor(vector, dtype=DTYPE)
        return self.residual_tensor(active, branch).detach().numpy()

    def value(self, vector, branch):
        active = torch.tensor(vector, dtype=DTYPE)
        residual = self.residual_tensor(active, branch)
        return float((0.5 * residual @ residual).detach())

    def gradient(self, vector, branch):
        active = torch.tensor(vector, dtype=DTYPE, requires_grad=True)
        residual = self.residual_tensor(active, branch)
        objective = 0.5 * residual @ residual
        return torch.autograd.grad(objective, active)[0].detach().numpy()

    def hessian(self, vector, branch):
        active = torch.tensor(vector, dtype=DTYPE)
        hessian = torch.autograd.functional.hessian(
            lambda z: 0.5 * self.residual_tensor(z, branch).square().sum(),
            active,
        )
        return hessian.detach().numpy()

    def jacobian(self, vector, branch):
        active = torch.tensor(vector, dtype=DTYPE)
        jacobian = torch.autograd.functional.jacobian(
            lambda z: self.residual_tensor(z, branch), active
        )
        return jacobian.detach().numpy()

    def boundary_matrix(self):
        theta = torch.tensor(
            np.concatenate((A_REFERENCE, self.w.reshape(-1), self.b)),
            dtype=DTYPE,
        )
        w = theta[2:6].reshape(2, 2)
        b = theta[6:8]
        return torch.tanh(self.y @ w.T + b).numpy()
