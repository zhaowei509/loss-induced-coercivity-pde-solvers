"""Experiment 2 residual and Ritz objectives; formulas adapted from Stage 4."""

from dataclasses import dataclass

import torch

from .features import require_double


def cubic_solution(x):
    require_double(x)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("Cubic manufactured solution needs (n,2) points.")
    return torch.sin(torch.pi * x[:, 0]) * torch.sin(torch.pi * x[:, 1])


def cubic_source(x):
    u = cubic_solution(x)
    return 2 * torch.pi**2 * u + u**3


def p4_source(x):
    require_double(x)
    if x.ndim != 2 or x.shape[1] != 1:
        raise ValueError("p4 source needs (n,1) points.")
    return 3 * torch.pi**4 * torch.cos(torch.pi * x[:, 0])**2 * torch.sin(torch.pi * x[:, 0])


@dataclass
class CubicResidual:
    values: torch.Tensor
    laplacian: torch.Tensor
    source: torch.Tensor

    def __post_init__(self):
        require_double(self.values, self.laplacian, self.source)
        if (self.values.ndim != 2 or self.laplacian.shape != self.values.shape
                or self.source.shape != (self.values.shape[0],)):
            raise ValueError("Incompatible cubic residual arrays.")

    def residual(self, a):
        require_double(a, self.values)
        if a.shape != (self.values.shape[1],):
            raise ValueError("Incorrect coefficient shape.")
        u = self.values @ a
        return -self.laplacian @ a + u**3 - self.source

    def __call__(self, a):
        r = self.residual(a)
        return 0.5 * r.square().mean()

    def analytic_value(self, a):
        return float(self(a))

    def analytic_value_gradient(self, a):
        u = self.values @ a
        r = -self.laplacian @ a + u**3 - self.source
        value = float(0.5 * r.square().mean())
        gradient = (-self.laplacian.T @ r
                    + self.values.T @ (3 * u.square() * r)) / self.values.shape[0]
        return value, gradient


@dataclass
class P4Ritz:
    values: torch.Tensor
    derivative: torch.Tensor
    source: torch.Tensor
    weights: torch.Tensor

    def __post_init__(self):
        require_double(self.values, self.derivative, self.source, self.weights)
        n, m = self.values.shape
        if (self.derivative.shape != (n, m) or self.source.shape != (n,)
                or self.weights.shape != (n,) or bool((self.weights <= 0).any())):
            raise ValueError("Incompatible p4 Ritz arrays.")

    def __call__(self, a):
        du, u = self.derivative @ a, self.values @ a
        return torch.dot(self.weights, 0.25 * du**4 - self.source * u)

    def analytic_value(self, a):
        return float(self(a))

    def analytic_value_gradient(self, a):
        du, u = self.derivative @ a, self.values @ a
        value = float(torch.dot(self.weights, 0.25 * du**4 - self.source * u))
        gradient = self.derivative.T @ (self.weights * du**3)
        gradient -= self.values.T @ (self.weights * self.source)
        return value, gradient


@dataclass
class AllenCahnRitz:
    values: torch.Tensor
    derivative: torch.Tensor
    weights: torch.Tensor
    epsilon: float = 0.1

    def __post_init__(self):
        require_double(self.values, self.derivative, self.weights)
        n, m = self.values.shape
        if (self.derivative.shape != (n, m) or self.weights.shape != (n,)
                or bool((self.weights <= 0).any()) or self.epsilon <= 0):
            raise ValueError("Incompatible Allen-Cahn arrays.")

    def __call__(self, a):
        du, u = self.derivative @ a, self.values @ a
        return torch.dot(self.weights,
                         0.5 * self.epsilon**2 * du.square() + 0.25 * (u.square() - 1).square())

    def analytic_value(self, a):
        return float(self(a))

    def analytic_value_gradient(self, a):
        du, u = self.derivative @ a, self.values @ a
        value = float(torch.dot(self.weights,
                                0.5 * self.epsilon**2 * du.square()
                                + 0.25 * (u.square() - 1).square()))
        gradient = self.derivative.T @ (self.weights * self.epsilon**2 * du)
        gradient += self.values.T @ (self.weights * (u.square() - 1) * u)
        return value, gradient
