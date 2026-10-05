"""Outward-rounded interval automatic differentiation for S2 certificates.

The validated object is the stored float64 empirical problem. Algebraic operations
are enlarged by one representable number in each direction; tanh endpoints and
small-matrix singular values are evaluated with 100-decimal mpmath arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Callable, Iterable

import mpmath as mp
import numpy as np


mp.mp.dps = 100


def _down(value):
    return np.nextafter(np.asarray(value, dtype=np.float64), -np.inf)


def _up(value):
    return np.nextafter(np.asarray(value, dtype=np.float64), np.inf)


def _mp_norm_upper(values) -> float:
    flattened = np.asarray(values, dtype=np.float64).ravel()
    exact = mp.sqrt(sum(mp.mpf(float(value))**2 for value in flattened))
    return float(np.nextafter(float(exact), np.inf))


@dataclass(frozen=True)
class Interval:
    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self):
        if np.any(self.lower > self.upper):
            raise ValueError("Invalid interval endpoints.")

    @classmethod
    def point(cls, value) -> "Interval":
        array = np.asarray(value, dtype=np.float64)
        return cls(array.copy(), array.copy())

    @classmethod
    def bounds(cls, lower, upper) -> "Interval":
        return cls(np.asarray(lower, dtype=np.float64),
                   np.asarray(upper, dtype=np.float64))

    def __add__(self, other) -> "Interval":
        other = as_interval(other)
        return Interval(_down(self.lower + other.lower),
                        _up(self.upper + other.upper))

    __radd__ = __add__

    def __neg__(self) -> "Interval":
        return Interval(_down(-self.upper), _up(-self.lower))

    def __sub__(self, other) -> "Interval":
        return self + (-as_interval(other))

    def __rsub__(self, other) -> "Interval":
        return as_interval(other) - self

    def __mul__(self, other) -> "Interval":
        other = as_interval(other)
        products = np.stack((self.lower * other.lower, self.lower * other.upper,
                             self.upper * other.lower, self.upper * other.upper))
        return Interval(_down(np.min(products, axis=0)),
                        _up(np.max(products, axis=0)))

    __rmul__ = __mul__

    def __truediv__(self, scalar) -> "Interval":
        return self * (1.0 / float(scalar))

    def square(self) -> "Interval":
        low_abs = np.minimum(np.abs(self.lower), np.abs(self.upper))
        low_abs = np.where((self.lower <= 0) & (self.upper >= 0), 0.0, low_abs)
        high_abs = np.maximum(np.abs(self.lower), np.abs(self.upper))
        return Interval(_down(low_abs * low_abs), _up(high_abs * high_abs))

    def cube(self) -> "Interval":
        return Interval(_down(self.lower**3), _up(self.upper**3))

    def max_abs(self) -> np.ndarray:
        return _up(np.maximum(np.abs(self.lower), np.abs(self.upper)))


def as_interval(value) -> Interval:
    return value if isinstance(value, Interval) else Interval.point(value)


def interval_outer(left: Interval, right: Interval) -> Interval:
    return Interval.bounds(left.lower[:, None], left.upper[:, None]) * Interval.bounds(
        right.lower[None, :], right.upper[None, :])


@dataclass(frozen=True)
class Jet:
    value: Interval
    gradient: Interval
    hessian: Interval

    @classmethod
    def constant(cls, value: float, dimension: int) -> "Jet":
        return cls(Interval.point(value), Interval.point(np.zeros(dimension)),
                   Interval.point(np.zeros((dimension, dimension))))

    @classmethod
    def variable(cls, lower: float, upper: float, index: int,
                 dimension: int) -> "Jet":
        gradient = np.zeros(dimension)
        gradient[index] = 1.0
        return cls(Interval.bounds(lower, upper), Interval.point(gradient),
                   Interval.point(np.zeros((dimension, dimension))))

    def __add__(self, other) -> "Jet":
        other = as_jet(other, self.gradient.lower.size)
        return Jet(self.value + other.value, self.gradient + other.gradient,
                   self.hessian + other.hessian)

    __radd__ = __add__

    def __neg__(self) -> "Jet":
        return Jet(-self.value, -self.gradient, -self.hessian)

    def __sub__(self, other) -> "Jet":
        return self + (-as_jet(other, self.gradient.lower.size))

    def __rsub__(self, other) -> "Jet":
        return as_jet(other, self.gradient.lower.size) - self

    def __mul__(self, other) -> "Jet":
        other = as_jet(other, self.gradient.lower.size)
        value = self.value * other.value
        gradient = self.gradient * other.value + other.gradient * self.value
        hessian = (self.hessian * other.value + other.hessian * self.value
                   + interval_outer(self.gradient, other.gradient)
                   + interval_outer(other.gradient, self.gradient))
        return Jet(value, gradient, hessian)

    __rmul__ = __mul__

    def __truediv__(self, scalar) -> "Jet":
        return self * (1.0 / float(scalar))

    def square(self) -> "Jet":
        value = self.value.square()
        gradient = 2.0 * self.value * self.gradient
        hessian = (2.0 * interval_outer(self.gradient, self.gradient)
                   + 2.0 * self.value * self.hessian)
        return Jet(value, gradient, hessian)

    def cube(self) -> "Jet":
        value = self.value.cube()
        first = 3.0 * self.value.square()
        second = 6.0 * self.value
        gradient = first * self.gradient
        hessian = (second * interval_outer(self.gradient, self.gradient)
                   + first * self.hessian)
        return Jet(value, gradient, hessian)

    def tanh(self) -> "Jet":
        lower = float(self.value.lower)
        upper = float(self.value.upper)
        tanh_lower_mp = mp.tanh(mp.mpf(lower))
        tanh_upper_mp = mp.tanh(mp.mpf(upper))
        tanh_lower = float(tanh_lower_mp)
        tanh_upper = float(tanh_upper_mp)
        value = Interval(_down(tanh_lower), _up(tanh_upper))
        far = max(abs(lower), abs(upper))
        near = 0.0 if lower <= 0 <= upper else min(abs(lower), abs(upper))
        far_tanh = mp.tanh(mp.mpf(far))
        near_tanh = mp.tanh(mp.mpf(near))
        first_lower = float(1 - far_tanh**2)
        first_upper = float(1 - near_tanh**2)
        first = Interval(_down(first_lower), _up(first_upper))
        candidates = [tanh_lower_mp, tanh_upper_mp]
        critical = 1 / mp.sqrt(3)
        if tanh_lower_mp <= critical <= tanh_upper_mp:
            candidates.append(critical)
        if tanh_lower_mp <= -critical <= tanh_upper_mp:
            candidates.append(-critical)
        second_values = [-2 * t * (1 - t * t) for t in candidates]
        second = Interval(_down(float(min(second_values))),
                          _up(float(max(second_values))))
        return Jet(value, first * self.gradient,
                   second * interval_outer(self.gradient, self.gradient)
                   + first * self.hessian)


def as_jet(value, dimension: int) -> Jet:
    return value if isinstance(value, Jet) else Jet.constant(float(value), dimension)


def box_jets(lower: np.ndarray, upper: np.ndarray) -> list[Jet]:
    dimension = lower.size
    return [Jet.variable(float(lower[index]), float(upper[index]), index, dimension)
            for index in range(dimension)]


def split_box(center: np.ndarray, radius: float, depth: int) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    """Cover the Euclidean ball by a deterministically subdivided enclosing cube."""
    if depth not in (0, 1):
        raise ValueError("The frozen certificate implementation supports depth 0 or 1.")
    if depth == 0:
        yield center - radius, center + radius
        return
    for signs in itertools.product((-1, 1), repeat=center.size):
        signs = np.asarray(signs)
        yield np.where(signs < 0, center - radius, center), np.where(
            signs < 0, center, center + radius)


def residual_interval_data(builder: Callable[[list[Jet]], list[Jet]], center: np.ndarray,
                           radius: float, *, subdivision_depth: int) -> dict:
    hessian_bounds = None
    box_count = 0
    for lower, upper in split_box(center, radius, subdivision_depth):
        residuals = builder(box_jets(lower, upper))
        current = np.asarray([_mp_norm_upper(residual.hessian.max_abs())
                              for residual in residuals])
        hessian_bounds = current if hessian_bounds is None else np.maximum(
            hessian_bounds, current)
        box_count += 1
    lipschitz = _mp_norm_upper(hessian_bounds)
    return {"hessian_operator_upper_bounds": _up(hessian_bounds).tolist(),
            "jacobian_lipschitz_upper": float(lipschitz),
            "subdivision_depth": subdivision_depth, "box_count": box_count,
            "enclosing_box_lower": (center - radius).tolist(),
            "enclosing_box_upper": (center + radius).tolist()}


def _mp_singular_values(matrix: np.ndarray) -> list[mp.mpf]:
    _, values, _ = mp.svd(mp.matrix([[mp.mpf(float(value)) for value in row]
                                     for row in matrix]))
    return sorted((mp.mpf(value) for value in values), reverse=True)


def point_bounds(builder: Callable[[list[Jet]], list[Jet]], center: np.ndarray) -> dict:
    residuals = builder(box_jets(center, center))
    residual_interval = Interval.bounds(
        np.asarray([float(row.value.lower) for row in residuals]),
        np.asarray([float(row.value.upper) for row in residuals]))
    jacobian_interval = Interval.bounds(
        np.stack([row.gradient.lower for row in residuals]),
        np.stack([row.gradient.upper for row in residuals]))
    midpoint = (jacobian_interval.lower + jacobian_interval.upper) / 2
    radius = np.maximum(jacobian_interval.upper - midpoint,
                        midpoint - jacobian_interval.lower)
    singular = _mp_singular_values(midpoint)
    radius_frobenius = mp.sqrt(sum(mp.mpf(float(value))**2 for value in radius.ravel()))
    sigma_min = max(mp.mpf("0"), singular[-1] - radius_frobenius)
    sigma_max = singular[0] + radius_frobenius
    sigma_min_lower = float(np.nextafter(float(sigma_min), -np.inf))
    sigma_max_upper = float(np.nextafter(float(sigma_max), np.inf))
    residual_upper = _mp_norm_upper(residual_interval.max_abs())
    gradient_components = []
    for column in range(center.size):
        total = Interval.point(0.0)
        for row in range(len(residuals)):
            total = total + Interval.bounds(
                jacobian_interval.lower[row, column],
                jacobian_interval.upper[row, column]) * Interval.bounds(
                    residual_interval.lower[row], residual_interval.upper[row])
        gradient_components.append(total)
    gradient_upper = _mp_norm_upper([
        float(component.max_abs()) for component in gradient_components])
    return {
        "sigma_min_lower": sigma_min_lower,
        "jacobian_operator_upper": sigma_max_upper,
        "residual_norm_upper": residual_upper,
        "gradient_norm_upper": gradient_upper,
        "residual_intervals": np.stack(
            (residual_interval.lower, residual_interval.upper), axis=1).tolist(),
        "jacobian_interval_lower": jacobian_interval.lower.tolist(),
        "jacobian_interval_upper": jacobian_interval.upper.tolist(),
        "jacobian_midpoint_singular_values_high_precision": [str(value) for value in singular],
        "jacobian_interval_radius_frobenius": str(radius_frobenius),
    }


def certificate(builder: Callable[[list[Jet]], list[Jet]], center: np.ndarray,
                radius: float, branch: str, *, subdivision_depth: int = 1) -> dict:
    if branch not in ("row", "column"):
        raise ValueError(branch)
    point = point_bounds(builder, center)
    neighborhood = residual_interval_data(builder, center, radius,
                                          subdivision_depth=subdivision_depth)
    delta = point["sigma_min_lower"]
    d0 = point["jacobian_operator_upper"]
    r0 = point["residual_norm_upper"]
    g0 = point["gradient_norm_upper"]
    lipschitz = neighborhood["jacobian_lipschitz_upper"]
    gamma = float(_down(delta - _up(lipschitz * radius)))
    result = {"branch": branch, "radius": radius, "point_bounds": point,
              "neighborhood_bounds": neighborhood, "gamma_lower": gamma}
    if branch == "row":
        invariant_margin = float(_down(_down(gamma * radius) - r0))
        result.update({
            "rank_margin": gamma,
            "invariant_ball_margin": invariant_margin,
            "necessary_margin": float(_down(_down(delta * radius) - r0)),
            "status": ("certified" if gamma > 0 and invariant_margin > 0
                       else "certificate_not_established"),
        })
        return result
    k_r = float(_up(d0 + _up(lipschitz * radius)))
    r_max = float(_up(r0 + _up(k_r * radius)))
    mu = float(_down(_down(gamma * gamma) - _up(lipschitz * r_max)))
    l_j = float(_up(_up(k_r * k_r) + _up(lipschitz * r_max)))
    gradient_margin = float(_down(_down(_down(mu * radius) / 2) - g0))
    result.update({
        "K_R_upper": k_r, "residual_ball_upper": r_max,
        "mu_lower": mu, "L_J_upper": l_j,
        "rank_margin": gamma, "gradient_invariant_margin": gradient_margin,
        "necessary_margin": float(_down(_down(_down(delta * delta) * radius / 2) - g0)),
        "status": ("certified" if gamma > 0 and mu > 0 and gradient_margin > 0
                   else "certificate_not_established"),
    })
    return result
