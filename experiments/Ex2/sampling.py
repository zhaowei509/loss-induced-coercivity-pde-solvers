"""Independent float64 quadrature, collocation points, and fixed unit directions."""

from dataclasses import dataclass

import numpy as np
from scipy.special import roots_legendre
from scipy.stats import qmc
import torch

from .features import DTYPE

FORMAL_SEEDS = tuple(range(10))
CALIBRATION_SEEDS = (30000, 30001)


@dataclass
class Quadrature:
    points: torch.Tensor
    weights: torch.Tensor


def gauss_legendre(order, dimension=1, *, device="cpu"):
    if order < 1 or dimension not in (1, 2):
        raise ValueError("Positive order and dimension 1 or 2 required.")
    nodes, weights = roots_legendre(order)
    nodes, weights = (nodes + 1) / 2, weights / 2
    grids = np.meshgrid(*([nodes] * dimension), indexing="ij")
    weight_grids = np.meshgrid(*([weights] * dimension), indexing="ij")
    points = np.stack([g.reshape(-1) for g in grids], axis=1)
    product_weights = np.prod(np.stack(weight_grids), axis=0).reshape(-1)
    return Quadrature(torch.as_tensor(points, dtype=DTYPE, device=device),
                      torch.as_tensor(product_weights, dtype=DTYPE, device=device))


def sobol_points(count, dimension, feature_seed, *, device="cpu"):
    if count < 1 or dimension not in (1, 2) or count & (count - 1):
        raise ValueError("The prescribed Sobol count must be a power of two.")
    sampler = qmc.Sobol(d=dimension, scramble=True, seed=1000 + feature_seed)
    return torch.as_tensor(sampler.random_base2(count.bit_length() - 1).copy(),
                           dtype=DTYPE, device=device)


def unit_direction(width, feature_seed, *, device="cpu"):
    if width < 1:
        raise ValueError("Positive width required.")
    generator = torch.Generator(device="cpu").manual_seed(10000 + feature_seed)
    direction = torch.randn(width, generator=generator, dtype=DTYPE)
    direction /= torch.linalg.vector_norm(direction)
    return direction.to(device)
