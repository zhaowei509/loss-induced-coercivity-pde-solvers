"""Fixed, independent random streams and quadrature on the unit interval/square."""

from dataclasses import dataclass
import math

import numpy as np
from scipy.special import roots_legendre
from scipy.stats import qmc
import torch

from .features import DTYPE

FORMAL_SEEDS = tuple(range(10))
CALIBRATION_SEEDS = (30000, 30001, 30002)


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
    if count < 1 or dimension < 1:
        raise ValueError("Positive count and dimension required.")
    sampler = qmc.Sobol(d=dimension, scramble=True, seed=1000 + feature_seed)
    # Prefix of one scrambled sequence; also handles the prescribed n=12.
    points = sampler.random_base2(math.ceil(math.log2(count)))[:count].copy()
    return torch.as_tensor(points, dtype=DTYPE, device=device)


def perimeter_midpoints(count, *, device="cpu"):
    """Counterclockwise from (0,0): bottom, right, top, left; no duplicated corners."""
    if count < 1:
        raise ValueError("Positive count required.")
    s = (np.arange(count, dtype=np.float64) + 0.5) / count
    t = 4 * s
    side = np.floor(t).astype(int)
    r = t - side
    points = np.empty((count, 2), dtype=np.float64)
    choices = ((r, np.zeros(count)), (np.ones(count), r),
               (1 - r, np.ones(count)), (np.zeros(count), 1 - r))
    for k, (x, y) in enumerate(choices):
        mask = side == k
        points[mask] = np.stack((x[mask], y[mask]), axis=1)
    return torch.as_tensor(points, dtype=DTYPE, device=device)


def circle_directions(count=8, *, device="cpu"):
    if count < 1:
        raise ValueError("Positive count required.")
    angles = 2 * torch.pi * torch.arange(count, dtype=DTYPE, device=device) / count
    return torch.stack((angles.cos(), angles.sin()), dim=1)
