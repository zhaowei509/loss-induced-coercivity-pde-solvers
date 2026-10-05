"""Float64 tanh features and their analytic spatial derivatives."""

from dataclasses import dataclass

import torch

DTYPE = torch.float64


def require_double(*tensors):
    if any(t.dtype != DTYPE for t in tensors):
        raise TypeError("Stage-four inputs must already be float64; do not upcast rounded data.")
    if tensors and any(t.device != tensors[0].device for t in tensors):
        raise ValueError("All tensors must be on the same device.")


@dataclass
class FeatureJet:
    values: torch.Tensor       # (points, features)
    gradient: torch.Tensor     # (points, features, spatial dimension)
    second_diagonal: torch.Tensor  # coordinatewise second derivatives
    laplacian: torch.Tensor    # (points, features)


def tanh_jet(x, weights, biases):
    """Return phi, grad(phi), and Delta(phi), without width normalization."""
    require_double(x, weights, biases)
    if (x.ndim != 2 or weights.ndim != 2 or biases.ndim != 1
            or weights.shape != (biases.numel(), x.shape[1]) or x.shape[1] == 0):
        raise ValueError("Expected x=(n,d), weights=(m,d), biases=(m,).")
    z = torch.tanh(x @ weights.T + biases)
    dz = 1 - z.square()
    ddz = -2 * z * dz
    gradient = dz[:, :, None] * weights[None, :, :]
    second_diagonal = ddz[:, :, None] * weights.square()[None, :, :]
    laplacian = second_diagonal.sum(dim=2)
    return FeatureJet(z, gradient, second_diagonal, laplacian)


@dataclass
class RandomFeatures:
    weights: torch.Tensor
    biases: torch.Tensor

    @classmethod
    def draw(cls, dimension, width, seed, *, device="cpu"):
        if dimension < 1 or width < 1:
            raise ValueError("Positive dimension and width required.")
        # Independent CPU stream makes feature draws identical on CPU and CUDA.
        generator = torch.Generator(device="cpu").manual_seed(seed)
        w = torch.randn(width, dimension, generator=generator, dtype=DTYPE)
        b = torch.randn(width, generator=generator, dtype=DTYPE)
        return cls(w.to(device), b.to(device))

    def jet(self, x):
        return tanh_jet(x, self.weights, self.biases)
