"""Float64 fixed tanh features with the hard-Dirichlet factor and exact spatial jets."""

from dataclasses import dataclass

import torch

DTYPE = torch.float64


def require_double(*tensors):
    if any(t.dtype != DTYPE for t in tensors):
        raise TypeError("Inputs must be generated in float64, not upcast after rounding.")
    if tensors and any(t.device != tensors[0].device for t in tensors):
        raise ValueError("All tensors must share a device.")


@dataclass
class FeatureJet:
    values: torch.Tensor
    gradient: torch.Tensor
    laplacian: torch.Tensor


def hard_tanh_jet(x, weights, biases):
    """Hard factor prod_d x_d(1-x_d), including derivatives on the boundary."""
    require_double(x, weights, biases)
    if (x.ndim != 2 or weights.ndim != 2 or biases.ndim != 1
            or weights.shape != (biases.numel(), x.shape[1])
            or x.shape[1] not in (1, 2)):
        raise ValueError("Expected x=(n,1|2), weights=(m,d), biases=(m,).")
    z = torch.tanh(x @ weights.T + biases)
    dz = 1 - z.square()
    ddz = -2 * z * dz
    raw_gradient = dz[:, :, None] * weights[None, :, :]
    raw_laplacian = ddz * weights.square().sum(dim=1)
    factors = x * (1 - x)
    factor = factors.prod(dim=1)
    first, second = [], []
    for axis in range(x.shape[1]):
        other = torch.cat((factors[:, :axis], factors[:, axis + 1:]), dim=1).prod(dim=1)
        first.append((1 - 2 * x[:, axis]) * other)
        second.append(-2 * other)
    factor_gradient = torch.stack(first, dim=1)
    factor_laplacian = torch.stack(second, dim=1).sum(dim=1)
    return FeatureJet(
        factor[:, None] * z,
        factor[:, None, None] * raw_gradient + z[:, :, None] * factor_gradient[:, None, :],
        factor[:, None] * raw_laplacian
        + 2 * (raw_gradient * factor_gradient[:, None, :]).sum(dim=2)
        + factor_laplacian[:, None] * z,
    )


@dataclass
class RandomFeatures:
    weights: torch.Tensor
    biases: torch.Tensor

    @classmethod
    def draw(cls, dimension, width, seed, *, device="cpu"):
        if dimension not in (1, 2) or width < 1:
            raise ValueError("Experiment 2 requires dimension 1 or 2 and positive width.")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        weights = torch.randn(width, dimension, generator=generator, dtype=DTYPE)
        biases = torch.randn(width, generator=generator, dtype=DTYPE)
        return cls(weights.to(device), biases.to(device))

    def jet(self, x):
        return hard_tanh_jet(x, self.weights, self.biases)


@dataclass
class Spectrum:
    numerical_rank: int
    threshold: float
    sigma_min: float
    full_column_rank: bool


def spectrum(matrix):
    """Experiment 1's machine-precision SVD rank criterion."""
    require_double(matrix)
    if matrix.ndim != 2 or not bool(torch.isfinite(matrix).all()):
        raise ValueError("A finite matrix is required.")
    singular = torch.linalg.svdvals(matrix.detach().cpu())
    if singular.numel() == 0:
        return Spectrum(0, 0.0, 0.0, matrix.shape[1] == 0)
    threshold = max(matrix.shape) * torch.finfo(matrix.dtype).eps * float(singular[0])
    rank = int((singular > threshold).sum())
    return Spectrum(rank, threshold, float(singular[-1]), rank == matrix.shape[1])


def qr_dual_residual(gradient, r_factor):
    """Dual norm of a coefficient gradient in the weighted feature metric."""
    require_double(gradient, r_factor)
    if gradient.shape != (r_factor.shape[0],) or r_factor.shape[0] != r_factor.shape[1]:
        raise ValueError("Incompatible gradient and thin-QR factor.")
    dual = torch.linalg.solve_triangular(r_factor.T, gradient[:, None], upper=False)
    return float(torch.linalg.vector_norm(dual))
