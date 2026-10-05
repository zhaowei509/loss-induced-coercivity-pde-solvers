"""Common finite-value and singular-value diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
try:
    import torch
except ModuleNotFoundError:
    torch = None


@dataclass(frozen=True)
class Spectrum:
    singular_values: list[float]
    numerical_rank: int
    threshold: float
    sigma_min: float
    sigma_max: float
    full_row_rank: bool
    full_column_rank: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def as_numpy(value: Any) -> np.ndarray:
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def spectrum(matrix: Any) -> Spectrum:
    array = np.asarray(as_numpy(matrix), dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("A finite two-dimensional matrix is required.")
    values = np.linalg.svd(array, compute_uv=False)
    if values.size == 0:
        rank = 0
        threshold = 0.0
        sigma_min = sigma_max = 0.0
    else:
        threshold = max(array.shape) * np.finfo(np.float64).eps * values[0]
        rank = int(np.count_nonzero(values > threshold))
        sigma_min, sigma_max = float(values[-1]), float(values[0])
    return Spectrum(values.tolist(), rank, float(threshold), sigma_min, sigma_max,
                    rank == array.shape[0], rank == array.shape[1])


def all_finite(*values: Any) -> bool:
    return all(bool(np.isfinite(as_numpy(value)).all()) for value in values)
