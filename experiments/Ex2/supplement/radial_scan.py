"""Pre-registered finite-ray Deep Ritz energy scan (supplement S4)."""

from __future__ import annotations

from collections import Counter
import math
from typing import Any

import numpy as np
import torch

from experiments.Ex2.experiment import make_problem
from experiments.Ex2.features import DTYPE
from experiments.Ex2.objectives import AllenCahnRitz, P4Ritz, p4_source
from experiments.Ex2.sampling import gauss_legendre


FORMAL_SEEDS = tuple(range(10))
DIRECTION_INDICES = (0, 1, 2)
P4_RADII = (0.0, 0.25, -0.25, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0,
            4.0, -4.0, 8.0, -8.0, 16.0, -16.0, 32.0, -32.0)
AC_RADII = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)


def direction(feature_seed: int, direction_index: int,
              *, device: str = "cpu") -> torch.Tensor:
    if direction_index not in DIRECTION_INDICES:
        raise ValueError("direction_index must be 0, 1, or 2.")
    seed = 20000 + 3 * feature_seed + direction_index
    generator = np.random.Generator(np.random.PCG64(seed))
    values = generator.standard_normal(8)
    norm = float(np.linalg.norm(values))
    if norm == 0 or not math.isfinite(norm):
        raise RuntimeError("Invalid PCG64 ray direction.")
    return torch.as_tensor(values / norm, dtype=DTYPE, device=device)


def _quadrature_objective(kind: str, features: Any, order: int,
                          *, device: str) -> tuple[Any, torch.Tensor, torch.Tensor]:
    quadrature = gauss_legendre(order, 1, device=device)
    jet = features.jet(quadrature.points)
    derivative = jet.gradient[:, :, 0]
    if kind == "p4":
        objective = P4Ritz(jet.values, derivative, p4_source(quadrature.points),
                           quadrature.weights)
    elif kind == "ac":
        objective = AllenCahnRitz(jet.values, derivative, quadrature.weights,
                                  epsilon=0.1)
    else:
        raise ValueError(f"Unknown radial-scan branch {kind!r}.")
    return objective, jet.values, derivative


def _diagnostics(objective: Any, values: torch.Tensor, derivative: torch.Tensor,
                 weights: torch.Tensor, coefficients: torch.Tensor) -> dict[str, float | None]:
    u = values @ coefficients
    du = derivative @ coefficients
    energy = float(objective.analytic_value(coefficients))
    return {
        "energy": energy,
        "coefficient_norm": float(torch.linalg.vector_norm(coefficients)),
        "u_l2": float(torch.dot(weights, u.square()).sqrt()),
        "du_l2": float(torch.dot(weights, du.square()).sqrt()),
        "du_l4": float(torch.dot(weights, du.abs().pow(4)).pow(0.25)),
    }


def scan_seed(feature_seed: int, *, device: str = "cpu",
              directions: tuple[int, ...] = DIRECTION_INDICES) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for kind, radii in (("p4", P4_RADII), ("ac", AC_RADII)):
        features = make_problem(kind, feature_seed, device=device)["features"]
        objective256, values256, derivative256 = _quadrature_objective(
            kind, features, 256, device=device)
        objective512, values512, derivative512 = _quadrature_objective(
            kind, features, 512, device=device)
        weights256 = gauss_legendre(256, 1, device=device).weights
        weights512 = gauss_legendre(512, 1, device=device).weights
        zero = torch.zeros(8, dtype=DTYPE, device=device)
        energy0_256 = float(objective256.analytic_value(zero))
        energy0_512 = float(objective512.analytic_value(zero))
        for direction_index in directions:
            ray = direction(feature_seed, direction_index, device=device)
            for radius in radii:
                coefficients = radius * ray
                q256 = _diagnostics(objective256, values256, derivative256,
                                    weights256, coefficients)
                q512 = _diagnostics(objective512, values512, derivative512,
                                    weights512, coefficients)
                discrepancy = abs(float(q256["energy"]) - float(q512["energy"]))
                tolerance = 1e-10 * max(1.0, abs(float(q512["energy"])))
                records.append({
                    "branch": kind,
                    "feature_seed": feature_seed,
                    "direction_index": direction_index,
                    "direction_seed": 20000 + 3 * feature_seed + direction_index,
                    "direction": ray,
                    "t": radius,
                    "abs_t": abs(radius),
                    "quadrature_256": q256,
                    "quadrature_512": q512,
                    "energy_difference_from_origin_256": float(q256["energy"]) - energy0_256,
                    "energy_difference_from_origin_512": float(q512["energy"]) - energy0_512,
                    "energy_over_abs_t_fourth_256": (
                        None if radius == 0 else float(q256["energy"]) / abs(radius)**4),
                    "energy_over_abs_t_fourth_512": (
                        None if radius == 0 else float(q512["energy"]) / abs(radius)**4),
                    "quadrature_absolute_difference": discrepancy,
                    "quadrature_tolerance": tolerance,
                    "quadrature_margin": tolerance - discrepancy,
                    "status": "pass" if discrepancy <= tolerance else "quadrature_failure",
                })
    return {
        "schema_version": 1,
        "supplement": "S4",
        "feature_seed": feature_seed,
        "device": device,
        "dtype": "float64",
        "directions": directions,
        "records": records,
        "status_counts": dict(Counter(row["status"] for row in records)),
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for result in results for row in result["records"]]
    failures = [row for row in rows if row["status"] != "pass"]
    return {
        "seed_count": len(results),
        "ray_count": len(results) * len(DIRECTION_INDICES) * 2,
        "evaluated_point_count": len(rows),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "quadrature_failure_count": len(failures),
        "minimum_quadrature_margin": min(row["quadrature_margin"] for row in rows),
        "interpretation": "finite-ray sampled diagnostic; not a coercivity proof",
    }

