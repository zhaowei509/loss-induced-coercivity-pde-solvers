"""Fast implementation checks for S1; these are not formal data."""

from __future__ import annotations

import math

from .inexact_proximal import InexactProximalConfig, initial_coefficients, run_trajectory


def run_checks() -> dict:
    a = initial_coefficients(0, "pcg64_direction")
    b = initial_coefficients(0, "pcg64_direction")
    if not bool((a == b).all()) or abs(float(a.norm()) - 0.1) > 2e-15:
        raise AssertionError("The PCG64 initialization is not reproducible and normalized.")
    config = InexactProximalConfig(max_outer_steps=1, checkpoints=(5, 10, 25))
    result = run_trajectory(30000, "zero", config=config, interior_count=128)
    for row in result["trajectory"]:
        if row.get("accepted"):
            if row["q_minus_anchor"] > 1e-12:
                raise AssertionError("An accepted S1 step violates sufficient descent.")
            if row["relative_residual_ratio"] > config.relative_residual + 1e-12:
                raise AssertionError("An accepted S1 step violates the residual test.")
    if not math.isfinite(result["final_objective"]):
        raise AssertionError("The S1 smoke objective is nonfinite.")
    return {
        "status": "pass",
        "checks": ["pcg64_reproducibility", "accepted_step_conditions", "finite_smoke"],
        "smoke_stop_reason": result["stop_reason"],
        "smoke_accepted_steps": result["accepted_steps"],
    }

