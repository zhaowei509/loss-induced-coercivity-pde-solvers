"""Small NumPy-only checks for outward interval automatic differentiation."""

from __future__ import annotations

import numpy as np

from .certificate_common import box_jets, certificate


def run_checks() -> dict:
    center = np.array([0.7, -0.4])
    jets = box_jets(center, center)
    value = jets[0] * jets[1] + jets[0].cube()
    exact_value = center[0] * center[1] + center[0]**3
    exact_gradient = np.array([center[1] + 3 * center[0]**2, center[0]])
    exact_hessian = np.array([[6 * center[0], 1.0], [1.0, 0.0]])
    for exact, interval in ((exact_value, value.value),
                            (exact_gradient, value.gradient),
                            (exact_hessian, value.hessian)):
        if np.any(np.asarray(exact) < interval.lower) or np.any(
                np.asarray(exact) > interval.upper):
            raise AssertionError("Interval AD failed to enclose an exact derivative.")

    def builder(active):
        return [active[0] + 2 * active[1],
                active[0] * active[1] + active[0].cube()]

    result = certificate(builder, center, 1e-4, "column", subdivision_depth=1)
    if result["neighborhood_bounds"]["box_count"] != 4:
        raise AssertionError("Deterministic subdivision did not cover all subboxes.")
    if result["neighborhood_bounds"]["jacobian_lipschitz_upper"] <= 0:
        raise AssertionError("Invalid Jacobian-Lipschitz upper bound.")
    return {"status": "pass", "checks": [
        "value_gradient_hessian_enclosure", "deterministic_subdivision",
        "positive_lipschitz_bound"]}


if __name__ == "__main__":
    print(run_checks())

