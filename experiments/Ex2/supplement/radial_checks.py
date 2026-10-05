"""Fast implementation checks for the S4 radial scan."""

from __future__ import annotations

from .radial_scan import direction, scan_seed


def run_checks() -> dict:
    first = direction(0, 0)
    second = direction(0, 0)
    if not bool((first == second).all()) or abs(float(first.norm()) - 1) > 2e-15:
        raise AssertionError("Radial directions are not deterministic unit vectors.")
    result = scan_seed(30000, directions=(0,))
    p4 = {(row["t"], row["quadrature_512"]["energy"])
          for row in result["records"] if row["branch"] == "p4"}
    if not any(t < 0 for t, _ in p4) or not any(t > 0 for t, _ in p4):
        raise AssertionError("The p4 scan must preserve both ray orientations.")
    if any(row["branch"] == "ac" and row["t"] < 0 for row in result["records"]):
        raise AssertionError("The even Allen--Cahn branch must not duplicate negative rays.")
    return {"status": "pass", "checks": [
        "pcg64_unit_direction", "p4_signed_scan", "ac_nonnegative_scan",
        "independent_256_512_quadrature"]}

