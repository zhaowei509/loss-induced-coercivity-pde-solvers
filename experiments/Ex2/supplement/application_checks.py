"""Non-formal mathematical and data-generation checks for S3."""

from __future__ import annotations

import torch

from .application_common import (CubicObjective, DTYPE, OptimizerConfig,
                                 hard_basis, make_master_bank, train_lbfgs)


def run_checks() -> dict:
    bank_a = make_master_bank(10003, train_count=16, validation_count=16,
                              rank_count=16, boundary_count=16)
    bank_b = make_master_bank(10003, train_count=16, validation_count=16,
                              rank_count=16, boundary_count=16)
    if bank_a.hashes() != bank_b.hashes():
        raise AssertionError("Master banks are not reproducible.")
    point = torch.tensor([[0.23, 0.61]], dtype=DTYPE, requires_grad=True)
    weights = bank_a.base_weights[:2]
    biases = bank_a.biases[:2]
    values, negative_laplacian = hard_basis(point, weights, biases)
    for feature in range(2):
        gradient = torch.autograd.grad(values[0, feature], point,
                                       create_graph=True, retain_graph=True)[0]
        laplacian = 0.0
        for axis in range(2):
            laplacian = laplacian + torch.autograd.grad(
                gradient[0, axis], point, retain_graph=True)[0][0, axis]
        if abs(float(laplacian + negative_laplacian[0, feature])) > 2e-11:
            raise AssertionError("Analytic hard-boundary Laplacian is inconsistent.")
    boundary_values, _ = hard_basis(bank_a.boundary_points, weights, biases)
    if float(boundary_values.abs().max()) != 0.0:
        raise AssertionError("Hard-boundary basis is nonzero on the boundary.")
    source = torch.zeros(16, dtype=DTYPE)
    training_values, training_laplacian = hard_basis(
        bank_a.training_points, weights, biases)
    objective = CubicObjective(training_values, training_laplacian, source)
    smoke = train_lbfgs(objective, 2, device="cpu",
                        config=OptimizerConfig(max_iterations=2, max_closures=10))
    if smoke["closure_count"] > 10:
        raise AssertionError("The closure budget was exceeded.")
    return {"status": "pass", "checks": [
        "master_bank_reproducibility", "analytic_laplacian", "hard_boundary",
        "closure_budget"]}

