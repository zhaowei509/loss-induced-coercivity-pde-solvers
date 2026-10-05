"""S2A validated neighborhood bounds for the frozen controlled Experiment 4 object."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from experiments.shared.artifacts import create_directory, freeze_file, sha256_file, write_json
from experiments.shared.manifests import build_manifest

from .certificate_common import Jet, certificate


RADII = (1e-4, 3e-4, 1e-3, 3e-3)
MAIN_RADIUS = 1e-3
FORMAL_RESULTS = "output/experiment4/formal_20260918_v1/results.json"


class ControlledProblem:
    """NumPy reconstruction of the exact stored Experiment 4 empirical problem."""

    def __init__(self, frozen: dict):
        configuration = frozen["configuration"]
        self.w = np.asarray(configuration["w"], dtype=np.float64)
        self.b = np.asarray(configuration["b"], dtype=np.float64)
        self.x = np.asarray(configuration["interior"], dtype=np.float64)
        self.y = np.asarray(configuration["boundary"], dtype=np.float64)
        self.f = np.asarray(configuration["f_delta"], dtype=np.float64)
        self.g = np.asarray(configuration["g"], dtype=np.float64)

    def starting_vector(self, coefficients: np.ndarray, branch: str) -> np.ndarray:
        if branch == "column":
            return np.concatenate((coefficients, self.b[:1]))
        if branch == "row":
            return np.concatenate((coefficients, self.w.reshape(-1), self.b))
        raise ValueError(branch)

    def _unpack(self, vector: np.ndarray, branch: str):
        coefficients = vector[:2]
        if branch == "column":
            biases = self.b.copy()
            biases[0] = vector[2]
            return coefficients, self.w, biases
        return coefficients, vector[2:6].reshape(2, 2), vector[6:8]

    def residual_jacobian(self, vector: np.ndarray, branch: str):
        coefficients, weights, biases = self._unpack(vector, branch)
        norm_squared = np.sum(weights**2, axis=1)

        def blocks(locations):
            activation = np.tanh(locations @ weights.T + biases)
            first = 1 - activation**2
            second = -2 * activation * first
            third = (-2 + 6 * activation**2) * first
            return activation, first, second, third

        ti, d1i, d2i, d3i = blocks(self.x)
        tb, d1b, _, _ = blocks(self.y)
        u = ti @ coefficients
        residual = np.concatenate((
            (-((d2i * norm_squared) @ coefficients) + u**3 - self.f) / math.sqrt(2),
            math.sqrt(0.5) * (tb @ coefficients - self.g)))
        output = np.vstack((
            (-d2i * norm_squared + 3 * u[:, None]**2 * ti) / math.sqrt(2),
            math.sqrt(0.5) * tb))
        common = -d3i * norm_squared + 3 * u[:, None]**2 * d1i
        bias = np.vstack((coefficients * common / math.sqrt(2),
                          math.sqrt(0.5) * coefficients * d1b))
        if branch == "column":
            hidden = bias[:, :1]
        else:
            weight_i = np.empty((2, 2, 2))
            weight_b = np.empty((4, 2, 2))
            for axis in range(2):
                weight_i[:, :, axis] = coefficients * (
                    common * self.x[:, [axis]] - 2 * d2i * weights[:, axis]) / math.sqrt(2)
                weight_b[:, :, axis] = (math.sqrt(0.5) * coefficients * d1b
                                        * self.y[:, [axis]])
            weight = np.vstack((weight_i.reshape(2, 4), weight_b.reshape(4, 4)))
            hidden = np.column_stack((weight, bias))
        return residual, np.column_stack((output, hidden))

    def residual(self, vector: np.ndarray, branch: str) -> np.ndarray:
        return self.residual_jacobian(vector, branch)[0]

    def value(self, vector: np.ndarray, branch: str) -> float:
        residual = self.residual(vector, branch)
        return 0.5 * float(residual @ residual)

    def gradient(self, vector: np.ndarray, branch: str) -> np.ndarray:
        residual, jacobian = self.residual_jacobian(vector, branch)
        return jacobian.T @ residual


def controlled_builder(problem: ControlledProblem, branch: str):
    f = problem.f
    g = problem.g
    fixed_w = problem.w.reshape(-1)
    fixed_b = problem.b

    def build(active: list[Jet]) -> list[Jet]:
        dimension = len(active)
        if branch == "row":
            full = active
        elif branch == "column":
            full = [active[0], active[1]]
            full += [Jet.constant(value, dimension) for value in fixed_w]
            full += [active[2], Jet.constant(fixed_b[1], dimension)]
        else:
            raise ValueError(branch)
        a = full[:2]
        w = (full[2:4], full[4:6])
        b = full[6:8]

        def field(point):
            u = Jet.constant(0.0, dimension)
            laplacian = Jet.constant(0.0, dimension)
            for index in range(2):
                q = point[0] * w[index][0] + point[1] * w[index][1] + b[index]
                activation = q.tanh()
                norm_squared = w[index][0].square() + w[index][1].square()
                second = -2.0 * (activation - activation.cube())
                u = u + a[index] * activation
                laplacian = laplacian + a[index] * second * norm_squared
            return u, laplacian

        residuals = []
        for index, point in enumerate(problem.x):
            u, laplacian = field(point)
            residuals.append((-laplacian + u.cube() - f[index]) / math.sqrt(2.0))
        for index, point in enumerate(problem.y):
            u, _ = field(point)
            residuals.append(math.sqrt(0.5) * (u - g[index]))
        return residuals

    return build


def theorem_guided_gd(problem: ControlledProblem, start: np.ndarray, certificate_row: dict,
                      reference: np.ndarray | None) -> dict:
    if certificate_row["status"] != "certified":
        return {"status": "not_run_certificate_not_established"}
    l_j = certificate_row["L_J_upper"]
    eta = float(np.nextafter(0.5 / l_j, 0.0))
    radius = certificate_row["radius"]
    mu = certificate_row["mu_lower"]
    theta = start.copy()
    rows = []
    started = time.perf_counter()
    reason = "step_budget"
    contraction = max(0.0, 1 - eta * mu)
    initial_reference_distance = (None if reference is None else
                                  float(np.linalg.norm(start - reference)))
    for step in range(200001):
        value = problem.value(theta, "column")
        gradient = problem.gradient(theta, "column")
        gradient_norm = float(np.linalg.norm(gradient))
        displacement = float(np.linalg.norm(theta - start))
        contraction_power = contraction**step
        rows.append({"step": step, "objective": value,
                     "gradient_norm": gradient_norm,
                     "displacement": displacement,
                     "reference_distance": (None if reference is None else
                                            float(np.linalg.norm(theta - reference))),
                     "theoretical_contraction_factor_power": contraction_power,
                     "theoretical_reference_distance_upper": (
                         None if initial_reference_distance is None else
                         initial_reference_distance * contraction_power)})
        if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
            reason = "nonfinite"
            break
        if gradient_norm <= 1e-10:
            reason = "gradient_tolerance"
            break
        if displacement > radius:
            reason = "trajectory_exit"
            break
        if step == 200000:
            break
        theta = theta - eta * gradient
    return {"status": reason, "eta": eta, "steps": step,
            "elapsed_seconds": time.perf_counter() - started,
            "final_parameters": theta.tolist(), "sampled_trajectory": rows}


def run(formal_results: Path, output: Path, *, subdivision_depth: int = 1) -> dict:
    if output.exists():
        raise FileExistsError(output)
    frozen = json.loads(formal_results.read_text(encoding="utf-8"))
    problem = ControlledProblem(frozen)
    a = np.asarray(frozen["warm_start"]["a"], dtype=np.float64)
    starts = {branch: problem.starting_vector(a, branch)
              for branch in ("column", "row")}
    reconstruction = {}
    for branch in ("column", "row"):
        residual, jacobian = problem.residual_jacobian(starts[branch], branch)
        residual_error = abs(float(np.linalg.norm(residual))
                             - frozen["geometry"]["initial_residual_norm"])
        sigma_error = abs(float(np.linalg.svd(jacobian, compute_uv=False)[-1])
                          - frozen["geometry"][branch]["joint"]["sigma_min"])
        reconstruction[branch] = {
            "residual_norm_absolute_error": residual_error,
            "sigma_min_absolute_error": sigma_error,
            "pass": residual_error <= 1e-14 and sigma_error <= 1e-12,
        }
    if not all(row["pass"] for row in reconstruction.values()):
        raise ValueError("NumPy reconstruction differs from the frozen Experiment 4 object.")
    certificates = {branch: [
        certificate(controlled_builder(problem, branch), starts[branch], radius,
                    branch, subdivision_depth=subdivision_depth)
        for radius in RADII] for branch in ("column", "row")}
    main_column = next(row for row in certificates["column"]
                       if row["radius"] == MAIN_RADIUS)
    reference_data = frozen["branches"]["column"].get("reference")
    reference = (None if not reference_data else
                 np.asarray(reference_data["theta_star"], dtype=np.float64))
    gd = theorem_guided_gd(problem, starts["column"], main_column, reference)
    root = create_directory(output)
    freeze_file(Path(__file__).resolve().parents[3]
                / "PAPER_A_SUPPLEMENTARY_EXPERIMENT_PLAN.md",
                root / "plan_frozen.md")
    manifest = build_manifest(
        project_root=Path(__file__).resolve().parents[3], supplement="S2A",
        run_kind="formal", plan_path=Path(__file__).resolve().parents[3]
        / "PAPER_A_SUPPLEMENTARY_EXPERIMENT_PLAN.md",
        source_paths=(Path(__file__), Path(__file__).with_name("certificate_common.py"),
                      Path(__file__).resolve().parents[1] / "core.py"),
        frozen_config={"radii": RADII, "main_radius": MAIN_RADIUS,
                       "subdivision_depth": subdivision_depth,
                       "formal_results": str(formal_results.resolve()),
                       "formal_results_sha256": sha256_file(formal_results),
                       "rounding": "IEEE float64 nextafter outward; tanh/SVD at 100 dps"})
    write_json(root / "manifest.json", manifest)
    result = {"schema_version": 1, "supplement": "S2A",
              "formal_results_sha256": sha256_file(formal_results),
              "starts": {key: value.tolist() for key, value in starts.items()},
              "reconstruction_check": reconstruction,
              "certificates": certificates, "full_column_gd": gd}
    write_json(root / "results.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--formal-results", type=Path,
                        default=Path(__file__).resolve().parents[3] / FORMAL_RESULTS)
    parser.add_argument("--subdivision-depth", type=int, default=1, choices=(0, 1))
    args = parser.parse_args()
    result = run(args.formal_results, args.output,
                 subdivision_depth=args.subdivision_depth)
    print({branch: [row["status"] for row in rows]
           for branch, rows in result["certificates"].items()})


if __name__ == "__main__":
    main()
