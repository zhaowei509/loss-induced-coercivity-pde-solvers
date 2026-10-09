"""Rerun the Burgers residual-NTK experiment used for manuscript Figure 2.

This script is a standalone, non-notebook version of
``figure2_burgers_ntk.ipynb``. It preserves the archived model, sampling,
optimization, and random seeds, records the complete numerical trajectory,
and exports Figure 2 as a genuine vector PDF.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import shutil
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, LogFormatterMathtext, NullLocator
import numpy as np
from pyDOE import lhs
import scipy
import scipy.io
import torch
import torch.autograd as autograd
import torch.nn as nn


SEED = 1234
SOURCE_DATA = Path(__file__).with_name("burgers_shock_mu_01_pi.mat")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def training_data(data: dict[str, np.ndarray], n_boundary: int, n_interior: int):
    """Reproduce the boundary and Latin-hypercube samples in the notebook."""
    x = data["x"]
    t = data["t"]
    solution = data["usol"]
    mesh_x, mesh_t = np.meshgrid(x, t)
    test_points = np.hstack((mesh_x.flatten()[:, None], mesh_t.flatten()[:, None]))
    lower = test_points[0]
    upper = test_points[-1]

    initial_x = np.hstack((mesh_x[0, :][:, None], mesh_t[0, :][:, None]))
    initial_u = solution[:, 0][:, None]
    left_x = np.hstack((mesh_x[:, 0][:, None], mesh_t[:, 0][:, None]))
    left_u = solution[-1, :][:, None]
    right_x = np.hstack((mesh_x[:, -1][:, None], mesh_t[:, 0][:, None]))
    right_u = solution[0, :][:, None]
    all_x = np.vstack((initial_x, left_x, right_x))
    all_u = np.vstack((initial_u, left_u, right_u))
    indices = np.random.choice(all_x.shape[0], n_boundary, replace=False)
    boundary_x = all_x[indices]
    boundary_u = all_u[indices]

    interior_x = lower + (upper - lower) * lhs(2, n_interior)
    residual_x = np.vstack((interior_x, boundary_x))
    return residual_x, boundary_x, boundary_u


class RandomFeatureBurgers(nn.Module):
    """Two-layer random-feature model from the archived notebook."""

    def __init__(self, width: int):
        super().__init__()
        self.hidden = nn.Linear(2, width, bias=False)
        self.output = nn.Linear(width, 1, bias=False)
        self.scale = 1.0 / torch.sqrt(torch.tensor(width, dtype=torch.float))
        self.hidden.weight.data.normal_(mean=0.0, std=1.0)
        self.output.weight.data.uniform_(-1.0, 1.0)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.output(torch.tanh(self.hidden(points))) * self.scale

    def boundary_loss(self, points: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        return torch.mean((self(points) - values) ** 2)

    def residual(self, points: torch.Tensor) -> torch.Tensor:
        viscosity = 0.01 / np.pi
        coordinates = points.clone().requires_grad_(True)
        prediction = self(coordinates)
        first = autograd.grad(
            prediction,
            coordinates,
            torch.ones_like(prediction),
            retain_graph=True,
            create_graph=True,
        )[0]
        second = autograd.grad(
            first,
            coordinates,
            torch.ones_like(coordinates),
            create_graph=True,
        )[0]
        u_x = first[:, [0]]
        u_t = first[:, [1]]
        u_xx = second[:, [0]]
        return u_t + prediction * u_x - viscosity * u_xx

    def loss(
        self,
        boundary_x: torch.Tensor,
        boundary_u: torch.Tensor,
        residual_x: torch.Tensor,
    ) -> torch.Tensor:
        return self.boundary_loss(boundary_x, boundary_u) + torch.mean(
            self.residual(residual_x) ** 2
        )


def residual_ntk(model: RandomFeatureBurgers, points: torch.Tensor) -> torch.Tensor:
    """Compute the residual NTK with respect to output coefficients."""
    residual = model.residual(points)
    rows = []
    for index in range(points.shape[0]):
        model.zero_grad(set_to_none=True)
        gradient = autograd.grad(
            residual[index], model.output.parameters(), retain_graph=True
        )
        rows.append(torch.cat([entry.flatten() for entry in gradient]).detach())
    jacobian = torch.stack(rows)
    return jacobian @ jacobian.T


def boundary_ntk(model: RandomFeatureBurgers, points: torch.Tensor) -> torch.Tensor:
    """Compute the boundary-value NTK with respect to output coefficients."""
    values = model(points)
    rows = []
    for index in range(points.shape[0]):
        model.zero_grad(set_to_none=True)
        gradient = autograd.grad(
            values[index], model.output.parameters(), retain_graph=True
        )
        rows.append(torch.cat([entry.flatten() for entry in gradient]).detach())
    jacobian = torch.stack(rows)
    return jacobian @ jacobian.T


def assign_parameters(model: nn.Module, values: list[torch.Tensor]) -> None:
    with torch.no_grad():
        for parameter, value in zip(model.parameters(), values):
            parameter.copy_(value)


def run_experiment(args: argparse.Namespace):
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(args.threads)

    data = scipy.io.loadmat(SOURCE_DATA)
    residual_np, boundary_np, values_np = training_data(
        data, args.boundary_points, args.interior_points
    )
    residual_x = torch.from_numpy(residual_np).double()
    boundary_x = torch.from_numpy(boundary_np).double()
    boundary_u = torch.from_numpy(values_np).double()

    model = RandomFeatureBurgers(args.width).double()
    for parameter in model.hidden.parameters():
        parameter.requires_grad_(False)
    for parameter in model.output.parameters():
        parameter.requires_grad_(True)

    current = [parameter.detach().clone() for parameter in model.parameters()]
    baseline_interior = residual_ntk(model, residual_x)
    baseline_boundary = boundary_ntk(model, boundary_x)
    baseline_interior_norm = torch.linalg.matrix_norm(baseline_interior, ord="fro")
    baseline_boundary_norm = torch.linalg.matrix_norm(baseline_boundary, ord="fro")

    records = []
    started = time.perf_counter()
    for step in range(1, args.outer_steps + 1):
        assign_parameters(model, current)
        optimizer = torch.optim.LBFGS(
            model.output.parameters(),
            lr=args.learning_rate,
            max_iter=args.inner_steps,
            max_eval=2 * args.inner_steps,
            history_size=100,
        )

        def closure():
            optimizer.zero_grad()
            parameters = list(model.parameters())
            proximal = 0.5 * sum(
                (parameter - previous).pow(2).sum()
                for parameter, previous in zip(parameters, current)
            )
            objective = model.loss(boundary_x, boundary_u, residual_x)
            total = proximal + args.eta * objective
            total.backward()
            return total

        optimizer.step(closure)
        current = [parameter.detach().clone() for parameter in model.parameters()]
        interior = residual_ntk(model, residual_x)
        boundary = boundary_ntk(model, boundary_x)
        interior_change = (
            torch.linalg.matrix_norm(interior - baseline_interior, ord="fro")
            / baseline_interior_norm
        )
        boundary_change = (
            torch.linalg.matrix_norm(boundary - baseline_boundary, ord="fro")
            / baseline_boundary_norm
        )
        loss = model.loss(boundary_x, boundary_u, residual_x)
        records.append(
            {
                "outer_iteration": step,
                "loss": float(loss.detach()),
                "interior_relative_change": float(interior_change),
                "boundary_relative_change": float(boundary_change),
            }
        )
        if step == 1 or step % args.progress_every == 0 or step == args.outer_steps:
            elapsed = time.perf_counter() - started
            print(
                f"step={step}/{args.outer_steps} loss={records[-1]['loss']:.8e} "
                f"interior={records[-1]['interior_relative_change']:.8e} "
                f"boundary={records[-1]['boundary_relative_change']:.8e} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
    return records, residual_np, boundary_np, values_np, time.perf_counter() - started


def write_results(
    output: Path,
    args: argparse.Namespace,
    records: list[dict[str, float]],
    residual_x: np.ndarray,
    boundary_x: np.ndarray,
    boundary_u: np.ndarray,
    elapsed: float,
) -> None:
    output.mkdir(parents=True, exist_ok=False)
    trajectory = output / "trajectory.csv"
    with trajectory.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    np.savez_compressed(
        output / "training_samples.npz",
        residual_points=residual_x,
        boundary_points=boundary_x,
        boundary_values=boundary_u,
    )
    metadata = {
        "experiment": "Burgers residual-NTK diagnostic for manuscript Figure 2",
        "algorithm_source": "experiments/legacy_burgers_ntk/figure2_burgers_ntk.ipynb",
        "source_data": SOURCE_DATA.name,
        "source_data_sha256": sha256(SOURCE_DATA),
        "seed": SEED,
        "device": "cpu",
        "dtype": "float64",
        "parameters": {
            "width": args.width,
            "boundary_points": args.boundary_points,
            "interior_lhs_points": args.interior_points,
            "residual_points_after_boundary_augmentation": int(residual_x.shape[0]),
            "outer_steps": args.outer_steps,
            "inner_lbfgs_steps": args.inner_steps,
            "learning_rate": args.learning_rate,
            "eta": args.eta,
            "threads": args.threads,
        },
        "software": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
            "pyDOE": importlib.metadata.version("pyDOE"),
        },
        "elapsed_seconds": elapsed,
        "trajectory_sha256": sha256(trajectory),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def plot_figure(records: list[dict[str, float]], output: Path) -> None:
    steps = np.asarray([row["outer_iteration"] for row in records])
    interior = np.asarray([row["interior_relative_change"] for row in records])
    boundary = np.asarray([row["boundary_relative_change"] for row in records])
    boundary_display = np.maximum(boundary, 1e-12)
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.labelsize": 13,
            "xtick.labelsize": 11.5,
            "ytick.labelsize": 11.5,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "dejavusans",
        }
    ):
        figure, axis = plt.subplots(figsize=(6.0, 3.55))
        figure.subplots_adjust(left=0.16, right=0.98, bottom=0.19, top=0.965)
        axis.set_yscale("log")
        axis.set_ylim(1e-13, 2.0)
        axis.yaxis.set_major_locator(FixedLocator([1, 1e-3, 1e-6, 1e-9, 1e-12]))
        axis.yaxis.set_major_formatter(LogFormatterMathtext())
        axis.yaxis.set_minor_locator(NullLocator())
        axis.set_xlabel("Outer iteration", labelpad=6)
        axis.set_ylabel(r"Relative change $\rho_Q(k)$", labelpad=7)
        axis.tick_params(direction="out", length=3.5, width=0.8, pad=4)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["bottom", "left"]].set_color("#555555")
        axis.set_axisbelow(True)
        axis.grid(
            which="major", color="#dce2e8", linewidth=0.6, linestyle=(0, (3, 3))
        )
        axis.plot(steps, interior, color="#0072B2", linewidth=2.0)
        axis.plot(steps, boundary_display, color="#E69F00", linewidth=2.0)
        axis.legend(
            handles=[
                Line2D([], [], color="#0072B2", linewidth=2, label="Interior residual NTK"),
                Line2D([], [], color="#E69F00", linewidth=2, label="Boundary residual NTK"),
            ],
            loc="center right",
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.97,
            fontsize=12,
            handlelength=2.1,
            labelspacing=0.8,
            borderpad=0.6,
            borderaxespad=0.3,
        )
        figure.savefig(
            output / "Figure_2.pdf",
            metadata={
                "Title": "Relative changes of residual NTK matrices",
                "Subject": "Numerically rerun Burgers residual-NTK trajectory",
            },
        )
        figure.savefig(output / "Figure_2.png", dpi=1000)
        plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=100_000)
    parser.add_argument("--boundary-points", type=int, default=3)
    parser.add_argument("--interior-points", type=int, default=3)
    parser.add_argument("--outer-steps", type=int, default=1000)
    parser.add_argument("--inner-steps", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--eta", type=float, default=0.01)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--publish-to", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records, residual_x, boundary_x, boundary_u, elapsed = run_experiment(args)
    write_results(
        args.output,
        args,
        records,
        residual_x,
        boundary_x,
        boundary_u,
        elapsed,
    )
    plot_figure(records, args.output)
    if args.publish_to is not None:
        args.publish_to.mkdir(parents=True, exist_ok=True)
        for name in ("Figure_2.pdf", "Figure_2.png"):
            shutil.copy2(args.output / name, args.publish_to / name)
    print(f"results={args.output}")


if __name__ == "__main__":
    main()
