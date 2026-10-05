"""Command-line entry point for S1 checks, smoke runs, and formal campaigns."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.shared.artifacts import create_directory, freeze_file, write_json
from experiments.shared.manifests import build_manifest

from .checks import run_checks
from .inexact_proximal import (FORMAL_SEEDS, INITIALIZATIONS,
                               InexactProximalConfig, run_trajectory, summarize)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PLAN_PATH = PROJECT_ROOT / "PAPER_A_SUPPLEMENTARY_EXPERIMENT_PLAN.md"
SOURCE_PATHS = (
    Path(__file__), Path(__file__).with_name("inexact_proximal.py"),
    Path(__file__).with_name("checks.py"),
    PROJECT_ROOT / "experiments/shared/artifacts.py",
    PROJECT_ROOT / "experiments/shared/manifests.py",
    PROJECT_ROOT / "experiments/Ex1/experiment.py",
    PROJECT_ROOT / "experiments/Ex1/problems.py",
    PROJECT_ROOT / "experiments/Ex1/features.py",
    PROJECT_ROOT / "experiments/Ex1/sampling.py",
    PROJECT_ROOT / "experiments/Ex1/diagnostics.py",
)


def _manifest(kind: str, config: InexactProximalConfig, seeds: tuple[int, ...],
              initializations: tuple[str, ...]) -> dict:
    return build_manifest(
        project_root=PROJECT_ROOT, supplement="S1", run_kind=kind,
        plan_path=PLAN_PATH, source_paths=SOURCE_PATHS,
        frozen_config={
            "seeds": seeds, "initializations": initializations,
            "algorithm": config.__dict__, "interior_count": 2048,
            "feature_generator": "existing Experiment 1 torch stream",
            "direction_generator": "NumPy PCG64(10000 + feature_seed)",
        })


def run_campaign(output: Path, *, kind: str, seeds: tuple[int, ...],
                 initializations: tuple[str, ...], max_outer_steps: int,
                 device: str) -> dict:
    config = InexactProximalConfig(max_outer_steps=max_outer_steps)
    config.validate()
    root = create_directory(output)
    freeze_file(PLAN_PATH, root / "plan_frozen.md")
    write_json(root / "manifest.json", _manifest(kind, config, seeds, initializations))
    results = []
    for seed in seeds:
        for initialization in initializations:
            result = run_trajectory(seed, initialization, device=device, config=config)
            results.append(result)
            write_json(root / f"seed_{seed:05d}_{initialization}.json", result)
    summary = summarize(results)
    write_json(root / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check", help="run fast non-formal implementation checks")
    smoke = subparsers.add_parser("smoke", help="run one non-formal full-size trajectory")
    smoke.add_argument("output", type=Path)
    smoke.add_argument("--max-outer", type=int, default=1)
    smoke.add_argument("--device", default="cpu")
    formal = subparsers.add_parser("formal", help="run the frozen 20-trajectory campaign")
    formal.add_argument("output", type=Path)
    formal.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.command == "check":
        print(run_checks())
    elif args.command == "smoke":
        print(run_campaign(args.output, kind="smoke", seeds=(30000,),
                           initializations=("zero",),
                           max_outer_steps=args.max_outer, device=args.device))
    else:
        print(run_campaign(args.output, kind="formal", seeds=FORMAL_SEEDS,
                           initializations=INITIALIZATIONS,
                           max_outer_steps=100, device=args.device))


if __name__ == "__main__":
    main()
