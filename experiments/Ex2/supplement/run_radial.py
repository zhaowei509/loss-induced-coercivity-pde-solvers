"""Command-line entry point for supplement S4."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.shared.artifacts import create_directory, freeze_file, write_json
from experiments.shared.manifests import build_manifest

from .radial_checks import run_checks
from .radial_scan import FORMAL_SEEDS, scan_seed, summarize


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PLAN_PATH = PROJECT_ROOT / "PAPER_A_SUPPLEMENTARY_EXPERIMENT_PLAN.md"
SOURCE_PATHS = (
    Path(__file__), Path(__file__).with_name("radial_scan.py"),
    Path(__file__).with_name("radial_checks.py"),
    PROJECT_ROOT / "experiments/shared/artifacts.py",
    PROJECT_ROOT / "experiments/shared/manifests.py",
    PROJECT_ROOT / "experiments/Ex2/experiment.py",
    PROJECT_ROOT / "experiments/Ex2/features.py",
    PROJECT_ROOT / "experiments/Ex2/objectives.py",
    PROJECT_ROOT / "experiments/Ex2/sampling.py",
)


def run_campaign(output: Path, *, kind: str, seeds: tuple[int, ...]) -> dict:
    root = create_directory(output)
    freeze_file(PLAN_PATH, root / "plan_frozen.md")
    manifest = build_manifest(
        project_root=PROJECT_ROOT, supplement="S4", run_kind=kind,
        plan_path=PLAN_PATH, source_paths=SOURCE_PATHS,
        frozen_config={
            "feature_seeds": seeds, "direction_indices": (0, 1, 2),
            "direction_seed": "20000 + 3 * feature_seed + direction_index",
            "quadrature_orders": (256, 512),
        })
    write_json(root / "manifest.json", manifest)
    results = []
    for seed in seeds:
        result = scan_seed(seed)
        results.append(result)
        write_json(root / f"seed_{seed:05d}.json", result)
    summary = summarize(results)
    write_json(root / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check")
    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("output", type=Path)
    formal = subparsers.add_parser("formal")
    formal.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "check":
        print(run_checks())
    elif args.command == "smoke":
        print(run_campaign(args.output, kind="smoke", seeds=(30000,)))
    else:
        print(run_campaign(args.output, kind="formal", seeds=FORMAL_SEEDS))


if __name__ == "__main__":
    main()
