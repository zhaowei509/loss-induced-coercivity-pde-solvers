"""S2B held-out constant-source geometry and certificate funnel."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from experiments.shared.artifacts import create_directory, freeze_file, replace_json, write_json
from experiments.shared.manifests import build_manifest

from .general_common import RADII, SoftConfiguration, run_seed


CONFIGURATION = SoftConfiguration(
    name="constant_source", width=8, interior_count=16, boundary_count=8,
    boundary_penalty=10.0, column_bias_count=4,
    equation="-Delta u + u^3 = 1, homogeneous Dirichlet")
CALIBRATION_SEEDS = (0, 1, 2)
FORMAL_SEEDS = tuple(range(10, 20))


def constant_source(points):
    return points[:, 0] * 0 + 1.0


def summarize(results: list[dict]) -> dict:
    summary = {"seed_count": len(results), "branches": {},
               "derivative_check_pass_count": sum(
                   all(row["pass"] for row in result["derivative_checks"].values())
                   for result in results)}
    for branch in ("column", "row"):
        rows = [result["branches"][branch] for result in results]
        summary["branches"][branch] = {
            "F1_count": sum(row["F1_pass"] for row in rows),
            "F2_count": sum(row["F2_pass"] for row in rows),
            "F3_count": sum(row["F3_pass"] for row in rows),
            "F4_by_radius": {str(radius): sum(
                next(item for item in row["radii"] if item["radius"] == radius)["F4_pass"]
                for row in rows) for radius in RADII},
            "F5_by_radius": {str(radius): sum(
                next(item for item in row["radii"] if item["radius"] == radius)["F5_pass"]
                for row in rows) for radius in RADII},
            "F6_count": sum(row["dynamics"] is not None for row in rows),
            "first_failures": dict(Counter(
                row["first_failure"] or "completed" for row in rows)),
        }
    return summary


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PLAN_PATH = PROJECT_ROOT / "PAPER_A_SUPPLEMENTARY_EXPERIMENT_PLAN.md"
SOURCE_PATHS = (
    Path(__file__), Path(__file__).with_name("general_common.py"),
    Path(__file__).with_name("certificate_common.py"),
    PROJECT_ROOT / "experiments/shared/artifacts.py",
    PROJECT_ROOT / "experiments/shared/numerics.py",
    PROJECT_ROOT / "experiments/shared/manifests.py",
)


def campaign(output: Path, *, mode: str, resume: bool = False,
             certify: bool = True) -> dict:
    seeds = CALIBRATION_SEEDS if mode == "calibration_recheck" else FORMAL_SEEDS
    if resume:
        root = output
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if manifest["frozen_config"]["seeds"] != list(seeds):
            raise ValueError("Resume seeds differ from the frozen manifest.")
    else:
        root = create_directory(output)
        freeze_file(PLAN_PATH, root / "plan_frozen.md")
        manifest = build_manifest(
            project_root=PROJECT_ROOT, supplement="S2B", run_kind=mode,
            plan_path=PLAN_PATH, source_paths=SOURCE_PATHS,
            frozen_config={"seeds": seeds, "configuration": CONFIGURATION.__dict__,
                           "radii": RADII, "certificates_enabled": certify,
                           "interval_subdivision_depth": 0,
                           "feature_draw": "PCG64(seed): weights N(0,1), then biases N(0,1)"})
        write_json(root / "manifest.json", manifest)
    results = []
    for seed in seeds:
        path = root / f"seed_{seed}.json"
        if path.is_file():
            results.append(json.loads(path.read_text(encoding="utf-8")))
            continue
        result = run_seed(seed, CONFIGURATION, constant_source,
                          certify=certify, subdivision_depth=0)
        write_json(path, result)
        results.append(result)
        replace_json(root / "summary.json", summarize(results))
    summary = summarize(results)
    replace_json(root / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("calibration_recheck", "formal"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-certificates", action="store_true",
                        help="implementation smoke only; never use for formal reporting")
    args = parser.parse_args()
    if args.mode == "formal" and args.skip_certificates:
        raise ValueError("Formal S2B must retain the pre-registered certificate funnel.")
    print(campaign(args.output, mode=args.mode, resume=args.resume,
                   certify=not args.skip_certificates))


if __name__ == "__main__":
    main()
