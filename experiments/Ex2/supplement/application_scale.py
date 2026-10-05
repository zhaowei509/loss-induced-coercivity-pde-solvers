"""S3 scale calibration, resource smoke, and S3A formal campaigns."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.shared.artifacts import (create_directory, freeze_file, replace_json,
                                          sha256_file, write_json)
from experiments.shared.manifests import build_manifest

from .application_checks import run_checks
from .application_common import (CALIBRATION_SEEDS, CONFIGURATIONS, FORMAL_SEEDS,
                                 SCALE_CANDIDATES, SMOKE_SEED, OptimizerConfig,
                                 choose_scale, resource_limit_bytes,
                                 run_calibration, run_configuration)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PLAN_PATH = PROJECT_ROOT / "PAPER_A_SUPPLEMENTARY_EXPERIMENT_PLAN.md"
SOURCE_PATHS = (
    Path(__file__), Path(__file__).with_name("application_common.py"),
    Path(__file__).with_name("application_checks.py"),
    PROJECT_ROOT / "experiments/shared/artifacts.py",
    PROJECT_ROOT / "experiments/shared/numerics.py",
    PROJECT_ROOT / "experiments/shared/manifests.py",
)


def _manifest(kind: str, device: str, config: dict[str, Any]) -> dict[str, Any]:
    return build_manifest(
        project_root=PROJECT_ROOT, supplement="S3A", run_kind=kind,
        plan_path=PLAN_PATH, source_paths=SOURCE_PATHS,
        frozen_config={**config, "device": device,
                       "resource_limit_bytes": resource_limit_bytes(device)})


def _summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_configuration = {}
    for label in CONFIGURATIONS:
        rows = [row for row in results if row["configuration"] == label]
        finite = [row for row in rows if row.get("evaluation") is not None]
        errors = [row["evaluation"]["gauss_legendre"]["relative_l2_error"]
                  for row in finite]
        by_configuration[label] = {
            "pre_registered_count": len(rows),
            "finite_count": len(finite),
            "stationarity_pass_count": sum(
                bool(row.get("training", {}).get("stationarity_pass")) for row in finite),
            "stop_reasons": dict(Counter(row["stop_reason"] for row in rows)),
            "relative_l2_error_median": float(np.median(errors)) if errors else None,
            "relative_l2_error_q1": float(np.quantile(errors, 0.25)) if errors else None,
            "relative_l2_error_q3": float(np.quantile(errors, 0.75)) if errors else None,
        }
    return {"run_count": len(results), "by_configuration": by_configuration}


def calibrate(output: Path, *, device: str) -> dict[str, Any]:
    root = create_directory(output)
    freeze_file(PLAN_PATH, root / "plan_frozen.md")
    write_json(root / "manifest.json", _manifest(
        "calibration", device,
        {"seeds": CALIBRATION_SEEDS, "scale_candidates": SCALE_CANDIDATES,
         "width": 128, "training_count": 16384, "closure_budget": 100}))
    results = []
    for scale in SCALE_CANDIDATES:
        for seed in CALIBRATION_SEEDS:
            result = run_calibration(seed, scale, device=device)
            results.append(result)
            write_json(root / f"scale_{scale:g}_seed_{seed}.json", result)
    selection = choose_scale(results)
    write_json(root / "selection.json", selection)
    return selection


def campaign(output: Path, *, kind: str, scale: float, device: str,
             resume: bool = False, resource_smoke: Path | None = None,
             stop_after: str | None = None) -> dict[str, Any]:
    seeds = (SMOKE_SEED,) if kind == "smoke" else FORMAL_SEEDS
    optimizer = (OptimizerConfig(max_iterations=100, max_closures=10)
                 if kind == "smoke" else OptimizerConfig())
    resource_failures: set[str] = set()
    smoke_hash = None
    if kind == "formal":
        if resource_smoke is None:
            raise ValueError("Formal S3A requires the completed resource-smoke directory.")
        smoke_manifest = resource_smoke / "manifest.json"
        if not smoke_manifest.is_file():
            raise FileNotFoundError(smoke_manifest)
        smoke_hash = sha256_file(smoke_manifest)
        for label in CONFIGURATIONS:
            smoke_result = resource_smoke / f"{label}_seed_{SMOKE_SEED}.json"
            if not smoke_result.is_file():
                raise FileNotFoundError(smoke_result)
            row = json.loads(smoke_result.read_text(encoding="utf-8"))
            if float(row["feature_scale"]) != scale or row["device"] != device:
                raise ValueError("Resource smoke scale/device differs from the formal request.")
            if row["stop_reason"] == "memory_failure":
                resource_failures.add(label)
    if resume:
        root = output
        if not (root / "manifest.json").is_file():
            raise FileNotFoundError("Cannot resume without manifest.json.")
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        frozen = manifest["frozen_config"]
        if (float(frozen["feature_scale"]) != scale or frozen["device"] != device
                or frozen["seeds"] != list(seeds)
                or frozen.get("resource_smoke_manifest_sha256") != smoke_hash):
            raise ValueError("Resume arguments differ from the frozen manifest.")
    else:
        root = create_directory(output)
        freeze_file(PLAN_PATH, root / "plan_frozen.md")
        write_json(root / "manifest.json", _manifest(
            kind, device, {"seeds": seeds, "configurations": CONFIGURATIONS,
                           "feature_scale": scale,
                           "optimizer": optimizer.__dict__, "frequency": 1,
                           "resource_smoke_manifest_sha256": smoke_hash,
                           "resource_gate_failures": sorted(resource_failures)}))
    results = []
    for label in CONFIGURATIONS:
        for seed in seeds:
            path = root / f"{label}_seed_{seed}.json"
            if path.is_file():
                results.append(json.loads(path.read_text(encoding="utf-8")))
                continue
            if label in resource_failures:
                width, count = CONFIGURATIONS[label]
                result = {"schema_version": 1, "feature_seed": seed,
                          "configuration": label, "width": width,
                          "training_count": count, "feature_scale": scale,
                          "frequency": 1, "device": device, "dtype": "float64",
                          "stop_reason": "memory_failure", "training": None,
                          "evaluation": None,
                          "error": "pre-registered configuration failed resource smoke"}
            else:
                result = run_configuration(seed, label, scale, 1, device=device,
                                           optimizer_config=optimizer)
            write_json(path, result)
            results.append(result)
            replace_json(root / "summary.json", _summary(results))
        if label == stop_after:
            break
    summary = _summary(results)
    replace_json(root / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check")
    calibration = subparsers.add_parser("calibrate")
    calibration.add_argument("output", type=Path)
    calibration.add_argument("--device", default="cuda")
    for command in ("smoke", "formal"):
        sub = subparsers.add_parser(command)
        sub.add_argument("output", type=Path)
        sub.add_argument("--scale", type=float, required=True,
                         choices=SCALE_CANDIDATES)
        sub.add_argument("--device", default="cuda")
        sub.add_argument("--resume", action="store_true")
        if command == "formal":
            sub.add_argument("--resource-smoke", type=Path, required=True)
            sub.add_argument("--stop-after", choices=tuple(CONFIGURATIONS))
    args = parser.parse_args()
    if args.command == "check":
        print(run_checks())
    elif args.command == "calibrate":
        print(calibrate(args.output, device=args.device))
    else:
        print(campaign(args.output, kind=args.command, scale=args.scale,
                       device=args.device, resume=args.resume,
                       resource_smoke=getattr(args, "resource_smoke", None),
                       stop_after=getattr(args, "stop_after", None)))


if __name__ == "__main__":
    main()
