"""S3C: paired k=2 target-frequency campaign at the frozen central setup."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from experiments.shared.artifacts import (create_directory, freeze_file, replace_json,
                                          sha256_file, write_json)
from experiments.shared.manifests import build_manifest

from .application_common import (FORMAL_SEEDS, SCALE_CANDIDATES, SMOKE_SEED,
                                 OptimizerConfig, run_configuration)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PLAN_PATH = PROJECT_ROOT / "PAPER_A_SUPPLEMENTARY_EXPERIMENT_PLAN.md"
SOURCE_PATHS = (
    Path(__file__), Path(__file__).with_name("application_common.py"),
    PROJECT_ROOT / "experiments/shared/artifacts.py",
    PROJECT_ROOT / "experiments/shared/numerics.py",
    PROJECT_ROOT / "experiments/shared/manifests.py",
)


def _summary(results: list[dict]) -> dict:
    finite = [row for row in results if row.get("evaluation") is not None]
    errors = [row["evaluation"]["gauss_legendre"]["relative_l2_error"] for row in finite]
    return {
        "pre_registered_count": len(results), "finite_count": len(finite),
        "stationarity_pass_count": sum(
            bool(row.get("training", {}).get("stationarity_pass")) for row in finite),
        "stop_reasons": dict(Counter(row["stop_reason"] for row in results)),
        "relative_l2_error_median": float(np.median(errors)) if errors else None,
        "relative_l2_error_q1": float(np.quantile(errors, 0.25)) if errors else None,
        "relative_l2_error_q3": float(np.quantile(errors, 0.75)) if errors else None,
    }


def campaign(output: Path, *, kind: str, scale: float, device: str,
             resume: bool = False, resource_smoke: Path | None = None) -> dict:
    seeds = (SMOKE_SEED,) if kind == "smoke" else FORMAL_SEEDS
    optimizer = (OptimizerConfig(max_iterations=100, max_closures=10)
                 if kind == "smoke" else OptimizerConfig())
    resource_failure = False
    smoke_hash = None
    if kind == "formal":
        if resource_smoke is None:
            raise ValueError("Formal S3C requires its completed k=2 resource smoke.")
        smoke_manifest = resource_smoke / "manifest.json"
        smoke_result = resource_smoke / f"W128_C_k2_seed_{SMOKE_SEED}.json"
        if not smoke_manifest.is_file() or not smoke_result.is_file():
            raise FileNotFoundError("Incomplete S3C resource smoke directory.")
        smoke_hash = sha256_file(smoke_manifest)
        smoke_row = json.loads(smoke_result.read_text(encoding="utf-8"))
        if float(smoke_row["feature_scale"]) != scale or smoke_row["device"] != device:
            raise ValueError("S3C smoke scale/device differs from the formal request.")
        resource_failure = smoke_row["stop_reason"] == "memory_failure"
    if resume:
        root = output
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        frozen = manifest["frozen_config"]
        if (float(frozen["feature_scale"]) != scale or frozen["device"] != device
                or frozen["seeds"] != list(seeds)
                or frozen.get("resource_smoke_manifest_sha256") != smoke_hash):
            raise ValueError("Resume arguments differ from the frozen manifest.")
    else:
        root = create_directory(output)
        freeze_file(PLAN_PATH, root / "plan_frozen.md")
        manifest = build_manifest(
            project_root=PROJECT_ROOT, supplement="S3C", run_kind=kind,
            plan_path=PLAN_PATH, source_paths=SOURCE_PATHS,
            frozen_config={"seeds": seeds, "configuration": "W128_C",
                           "feature_scale": scale, "frequency": 2,
                           "optimizer": optimizer.__dict__, "device": device,
                           "resource_smoke_manifest_sha256": smoke_hash,
                           "resource_gate_failure": resource_failure})
        write_json(root / "manifest.json", manifest)
    results = []
    for seed in seeds:
        path = root / f"W128_C_k2_seed_{seed}.json"
        if path.is_file():
            results.append(json.loads(path.read_text(encoding="utf-8")))
            continue
        if resource_failure:
            result = {"schema_version": 1, "feature_seed": seed,
                      "configuration": "W128_C", "width": 128,
                      "training_count": 16384, "feature_scale": scale,
                      "frequency": 2, "device": device, "dtype": "float64",
                      "stop_reason": "memory_failure", "training": None,
                      "evaluation": None,
                      "error": "pre-registered k=2 configuration failed resource smoke"}
        else:
            result = run_configuration(seed, "W128_C", scale, 2, device=device,
                                       optimizer_config=optimizer)
        write_json(path, result)
        results.append(result)
        replace_json(root / "summary.json", _summary(results))
    summary = _summary(results)
    replace_json(root / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "formal"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--scale", type=float, required=True, choices=SCALE_CANDIDATES)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resource-smoke", type=Path)
    args = parser.parse_args()
    print(campaign(args.output, kind=args.mode, scale=args.scale,
                   device=args.device, resume=args.resume,
                   resource_smoke=args.resource_smoke))


if __name__ == "__main__":
    main()
