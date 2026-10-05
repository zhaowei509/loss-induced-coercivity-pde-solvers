"""Prespecified two-start Allen--Cahn supplement to the frozen Ex2 campaign.

This module never writes inside the original formal campaign. Its two new
trajectories use the same constructed features but distinct fixed directions.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np
import torch

from .analysis import _ritz_record
from .dynamics import armijo
from .experiment import ARMIJO, CONTROL_SEED, GRADIENT_TOLERANCE, WIDTH, make_problem
from .io import atomic_json, file_hash, load_run, save_run
from .sampling import unit_direction


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "EXPERIMENT2_C_SUPPLEMENT_PLAN.md"
FORMAL = ROOT / "output" / "experiment2" / "formal_20260917_v1"
FORMAL_PLAN_SHA256 = "44f4e9e03940d4bf6af342f190c49fc567e96cf0db97989782e891bc9b4bc700"
INITIAL_SEEDS = (30001, 30002, 30003)
NEW_SEEDS = INITIAL_SEEDS[1:]
FORMAL_NAME = "control_ac_seed30001_plus"


def _sources():
    return {path.name: file_hash(path)
            for path in sorted(Path(__file__).resolve().parent.glob("*.py"))}


def _reference(formal):
    formal = Path(formal)
    summary = json.loads((formal / "summary.json").read_text(encoding="utf-8"))
    frozen = formal / "EXPERIMENT2_PLAN_frozen.md"
    reference = formal / FORMAL_NAME
    metadata, arrays = load_run(reference)
    if (summary["mode"] != "formal"
            or summary["plan_sha256"] != FORMAL_PLAN_SHA256
            or (frozen.is_file() and file_hash(frozen) != FORMAL_PLAN_SHA256)
            or metadata["plan_sha256"] != summary["plan_sha256"]
            or metadata["config"]["branch"] != "ac"
            or metadata["config"]["basis"] != "control"
            or metadata["config"]["feature_seed"] != CONTROL_SEED
            or metadata["config"]["initialization"] != "plus"
            or metadata["config"]["coordinate"] != "a"
            or metadata["config"]["accepted_step_limit"] != 50000):
        raise ValueError("The saved constructed Allen--Cahn reference is inconsistent.")
    hashes = {str(path.relative_to(formal)): file_hash(path)
              for path in (reference / "metadata.json", reference / "trajectory.npz")}
    return metadata, arrays, hashes


def precheck(formal=FORMAL):
    if not PLAN.is_file():
        raise FileNotFoundError(PLAN)
    original, arrays, hashes = _reference(formal)
    problem = make_problem("ac", CONTROL_SEED, device="cpu", basis="control")
    cfg = original["config"]
    if (not np.array_equal(problem["features"].weights.numpy(),
                           np.asarray(cfg["feature_weights"]))
            or not np.array_equal(problem["features"].biases.numpy(),
                                  np.asarray(cfg["feature_biases"]))):
        raise ValueError("Constructed features differ from the frozen reference.")
    directions = {seed: unit_direction(WIDTH["ac"], seed, device="cpu")
                  for seed in INITIAL_SEEDS}
    if not torch.allclose(directions[CONTROL_SEED],
                          torch.as_tensor(arrays["parameters"][0], dtype=torch.float64),
                          rtol=0, atol=1e-15):
        raise ValueError("The frozen reference does not start from d_30001.")
    max_alignment = max(abs(float(torch.dot(directions[a], directions[b])))
                        for index, a in enumerate(INITIAL_SEEDS)
                        for b in INITIAL_SEEDS[index + 1:])
    if max_alignment >= 1 - 1e-10:
        raise ValueError("Two prescribed directions are identical or sign mirrors.")
    return problem, directions, hashes, max_alignment


def _run_name(seed):
    return f"control_ac_init{seed}_plus"


def run(output, *, mode, steps, formal=FORMAL, resume=False):
    if mode not in ("smoke", "formal") or steps not in (10, 50000):
        raise ValueError("Use smoke/10 or formal/50000 exactly.")
    if (mode, steps) not in (("smoke", 10), ("formal", 50000)):
        raise ValueError("Smoke and formal budgets may not be mixed.")
    output = Path(output)
    if output.resolve() == Path(formal).resolve() or Path(formal).resolve() in output.resolve().parents:
        raise ValueError("Supplement output cannot be inside the original formal run.")
    problem, directions, reference_hashes, alignment = precheck(formal)
    plan_hash, source_hashes = file_hash(PLAN), _sources()
    summary_path, stop_file = output / "summary.json", output / "STOP_REQUESTED"
    if stop_file.exists():
        raise InterruptedError(f"Stop marker is present: {stop_file}")
    if resume:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        expected = dict(mode=mode, max_steps=steps, plan_sha256=plan_hash,
                        source_sha256=source_hashes, reference_sha256=reference_hashes,
                        initial_seeds=list(INITIAL_SEEDS))
        if any(summary.get(key) != value for key, value in expected.items()):
            raise ValueError("Supplement resume provenance differs.")
    else:
        output.mkdir(parents=True, exist_ok=False)
        summary = dict(mode=mode, created_utc=datetime.now(timezone.utc).isoformat(),
                       max_steps=steps, plan_sha256=plan_hash,
                       source_sha256=source_hashes,
                       reference_sha256=reference_hashes,
                       initial_seeds=list(INITIAL_SEEDS),
                       maximum_pairwise_direction_alignment=alignment,
                       planned_runs=[_run_name(seed) for seed in NEW_SEEDS], records=[])
        atomic_json(summary_path, summary)
    completed = {item["run"] for item in summary["records"]}
    if len(completed) != len(summary["records"]):
        raise ValueError("Duplicate saved supplement record.")
    for seed in NEW_SEEDS:
        name = _run_name(seed)
        if name in completed:
            metadata, arrays = load_run(output / name)
            if (metadata["plan_sha256"] != plan_hash
                    or metadata["source_sha256"] != source_hashes
                    or metadata["config"]["initial_seed"] != seed
                    or metadata["config"]["accepted_step_limit"] != steps):
                raise ValueError(f"Saved supplement run differs: {name}")
            print(f"already completed {name}", flush=True)
            continue
        if (output / name).exists():
            raise FileExistsError(f"Incomplete run was preserved, not overwritten: {name}")
        if stop_file.exists():
            raise InterruptedError(f"Stop marker is present: {stop_file}")
        result = armijo(problem["function"], directions[seed], max_steps=steps,
                        gradient_tolerance=GRADIENT_TOLERANCE["ac"],
                        stop_requested=stop_file.exists, **ARMIJO)
        values = [row["objective"] for row in result.trajectory]
        if (result.stop_reason in ("nonfinite_objective_or_gradient", "line_search_failure")
                or any(not math.isfinite(row["gradient_norm"])
                       for row in result.trajectory)
                or any(right > left + 1e-10 * max(1.0, abs(left))
                       for left, right in zip(values, values[1:]))):
            raise RuntimeError(f"Numerical anomaly in {name}: {result.stop_reason}")
        config = dict(run_kind="unit_smoke" if mode == "smoke" else "formal",
                      supplement="C_three_initializations", branch="ac",
                      feature_seed=CONTROL_SEED, initial_seed=seed,
                      initialization="plus", basis="control", coordinate="a",
                      qr_r_factor=None, device="cpu", dtype="float64", width=WIDTH["ac"],
                      accepted_step_limit=steps,
                      gradient_tolerance=GRADIENT_TOLERANCE["ac"], armijo=ARMIJO,
                      feature_weights=problem["features"].weights,
                      feature_biases=problem["features"].biases,
                      training_points=problem["points"],
                      training_weights=problem["weights"])
        save_run(output / name, config, result, PLAN)
        record = dict(run=name, initial_seed=seed, accepted_steps=result.accepted_steps,
                      stop_reason=result.stop_reason, elapsed_seconds=result.elapsed_seconds)
        summary["records"].append(record)
        atomic_json(summary_path, summary)
        print(f"completed {name}: {result.accepted_steps} steps, "
              f"{result.stop_reason}", flush=True)
    return summary


def analyze(output, *, formal=FORMAL):
    output = Path(output)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    if set(item["run"] for item in summary["records"]) != {
            _run_name(seed) for seed in NEW_SEEDS}:
        raise ValueError("Both supplement trajectories must be complete before analysis.")
    if summary["reference_sha256"] != _reference(formal)[2]:
        raise ValueError("Original reference changed since supplement launch.")
    if summary["plan_sha256"] != file_hash(PLAN):
        raise ValueError("Supplement plan changed since launch.")
    reference_meta, reference_arrays, _ = _reference(formal)
    rows = []
    for seed in INITIAL_SEEDS:
        metadata, arrays = ((reference_meta, reference_arrays) if seed == CONTROL_SEED
                            else load_run(output / _run_name(seed)))
        derived = _ritz_record(metadata, arrays)
        initial = derived["checkpoints"]["0"]
        final = derived["checkpoints"][str(derived["accepted_steps"])]
        if not derived["feature_full_rank"] or not derived["metric_full_rank"]:
            raise ValueError(f"Constructed feature metric lost rank for seed {seed}.")
        rows.append(dict(initial_seed=seed,
                         source="frozen formal reference" if seed == CONTROL_SEED
                         else "new supplement", rho_initial=initial["rho"],
                         rho_final=final["rho"], energy_initial=initial["energy_256"],
                         energy_final=final["energy_256"],
                         energy_final_512=derived["final_energy_512"],
                         rho_final_512=derived["final_rho_512"],
                         energy_256_512_difference=derived["energy_256_512_difference"],
                         rho_256_512_difference=derived["rho_256_512_difference"],
                         accepted_steps=derived["accepted_steps"],
                         stop_reason=derived["stop_reason"],
                         final_gradient=derived["final_coordinate_gradient"]))
    result = dict(mode=summary["mode"], plan_sha256=summary["plan_sha256"],
                  reference_sha256=summary["reference_sha256"], rows=rows,
                  note="Finite-step diagnostics only; numerical underflow is not convergence.")
    destination = output / "analysis.json"
    if destination.exists():
        raise FileExistsError(destination)
    atomic_json(destination, result)
    with (output / "appendix_c_three_starts.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "run", "analyze"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--formal", type=Path, default=FORMAL)
    parser.add_argument("--mode", choices=("smoke", "formal"), default="formal")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.action == "check":
        _, _, hashes, alignment = precheck(args.formal)
        print(json.dumps(dict(reference_sha256=hashes,
                              maximum_pairwise_direction_alignment=alignment)))
    else:
        if args.output is None:
            parser.error("--output is required for run and analyze")
        if args.action == "run":
            run(args.output, mode=args.mode, steps=10 if args.mode == "smoke" else 50000,
                formal=args.formal, resume=args.resume)
        else:
            print(json.dumps(analyze(args.output, formal=args.formal)))
