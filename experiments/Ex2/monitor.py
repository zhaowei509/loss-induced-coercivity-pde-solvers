"""Read-only audit of completed formal trajectories; optional cooperative stop."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil

import numpy as np
import torch

from .analysis import _cubic_record, _ritz_record
from .experiment import _source_hashes, default_plan
from .io import file_hash, load_run


def audit(root):
    root = Path(root)
    summary_path = root / "summary.json"
    if not summary_path.is_file():
        return dict(status="not_started", completed=0, critical=[], warnings=[], runs=[])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    critical, warnings, runs = [], [], []
    if summary.get("mode") != "formal" or summary.get("expected_runs") != 73:
        critical.append("Unexpected campaign mode or manifest size.")
    if summary.get("plan_sha256") != file_hash(default_plan()):
        critical.append("Formal plan hash changed.")
    if summary.get("source_sha256") != _source_hashes():
        critical.append("Formal source hash changed.")
    records = summary.get("records", [])
    names = [item.get("run") for item in records]
    if len(names) != len(set(names)) or not set(names).issubset(summary.get("planned_runs", [])):
        critical.append("Duplicate or unexpected completed trajectory.")
    if len(records) + len(summary.get("skipped", [])) > 73:
        critical.append("Campaign exceeds its frozen manifest.")
    for item in records:
        name = item["run"]
        try:
            meta, arrays = load_run(root / name)
            if (meta["plan_sha256"] != summary.get("plan_sha256")
                    or meta["source_sha256"] != summary.get("source_sha256")):
                raise ValueError("Saved trajectory hashes differ from campaign.")
            steps = meta["result"]["accepted_steps"]
            if steps != item["accepted_steps"] or steps > 50000:
                raise ValueError("Invalid accepted-step count.")
            for key in ("parameters", "objective", "gradient_norm", "parameter_norm",
                        "elapsed_seconds"):
                if not np.isfinite(arrays[key]).all():
                    raise ValueError(f"Nonfinite {key}.")
            objective = arrays["objective"]
            tolerance = 1e-10 * np.maximum(1.0, np.abs(objective[:-1]))
            if np.any(np.diff(objective) > tolerance):
                raise ValueError("Accepted-step objective increased.")
            if (np.any(arrays["gradient_norm"] < 0)
                    or np.any(np.diff(arrays["elapsed_seconds"]) < 0)
                    or np.any(~np.isfinite(arrays["step_size"][1:]))
                    or np.any(arrays["step_size"][1:] <= 0)):
                raise ValueError("Invalid trajectory diagnostics.")
            reason = meta["result"]["stop_reason"]
            if reason in ("nonfinite_objective_or_gradient", "line_search_failure"):
                critical.append(f"{name}: {reason}.")
            elif reason == "step_underflow":
                warnings.append(f"{name}: step underflow, reported as numerical stop.")
            if meta["config"]["branch"] == "cubic":
                derived = _cubic_record(meta, arrays)
                error, bound = derived["final_E256"], derived["final_B256"]
                if not all(map(math.isfinite, (error, bound))):
                    raise ValueError("Nonfinite independent cubic diagnostic.")
                if error > bound + max(1e-6, 1e-3 * max(1.0, error, bound)):
                    critical.append(f"{name}: large independent error-bound violation.")
                elif error > bound:
                    warnings.append(f"{name}: small discrete error-bound deficit; inspect quadrature.")
                diagnostic = dict(E256=error, B256=bound,
                                  Q_full_rank=derived["Q_full_rank"],
                                  P_full_rank=derived["P_full_rank"])
            else:
                derived = _ritz_record(meta, arrays)
                energy, rho = derived["final_energy_256"], derived["final_rho"]
                if not math.isfinite(energy) or (rho is not None and not math.isfinite(rho)):
                    raise ValueError("Nonfinite independent Ritz diagnostic.")
                if rho is None:
                    warnings.append(f"{name}: independent residual unavailable after rank check.")
                condition = derived["metric_condition_number"]
                if not math.isfinite(condition):
                    warnings.append(f"{name}: nonfinite training-metric condition number.")
                    condition = None
                diagnostic = dict(energy256=energy, rho256=rho,
                                  feature_full_rank=derived["feature_full_rank"],
                                  condition=condition)
            runs.append(dict(run=name, accepted_steps=steps, stop_reason=reason,
                             final_gradient=float(arrays["gradient_norm"][-1]),
                             **diagnostic))
        except Exception as exc:
            critical.append(f"{name}: {type(exc).__name__}: {exc}")
    free_gib = shutil.disk_usage(root).free / 2**30
    if free_gib < 2:
        critical.append(f"Only {free_gib:.2f} GiB of disk space remains.")
    status = "critical" if critical else "complete" if (
        len(records) + len(summary.get("skipped", [])) == 73) else "running"
    return dict(status=status, completed=len(records), expected=73,
                skipped=len(summary.get("skipped", [])), free_gib=free_gib,
                critical=critical, warnings=warnings, runs=runs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stop-on-critical", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = audit(args.root)
    if args.stop_on_critical and report["critical"]:
        marker = args.root / "STOP_REQUESTED"
        try:
            with marker.open("x", encoding="utf-8") as stream:
                stream.write(datetime.now(timezone.utc).isoformat() + "\n")
                stream.write("\n".join(report["critical"]) + "\n")
        except FileExistsError:
            pass
        report["stop_requested"] = True
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
