"""Lossless trajectory artifacts, provenance hashes, and environment snapshots."""

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform

import matplotlib
import numpy as np
import scipy
import torch


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment_snapshot(device):
    device = torch.device(device)
    return dict(python=platform.python_version(), pytorch=torch.__version__,
                torch_cuda=torch.version.cuda, numpy=np.__version__, scipy=scipy.__version__,
                matplotlib=matplotlib.__version__, platform=platform.platform(), device=str(device),
                torch_num_threads=torch.get_num_threads(),
                torch_num_interop_threads=torch.get_num_interop_threads(),
                device_name=(torch.cuda.get_device_name(device)
                             if device.type == "cuda" else platform.processor()))


def json_safe(value):
    if isinstance(value, torch.Tensor):
        return json_safe(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    temporary.replace(path)


def save_run(directory, config, result, plan_path):
    """Create a new run directory and refuse to overwrite an existing run."""
    if config.get("run_kind") not in ("unit_smoke", "calibration", "formal"):
        raise ValueError("Explicit run_kind is required.")
    if not result.trajectory:
        raise ValueError("A run must retain its initial state.")
    directory = Path(directory)
    module_root = Path(__file__).resolve().parent
    source_hashes = {path.relative_to(module_root).as_posix(): file_hash(path)
                     for path in sorted(module_root.glob("*.py"))}
    rows = result.trajectory
    if [row["step"] for row in rows] != list(range(result.accepted_steps + 1)):
        raise ValueError("Trajectory must include every accepted step exactly once.")
    arrays = dict(
        step=np.asarray([row["step"] for row in rows], dtype=np.int64),
        parameters=np.asarray([row["parameters"] for row in rows], dtype=np.float64),
        objective=np.asarray([row["objective"] for row in rows], dtype=np.float64),
        gradient_norm=np.asarray([row["gradient_norm"] for row in rows], dtype=np.float64),
        parameter_norm=np.asarray([row["parameter_norm"] for row in rows], dtype=np.float64),
        elapsed_seconds=np.asarray([row["elapsed_seconds"] for row in rows], dtype=np.float64),
        step_size=np.asarray([row.get("step_size", np.nan) for row in rows], dtype=np.float64),
        backtracks=np.asarray([row.get("backtracks", -1) for row in rows], dtype=np.int64),
        armijo_margin=np.asarray([row.get("armijo_margin", np.nan) for row in rows], dtype=np.float64))
    if arrays["parameters"].ndim != 2 or not np.isfinite(arrays["parameters"]).all():
        raise ValueError("Invalid coefficient trajectory.")
    if not np.isfinite(arrays["objective"]).all() or not np.isfinite(arrays["gradient_norm"]).all():
        raise ValueError("Nonfinite objective or gradient trajectory.")
    raw = dict(schema_version=2, created_utc=datetime.now(timezone.utc).isoformat(),
               config=config, environment=environment_snapshot(result.parameters.device),
               plan_sha256=file_hash(plan_path), source_sha256=source_hashes,
               trajectory_file="trajectory.npz", trajectory_encoding="lossless float64 npz",
               result=dict(stop_reason=result.stop_reason,
                           accepted_steps=result.accepted_steps,
                           elapsed_seconds=result.elapsed_seconds))
    directory.mkdir(parents=True, exist_ok=False)
    temporary = directory / "trajectory.tmp.npz"
    np.savez_compressed(temporary, **arrays)
    temporary.replace(directory / "trajectory.npz")
    atomic_json(directory / "metadata.json", raw)
    return directory / "metadata.json"


def load_run(directory):
    """Read a complete schema-2 run without dropping any accepted steps."""
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 2:
        raise ValueError(f"Unsupported trajectory schema in {directory}.")
    with np.load(directory / metadata["trajectory_file"], allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    required = {"step", "parameters", "objective", "gradient_norm", "parameter_norm",
                "elapsed_seconds", "step_size", "backtracks", "armijo_margin"}
    if set(arrays) != required:
        raise ValueError(f"Incomplete trajectory arrays in {directory}.")
    count = metadata["result"]["accepted_steps"] + 1
    if any(array.shape[0] != count for array in arrays.values()):
        raise ValueError(f"Trajectory length mismatch in {directory}.")
    if arrays["parameters"].shape != (count, metadata["config"]["width"]):
        raise ValueError(f"Coefficient shape mismatch in {directory}.")
    if not np.array_equal(arrays["step"], np.arange(count, dtype=np.int64)):
        raise ValueError(f"Missing or unordered accepted steps in {directory}.")
    return metadata, arrays
