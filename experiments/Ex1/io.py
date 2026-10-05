"""Non-overwriting JSON artifacts, provenance hashes, and environment snapshots."""

from dataclasses import asdict
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
    raw = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
               config=config, environment=environment_snapshot(result.parameters.device),
               plan_sha256=file_hash(plan_path), source_sha256=source_hashes,
               nonfinite_encoding="null", result=asdict(result))
    directory.mkdir(parents=True, exist_ok=False)
    atomic_json(directory / "raw.json", raw)
    return directory / "raw.json"
