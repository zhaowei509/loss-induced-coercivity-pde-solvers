"""Immutable JSON artifacts and reproducibility metadata for supplements."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any, Iterable

import numpy as np
import scipy
try:
    import torch
except ModuleNotFoundError:  # NumPy-only certificate jobs may run without PyTorch.
    torch = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if torch is not None and isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return value.item() if value.ndim == 0 else value.tolist()
    if isinstance(value, np.ndarray):
        return value.item() if value.ndim == 0 else value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: Any) -> str:
    """Hash shape, dtype, and C-order bytes of an exact numerical array."""
    array = np.ascontiguousarray(
        value.detach().cpu().numpy()
        if torch is not None and isinstance(value, torch.Tensor) else value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def source_hashes(paths: Iterable[str | Path], *, root: str | Path) -> dict[str, str]:
    root = Path(root).resolve()
    result: dict[str, str] = {}
    for raw in sorted({Path(path).resolve() for path in paths}):
        if not raw.is_file():
            raise FileNotFoundError(raw)
        try:
            name = str(raw.relative_to(root))
        except ValueError:
            name = str(raw)
        result[name] = sha256_file(raw)
    return result


def environment_snapshot() -> dict[str, Any]:
    cuda = torch is not None and torch.cuda.is_available()
    return {
        "created_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": None if torch is None else torch.__version__,
        "torch_cuda_version": None if torch is None else torch.version.cuda,
        "cuda_available": cuda,
        "cuda_device": torch.cuda.get_device_name(0) if cuda else None,
        "pid": os.getpid(),
    }


def git_snapshot(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    snapshot: dict[str, Any] = {"commit": None, "dirty": None}
    try:
        snapshot["commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True).stdout.strip()
        snapshot["dirty"] = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True,
            capture_output=True, text=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        pass
    return snapshot


def create_directory(path: str | Path) -> Path:
    """Create a run directory and refuse to overwrite any prior artifact."""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=False)
    return target


def freeze_file(source: str | Path, destination: str | Path) -> Path:
    """Copy a frozen input into a new artifact directory without overwriting."""
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return destination


def write_json(path: str | Path, payload: Any) -> Path:
    """Atomically create JSON; an existing path is always an error."""
    target = Path(path)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(json_safe(payload), handle, ensure_ascii=False, indent=2,
                      sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def replace_json(path: str | Path, payload: Any) -> Path:
    """Atomically create or replace a mutable campaign index such as summary.json."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(json_safe(payload), handle, ensure_ascii=False, indent=2,
                      sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target
