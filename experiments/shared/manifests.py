"""Manifest builder shared by independently versioned supplement campaigns."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .artifacts import environment_snapshot, git_snapshot, sha256_file, source_hashes, utc_now


def build_manifest(*, project_root: str | Path, supplement: str,
                   run_kind: str, plan_path: str | Path,
                   source_paths: Iterable[str | Path],
                   frozen_config: dict[str, Any]) -> dict[str, Any]:
    root = Path(project_root).resolve()
    plan = Path(plan_path).resolve()
    return {
        "schema_version": 1,
        "supplement": supplement,
        "run_kind": run_kind,
        "created_utc": utc_now(),
        "plan_path": str(plan),
        "plan_sha256": sha256_file(plan),
        "source_sha256": source_hashes(source_paths, root=root),
        "environment": environment_snapshot(),
        "git": git_snapshot(root),
        "frozen_config": frozen_config,
    }

