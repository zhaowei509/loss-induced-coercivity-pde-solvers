#!/usr/bin/env python3
"""Verify checksums, inventory, and readability of the experiment archive."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "file_manifest.csv"
IGNORED_PARTS = {
    ".git",
    "__pycache__",
    ".ipynb_checkpoints",
    ".pytest_cache",
    ".mypy_cache",
}


def is_transient(path: Path) -> bool:
    """Return whether a path is a local cache or repository-control file."""
    relative = path.relative_to(ROOT)
    return (
        any(part in IGNORED_PARTS for part in relative.parts)
        or path.name == ".DS_Store"
        or path.suffix in {".pyc", ".pyo"}
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    with MANIFEST.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    paths = [row["path"] for row in rows]
    if len(paths) != len(set(paths)):
        raise ValueError("The file manifest contains duplicate paths.")

    expected = set(paths)
    actual = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file() and path != MANIFEST and not is_transient(path)
    }
    if actual != expected:
        missing = sorted(expected - actual)
        unlisted = sorted(actual - expected)
        raise ValueError(f"Inventory mismatch; missing={missing}, unlisted={unlisted}")

    for row in rows:
        path = ROOT / row["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256(path) != row["sha256"]:
            raise ValueError(f"Checksum mismatch: {row['path']}")
        suffix = path.suffix.lower()
        if suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
        elif suffix == ".csv":
            with path.open(newline="", encoding="utf-8") as data_stream:
                list(csv.reader(data_stream))
        elif suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                for name in archive.files:
                    archive[name]
        elif suffix == ".py":
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        elif suffix == ".ipynb":
            json.loads(path.read_text(encoding="utf-8"))
    provenance = ROOT / "source_provenance.csv"
    unavailable = 0
    if provenance.is_file():
        with provenance.open(newline="", encoding="utf-8") as stream:
            unavailable = sum(
                row["status"] == "historical_revision_unavailable"
                for row in csv.DictReader(stream)
            )
    print(
        f"Verified {len(rows)} listed files and the complete archive inventory; "
        f"{unavailable} unavailable historical source revisions are documented."
    )


if __name__ == "__main__":
    main()
