"""Small, dependency-light provenance receipts for datasets and experiments.

The receipt format is intentionally JSON compatible so an experiment can be
verified after it has been copied to another machine or attached to a model
artifact.  It records hashes, not source contents.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
import hashlib
import platform
from pathlib import Path
import sys
from typing import Any


def file_digest(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it all in memory."""

    target = Path(path)
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "executable": sys.executable,
    }


def build_receipt(
    *,
    root: str | Path,
    files: Iterable[str | Path],
    sources: Iterable[Mapping[str, Any]] | None = None,
    environment: Mapping[str, Any] | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic file/source receipt rooted at ``root``.

    File paths are stored relative to ``root``.  A path outside the root is
    rejected instead of silently producing an absolute, machine-specific
    receipt.
    """

    base = Path(root).resolve()
    entries: dict[str, dict[str, Any]] = {}
    for value in files:
        path = Path(value).resolve()
        try:
            relative = path.relative_to(base)
        except ValueError as exc:
            raise ValueError(f"file is outside receipt root: {path}") from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        key = relative.as_posix()
        entries[key] = {"sha256": file_digest(path), "size": path.stat().st_size}

    return {
        "format_version": 1,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "files": dict(sorted(entries.items())),
        "sources": [dict(source) for source in (sources or [])],
        "environment": dict(environment or _environment()),
    }


def validate_receipt(receipt: Mapping[str, Any], root: str | Path) -> list[str]:
    """Return stable validation errors; an empty list means the receipt matches."""

    if receipt.get("format_version") != 1:
        raise ValueError("receipt format_version must be 1")
    files = receipt.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("receipt files must be a mapping")

    base = Path(root).resolve()
    errors: list[str] = []
    for name, metadata in sorted(files.items()):
        path = base / str(name)
        if not path.is_file():
            errors.append(f"{name}: file is missing")
            continue
        if not isinstance(metadata, Mapping) or not isinstance(metadata.get("sha256"), str):
            errors.append(f"{name}: invalid file metadata")
            continue
        if file_digest(path) != metadata["sha256"]:
            errors.append(f"{name}: sha256 mismatch")
            continue
        expected_size = metadata.get("size")
        if expected_size is not None and path.stat().st_size != expected_size:
            errors.append(f"{name}: size mismatch")
    return errors
