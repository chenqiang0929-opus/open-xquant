"""Data manifest — provenance and integrity metadata for parquet files."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal


@dataclass
class ManifestVerification:
    """Result of verifying a parquet file against its manifest."""

    status: Literal["real", "mock", "missing", "corrupted"]
    provider: str | None
    detail: str


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(
    parquet_path: Path,
    symbol: str,
    provider: str,
    start: str,
    end: str,
    rows: int,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Compute parquet sha256, write {symbol}.manifest.json, return manifest path."""
    manifest: dict[str, Any] = {
        "symbol": symbol,
        "provider": provider,
        "start": start,
        "end": end,
        "rows": rows,
        "sha256": _sha256(parquet_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "extra": extra,
    }
    manifest_path = parquet_path.parent / f"{symbol}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest_path


def read_manifest(manifest_path: Path) -> dict[str, Any] | None:
    """Read and parse a manifest file. Return None if missing or malformed."""
    try:
        return json.loads(manifest_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
