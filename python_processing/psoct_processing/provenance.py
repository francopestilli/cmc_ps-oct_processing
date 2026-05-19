from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from . import __version__
except Exception:  # pragma: no cover
    __version__ = "unknown"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def describe_file(path: str | Path, hash_file: bool = False) -> dict[str, Any]:
    p = Path(path)
    stat = p.stat()
    out: dict[str, Any] = {
        "path": str(p),
        "name": p.name,
        "size_bytes": stat.st_size,
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }
    if hash_file:
        out["sha256"] = sha256_file(p)
    return out


def runtime_provenance() -> dict[str, Any]:
    return {
        "psoct_processing_version": __version__,
        "created_utc": utc_now_iso(),
        "python": sys.version,
        "platform": platform.platform(),
    }


def build_run_manifest(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    config: Any,
    products: dict[str, str],
    notes: list[str] | None = None,
    hash_inputs: bool = False,
) -> dict[str, Any]:
    cfg = config.model_dump(mode="json") if hasattr(config, "model_dump") else config
    return {
        "runtime": runtime_provenance(),
        "input": describe_file(input_path, hash_file=hash_inputs),
        "output_dir": str(Path(output_dir)),
        "configuration": cfg,
        "products": products,
        "notes": notes or [],
    }


def save_manifest(manifest: dict[str, Any], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return p
