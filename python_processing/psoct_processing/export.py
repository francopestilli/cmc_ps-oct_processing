from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

try:
    import h5py
except ModuleNotFoundError:  # pragma: no cover
    h5py = None

try:
    import zarr
except ModuleNotFoundError:  # pragma: no cover
    zarr = None


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_tiff(array: np.ndarray, path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    tifffile.imwrite(str(path), array.astype(np.float32), photometric="minisblack")
    return path


def save_zarr(array: np.ndarray, path: str | Path, dataset: str = "data", chunks: tuple[int, ...] | str | None = "auto") -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    if zarr is None:
        # Lightweight fallback for development environments where zarr is not installed.
        # The directory keeps the .zarr suffix and stores the requested dataset as a NumPy array.
        ensure_dir(path)
        np.save(path / f"{dataset}.npy", array)
        return path
    root = zarr.open_group(str(path), mode="w")
    chunk_spec = True if chunks in ("auto", None) else chunks
    root.create_dataset(dataset, data=array, overwrite=True, chunks=chunk_spec)
    return path


def save_hdf5(array: np.ndarray, path: str | Path, dataset: str = "data") -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    if h5py is None:
        raise RuntimeError("h5py is required for HDF5 export but is not installed.")
    with h5py.File(path, "w") as f:
        f.create_dataset(dataset, data=array, compression="gzip")
    return path


def save_json(metadata: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return path
