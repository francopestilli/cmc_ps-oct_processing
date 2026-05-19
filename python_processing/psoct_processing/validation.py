from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

import numpy as np


@dataclass
class StageComparison:
    stage: str
    shape_matlab: tuple[int, ...]
    shape_python: tuple[int, ...]
    dtype_matlab: str
    dtype_python: str
    max_abs_error: float
    mean_abs_error: float
    rmse: float
    relative_rmse: float
    pearson_r: float | None
    n_values: int


def _load_array(path: str | Path) -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path)
    if suffix == ".npz":
        loaded = np.load(path)
        keys = list(loaded.keys())
        if len(keys) != 1:
            raise ValueError(f"NPZ reference {path} has {len(keys)} arrays; expected exactly one.")
        return loaded[keys[0]]
    if suffix == ".mat":
        try:
            from scipy.io import loadmat
        except ImportError as exc:
            raise ImportError("scipy is required to compare MATLAB .mat files") from exc
        data = loadmat(path)
        candidates = [k for k in data if not k.startswith("__")]
        if len(candidates) != 1:
            raise ValueError(f"MAT file {path} has variables {candidates}; expected exactly one data variable.")
        return np.asarray(data[candidates[0]])
    raise ValueError(f"Unsupported array format for validation: {path}")


def compare_arrays(matlab_array: np.ndarray, python_array: np.ndarray, stage: str) -> StageComparison:
    m = np.asarray(matlab_array)
    p = np.asarray(python_array)
    if m.shape != p.shape:
        raise ValueError(f"Shape mismatch for {stage}: MATLAB {m.shape} vs Python {p.shape}")
    m_float = m.astype(np.float64, copy=False)
    p_float = p.astype(np.float64, copy=False)
    valid = np.isfinite(m_float) & np.isfinite(p_float)
    if not np.any(valid):
        raise ValueError(f"No finite values to compare for stage {stage}")
    diff = p_float[valid] - m_float[valid]
    max_abs = float(np.max(np.abs(diff)))
    mean_abs = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    denom = float(np.sqrt(np.mean(m_float[valid] ** 2)))
    relative_rmse = float(rmse / denom) if denom > 0 else float("nan")
    r = None
    if valid.sum() > 1 and np.std(m_float[valid]) > 0 and np.std(p_float[valid]) > 0:
        r = float(np.corrcoef(m_float[valid].ravel(), p_float[valid].ravel())[0, 1])
    return StageComparison(
        stage=stage,
        shape_matlab=tuple(int(x) for x in m.shape),
        shape_python=tuple(int(x) for x in p.shape),
        dtype_matlab=str(m.dtype),
        dtype_python=str(p.dtype),
        max_abs_error=max_abs,
        mean_abs_error=mean_abs,
        rmse=rmse,
        relative_rmse=relative_rmse,
        pearson_r=r,
        n_values=int(valid.sum()),
    )


def compare_stage_files(matlab_path: str | Path, python_path: str | Path, stage: str | None = None) -> StageComparison:
    matlab_path = Path(matlab_path)
    python_path = Path(python_path)
    return compare_arrays(_load_array(matlab_path), _load_array(python_path), stage or matlab_path.stem)


def compare_stage_directories(matlab_dir: str | Path, python_dir: str | Path, output_json: str | Path | None = None) -> list[StageComparison]:
    matlab_dir = Path(matlab_dir)
    python_dir = Path(python_dir)
    results: list[StageComparison] = []
    matlab_files = sorted([p for p in matlab_dir.iterdir() if p.suffix.lower() in {".npy", ".npz", ".mat"}])
    if not matlab_files:
        raise FileNotFoundError(f"No .npy, .npz, or .mat files found in {matlab_dir}")
    for mpath in matlab_files:
        candidates = [python_dir / mpath.name]
        if mpath.suffix.lower() == ".mat":
            candidates.extend([python_dir / f"{mpath.stem}.npy", python_dir / f"{mpath.stem}.npz"])
        ppath = next((p for p in candidates if p.exists()), None)
        if ppath is None:
            continue
        results.append(compare_stage_files(mpath, ppath, stage=mpath.stem))
    if output_json is not None:
        payload = [asdict(item) for item in results]
        Path(output_json).write_text(json.dumps(payload, indent=2))
    return results


def save_diagnostic_plot(matlab_array: np.ndarray, python_array: np.ndarray, path: str | Path, max_points: int = 20000) -> Path:
    import matplotlib.pyplot as plt

    m = np.asarray(matlab_array).astype(float).ravel()
    p = np.asarray(python_array).astype(float).ravel()
    valid = np.isfinite(m) & np.isfinite(p)
    m = m[valid]
    p = p[valid]
    if m.size > max_points:
        idx = np.linspace(0, m.size - 1, max_points).astype(int)
        m = m[idx]
        p = p[idx]
    fig = plt.figure()
    plt.scatter(m, p, s=2, alpha=0.25)
    plt.xlabel("MATLAB")
    plt.ylabel("Python")
    plt.title("MATLAB vs Python stage comparison")
    plt.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
