from __future__ import annotations

import numpy as np


def crop_depth(volume: np.ndarray, depth_range: tuple[int, int] | None, depth_axis: int = -1) -> np.ndarray:
    if depth_range is None:
        return volume
    start, stop = depth_range
    slicer = [slice(None)] * volume.ndim
    slicer[depth_axis] = slice(start, stop)
    return volume[tuple(slicer)]


def create_enface_projection(volume: np.ndarray, method: str = "mean", depth_range: tuple[int, int] | None = None, depth_axis: int = -1) -> np.ndarray:
    work = crop_depth(volume, depth_range, depth_axis=depth_axis)
    if method == "mean":
        return np.mean(work, axis=depth_axis).astype(np.float32)
    if method == "max":
        return np.max(work, axis=depth_axis).astype(np.float32)
    if method == "median":
        return np.median(work, axis=depth_axis).astype(np.float32)
    if method == "sum":
        return np.sum(work, axis=depth_axis).astype(np.float32)
    raise ValueError(f"Unknown projection method: {method}")


def combo_mask_cross_d(db_volume: np.ndarray, db_limit: float, endp: int) -> np.ndarray:
    """Vectorized implementation of MATLAB CombomaskCrossD.m.

    Input is expected as [depth, x, y]. Values above ``db_limit`` are averaged
    across depth up to ``endp``. Pixels without suprathreshold samples are set to
    ``db_limit / 2``, matching the MATLAB fallback.
    """
    if db_volume is None:
        raise ValueError("db_volume cannot be None.")
    volume = np.asarray(db_volume)
    if volume.ndim != 3:
        raise ValueError("combo_mask_cross_d expects [depth, x, y].")
    limited = volume[:endp, :, :]
    mask = limited > db_limit
    sums = np.sum(np.where(mask, limited, 0.0), axis=0)
    counts = np.sum(mask, axis=0)
    out = sums / np.maximum(counts, 1)
    out = np.where(counts == 0, db_limit / 2.0, out)
    return out.astype(np.float32)
