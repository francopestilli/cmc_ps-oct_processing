from __future__ import annotations

import numpy as np


def estimate_background(volume: np.ndarray, strategy: str = "mean_along_bscan") -> np.ndarray:
    if strategy == "mean_along_bscan":
        if volume.ndim < 2:
            return np.mean(volume, keepdims=True)
        return np.mean(volume, axis=-2, keepdims=True)
    if strategy == "global_mean":
        return np.mean(volume, keepdims=True)
    if strategy == "none":
        return np.zeros_like(volume)
    raise ValueError(f"Unknown background strategy: {strategy}")


def subtract_background(volume: np.ndarray, strategy: str = "mean_along_bscan") -> np.ndarray:
    bg = estimate_background(volume, strategy=strategy)
    return volume - bg


def normalize_percentile(volume: np.ndarray, lower: float = 1.0, upper: float = 99.0) -> np.ndarray:
    lo, hi = np.percentile(volume, [lower, upper])
    if hi <= lo:
        return np.zeros_like(volume, dtype=np.float32)
    return np.clip((volume - lo) / (hi - lo), 0, 1).astype(np.float32)
