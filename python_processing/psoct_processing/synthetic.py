from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import PipelineConfig
from .export import ensure_dir, save_json


def make_synthetic_spectral_cube(
    *,
    n_channels: int = 2,
    spectral_samples: int = 1024,
    x: int = 16,
    y: int = 8,
    seed: int = 42,
) -> np.ndarray:
    """Create a deterministic synthetic channel-first spectral cube.

    The generated data are not a biophysical PS-OCT simulator. They are a
    controlled engineering phantom with smooth spectral structure, channel
    gain differences, spatial modulation, and noise. It is intended for testing
    I/O, reconstruction, projections, and provenance before real acquisitions
    are available.
    """
    rng = np.random.default_rng(seed)
    z = np.linspace(0.0, 1.0, spectral_samples, dtype=np.float32)
    xx = np.linspace(-1.0, 1.0, x, dtype=np.float32)
    yy = np.linspace(-1.0, 1.0, y, dtype=np.float32)
    X, Y = np.meshgrid(xx, yy, indexing="ij")
    cube = np.empty((n_channels, spectral_samples, x, y), dtype=np.float32)
    for c in range(n_channels):
        carrier = np.sin(2 * np.pi * (8 + 2 * c) * z) + 0.5 * np.sin(2 * np.pi * (23 + c) * z)
        envelope = np.exp(-((z - (0.35 + 0.05 * c)) ** 2) / 0.035)
        spatial = 1.0 + 0.2 * X + 0.15 * Y + 0.1 * np.sin(np.pi * (c + 1) * X * Y)
        signal = (carrier * envelope)[:, None, None] * spatial[None, :, :]
        background = 200.0 + 25.0 * c + 10.0 * z[:, None, None]
        noise = rng.normal(0.0, 2.0, size=(spectral_samples, x, y)).astype(np.float32)
        cube[c] = background + 1000.0 * signal + noise
    return cube


def write_synthetic_dat(
    output_dir: str | Path,
    *,
    filename: str = "synthetic_psoct.dat",
    n_channels: int = 2,
    spectral_samples: int = 1024,
    x: int = 16,
    y: int = 8,
    seed: int = 42,
) -> dict[str, str]:
    out = ensure_dir(output_dir)
    cube = make_synthetic_spectral_cube(
        n_channels=n_channels, spectral_samples=spectral_samples, x=x, y=y, seed=seed
    )
    clipped = np.clip(np.rint(cube), np.iinfo(np.int16).min, np.iinfo(np.int16).max).astype("<i2")
    dat_path = out / filename
    clipped.tofile(dat_path)
    meta = {
        "filename": filename,
        "shape_channel_first": [n_channels, spectral_samples, x, y],
        "dtype": "int16",
        "byte_order": "little",
        "layout": "axis",
        "channel_axis": 0,
        "spectral_axis": 1,
        "seed": seed,
        "note": "Engineering phantom, not a physical PS-OCT simulation.",
    }
    meta_path = save_json(meta, out / "synthetic_metadata.json")
    return {"dat": str(dat_path), "metadata": str(meta_path)}


def synthetic_config(n_channels: int = 2, spectral_samples: int = 1024, x: int = 16, y: int = 8) -> PipelineConfig:
    data = {
        "io": {
            "raw_extension": ".dat",
            "dtype": "int16",
            "byte_order": "little",
            "shape": [n_channels, spectral_samples, x, y],
            "spectral_axis": 1,
            "background_strategy": "mean_along_bscan",
            "channels": {
                "n_channels": n_channels,
                "layout": "axis",
                "channel_axis": 0,
                "channels": [
                    {"name": "co_pol" if i == 0 else "cross_pol" if i == 1 else f"channel_{i}", "index": i}
                    for i in range(n_channels)
                ],
            },
        },
        "reconstruction": {"fft_size": spectral_samples, "window": "hann", "subtract_dc": True},
        "export": {"save_tiff": True, "save_zarr": True, "save_hdf5": False, "save_intermediates": True},
    }
    return PipelineConfig.model_validate(data)
