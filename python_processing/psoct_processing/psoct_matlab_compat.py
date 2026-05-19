from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .contrasts import (
    compute_psoct_cross_polarization,
    compute_psoct_reflectivity,
    compute_psoct_retardance_complex,
    compute_weighted_delta_phase,
)
from .enface import combo_mask_cross_d
from .export import ensure_dir, save_tiff
from .io import read2024_scan_stack
from .reconstruct import interp_and_cdp, interpolate_wavelengths_octoplus_2024


@dataclass(frozen=True)
class TileProducts:
    reflectivity: np.ndarray | None
    cross_polarization: np.ndarray | None
    retardance: np.ndarray | None
    weighted_delta_phase: np.ndarray | None
    enface_reflectivity: np.ndarray | None
    enface_cross: np.ndarray | None
    enface_retardance: np.ndarray | None
    metadata: dict[str, object]


def process_tile_2025(
    filename_prefix: str | Path,
    scans: list[int],
    *,
    depth_start: int = 0,
    depth_stop: int | None = None,
    noise_floor_db: float = 0.0,
    cut: int = 185,
    calc_reflectivity: bool = True,
    calc_cross_polarization: bool = True,
    calc_retardance: bool = True,
    calc_orientation_basis: bool = True,
    window_data: bool = False,
    dispersion_compensation: bool = False,
    byte_order: str = "big",
) -> TileProducts:
    """Process one PS-OCT tile using the main MATLAB 2025 algorithmic path.

    This function ports the central loop of ``PSOCT_2025_FCN.m``: Read2024 raw
    loading, channel splitting, InterpandCDP2 reconstruction, reflectivity,
    cross-polarization, complex retardance, weighted phase-difference basis, and
    en-face projections. It deliberately does not yet implement the CFM2 absolute
    orientation correction, because that requires acquisition-specific calibration
    files and policy decisions.
    """
    raw, bg, bg_lines, read_meta = read2024_scan_stack(filename_prefix, scans, byte_order=byte_order)
    spectral2, bline_length, n_lines = raw.shape
    if spectral2 % 2 != 0:
        raise ValueError("Expected two concatenated spectral channels along axis 0.")
    aline_length = spectral2 // 2
    depth_stop = depth_stop if depth_stop is not None else aline_length // 2 - 10
    calibration = interpolate_wavelengths_octoplus_2024(4 * aline_length, aline_length, 1, aline_length + 1)

    out_depth = depth_stop - depth_start
    shape = (out_depth, bline_length, n_lines)
    tile_ref = np.zeros(shape, dtype=np.float32) if calc_reflectivity else None
    tile_cross = np.zeros(shape, dtype=np.float32) if calc_cross_polarization else None
    tile_ret = np.zeros(shape, dtype=np.complex64) if calc_retardance else None
    tile_wdp = np.zeros(shape, dtype=np.complex64) if calc_orientation_basis else None

    for line in range(n_lines):
        bscan = raw[:, :, line]
        ch1 = bscan[:aline_length, :]
        ch2 = bscan[aline_length:, :]
        cdp1, cdp2 = interp_and_cdp(
            ch1,
            ch2,
            calibration,
            window_data=window_data,
            dispersion_compensation=dispersion_compensation,
        )
        reflectivity_db = compute_psoct_reflectivity(cdp1, cdp2, log_db=True)
        if calc_reflectivity:
            tile_ref[:, :, line] = reflectivity_db[depth_start:depth_stop, :]
        if calc_cross_polarization:
            tile_cross[:, :, line] = compute_psoct_cross_polarization(cdp2, log_db=True)[depth_start:depth_stop, :]
        if calc_retardance:
            tile_ret[:, :, line] = compute_psoct_retardance_complex(cdp1, cdp2, reflectivity_db, noise_floor_db)[depth_start:depth_stop, :]
        if calc_orientation_basis:
            tile_wdp[:, :, line] = compute_weighted_delta_phase(cdp1, cdp2)[depth_start:depth_stop, :]

    en_ref = combo_mask_cross_d(tile_ref, noise_floor_db + 10, cut) if tile_ref is not None else None
    en_cross = combo_mask_cross_d(tile_cross, noise_floor_db, cut) if tile_cross is not None else None
    en_ret = np.squeeze(np.sum(tile_ret[:cut, :, :], axis=0)) if tile_ret is not None else None

    metadata = {
        "filename_prefix": str(filename_prefix),
        "scans": scans,
        "raw_shape": tuple(int(v) for v in raw.shape),
        "aline_length": aline_length,
        "bline_length": bline_length,
        "n_lines": n_lines,
        "depth_start": depth_start,
        "depth_stop": depth_stop,
        "read2024": read_meta,
        "implemented_from": ["Read2024.m", "InterpolateWavelengths5.m", "InterpandCDP2.m", "PSOCT_2025_FCN.m"],
        "not_yet_implemented": ["CFM2 absolute-orientation correction", "MATLAB .mat saving parity", "batch slice/tile folder orchestration"],
    }
    return TileProducts(tile_ref, tile_cross, tile_ret, tile_wdp, en_ref, en_cross, en_ret, metadata)


def save_tile_products(products: TileProducts, output_dir: str | Path, stem: str = "tile") -> dict[str, str]:
    out = ensure_dir(output_dir)
    paths: dict[str, str] = {}
    if products.reflectivity is not None:
        paths["reflectivity"] = str(save_tiff(products.reflectivity, out / f"{stem}_reflectivity.tiff"))
    if products.cross_polarization is not None:
        paths["cross_polarization"] = str(save_tiff(products.cross_polarization, out / f"{stem}_cross_polarization.tiff"))
    if products.enface_reflectivity is not None:
        paths["enface_reflectivity"] = str(save_tiff(products.enface_reflectivity, out / f"{stem}_enface_reflectivity.tiff"))
    if products.enface_cross is not None:
        paths["enface_cross"] = str(save_tiff(products.enface_cross, out / f"{stem}_enface_cross.tiff"))
    return paths
