from __future__ import annotations

import numpy as np

from .data import ReconstructedVolume


def compute_reflectivity(complex_volume: np.ndarray, log_db: bool = True, eps: float = 1e-12) -> np.ndarray:
    power = np.abs(complex_volume) ** 2
    if log_db:
        return (10.0 * np.log10(power + eps)).astype(np.float32)
    return power.astype(np.float32)


def compute_phase(complex_volume: np.ndarray) -> np.ndarray:
    return np.angle(complex_volume).astype(np.float32)


def compute_channel_reflectivity(volume: ReconstructedVolume, channel: str, log_db: bool = True, eps: float = 1e-12) -> np.ndarray:
    return compute_reflectivity(volume.get_channel(channel), log_db=log_db, eps=eps)


def compute_polarization_ratio(
    volume: ReconstructedVolume,
    co_pol_channel: str = "co_pol",
    cross_pol_channel: str = "cross_pol",
    eps: float = 1e-12,
    log_db: bool = True,
) -> np.ndarray:
    co = volume.get_channel(co_pol_channel)
    cross = volume.get_channel(cross_pol_channel)
    ratio = (np.abs(cross) ** 2 + eps) / (np.abs(co) ** 2 + eps)
    if log_db:
        ratio = 10.0 * np.log10(ratio)
    return ratio.astype(np.float32)


def compute_retardance(
    volume: ReconstructedVolume,
    co_pol_channel: str = "co_pol",
    cross_pol_channel: str = "cross_pol",
    eps: float = 1e-12,
) -> np.ndarray:
    co = np.abs(volume.get_channel(co_pol_channel))
    cross = np.abs(volume.get_channel(cross_pol_channel))
    return np.arctan2(cross, co + eps).astype(np.float32)


# MATLAB-derived contrast primitives from PSOCT_2025_FCN.m.

def compute_psoct_reflectivity(cdp1: np.ndarray, cdp2: np.ndarray, eps: float = 1e-12, log_db: bool = True) -> np.ndarray:
    reflectivity = np.abs(cdp1) ** 2 + np.abs(cdp2) ** 2
    if log_db:
        reflectivity = 10.0 * np.log10(reflectivity + eps)
    return reflectivity.astype(np.float32)


def compute_psoct_cross_polarization(cdp2: np.ndarray, eps: float = 1e-12, log_db: bool = True) -> np.ndarray:
    cross = np.abs(cdp2) ** 2
    if log_db:
        cross = 10.0 * np.log10(cross + eps)
    return cross.astype(np.float32)


def compute_psoct_retardance_complex(cdp1: np.ndarray, cdp2: np.ndarray, reflectivity_db: np.ndarray, noise_floor_db: float) -> np.ndarray:
    """Approximate the complex retardance expression used in PSOCT_2025_FCN.m."""
    amp1 = np.abs(cdp1)
    amp2 = np.abs(cdp2)
    gated = (reflectivity_db - noise_floor_db) * (reflectivity_db >= noise_floor_db)
    return (gated * np.exp(1j * np.arctan2(amp2, amp1))).astype(np.complex64)


def compute_weighted_delta_phase(cdp1: np.ndarray, cdp2: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Weighted phase-difference quantity used as the basis for orientation."""
    amp1_db = 10.0 * np.log10(np.abs(cdp1) ** 2 + eps)
    amp2_db = 10.0 * np.log10(np.abs(cdp2) ** 2 + eps)
    weighted = cdp2 * np.conj(cdp1)
    denom = amp1_db * amp2_db
    return (weighted / (denom + eps)) * (np.minimum(amp1_db, amp2_db) ** 2)


def compute_orientation(*_args, **_kwargs) -> np.ndarray:
    raise NotImplementedError(
        "Absolute orientation still requires the instrument-specific calibration correction "
        "factor CFM2 and the acquisition-specific reference-line policy."
    )
