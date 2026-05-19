from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import interp1d

from .config import ReconstructionConfig


@dataclass(frozen=True)
class WavelengthCalibration:
    wavelengths_l: np.ndarray
    wavelengths_r: np.ndarray
    interpolated_wavelengths: np.ndarray
    ks: np.ndarray


def make_window(n: int, kind: str) -> np.ndarray:
    if kind == "hann":
        return np.hanning(n)
    if kind == "hamming":
        return np.hamming(n)
    if kind == "none":
        return np.ones(n)
    raise ValueError(f"Unknown window: {kind}")


def linearize_k_space(volume: np.ndarray, target_k: np.ndarray | None = None, spectral_axis: int = -1) -> np.ndarray:
    if target_k is None:
        return volume
    n = volume.shape[spectral_axis]
    source = np.linspace(0.0, 1.0, n)
    target = np.asarray(target_k, dtype=np.float64)
    if target.shape[0] != n:
        raise ValueError("k-linearization vector length must match spectral dimension.")
    f = interp1d(source, np.moveaxis(volume, spectral_axis, -1), axis=-1, bounds_error=False, fill_value="extrapolate")
    out = f(target)
    return np.moveaxis(out, -1, spectral_axis)


def apply_dispersion_compensation(volume: np.ndarray, coefficients: tuple[float, float, float], spectral_axis: int = -1) -> np.ndarray:
    c0, c1, c2 = coefficients
    if c0 == c1 == c2 == 0.0:
        return volume
    n = volume.shape[spectral_axis]
    x = np.linspace(-1.0, 1.0, n)
    phase = c0 + c1 * x + c2 * x**2
    correction = np.exp(1j * phase)
    shape = [1] * volume.ndim
    shape[spectral_axis] = n
    return volume * correction.reshape(shape)


def reconstruct_volume(volume: np.ndarray, config: ReconstructionConfig, spectral_axis: int = -1) -> np.ndarray:
    work = np.asarray(volume, dtype=np.float32)
    work = linearize_k_space(work, config.k_linearization, spectral_axis=spectral_axis)

    if config.subtract_dc:
        work = work - np.mean(work, axis=spectral_axis, keepdims=True)

    n = work.shape[spectral_axis]
    win = make_window(n, config.window).astype(np.float32)
    shape = [1] * work.ndim
    shape[spectral_axis] = n
    work = work * win.reshape(shape)

    work = apply_dispersion_compensation(work, config.dispersion_coefficients, spectral_axis=spectral_axis)
    fft_size = config.fft_size or n
    complex_volume = np.fft.fft(work, n=fft_size, axis=spectral_axis)

    half = fft_size // 2
    slicer = [slice(None)] * complex_volume.ndim
    slicer[spectral_axis] = slice(0, half)
    complex_volume = complex_volume[tuple(slicer)]

    if config.crop_depth is not None:
        start, stop = config.crop_depth
        slicer = [slice(None)] * complex_volume.ndim
        slicer[spectral_axis] = slice(start, stop)
        complex_volume = complex_volume[tuple(slicer)]

    return complex_volume.astype(np.complex64, copy=False)


# -----------------------------------------------------------------------------
# MATLAB-compatible PS-OCT reconstruction primitives.
# These implement the numerical structure of InterpolateWavelengths5.m and
# InterpandCDP2.m, while keeping the functions explicit and testable in Python.
# -----------------------------------------------------------------------------


def interpolate_wavelengths_octoplus_2024(
    padding_length: int = 4096,
    original_line_length: int = 1024,
    start1: int = 1,
    start2: int = 1025,
) -> WavelengthCalibration:
    """Return the wavelength/k-space calibration used by InterpolateWavelengths5.m.

    The constants are the 11/20/2024 Octoplus new-lens calibration embedded in
    the MATLAB repository. Pixel coordinates are MATLAB-style one-based values;
    interpolation is performed on the same physical grid, then returned as SI
    wavelengths in meters.
    """
    position_l = np.array([5, 42.4, 115.7, 188.8, 262.8, 336.2, 409.7, 483.8, 558, 632.9, 707.5, 781.9, 857, 933, 993], dtype=float)
    position_r = np.array([56.5, 92.1, 162.6, 232.2, 303.9, 375, 445.8, 517.8, 589.9, 662.9, 735.8, 808.6, 882, 956.8, 1015.8], dtype=float) + 1024
    wave_sample = np.array([802.45, 805.01, 810.04, 815.02, 820.06, 825.07, 830.04, 835.05, 840.04, 845.07, 850.06, 855.01, 860.01, 865.05, 869], dtype=float)

    pcoeff1 = np.polyfit(position_l, wave_sample, 2)
    x1 = np.arange(1, original_line_length + 1, dtype=float)
    lamda_l = np.polyval(pcoeff1, x1)

    pcoeff2 = np.polyfit(position_r, wave_sample, 2)
    x2 = np.arange(1025, 1025 + original_line_length, dtype=float)
    lamda_r = np.polyval(pcoeff2, x2)

    xx1 = np.linspace(start1, start1 + original_line_length - 1, padding_length)
    wavelengths_l = 1e-9 * np.interp(xx1, x1, lamda_l)

    xx2 = np.linspace(start2 - 1024, start2 + original_line_length - 1025, padding_length)
    wavelengths_r = 1e-9 * np.interp(xx2, np.arange(1, original_line_length + 1, dtype=float), lamda_r)

    min_k = 2 * np.pi / min(wavelengths_l[-1], wavelengths_r[-1])
    max_k = 2 * np.pi / max(wavelengths_l[0], wavelengths_r[0])
    ks = np.linspace(max_k, min_k, original_line_length)
    interpolated_wavelengths = (2 * np.pi) / ks
    return WavelengthCalibration(wavelengths_l, wavelengths_r, interpolated_wavelengths, ks)


def _interp_columns(source_x: np.ndarray, source_y: np.ndarray, target_x: np.ndarray) -> np.ndarray:
    """Column-wise linear interpolation with linear extrapolation.

    SciPy requires unique x coordinates. Real PS-OCT calibration vectors are
    well behaved, but synthetic/unit-test grids can contain repeated values after
    low-order polynomial fitting; those are collapsed conservatively here.
    """
    x = np.asarray(source_x, dtype=float)
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = np.asarray(source_y)[order, ...]
    x_unique, unique_idx = np.unique(x_sorted, return_index=True)
    y_unique = y_sorted[unique_idx, ...]
    f = interp1d(x_unique, y_unique, axis=0, kind="linear", bounds_error=False, fill_value="extrapolate", assume_sorted=True)
    return f(target_x)


def interp_and_cdp(
    channel1_bscan: np.ndarray,
    channel2_bscan: np.ndarray,
    calibration: WavelengthCalibration | None = None,
    *,
    interp_zero_padding_factor: int = 4,
    cdp_zero_padding_factor: int = 1,
    dispersion_compensation: bool = False,
    phase_correction1: np.ndarray | None = None,
    phase_correction2: np.ndarray | None = None,
    window_data: bool = False,
    auto_peak_corr_cut: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Python implementation of MATLAB InterpandCDP2.m.

    Parameters are named after the MATLAB code. Input arrays are spectral x A-line
    matrices, normally 1024 x blineLength. The return order follows the MATLAB
    function signature: (CDP1, CDP2).
    """
    b1 = np.asarray(channel1_bscan)
    b2 = np.asarray(channel2_bscan)
    if b1.shape != b2.shape:
        raise ValueError("channel1_bscan and channel2_bscan must have the same shape.")
    if b1.ndim != 2:
        raise ValueError("Expected 2D arrays with shape [spectral, aline].")

    aline_length = b1.shape[0]
    original_line_length = aline_length
    mid_length = aline_length // 2
    padding_length = interp_zero_padding_factor * aline_length
    if calibration is None:
        calibration = interpolate_wavelengths_octoplus_2024(padding_length, original_line_length, 1, aline_length + 1)

    def zero_pad_and_ifft(ch: np.ndarray) -> np.ndarray:
        transformed = np.fft.fft(ch, axis=0)[:mid_length, :]
        padded = np.vstack([transformed, np.zeros((padding_length - mid_length, ch.shape[1]), dtype=complex)])
        return np.real(np.fft.ifft(padded, axis=0)) * interp_zero_padding_factor

    zp1 = zero_pad_and_ifft(b1)
    zp2 = zero_pad_and_ifft(b2)

    ib1 = _interp_columns(calibration.wavelengths_l, zp1, calibration.interpolated_wavelengths)
    ib2 = _interp_columns(calibration.wavelengths_r, zp2, calibration.interpolated_wavelengths)

    if dispersion_compensation:
        if phase_correction1 is not None:
            ib1 = ib1 * np.asarray(phase_correction1).reshape(-1, 1)
        if phase_correction2 is not None:
            ib2 = ib2 * np.asarray(phase_correction2).reshape(-1, 1)

    if window_data:
        win = np.hamming(original_line_length).reshape(-1, 1)
        ib1 = ib1 * win
        ib2 = ib2 * win

    fft_n = original_line_length * cdp_zero_padding_factor
    cut_half = fft_n // 2
    cdp1 = np.fft.fft(ib1, n=fft_n, axis=0)[:cut_half, :]
    cdp2 = np.fft.fft(ib2, n=fft_n, axis=0)[:cut_half, :]

    dc_cut = auto_peak_corr_cut * cdp_zero_padding_factor
    if dc_cut > 0:
        cdp1 = cdp1[dc_cut:, :]
        cdp2 = cdp2[dc_cut:, :]
    return cdp1.astype(np.complex64, copy=False), cdp2.astype(np.complex64, copy=False)
