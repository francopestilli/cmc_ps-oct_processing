from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import IOConfig
from .data import ChannelMap, RawAcquisition


def discover_raw_files(input_dir: str | Path, extension: str = ".dat") -> list[Path]:
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {root}")
    return sorted(root.rglob(f"*{extension}"))


def numpy_dtype(dtype: str, byte_order: str = "little") -> np.dtype:
    dt = np.dtype(dtype)
    if byte_order == "little":
        return dt.newbyteorder("<")
    if byte_order == "big":
        return dt.newbyteorder(">")
    return dt


def _normalize_axis(axis: int, ndim: int) -> int:
    return axis if axis >= 0 else ndim + axis


def _move_spectral_axis_after_channel(data: np.ndarray, spectral_axis: int) -> tuple[np.ndarray, int]:
    """Move the spectral dimension to axis 1, leaving channel at axis 0."""
    spectral_axis = _normalize_axis(spectral_axis, data.ndim)
    if spectral_axis == 0:
        raise ValueError("spectral_axis cannot be the channel axis after canonicalization.")
    if spectral_axis == 1:
        return data, 1
    data = np.moveaxis(data, spectral_axis, 1)
    return data, 1


def canonicalize_channels(data: np.ndarray, config: IOConfig) -> tuple[np.ndarray, ChannelMap, int]:
    """Return data with channel axis 0 and spectral axis 1.

    Supported raw layouts:
    - none: single-channel data; a singleton channel axis is inserted.
    - axis: an existing raw axis contains channels.
    - interleaved_alines: the A-line axis contains alternating channels.
    - interleaved_samples: spectral samples are interleaved by channel.
    """
    ch = config.channels
    names = tuple(ch.names)

    if ch.layout == "none":
        if ch.n_channels != 1:
            raise ValueError("layout='none' is only valid for one channel.")
        data = np.expand_dims(data, axis=0)
        spectral_axis = _normalize_axis(config.spectral_axis, data.ndim - 1) + 1
        return _move_spectral_axis_after_channel(data, spectral_axis)[0], ChannelMap(names=names), 1

    if ch.layout == "axis":
        if ch.channel_axis is None:
            raise ValueError("channel_axis must be provided when layout='axis'.")
        channel_axis = _normalize_axis(ch.channel_axis, data.ndim)
        if data.shape[channel_axis] != ch.n_channels:
            raise ValueError(
                f"Channel axis has length {data.shape[channel_axis]}, expected {ch.n_channels}."
            )
        data = np.moveaxis(data, channel_axis, 0)
        spectral_axis = _normalize_axis(config.spectral_axis, data.ndim)
        if spectral_axis == channel_axis:
            raise ValueError("spectral_axis and channel_axis cannot be the same.")
        spectral_axis = 0 if spectral_axis == channel_axis else spectral_axis
        # Account for moving the channel axis to the front.
        old_axes = list(range(data.ndim))
        # Easier: identify spectral dimension by original axis length/moveaxis mapping.
        original_axes_after_move = [channel_axis] + [ax for ax in range(data.ndim) if ax != channel_axis]
        spectral_axis = original_axes_after_move.index(_normalize_axis(config.spectral_axis, data.ndim))
        data, spectral_axis = _move_spectral_axis_after_channel(data, spectral_axis)
        return data, ChannelMap(names=names), spectral_axis

    if ch.layout == "interleaved_alines":
        # Assumes axis 1 is the A-line axis if data is [bscan, aline, spectral], otherwise
        # use the first non-spectral axis. The result is [channel, spectral, ...].
        spectral_axis = _normalize_axis(config.spectral_axis, data.ndim)
        candidate_axes = [ax for ax in range(data.ndim) if ax != spectral_axis]
        aline_axis = candidate_axes[-1] if len(candidate_axes) == 1 else candidate_axes[1 if len(candidate_axes) > 1 else 0]
        n = data.shape[aline_axis]
        if n % ch.n_channels != 0:
            raise ValueError("A-line axis length is not divisible by n_channels.")
        parts = [np.take(data, np.arange(i, n, ch.n_channels), axis=aline_axis) for i in range(ch.n_channels)]
        data = np.stack(parts, axis=0)
        spectral_axis = spectral_axis + 1
        data, spectral_axis = _move_spectral_axis_after_channel(data, spectral_axis)
        return data, ChannelMap(names=names), spectral_axis

    if ch.layout == "interleaved_samples":
        spectral_axis = _normalize_axis(config.spectral_axis, data.ndim)
        n = data.shape[spectral_axis]
        if n % ch.n_channels != 0:
            raise ValueError("Spectral axis length is not divisible by n_channels.")
        parts = [np.take(data, np.arange(i, n, ch.n_channels), axis=spectral_axis) for i in range(ch.n_channels)]
        data = np.stack(parts, axis=0)
        spectral_axis = spectral_axis + 1
        data, spectral_axis = _move_spectral_axis_after_channel(data, spectral_axis)
        return data, ChannelMap(names=names), spectral_axis

    raise ValueError(f"Unsupported channel layout: {ch.layout}")


def read_raw_dat(path: str | Path, config: IOConfig) -> RawAcquisition:
    """Read a raw LabView-style binary `.dat` file and apply channel schema.

    The returned data always uses the internal convention [channel, spectral, ...].
    """
    path = Path(path)
    dt = numpy_dtype(config.dtype, config.byte_order)
    with path.open("rb") as f:
        header = f.read(config.header_bytes) if config.header_bytes else b""
        data = np.fromfile(f, dtype=dt)

    metadata = {
        "filename": path.name,
        "header_bytes": config.header_bytes,
        "header_preview": header[:512].decode("utf-8", errors="replace") if header else "",
        "dtype": str(dt),
        "n_values": int(data.size),
        "configured_shape": config.shape,
        "channel_layout": config.channels.model_dump(),
    }

    if config.shape is not None:
        expected = int(np.prod(config.shape))
        if expected != data.size:
            raise ValueError(f"Configured shape {config.shape} expects {expected} values; file has {data.size}.")
        data = data.reshape(config.shape)

    data, channels, spectral_axis = canonicalize_channels(data.astype(np.float32, copy=False), config)
    metadata["canonical_shape"] = tuple(int(v) for v in data.shape)
    metadata["canonical_axis_order"] = "channel, spectral, remaining acquisition axes"

    return RawAcquisition(data=data, path=path, metadata=metadata, channels=channels, spectral_axis=spectral_axis)

# -----------------------------------------------------------------------------
# MATLAB Read2024.m compatible utilities.
# -----------------------------------------------------------------------------

import re
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class LabViewHeader:
    text: str
    values: dict[str, float | int | str]

    def get_int(self, name: str) -> int:
        if name not in self.values:
            raise KeyError(f"Header variable {name!r} was not found.")
        return int(self.values[name])


_HEADER_ASSIGNMENT_RE = re.compile(r"\b([A-Za-z_]\w*)\s*=\s*([^;\n\r]+)")


def parse_labview_header(header_text: str) -> LabViewHeader:
    """Parse simple LabView/MATLAB-style assignment headers safely.

    The MATLAB code uses evalc(headerStr). This Python function intentionally
    avoids eval and extracts assignments such as ``alineLength=1024;``.
    Non-numeric values are retained as stripped strings.
    """
    values: dict[str, float | int | str] = {}
    for name, raw_value in _HEADER_ASSIGNMENT_RE.findall(header_text):
        value = raw_value.strip().strip("'").strip('"')
        try:
            f = float(value)
            values[name] = int(f) if f.is_integer() else f
        except ValueError:
            values[name] = value
    return LabViewHeader(text=header_text, values=values)


def read_labview_dat_file(
    path: str | Path,
    *,
    dtype: str = "int16",
    byte_order: str = "little",
    shape: tuple[int, ...] | None = None,
    order: str = "F",
) -> tuple[np.ndarray, LabViewHeader]:
    """Read one LabView .dat file with a one-line text header.

    When ``shape`` is provided the binary payload is reshaped using MATLAB-like
    column-major order by default.
    """
    path = Path(path)
    dt = numpy_dtype(dtype, byte_order)
    with path.open("rb") as f:
        header_line = f.readline().decode("utf-8", errors="replace")
        payload = np.fromfile(f, dtype=dt)
    header = parse_labview_header(header_line)
    if shape is not None:
        expected = int(np.prod(shape))
        if payload.size != expected:
            raise ValueError(f"{path} contains {payload.size} values; expected {expected} for shape {shape}.")
        payload = payload.reshape(shape, order=order)
    return payload.astype(np.float32, copy=False), header


def read2024_scan_stack(
    filename_prefix: str | Path,
    scans: Iterable[int],
    *,
    aline_length: int | None = None,
    bline_length: int | None = None,
    num_bscans: int | None = None,
    dtype: str = "int16",
    byte_order: str = "big",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Python implementation of MATLAB Read2024.m.

    ``filename_prefix`` corresponds to the MATLAB ``filename_temp`` argument;
    files are read as ``{prefix}{scan}.dat``. The returned tuple is
    ``(raw_tile_minus_bg, BG, BGlines, metadata)``. Shapes follow MATLAB:
    ``Raw_Tile == [alineLength*2, blineLength, len(scans)*num_bscans]``.

    The original MATLAB function indexes output blocks using ``scan*10`` and
    assumes ten b-scans per file. This implementation generalizes to
    ``num_bscans`` and appends scans in the order requested.
    """
    scans = list(scans)
    if not scans:
        raise ValueError("At least one scan index is required.")

    raw_blocks: list[np.ndarray] = []
    bg_lines_blocks: list[np.ndarray] = []
    headers: list[dict[str, object]] = []

    for scan in scans:
        path = Path(f"{filename_prefix}{scan}.dat")
        # Read the header first to infer dimensions if they were not supplied.
        data_1d, header = read_labview_dat_file(path, dtype=dtype, byte_order=byte_order, shape=None)
        a_len = int(aline_length or header.values.get("alineLength", 0))
        b_len = int(bline_length or header.values.get("blineLength", 0))
        n_b = int(num_bscans or header.values.get("buffersPerFile", header.values.get("num_bscans", 0)))
        if a_len <= 0 or b_len <= 0 or n_b <= 0:
            raise ValueError(f"Could not infer alineLength, blineLength, and buffersPerFile from {path}.")
        shape = (a_len * 2, b_len, n_b)
        expected = int(np.prod(shape))
        if data_1d.size != expected:
            raise ValueError(f"{path} contains {data_1d.size} values; expected {expected} for shape {shape}.")
        all_bscans = data_1d.reshape(shape, order="F")
        raw_blocks.append(all_bscans)
        bg_lines_blocks.append(np.mean(all_bscans, axis=1))
        headers.append(header.values)

    raw_tile = np.concatenate(raw_blocks, axis=2)
    bg_lines = np.concatenate(bg_lines_blocks, axis=1)
    bg = np.mean(raw_tile, axis=(1, 2), keepdims=True)
    raw_tile = raw_tile - bg
    metadata = {
        "filename_prefix": str(filename_prefix),
        "scans": scans,
        "headers": headers,
        "shape": tuple(int(v) for v in raw_tile.shape),
        "matlab_source": "Read2024.m",
    }
    return raw_tile.astype(np.float32, copy=False), bg.squeeze().astype(np.float32), bg_lines.astype(np.float32), metadata
