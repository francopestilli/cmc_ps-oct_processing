from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class ChannelConfig(BaseModel):
    """Description of one physical or logical acquisition channel."""

    name: str
    index: int
    role: str | None = None
    gain: float = 1.0
    background_strategy: str | None = None
    dispersion_coefficients: tuple[float, float, float] | None = None


class ChannelLayoutConfig(BaseModel):
    """How channel samples are arranged in the raw acquisition."""

    n_channels: int = 1
    names: list[str] = Field(default_factory=lambda: ["intensity"])
    layout: Literal["none", "axis", "interleaved_alines", "interleaved_samples"] = "none"
    channel_axis: int | None = None
    canonical_axis_order: Literal["czyx", "czxy", "cxyz"] = "czyx"
    channels: list[ChannelConfig] | None = None

    @model_validator(mode="after")
    def _validate_channels(self) -> "ChannelLayoutConfig":
        if self.n_channels < 1:
            raise ValueError("n_channels must be at least 1.")
        if self.channels is None:
            self.channels = [ChannelConfig(name=name, index=i) for i, name in enumerate(self.names)]
        if len(self.channels) != self.n_channels:
            raise ValueError("Number of channel definitions must match n_channels.")
        if sorted(ch.index for ch in self.channels) != list(range(self.n_channels)):
            raise ValueError("Channel indices must be contiguous and zero-based.")
        self.names = [ch.name for ch in sorted(self.channels, key=lambda ch: ch.index)]
        return self


class IOConfig(BaseModel):
    raw_extension: str = ".dat"
    dtype: str = "int16"
    byte_order: Literal["little", "big", "native"] = "little"
    header_bytes: int = 0
    shape: tuple[int, ...] | None = None
    spectral_axis: int = -1
    background_strategy: str = "mean_along_bscan"
    channels: ChannelLayoutConfig = ChannelLayoutConfig()


class ReconstructionConfig(BaseModel):
    fft_size: int | None = None
    window: Literal["hann", "hamming", "none"] = "hann"
    crop_depth: tuple[int, int] | None = None
    subtract_dc: bool = True
    dispersion_coefficients: tuple[float, float, float] = (0.0, 0.0, 0.0)
    k_linearization: list[float] | None = None
    reconstruct_per_channel: bool = True


class ContrastConfig(BaseModel):
    reflectivity_log_db: bool = True
    eps: float = 1e-12
    co_pol_channel: str | None = "co_pol"
    cross_pol_channel: str | None = "cross_pol"


class EnfaceConfig(BaseModel):
    method: Literal["mean", "max", "median", "sum"] = "mean"
    depth_range: tuple[int, int] | None = None


class StitchingConfig(BaseModel):
    overlap_fraction: float = Field(default=0.15, ge=0.0, lt=0.9)
    blending: Literal["linear", "mean", "overwrite"] = "linear"
    tile_grid: tuple[int, int] | None = None


class ExportConfig(BaseModel):
    save_tiff: bool = True
    save_zarr: bool = True
    save_hdf5: bool = False
    save_intermediates: bool = False
    zarr_chunks: tuple[int, ...] | str | None = "auto"


class ProvenanceConfig(BaseModel):
    save_manifest: bool = True
    hash_inputs: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class PipelineConfig(BaseModel):
    io: IOConfig = IOConfig()
    reconstruction: ReconstructionConfig = ReconstructionConfig()
    contrasts: ContrastConfig = ContrastConfig()
    enface: EnfaceConfig = EnfaceConfig()
    stitching: StitchingConfig = StitchingConfig()
    export: ExportConfig = ExportConfig()
    provenance: ProvenanceConfig = ProvenanceConfig()

    @model_validator(mode="after")
    def _validate_axis_consistency(self) -> "PipelineConfig":
        if self.io.shape is not None:
            ndim = len(self.io.shape)
            spectral_axis = self.io.spectral_axis if self.io.spectral_axis >= 0 else ndim + self.io.spectral_axis
            if spectral_axis < 0 or spectral_axis >= ndim:
                raise ValueError("io.spectral_axis is outside the configured raw shape.")
            if self.io.channels.layout == "axis":
                axis = self.io.channels.channel_axis
                if axis is None:
                    raise ValueError("channel_axis is required when channel layout is 'axis'.")
                axis = axis if axis >= 0 else ndim + axis
                if axis < 0 or axis >= ndim:
                    raise ValueError("channel_axis is outside the configured raw shape.")
                if self.io.shape[axis] != self.io.channels.n_channels:
                    raise ValueError("Configured raw shape does not match n_channels along channel_axis.")
        return self


def load_config(path: str | Path | None = None) -> PipelineConfig:
    if path is None:
        return PipelineConfig()
    with Path(path).open("r", encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}
    return PipelineConfig.model_validate(data)


def write_default_config(path: str | Path) -> Path:
    """Write a documented default YAML configuration."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cfg = PipelineConfig()
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.model_dump(mode="json"), f, sort_keys=False)
    return p
