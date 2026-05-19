from __future__ import annotations

from pathlib import Path

from .config import PipelineConfig
from .contrasts import compute_channel_reflectivity, compute_polarization_ratio, compute_retardance
from .data import ReconstructedVolume
from .enface import create_enface_projection
from .export import ensure_dir, save_json, save_tiff, save_zarr
from .logging_utils import configure_logging
from .provenance import build_run_manifest, save_manifest
from .io import discover_raw_files, read_raw_dat
from .preprocess import subtract_background
from .reconstruct import reconstruct_volume


def process_file(path: str | Path, output_dir: str | Path, config: PipelineConfig) -> dict:
    output_dir = ensure_dir(output_dir)
    logger = configure_logging(output_dir / "psoct_processing.log", level=config.provenance.log_level)
    logger.info("Processing %s", path)
    raw = read_raw_dat(path, config.io)
    corrected = subtract_background(raw.data, strategy=config.io.background_strategy)
    if config.export.save_intermediates:
        save_zarr(corrected, output_dir / f"{Path(path).stem}_stage_background_corrected.zarr", chunks=config.export.zarr_chunks)
    complex_data = reconstruct_volume(corrected, config.reconstruction, spectral_axis=raw.spectral_axis)
    if config.export.save_intermediates:
        save_zarr(complex_data, output_dir / f"{Path(path).stem}_stage_complex_volume.zarr", chunks=config.export.zarr_chunks)
    volume = ReconstructedVolume(
        data=complex_data,
        channels=raw.channels,
        depth_axis=raw.spectral_axis,
        metadata={"input": raw.metadata},
    )

    stem = Path(path).stem
    products = {}

    primary_channel = raw.channels.names[0]
    reflectivity = compute_channel_reflectivity(
        volume,
        channel=primary_channel,
        log_db=config.contrasts.reflectivity_log_db,
        eps=config.contrasts.eps,
    )
    enface = create_enface_projection(
        reflectivity,
        method=config.enface.method,
        depth_range=config.enface.depth_range,
        depth_axis=volume.depth_axis - 1,  # channel removed for per-channel product
    )

    if config.export.save_tiff:
        products[f"{primary_channel}_reflectivity_tiff"] = str(
            save_tiff(reflectivity, output_dir / f"{stem}_{primary_channel}_reflectivity.tiff")
        )
        products[f"{primary_channel}_enface_tiff"] = str(
            save_tiff(enface, output_dir / f"{stem}_{primary_channel}_enface.tiff")
        )
    if config.export.save_zarr:
        products["complex_volume_zarr"] = str(save_zarr(volume.data, output_dir / f"{stem}_complex_channels.zarr", chunks=config.export.zarr_chunks))
        products[f"{primary_channel}_reflectivity_zarr"] = str(
            save_zarr(reflectivity, output_dir / f"{stem}_{primary_channel}_reflectivity.zarr", chunks=config.export.zarr_chunks)
        )

    # Optional two-channel polarization diagnostics when co/cross channel names are present.
    available = set(raw.channels.names)
    co = config.contrasts.co_pol_channel
    cross = config.contrasts.cross_pol_channel
    if co in available and cross in available:
        pol_ratio = compute_polarization_ratio(volume, co_pol_channel=co, cross_pol_channel=cross, eps=config.contrasts.eps)
        apparent_retardance = compute_retardance(volume, co_pol_channel=co, cross_pol_channel=cross, eps=config.contrasts.eps)
        if config.export.save_tiff:
            products["polarization_ratio_tiff"] = str(save_tiff(pol_ratio, output_dir / f"{stem}_polarization_ratio.tiff"))
            products["apparent_retardance_tiff"] = str(save_tiff(apparent_retardance, output_dir / f"{stem}_apparent_retardance.tiff"))
        if config.export.save_zarr:
            products["polarization_ratio_zarr"] = str(save_zarr(pol_ratio, output_dir / f"{stem}_polarization_ratio.zarr", chunks=config.export.zarr_chunks))
            products["apparent_retardance_zarr"] = str(save_zarr(apparent_retardance, output_dir / f"{stem}_apparent_retardance.zarr", chunks=config.export.zarr_chunks))

    metadata = {
        "input": raw.metadata,
        "channels": list(raw.channels.names),
        "depth_axis": volume.depth_axis,
        "products": products,
        "notes": [
            "Channel-aware arrays are stored internally as [channel, spectral/depth, ...].",
            "apparent_retardance is an uncalibrated two-channel approximation until instrument calibration is added.",
        ],
    }
    save_json(metadata, output_dir / f"{stem}_metadata.json")
    if config.provenance.save_manifest:
        manifest = build_run_manifest(
            input_path=path,
            output_dir=output_dir,
            config=config,
            products=products,
            notes=metadata["notes"],
            hash_inputs=config.provenance.hash_inputs,
        )
        products["manifest_json"] = str(save_manifest(manifest, output_dir / f"{stem}_manifest.json"))
    logger.info("Finished %s", path)
    return metadata


def run_pipeline(input_dir: str | Path, output_dir: str | Path, config: PipelineConfig) -> list[dict]:
    files = discover_raw_files(input_dir, extension=config.io.raw_extension)
    if not files:
        raise FileNotFoundError(f"No raw files with extension {config.io.raw_extension!r} found under {input_dir}")
    return [process_file(path, output_dir, config) for path in files]
