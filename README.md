# PS-OCT Processing Toolbox

A mesoscale Polarization-Sensitive Optical Coherence Tomography (PS-OCT) processing framework for reconstruction, polarization contrast estimation, en-face projection, stitching, visualization, and scalable volumetric data analysis.

This repository contains both the original MATLAB processing pipeline and a modern Python re-implementation designed for reproducibility, scalability, validation, and integration with contemporary neuroinformatics infrastructure.

## Overview

Polarization-Sensitive Optical Coherence Tomography enables high-resolution volumetric imaging of biological tissue while also estimating polarization-derived contrasts. These contrasts can support the study of mesoscale tissue architecture, fiber organization, and structural organization across large specimens.

The toolbox is intended to support processing pipelines for large volumetric acquisitions and tile-based mesoscale reconstructions. The MATLAB code represents the original implementation. The Python code is being developed as a cleaner, modular, and scalable re-implementation.

## Repository Structure

```text
.
├── matlab_processing/
│   ├── PSOCT_2025_FCN.m
│   ├── Read2024.m
│   ├── MStitchFCN_Vlad.m
│   └── ...
│
├── python_processing/
│   ├── psoct_processing/
│   ├── notebooks/
│   ├── tests/
│   └── ...
│
├── calibration/
├── example_configs/
├── example_data/
└── README.md
```

The exact folder layout may evolve as the Python implementation matures. The separation between `matlab_processing` and `python_processing` is recommended so that the original pipeline remains available while the Python implementation is developed and validated.

## MATLAB Pipeline

The MATLAB implementation contains the original reconstruction and processing framework developed for PS-OCT mesoscale imaging experiments.

### Main Features

The MATLAB workflow supports raw acquisition reading, spectral calibration, wavelength interpolation, dispersion compensation, FFT-based OCT reconstruction, polarization contrast estimation, en-face image generation, tile stitching, and image export.

### Main Entry Point

```matlab
PSOCT_2025_FCN(...)
```

Additional MATLAB functions include raw data readers, interpolation utilities, contrast computation routines, and stitching functions.

### MATLAB Requirements

Recommended MATLAB toolboxes include the Signal Processing Toolbox and Image Processing Toolbox. The Parallel Computing Toolbox may be useful for large datasets but is not necessarily required for all processing steps.

## Python Pipeline

The Python implementation is a modern re-architecture of the MATLAB workflow. It emphasizes modularity, reproducibility, scalability, command-line execution, validation, and compatibility with large-volume data systems.

### Current Status

The Python implementation currently includes an installable package architecture, command-line tools, a reconstruction framework, synthetic dataset generation, channel-aware processing, TIFF and Zarr export, a validation harness, unit tests, and Jupyter notebook tutorials.

Scientific validation against real PS-OCT acquisitions is still required before the Python implementation should be considered a complete replacement for the MATLAB pipeline.

## Installation

### MATLAB

Clone the repository and add the MATLAB processing folder to the MATLAB path:

```matlab
addpath(genpath('matlab_processing'))
```

### Python

Create and activate a Python environment:

```bash
conda create -n psoct python=3.11
conda activate psoct
```

Install the Python package from the repository root:

```bash
pip install -e .
```

Optional dependencies can be installed for large-volume processing, acceleration, and visualization:

```bash
pip install zarr dask numba napari
```

GPU support may require additional installation steps depending on the CUDA version and local computing environment.

## Quick Start

### Generate a Synthetic Dataset

```bash
psoct-generate-synthetic --output ./synthetic_data
```

### Initialize or Edit a Configuration File

```bash
psoct-init-config --output ./config.yaml
```

### Run the Python Processing Pipeline

```bash
psoct-process --config ./config.yaml
```

### Run MATLAB-to-Python Validation

```bash
psoct-validate \
  --matlab-output ./matlab_reference \
  --python-output ./python_output
```

The validation tools are designed to compare intermediate and final outputs between MATLAB and Python. This is the recommended pathway for establishing scientific equivalence.

## Jupyter Tutorials

Tutorial notebooks are provided in:

```text
python_processing/notebooks/
```

The tutorials demonstrate installation, configuration, synthetic data generation, processing, validation, and visualization. They are intended as the easiest starting point for new users.

## Data Model

The Python implementation uses an explicit channel-aware acquisition model. Internally, reconstructed and raw data can be represented using named channels, for example:

```python
volume.shape == (channels, spectral_samples, x, y)
```

Example channel names may include:

```yaml
channel_names:
  - co_pol
  - cross_pol
```

The channel abstraction is intended to make acquisition assumptions explicit and to reduce ambiguity when handling polarization-sensitive data.

## Configuration

Processing parameters should be stored in YAML configuration files rather than hard-coded in scripts. A typical configuration includes input paths, output paths, acquisition dimensions, channel names, reconstruction parameters, projection settings, stitching settings, and export options.

Example:

```yaml
input_dir: ./raw
output_dir: ./processed

acquisition:
  n_channels: 2
  channel_names:
    - co_pol
    - cross_pol

reconstruction:
  fft_size: 2048
  dispersion_coefficients: [0.0, 0.0, 0.0]

projection:
  method: mean
  depth_range: [20, 200]

stitching:
  overlap: 0.15
  blending: linear
```

## Export Formats

Supported or planned output formats include TIFF, JPEG, NumPy arrays, HDF5, and Zarr. Zarr is recommended for large volumetric datasets because it supports chunked storage, parallel access, and cloud-compatible workflows.

Planned extensions include OME-Zarr, Neuroglancer-compatible multiscale pyramids, and napari-compatible visualization outputs.

## Validation Strategy

The migration strategy follows three phases. First, the architecture is modernized and modularized. Second, the Python implementation is validated against real PS-OCT acquisition data and MATLAB outputs. Third, numerical equivalence is quantified and improved where necessary.

This means that the Python implementation should be interpreted as a clean scientific re-implementation rather than a strict line-by-line translation of the MATLAB code.

## Intended Use Cases

This toolbox is intended for mesoscale neuroimaging, white matter architecture analysis, fiber orientation mapping, histological imaging workflows, large volumetric microscopy, scalable imaging infrastructure, and reproducible computational imaging pipelines.

## Planned Features

Future development priorities include full polarization validation, advanced retardance estimation, fiber orientation refinement, registration and alignment, GPU acceleration, distributed reconstruction, Dask-based chunked processing, cloud-native workflows, brainlife.io integration, Docker and Apptainer containers, napari integration, Neuroglancer export, and interactive mesoscale visualization.

## Development Philosophy

The MATLAB implementation remains the reference implementation until the Python implementation has been validated on real acquisition data. The Python implementation is intended to become a more maintainable and scalable processing framework suitable for modern computational environments, including workstations, institutional clusters, cloud infrastructure, and neuroinformatics platforms.

## Citation

If you use this software in scientific work, please cite the relevant PS-OCT and mesoscale imaging publications associated with this repository. Additional citation information should be added as the Python implementation stabilizes.

## Contributing

Contributions are welcome. Useful development areas include reconstruction optimization, GPU acceleration, visualization, validation datasets, documentation, containerization, cloud and HPC support, and scientific validation against MATLAB outputs.

## License

See the repository license information.

## Acknowledgments

Development of this framework is part of broader efforts in mesoscale neuroimaging, computational neuroscience, scalable neuroinformatics, open scientific infrastructure, and reproducible imaging science.
