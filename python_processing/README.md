# PS-OCT Processing Toolbox

Python re-implementation scaffold for the Center for Mesoscale Connectivity PS-OCT processing workflow.

This repository is intended to replace the original MATLAB-oriented processing workflow with a modular, testable, and extensible Python toolbox. The present version should be understood as an engineering-ready foundation rather than a fully scientifically validated replacement for the MATLAB pipeline. It includes raw-data reading abstractions, channel-aware data models, reconstruction utilities, contrast computation hooks, en-face projection, stitching, export tools, synthetic-data generation, provenance capture, and a MATLAB-to-Python validation harness.

The next scientific step is to validate the package against a small real PS-OCT acquisition and the corresponding MATLAB intermediate outputs.

---

## 1. What this toolbox currently provides

The package currently includes:

- an installable Python package named `psoct_processing`;
- a command-line interface exposed through `psoct-process`;
- YAML-based configuration;
- channel-aware raw data handling;
- basic background subtraction and reconstruction functions;
- FFT-based OCT reconstruction utilities;
- preliminary contrast functions for reflectivity and polarization-channel diagnostics;
- en-face projection tools;
- MATLAB-style tile-stitching utilities;
- TIFF, Zarr, HDF5, JSON, and NumPy export helpers;
- synthetic PS-OCT-like data generation;
- provenance and logging utilities;
- MATLAB reference export helpers;
- validation tools for comparing MATLAB and Python stage outputs;
- an initial test suite.

This version does **not** yet guarantee scientific equivalence with the MATLAB pipeline. In particular, real acquisition-specific conventions such as header interpretation, exact channel layout, calibration coefficients, k-space interpolation, dispersion compensation, and polarization contrast definitions must still be validated using real data.

---

## 2. Recommended installation

Create a clean Python environment. Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the package in editable/development mode from the repository root:

```bash
pip install -e .
```

For development and testing, install optional development dependencies:

```bash
pip install -e '.[dev]'
```

For performance-oriented development, you may also install the optional performance dependencies:

```bash
pip install -e '.[performance]'
```

These optional dependencies are not required for the first runs.

---

## 3. Repository layout

```text
psoct_pkg_v014/
├── README.md
├── pyproject.toml
├── config/
│   └── default.yaml
├── examples/
│   └── synthetic_run.py
├── matlab_export/
│   ├── README.md
│   └── export_stage_array.m
├── notebooks/
├── psoct_processing/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── contrasts.py
│   ├── data.py
│   ├── enface.py
│   ├── export.py
│   ├── io.py
│   ├── logging_utils.py
│   ├── pipeline.py
│   ├── preprocess.py
│   ├── provenance.py
│   ├── psoct_matlab_compat.py
│   ├── reconstruct.py
│   ├── stitch.py
│   ├── synthetic.py
│   └── validation.py
└── tests/
```

The most important files for a new user are `config/default.yaml`, `psoct_processing/cli.py`, `psoct_processing/pipeline.py`, `psoct_processing/io.py`, and `psoct_processing/reconstruct.py`.

---

## 4. Quick start with synthetic data

The synthetic-data workflow is the safest way to verify that the package is installed correctly before attempting to process real PS-OCT acquisitions.

From the repository root, run:

```bash
psoct-process generate-synthetic --output-dir ./demo_synthetic
```

This creates a synthetic raw `.dat` file and a matching YAML configuration file in `./demo_synthetic`.

Inspect the generated data:

```bash
psoct-process inspect --input-dir ./demo_synthetic --config ./demo_synthetic/synthetic_config.yaml
```

Run the processing pipeline:

```bash
psoct-process run \
  --input-dir ./demo_synthetic \
  --output-dir ./demo_output \
  --config ./demo_synthetic/synthetic_config.yaml
```

After completion, inspect the output folder:

```bash
find ./demo_output -maxdepth 3 -type f
```

Depending on the configuration, the output may include reconstructed arrays, TIFF images, Zarr stores, JSON metadata, and provenance manifests.

---

## 5. Command-line interface

The package exposes a single command group:

```bash
psoct-process --help
```

The main subcommands are:

```bash
psoct-process init-config
psoct-process generate-synthetic
psoct-process inspect
psoct-process run
psoct-process validate
```

### 5.1 Create a default configuration

```bash
psoct-process init-config --output psoct_config.yaml
```

This writes a configurable YAML file that can be edited for a specific acquisition.

### 5.2 Generate synthetic data

```bash
psoct-process generate-synthetic \
  --output-dir ./demo_synthetic \
  --spectral-samples 1024 \
  --x 16 \
  --y 8 \
  --n-channels 2 \
  --seed 42
```

This command is useful for smoke testing, continuous integration, and development without private experimental data.

### 5.3 Inspect raw files

```bash
psoct-process inspect \
  --input-dir ./raw_data \
  --config ./psoct_config.yaml
```

This reports discovered raw files, inferred data shape, and metadata extracted by the reader.

### 5.4 Run the processing pipeline

```bash
psoct-process run \
  --input-dir ./raw_data \
  --output-dir ./processed_output \
  --config ./psoct_config.yaml
```

The pipeline currently performs a staged reconstruction workflow using the assumptions encoded in the YAML configuration.

### 5.5 Validate against MATLAB outputs

```bash
psoct-process validate \
  --matlab-output ./matlab_reference \
  --python-output ./python_output \
  --report ./validation_report.json
```

The validation command compares files with matching stage names and reports metrics such as RMSE, relative RMSE, and Pearson correlation.

---

## 6. Configuration file

The YAML configuration controls the assumptions used by the raw reader, reconstruction pipeline, contrast computation, stitching, export, and provenance capture.

A typical configuration looks like this:

```yaml
io:
  raw_extension: .dat
  dtype: int16
  byte_order: little
  header_bytes: 0
  shape: null
  spectral_axis: -1
  background_strategy: mean_along_bscan
  channels:
    n_channels: 2
    names: [co_pol, cross_pol]
    layout: axis
    channel_axis: 0
    canonical_axis_order: czyx
    channels:
      - name: co_pol
        index: 0
        role: co-polarized detector
        gain: 1.0
      - name: cross_pol
        index: 1
        role: cross-polarized detector
        gain: 1.0

reconstruction:
  fft_size: null
  window: hann
  crop_depth: null
  subtract_dc: true
  dispersion_coefficients: [0.0, 0.0, 0.0]
  k_linearization: null
  reconstruct_per_channel: true

contrasts:
  reflectivity_log_db: true
  eps: 1.0e-12
  co_pol_channel: co_pol
  cross_pol_channel: cross_pol

enface:
  method: mean
  depth_range: null

stitching:
  overlap_fraction: 0.15
  blending: linear
  tile_grid: null

export:
  save_tiff: true
  save_zarr: true
  save_hdf5: false
  save_intermediates: false
  zarr_chunks: auto

provenance:
  save_manifest: true
  hash_inputs: false
  log_level: INFO
```

The most important parameters to adjust for a real acquisition are `io.header_bytes`, `io.shape`, `io.spectral_axis`, and the channel schema under `io.channels`.

---

## 7. Channel model

The package uses a named-channel model. This is essential for PS-OCT because polarization contrasts depend on knowing which raw data stream corresponds to each optical or detector channel.

The canonical internal orientation is controlled by:

```yaml
io:
  channels:
    canonical_axis_order: czyx
```

where `c` is channel, `z` is spectral/depth dimension, and `y`/`x` are spatial dimensions. The actual raw file can use another layout, but the reader should convert it into this canonical representation.

For a two-channel co-/cross-polarized acquisition, use:

```yaml
channels:
  n_channels: 2
  names: [co_pol, cross_pol]
  channels:
    - name: co_pol
      index: 0
      role: co-polarized detector
      gain: 1.0
    - name: cross_pol
      index: 1
      role: cross-polarized detector
      gain: 1.0
```

For real data, the critical information to determine is whether channels are stored by axis, interleaved samples, interleaved A-lines, interleaved B-scans, or separate files. This must be encoded in the configuration and validated against MATLAB outputs.

---

## 8. Processing real data

To process a real acquisition, first create a configuration file:

```bash
psoct-process init-config --output real_acquisition.yaml
```

Edit the configuration to match the acquisition. At minimum, define:

```yaml
io:
  header_bytes: <number_of_header_bytes>
  shape: [<channels>, <spectral_samples>, <y>, <x>]
  spectral_axis: 1
  channels:
    n_channels: <number_of_channels>
    names: [<channel_1>, <channel_2>]
```

Then inspect the raw data:

```bash
psoct-process inspect \
  --input-dir /path/to/raw/acquisition \
  --config real_acquisition.yaml
```

If the reported shape and metadata look reasonable, run:

```bash
psoct-process run \
  --input-dir /path/to/raw/acquisition \
  --output-dir /path/to/output \
  --config real_acquisition.yaml
```

For the first real dataset, use a small acquisition or a single tile. Avoid starting with a whole-brain or whole-sample volume until the data conventions have been validated.

---

## 9. MATLAB-to-Python validation workflow

The recommended validation strategy is stage-by-stage comparison rather than only comparing final images.

### 9.1 Export reference arrays from MATLAB

The folder `matlab_export/` contains a helper function:

```matlab
export_stage_array(stage_name, array, output_dir)
```

Use it inside the MATLAB pipeline to save intermediate arrays, for example:

```matlab
export_stage_array('raw_reshaped', raw_reshaped, './matlab_reference')
export_stage_array('background_subtracted', bg_subtracted, './matlab_reference')
export_stage_array('fft_magnitude', fft_magnitude, './matlab_reference')
export_stage_array('reflectivity', reflectivity, './matlab_reference')
export_stage_array('enface_mean', enface_mean, './matlab_reference')
```

### 9.2 Run the Python pipeline

Run the Python implementation on the same raw data and save comparable intermediate arrays.

```bash
psoct-process run \
  --input-dir ./raw_reference_tile \
  --output-dir ./python_output \
  --config ./real_acquisition.yaml
```

### 9.3 Compare stages

```bash
psoct-process validate \
  --matlab-output ./matlab_reference \
  --python-output ./python_output \
  --report ./validation_report.json
```

The validation report helps identify whether discrepancies arise from raw reshaping, background correction, interpolation, FFT conventions, contrast computation, or export conventions.

---

## 10. Running tests

Install development dependencies and run:

```bash
pytest
```

The tests currently cover configuration handling, channel-aware data handling, MATLAB compatibility utilities, reconstruction basics, stitching, synthetic data generation, and validation metrics.

A successful test run indicates that the package infrastructure is functioning. It does not imply that the scientific reconstruction has been fully validated for a real PS-OCT acquisition.

---

## 11. Suggested first validation dataset

The ideal first validation dataset should be small. A recommended minimal package would include:

```text
validation_dataset/
├── raw/
│   └── one_tile.dat
├── calibration/
│   ├── wavelength_or_k_linearization.mat
│   ├── dispersion_coefficients.mat
│   └── any_other_required_files.mat
├── matlab_reference/
│   ├── raw_reshaped.npy or raw_reshaped.mat
│   ├── background_subtracted.npy or .mat
│   ├── fft_magnitude.npy or .mat
│   ├── reflectivity.npy or .mat
│   └── enface_mean.npy or .mat
└── run_parameters.txt
```

The most useful MATLAB reference outputs are the earliest intermediate arrays. If `raw_reshaped` differs, then later reconstruction differences are not interpretable.

---

## 12. Current scientific limitations

The package is intentionally conservative. Several components are placeholders or approximate implementations until real data validation is performed:

- full LabView `.dat` header parsing;
- exact spectral dimension inference;
- acquisition-specific reshaping;
- channel interleaving conventions;
- wavelength-to-k interpolation;
- dispersion compensation;
- calibration-file loading;
- exact retardance and orientation definitions;
- exact MATLAB FFT normalization and indexing conventions;
- final validated tile-stitching behavior.

These are not architectural problems. They are empirical acquisition details that should be resolved by comparing MATLAB and Python outputs on one or more real acquisitions.

---

## 13. Recommended development sequence

The most efficient next development sequence is:

1. validate raw file reading on one real tile;
2. confirm shape and channel order;
3. validate background subtraction;
4. validate FFT magnitude;
5. add calibration-file loading;
6. validate k-space interpolation and dispersion compensation;
7. validate reflectivity and polarization contrasts;
8. validate en-face projections;
9. validate tile stitching;
10. add performance optimizations only after scientific correctness is established.

Performance optimization should not precede scientific validation.

---

## 14. Output formats

The package supports several output formats.

TIFF is useful for visual inspection and compatibility with image-processing software. Zarr is recommended for large volumetric data because it supports chunked storage, partial reading, cloud/HPC workflows, and eventual integration with multiscale visualization systems. HDF5 is supported for compatibility with traditional scientific workflows. JSON is used for metadata, provenance, and validation reports.

For large PS-OCT volumes, Zarr should become the primary internal format, while TIFF/JPEG should be treated as visualization exports.

---

## 15. Provenance

When enabled, the pipeline records provenance information such as configuration values, input paths, output paths, timestamps, and processing products. This is important for scientific reproducibility and for eventual deployment on brainlife.io or another workflow platform.

To enable input hashing, set:

```yaml
provenance:
  save_manifest: true
  hash_inputs: true
```

Hashing may be slow for very large acquisitions, so it is disabled by default.

---

## 16. Example end-to-end smoke test

The following commands exercise the package without real data:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

psoct-process generate-synthetic --output-dir ./demo_synthetic
psoct-process inspect --input-dir ./demo_synthetic --config ./demo_synthetic/synthetic_config.yaml
psoct-process run --input-dir ./demo_synthetic --output-dir ./demo_output --config ./demo_synthetic/synthetic_config.yaml
pytest
```

If these commands complete successfully, the software infrastructure is working and the toolbox is ready for acquisition-specific validation.

---

## 17. How to contribute new acquisition details

When adding support for a new scanner or acquisition protocol, document the following information:

- raw file extension;
- header size and header structure;
- data type and byte order;
- spectral samples per A-line;
- A-lines per B-scan;
- number of B-scans;
- number of polarization or detector channels;
- channel storage convention;
- calibration files required;
- units and scaling factors;
- MATLAB command used for the reference run;
- example output dimensions.

These details should be reflected in a dedicated YAML configuration file and, if necessary, a specialized reader function.

---

## 18. Project status

Current status:

- software architecture: substantially complete;
- package infrastructure: functional;
- CLI: functional;
- synthetic testing: available;
- validation harness: available;
- real-data scientific validation: pending;
- complete MATLAB replacement: pending validation.

The project is ready for the first small real-data validation cycle.

## Notebook tutorial

A Jupyter notebook tutorial is provided at:

```text
notebooks/psoct_processing_tutorial.ipynb
```

The notebook demonstrates how to generate a synthetic acquisition, create a configuration file, run the processing pipeline through the Python API, inspect outputs, visualize TIFF products, and prepare for MATLAB-to-Python validation.
