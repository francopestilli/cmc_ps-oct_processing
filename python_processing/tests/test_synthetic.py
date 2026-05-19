from pathlib import Path

from typer.testing import CliRunner

from psoct_processing.cli import app
from psoct_processing.config import load_config, write_default_config
from psoct_processing.pipeline import run_pipeline
from psoct_processing.synthetic import synthetic_config, write_synthetic_dat


def test_default_config_roundtrip(tmp_path: Path):
    cfg_path = write_default_config(tmp_path / "config.yaml")
    cfg = load_config(cfg_path)
    assert cfg.io.channels.n_channels >= 1
    assert cfg.provenance.save_manifest is True


def test_synthetic_pipeline_end_to_end(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    write_synthetic_dat(raw_dir, spectral_samples=64, x=4, y=3)
    cfg = synthetic_config(spectral_samples=64, x=4, y=3)
    results = run_pipeline(raw_dir, out_dir, cfg)
    assert len(results) == 1
    assert list(out_dir.glob("*_manifest.json"))
    assert list(out_dir.glob("*_complex_channels.zarr"))


def test_cli_generate_synthetic(tmp_path: Path):
    runner = CliRunner()
    result = runner.invoke(app, ["generate-synthetic", "--output-dir", str(tmp_path / "raw"), "--spectral-samples", "64"])
    assert result.exit_code == 0
    assert (tmp_path / "raw" / "synthetic_psoct.dat").exists()
    assert (tmp_path / "raw" / "synthetic_config.yaml").exists()
