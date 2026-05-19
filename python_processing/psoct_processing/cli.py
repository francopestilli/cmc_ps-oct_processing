from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from .config import load_config, write_default_config
from .pipeline import run_pipeline
from .validation import compare_stage_directories

app = typer.Typer(help="PS-OCT Python processing pipeline scaffold.")
console = Console()


@app.command("init-config")
def init_config(
    output: Path = typer.Option(Path("psoct_config.yaml"), help="Path for the generated YAML configuration."),
) -> None:
    path = write_default_config(output)
    console.print(f"[green]Wrote default configuration to {path}[/green]")


@app.command("generate-synthetic")
def generate_synthetic(
    output_dir: Path = typer.Option(..., help="Directory where synthetic raw data and config will be written."),
    spectral_samples: int = typer.Option(1024, min=16),
    x: int = typer.Option(16, min=1),
    y: int = typer.Option(8, min=1),
    n_channels: int = typer.Option(2, min=1),
    seed: int = typer.Option(42),
) -> None:
    from .synthetic import synthetic_config, write_synthetic_dat
    from .export import save_json

    products = write_synthetic_dat(
        output_dir, n_channels=n_channels, spectral_samples=spectral_samples, x=x, y=y, seed=seed
    )
    cfg = synthetic_config(n_channels=n_channels, spectral_samples=spectral_samples, x=x, y=y)
    cfg_path = output_dir / "synthetic_config.yaml"
    import yaml
    output_dir.mkdir(parents=True, exist_ok=True)
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.model_dump(mode="json"), f, sort_keys=False)
    console.print({"products": products, "config": str(cfg_path)})


@app.command()
def run(
    input_dir: Path = typer.Option(..., exists=True, file_okay=False, help="Directory containing raw acquisition files."),
    output_dir: Path = typer.Option(..., help="Directory for processed outputs."),
    config: Path | None = typer.Option(None, exists=True, dir_okay=False, help="YAML configuration file."),
) -> None:
    cfg = load_config(config)
    results = run_pipeline(input_dir, output_dir, cfg)
    console.print(f"[green]Processed {len(results)} file(s).[/green]")
    for item in results:
        console.print(item["products"])


@app.command()
def inspect(
    input_dir: Path = typer.Option(..., exists=True, file_okay=False),
    config: Path | None = typer.Option(None, exists=True, dir_okay=False),
) -> None:
    from .io import discover_raw_files, read_raw_dat

    cfg = load_config(config)
    files = discover_raw_files(input_dir, extension=cfg.io.raw_extension)
    console.print(f"Found {len(files)} raw file(s).")
    for path in files[:10]:
        raw = read_raw_dat(path, cfg.io)
        console.print({"path": str(path), "shape": raw.data.shape, "metadata": raw.metadata})


@app.command()
def validate(
    matlab_output: Path = typer.Option(..., exists=True, file_okay=False, help="Directory containing MATLAB reference arrays."),
    python_output: Path = typer.Option(..., exists=True, file_okay=False, help="Directory containing Python arrays with matching stage names."),
    report: Path | None = typer.Option(None, help="Optional JSON report path."),
) -> None:
    comparisons = compare_stage_directories(matlab_output, python_output, output_json=report)
    if not comparisons:
        console.print("[yellow]No matching stage files were found.[/yellow]")
        raise typer.Exit(code=1)
    for item in comparisons:
        console.print({
            "stage": item.stage,
            "shape": item.shape_matlab,
            "rmse": item.rmse,
            "relative_rmse": item.relative_rmse,
            "pearson_r": item.pearson_r,
        })
    if report is not None:
        console.print(f"[green]Wrote validation report to {report}[/green]")
