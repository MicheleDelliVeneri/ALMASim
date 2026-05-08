"""Simulation commands for the ALMASim CLI."""

from __future__ import annotations

from pathlib import Path
from threading import Event, Thread
from time import sleep
from typing import Any

import numpy as np
import pandas as pd
import typer
from tqdm.auto import tqdm

from . import export_results, generate_clean_cube, simulate_observation
from .services import astro
from .services.astro.spectral import sample_given_redshift
from .services.compute import create_backend
from .services.simulation import SimulationParams

simulation_app = typer.Typer(
    help="Simulation workflow commands.",
    no_args_is_help=True,
)

_SOURCE_TYPES = [
    "point",
    "gaussian",
    "diffuse",
    "galaxy-zoo",
    "molecular",
    "hubble-100",
]
_SAVE_MODES = ["memory", "npz", "h5", "fits"]
_BACKEND_TYPES = ["sync", "local"]
_IMAGING_ALGORITHMS = ["legacy", "ducc0"]


def _path_with_suffix(path: Path, *parts: Any) -> Path:
    suffix = "_".join(str(part) for part in parts)
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def _update_progress(progress_bar: tqdm, target: int) -> None:
    clamped = max(0, min(100, int(target)))
    if clamped > progress_bar.n:
        progress_bar.update(clamped - progress_bar.n)


def _progress_callback(progress_bar: tqdm, start: int, span: int):
    def callback(value: int | float) -> None:
        _update_progress(progress_bar, start + int(round((float(value) / 100.0) * span)))

    return callback


def _progress_logger(progress_bar: tqdm):
    def logger(message: str) -> None:
        progress_bar.write(str(message))

    return logger


def _run_stage_with_feedback(
    progress_bar: tqdm,
    *,
    label: str,
    work,
    smooth_target: int | None = None,
):
    stop_event = Event()
    spinner_frames = "|/-\\"

    def refresh() -> None:
        spinner_index = 0
        while not stop_event.wait(0.2):
            progress_bar.set_postfix_str(
                f"{label} {spinner_frames[spinner_index % len(spinner_frames)]}"
            )
            spinner_index += 1
            if smooth_target is not None and progress_bar.n < smooth_target:
                progress_bar.update(1)
                sleep(0.8)

    thread = Thread(target=refresh, daemon=True)
    thread.start()
    try:
        return work()
    finally:
        stop_event.set()
        thread.join(timeout=1)
        progress_bar.set_postfix_str("")


def _build_simulation_params(
    *,
    metadata: pd.DataFrame,
    row_idx: int,
    run_idx: int,
    main_dir: Path,
    output_dir: Path,
    project_name: str,
    source_type: str,
    save_mode: str,
    persist_standard_outputs: bool,
    ml_shard_path: Path,
    n_pix: int,
    n_channels: int,
    n_lines: int | None,
    imaging_algorithm: str,
    background_level: float,
    background_seed: int | None,
    output_subdir_name: str | None,
) -> SimulationParams:
    rest_frequency, _ = astro.get_line_info(main_dir)
    sampled = sample_given_redshift(
        metadata,
        n=max(row_idx + 1, 1),
        rest_frequency=rest_frequency,
        extended=(source_type == "extended"),
        zmax=None,
    )

    if row_idx >= len(sampled):
        typer.echo(
            f"--row-idx {row_idx} is out of range for {len(sampled)} metadata rows",
            err=True,
        )
        raise typer.Exit(code=2)

    row = sampled.iloc[row_idx]
    output_dir.mkdir(parents=True, exist_ok=True)

    return SimulationParams.from_metadata_row(
        row,
        idx=run_idx,
        main_dir=main_dir,
        output_dir=output_dir,
        tng_dir=output_dir / "tng",
        galaxy_zoo_dir=output_dir / "galaxy_zoo",
        hubble_dir=output_dir / "hubble",
        project_name=project_name,
        source_type=source_type,
        save_mode=save_mode,
        persist=persist_standard_outputs,
        ml_dataset_path=ml_shard_path,
        output_subdir_name=output_subdir_name,
        n_pix=n_pix,
        n_channels=n_channels,
        n_lines=n_lines,
        imaging_algorithm=imaging_algorithm,
        background_level=background_level,
        background_seed=background_seed,
        # Disable MS export by default; requires casatools or python-casacore.
        ms_export=False,
        ms_save_mode="none",
    )


@simulation_app.command("run")
def simulation_run(
    metadata_path: Path = typer.Option(
        ...,
        "--metadata-path",
        help="CSV file with pre-fetched ALMA metadata rows (from 'almasim metadata query').",
    ),
    row_idx: int | None = typer.Option(
        None,
        "--row-idx",
        min=0,
        help="Single metadata row index to simulate. Omit to simulate all rows.",
    ),
    project_name: str = typer.Option(
        "sim_demo",
        "--project-name",
        help="Project name used in outputs.",
    ),
    output_dir: Path = typer.Option(
        Path.home() / "almasim_outputs" / "simulation",
        "--output-dir",
        help="Directory for simulation outputs.",
    ),
    main_dir: Path = typer.Option(
        Path(__file__).resolve().parents[0],
        "--main-dir",
        help="Path to ALMASim data directory (defaults to src/almasim).",
    ),
    source_type: str = typer.Option(
        "point",
        "--source-type",
        help="Sky model family. Choices: " + ", ".join(_SOURCE_TYPES) + ".",
        case_sensitive=False,
    ),
    n_pix: int = typer.Option(
        128,
        "--n-pix",
        min=1,
        help="Spatial cube size override.",
    ),
    n_channels: int = typer.Option(
        32,
        "--n-channels",
        min=1,
        help="Channel count override.",
    ),
    n_lines: int | None = typer.Option(
        None,
        "--n-lines",
        min=1,
        help="Number of spectral lines to inject. Omit to use metadata default.",
    ),
    seed: int | None = typer.Option(
        None,
        "--seed",
        help=(
            "Base random seed for reproducible simulations. When running multiple "
            "simulations, each run uses seed + run_index."
        ),
    ),
    simulations_per_row: int = typer.Option(
        1,
        "--simulations-per-row",
        min=1,
        help="How many simulations to run for each selected metadata row.",
    ),
    imaging_algorithm: str = typer.Option(
        "legacy",
        "--imaging-algorithm",
        help=("Channel imaging algorithm. Choices: " + ", ".join(_IMAGING_ALGORITHMS) + "."),
        case_sensitive=False,
    ),
    robust: float = typer.Option(
        0.0,
        "--robust",
        help="Robust weighting parameter for imaging.",
    ),
    background_level: float = typer.Option(
        0.0,
        "--background-level",
        help="Background sky level scaling factor (0 = no background).",
    ),
    background_seed: int | None = typer.Option(
        None,
        "--background-seed",
        help="Random seed for background sky generation.",
    ),
    save_mode: str = typer.Option(
        "memory",
        "--save-mode",
        help="Output save mode. Choices: " + ", ".join(_SAVE_MODES) + ".",
        case_sensitive=False,
    ),
    persist_standard_outputs: bool = typer.Option(
        False,
        "--persist-standard-outputs",
        help="Persist standard outputs in addition to ML shard.",
    ),
    ml_shard_path: Path = typer.Option(
        Path.home() / "almasim_outputs" / "ddrm_training_sample.h5",
        "--ml-shard-path",
        help="Destination HDF5 shard path.",
    ),
    backend: str = typer.Option(
        "sync",
        "--backend",
        help="Compute backend. Choices: " + ", ".join(_BACKEND_TYPES) + ".",
        case_sensitive=False,
    ),
    n_workers: int = typer.Option(
        1,
        "--n-workers",
        min=1,
        help="Worker count for local backend.",
    ),
) -> None:
    """Run staged ALMASim simulation from a pre-queried metadata CSV."""
    source_type_normalized = source_type.lower()
    if source_type_normalized not in _SOURCE_TYPES:
        typer.echo("--source-type must be one of: " + ", ".join(_SOURCE_TYPES), err=True)
        raise typer.Exit(code=2)

    save_mode_normalized = save_mode.lower()
    if save_mode_normalized not in _SAVE_MODES:
        typer.echo("--save-mode must be one of: " + ", ".join(_SAVE_MODES), err=True)
        raise typer.Exit(code=2)

    backend_normalized = backend.lower()
    if backend_normalized not in _BACKEND_TYPES:
        typer.echo("--backend must be one of: " + ", ".join(_BACKEND_TYPES), err=True)
        raise typer.Exit(code=2)

    imaging_algorithm_normalized = imaging_algorithm.lower()
    if imaging_algorithm_normalized not in _IMAGING_ALGORITHMS:
        typer.echo(
            "--imaging-algorithm must be one of: " + ", ".join(_IMAGING_ALGORITHMS),
            err=True,
        )
        raise typer.Exit(code=2)

    if not metadata_path.exists():
        typer.echo(f"--metadata-path not found: {metadata_path}", err=True)
        raise typer.Exit(code=2)

    metadata = pd.read_csv(metadata_path)
    total_rows = len(metadata)
    typer.echo(f"Metadata rows available: {total_rows}")

    indices = [row_idx] if row_idx is not None else list(range(total_rows))
    total_runs = len(indices) * simulations_per_row
    typer.echo(f"Rows to simulate: {len(indices)}")
    typer.echo(f"Simulations per row: {simulations_per_row}")
    typer.echo(f"Total simulations: {total_runs}")

    main_dir_resolved = main_dir.expanduser().resolve()
    output_dir_resolved = output_dir.expanduser().resolve()
    ml_shard_resolved = ml_shard_path.expanduser().resolve()
    persist_outputs = persist_standard_outputs or save_mode_normalized != "memory"

    backend_kwargs: dict[str, int] = {}
    if backend_normalized == "local":
        backend_kwargs = {"n_workers": n_workers}

    with create_backend(backend_normalized, **backend_kwargs) as compute_backend:
        run_number = 0
        for idx in indices:
            for simulation_idx in range(simulations_per_row):
                run_number += 1
                typer.echo(
                    f"\n--- Simulation {run_number}/{total_runs}: row {idx + 1}/{total_rows} "
                    f"(index {idx}), repeat {simulation_idx + 1}/{simulations_per_row} ---"
                )
                run_idx = idx * simulations_per_row + simulation_idx
                run_seed = seed + run_idx if seed is not None else None
                if run_seed is not None:
                    np.random.seed(run_seed)
                    typer.echo(f"Seed: {run_seed}")
                run_ml_shard_path = (
                    _path_with_suffix(
                        ml_shard_resolved, f"row{idx:04d}", f"sim{simulation_idx:03d}"
                    )
                    if total_runs > 1
                    else ml_shard_resolved
                )
                output_subdir_name = (
                    f"{project_name}_{run_idx}" if persist_outputs and total_runs > 1 else None
                )
                params = _build_simulation_params(
                    metadata=metadata,
                    row_idx=idx,
                    run_idx=run_idx,
                    main_dir=main_dir_resolved,
                    output_dir=output_dir_resolved,
                    project_name=project_name,
                    source_type=source_type_normalized,
                    save_mode=save_mode_normalized,
                    persist_standard_outputs=persist_outputs,
                    ml_shard_path=run_ml_shard_path,
                    n_pix=n_pix,
                    n_channels=n_channels,
                    n_lines=n_lines,
                    imaging_algorithm=imaging_algorithm_normalized,
                    background_level=background_level,
                    background_seed=(background_seed if background_seed is not None else run_seed),
                    output_subdir_name=output_subdir_name,
                )

                with tqdm(
                    total=100,
                    desc=f"Simulation {run_number}/{total_runs}",
                    unit="%",
                    leave=True,
                ) as progress_bar:
                    logger = _progress_logger(progress_bar)
                    is_sync_backend = backend_normalized == "sync"

                    clean_stage = _run_stage_with_feedback(
                        progress_bar,
                        label="clean-cube",
                        smooth_target=(34 if is_sync_backend else None),
                        work=lambda: generate_clean_cube(
                            params,
                            logger=logger,
                            compute_backend=compute_backend,
                            progress_emitter=_progress_callback(progress_bar, 0, 35),
                        ),
                    )
                    _update_progress(progress_bar, 35)
                    logger(f"Clean cube shape: {clean_stage.model_cube.shape}")
                    logger(f"Target ML shard: {params.ml_dataset_path}")

                    simulation_results = _run_stage_with_feedback(
                        progress_bar,
                        label="observation",
                        smooth_target=(89 if is_sync_backend else None),
                        work=lambda: simulate_observation(
                            clean_stage,
                            compute_backend=compute_backend,
                            robust=robust,
                            terminal_logger=logger,
                            interferometer_progress_callback=_progress_callback(
                                progress_bar, 35, 55
                            ),
                        ),
                    )
                    _update_progress(progress_bar, 90)
                    logger(f"Dirty cube shape: {simulation_results['dirty_cube'].shape}")
                    logger(f"UV mask cube shape: {simulation_results['uv_mask_cube'].shape}")

                    exported_results = _run_stage_with_feedback(
                        progress_bar,
                        label="export",
                        work=lambda: export_results(
                            params,
                            clean_stage,
                            simulation_results,
                            logger=logger,
                        ),
                    )
                    _update_progress(progress_bar, 100)
                typer.echo(f"ML shard written to: {exported_results.get('ml_dataset_path')}")
                if save_mode_normalized != "memory":
                    for key in sorted(exported_results):
                        if key in {"ml_dataset_path", "antenna_config_paths"}:
                            continue
                        if key.endswith("_path"):
                            typer.echo(f"{key}: {exported_results[key]}")
                        elif key.endswith("_paths"):
                            for path in exported_results[key]:
                                typer.echo(f"{key}: {path}")

    typer.echo(f"\nAll done. Simulated {total_runs} run(s) across {len(indices)} row(s).")
