"""Simulation commands for the ALMASim CLI."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from threading import Event, Thread
from time import sleep
from typing import Any

import pandas as pd
import typer
from tqdm.auto import tqdm

from ..services import astro
from ..services.astro.spectral import sample_given_redshift
from ..services.compute import create_backend
from ..services.external_skymodel import EXTERNAL_SOURCE_TYPES, is_external_source_type
from ..services.simulation import (
    SimulationParams,
    export_results,
    generate_clean_cube,
    image_products,
    simulate_observation,
)
from ..services.simulation.rng import run_rng
from ..skymodels.registry import SKYMODELS
from ._shared import usage_error


class InstrumentGeometry(StrEnum):
    """How a simulation sets its beam, field of view and cell (MADIV-28 T2.11)."""

    legacy = "legacy"
    planner = "planner"


simulation_app = typer.Typer(
    help="Simulation workflow commands.",
    no_args_is_help=True,
)

_SOURCE_TYPES = [name for name, spec in SKYMODELS.items() if spec.in_cli and not spec.external]
_SAVE_MODES = ["memory", "npz", "h5", "fits"]
_BACKEND_TYPES = ["sync", "local"]
# "ducc0" is disabled until its uv geometry is fixed (MADIV-28 T1.3, T3.7).
_IMAGING_ALGORITHMS = ["legacy"]
_EXTERNAL_INPUT_SUFFIXES = {".fits", ".fit", ".fts", ".fz"}


def _path_with_suffix(path: Path, *parts: Any) -> Path:
    suffix = "_".join(str(part) for part in parts)
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def _normalize_match_token(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _contains_token(name: str, compact_name: str, token: str | None) -> bool:
    normalized = _normalize_match_token(token)
    raw = str(token or "").lower()
    return bool((normalized and normalized in compact_name) or (raw and raw in name))


def _resolve_input_path_for_row(input_path: Path | None, row: pd.Series) -> Path | None:  # noqa: C901
    if input_path is None:
        return None

    resolved = input_path.expanduser().resolve()
    if resolved.is_file():
        return resolved
    if not resolved.is_dir():
        raise typer.BadParameter(f"--input must be an existing file or directory: {resolved}")

    candidates = [
        path
        for path in resolved.rglob("*")
        if path.is_file() and path.suffix.lower() in _EXTERNAL_INPUT_SUFFIXES
    ]
    if not candidates:
        raise typer.BadParameter(
            f"No FITS skymodel files found under --input directory: {resolved}"
        )

    member_uid = row.get("member_ous_uid") if hasattr(row, "get") else None
    group_uid = row.get("group_ous_uid") if hasattr(row, "get") else None

    ranked: dict[int, list[Path]] = {3: [], 2: [], 1: []}
    for candidate in candidates:
        name = candidate.name.lower()
        compact_name = _normalize_match_token(candidate.name)
        has_member = _contains_token(name, compact_name, member_uid)
        has_group = _contains_token(name, compact_name, group_uid)
        if has_member and has_group:
            ranked[3].append(candidate)
        elif has_member:
            ranked[2].append(candidate)
        elif has_group:
            ranked[1].append(candidate)

    for score in (3, 2, 1):
        matches = ranked[score]
        if len(matches) == 1:
            return matches[0].resolve()
        if len(matches) > 1:
            options = ", ".join(str(path.name) for path in sorted(matches))
            raise typer.BadParameter(
                "Ambiguous --input directory match for metadata row "
                f"member_ous_uid={member_uid!r}, group_ous_uid={group_uid!r}: {options}"
            )

    raise typer.BadParameter(
        "Could not match any input skymodel file in directory "
        f"{resolved} for metadata row member_ous_uid={member_uid!r}, group_ous_uid={group_uid!r}"
    )


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
    seed: int | None,
    instrument_geometry: str = "legacy",
    inject_serendipitous: bool,
    output_subdir_name: str | None,
    input_path: Path | None,
    kaggle_token: Path | None = None,
) -> SimulationParams:
    rest_frequency, _ = astro.get_line_info(main_dir)
    sampled = sample_given_redshift(
        metadata,
        n=max(row_idx + 1, 1),
        rest_frequency=rest_frequency,
        extended=(source_type == "extended"),
        zmax=None,
        rng=run_rng(seed, "metadata"),
    )

    if row_idx >= len(sampled):
        usage_error(f"--row-idx {row_idx} is out of range for {len(sampled)} metadata rows")

    row = sampled.iloc[row_idx]
    output_dir.mkdir(parents=True, exist_ok=True)
    external_input_path = _resolve_input_path_for_row(input_path, row)
    effective_source_type = (
        source_type
        if external_input_path is None or is_external_source_type(source_type)
        else "external-fits-cube"
    )

    return SimulationParams.from_metadata_row(
        row,
        idx=run_idx,
        main_dir=main_dir,
        output_dir=output_dir,
        tng_dir=output_dir / "tng",
        galaxy_zoo_dir=output_dir / "galaxy_zoo",
        hubble_dir=output_dir / "hubble",
        project_name=project_name,
        source_type=effective_source_type,
        save_mode=save_mode,
        persist=persist_standard_outputs,
        ml_dataset_path=ml_shard_path,
        output_subdir_name=output_subdir_name,
        external_skymodel_path=external_input_path,
        n_pix=n_pix,
        n_channels=n_channels,
        n_lines=n_lines,
        imaging_algorithm=imaging_algorithm,
        background_level=background_level,
        background_seed=background_seed,
        seed=seed,
        instrument_geometry=instrument_geometry,
        inject_serendipitous=inject_serendipitous,
        kaggle_token=kaggle_token,
        # Disable MS export by default; requires casatools or python-casacore.
        ms_export=False,
        ms_save_mode="none",
    )


@dataclass(frozen=True)
class SimulationRunOptions:
    """Every option of ``simulation run`` (MADIV-28 T3.8)."""

    metadata_path: Path
    row_idx: int | None
    project_name: str
    output_dir: Path
    input_path: Path | None
    main_dir: Path
    source_type: str
    n_pix: int
    n_channels: int
    n_lines: int | None
    seed: int | None
    simulations_per_row: int
    imaging_algorithm: str
    robust: float
    instrument_geometry: InstrumentGeometry
    background_level: float
    background_seed: int | None
    inject_serendipitous: bool
    save_mode: str
    persist_standard_outputs: bool
    ml_shard_path: Path
    backend: str
    n_workers: int
    kaggle_token: Path | None


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
    input_path: Path | None = typer.Option(
        None,
        "--input",
        help=(
            "External skymodel input. If a file is provided, it is reused for all rows. "
            "If a directory is provided, files are matched to each metadata row using "
            "member_ous_uid and group_ous_uid."
        ),
    ),
    main_dir: Path = typer.Option(
        Path(__file__).resolve().parents[1],
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
    instrument_geometry: InstrumentGeometry = typer.Option(
        InstrumentGeometry.legacy,
        "--instrument-geometry",
        case_sensitive=False,
        help=(
            "Beam, field of view and cell: legacy (the metadata FOV, beam / 5) or planner "
            "(the imaging planner's: the primary beam's FOV, beam / 8)."
        ),
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
    inject_serendipitous: bool = typer.Option(
        False,
        "--inject-serendipitous",
        help="Add a few random serendipitous sources around the target.",
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
    kaggle_token: Path | None = typer.Option(
        None,
        "--kaggle-token",
        help=(
            "Path to a Kaggle credentials file (JSON with username/key, e.g. "
            "kaggle.json or access_token) or the directory containing it, used to "
            "download morphology images for the galaxy-zoo / hubble-100 source "
            "types. Defaults to the standard Kaggle locations if omitted."
        ),
    ),
) -> None:
    """Run staged ALMASim simulation from a pre-queried metadata CSV."""
    options = _validated_simulation_options(SimulationRunOptions(**locals()))
    o = options
    metadata = pd.read_csv(o.metadata_path)
    total_rows = len(metadata)
    typer.echo(f"Metadata rows available: {total_rows}")

    indices = [o.row_idx] if o.row_idx is not None else list(range(total_rows))
    total_runs = len(indices) * o.simulations_per_row
    typer.echo(f"Rows to simulate: {len(indices)}")
    typer.echo(f"Simulations per row: {o.simulations_per_row}")
    typer.echo(f"Total simulations: {total_runs}")

    ml_shard_resolved = o.ml_shard_path.expanduser().resolve()
    persist_outputs = o.persist_standard_outputs or o.save_mode != "memory"
    backend_kwargs = {"n_workers": o.n_workers} if o.backend == "local" else {}

    with create_backend(o.backend, **backend_kwargs) as compute_backend:
        run_number = 0
        for idx in indices:
            for simulation_idx in range(o.simulations_per_row):
                run_number += 1
                typer.echo(
                    f"\n--- Simulation {run_number}/{total_runs}: row {idx + 1}/{total_rows} "
                    f"(index {idx}), repeat {simulation_idx + 1}/{o.simulations_per_row} ---"
                )
                run_idx = idx * o.simulations_per_row + simulation_idx
                run_seed = o.seed + run_idx if o.seed is not None else None
                if run_seed is not None:
                    typer.echo(f"Seed: {run_seed}")
                params = _build_simulation_params(
                    metadata=metadata,
                    row_idx=idx,
                    run_idx=run_idx,
                    main_dir=o.main_dir.expanduser().resolve(),
                    output_dir=o.output_dir.expanduser().resolve(),
                    project_name=o.project_name,
                    source_type=o.source_type,
                    save_mode=o.save_mode,
                    persist_standard_outputs=persist_outputs,
                    ml_shard_path=(
                        _path_with_suffix(
                            ml_shard_resolved, f"row{idx:04d}", f"sim{simulation_idx:03d}"
                        )
                        if total_runs > 1
                        else ml_shard_resolved
                    ),
                    n_pix=o.n_pix,
                    n_channels=o.n_channels,
                    n_lines=o.n_lines,
                    imaging_algorithm=o.imaging_algorithm,
                    background_level=o.background_level,
                    background_seed=(
                        o.background_seed if o.background_seed is not None else run_seed
                    ),
                    seed=run_seed,
                    instrument_geometry=str(o.instrument_geometry),
                    inject_serendipitous=o.inject_serendipitous,
                    output_subdir_name=(
                        f"{o.project_name}_{run_idx}"
                        if persist_outputs and total_runs > 1
                        else None
                    ),
                    input_path=o.input_path,
                    kaggle_token=o.kaggle_token,
                )
                _run_simulation_stages(
                    o, params, compute_backend, label=f"Simulation {run_number}/{total_runs}"
                )

    typer.echo(f"\nAll done. Simulated {total_runs} run(s) across {len(indices)} row(s).")


def _validated_simulation_options(o: SimulationRunOptions) -> SimulationRunOptions:
    """``o`` with its choices normalised; a usage error for anything invalid."""
    source_types = _SOURCE_TYPES + sorted(EXTERNAL_SOURCE_TYPES)
    o = replace(
        o,
        source_type=o.source_type.lower(),
        save_mode=o.save_mode.lower(),
        backend=o.backend.lower(),
        imaging_algorithm=o.imaging_algorithm.lower(),
    )
    if o.source_type not in source_types:
        usage_error("--source-type must be one of: " + ", ".join(source_types))
    if o.save_mode not in _SAVE_MODES:
        usage_error("--save-mode must be one of: " + ", ".join(_SAVE_MODES))
    if o.backend not in _BACKEND_TYPES:
        usage_error("--backend must be one of: " + ", ".join(_BACKEND_TYPES))
    if o.imaging_algorithm not in _IMAGING_ALGORITHMS:
        usage_error("--imaging-algorithm must be one of: " + ", ".join(_IMAGING_ALGORITHMS))
    if not o.metadata_path.exists():
        usage_error(f"--metadata-path not found: {o.metadata_path}")
    if o.input_path is not None and not o.input_path.expanduser().exists():
        usage_error(f"--input not found: {o.input_path}")
    if o.kaggle_token is not None and not o.kaggle_token.expanduser().exists():
        usage_error(f"--kaggle-token not found: {o.kaggle_token}")
    return o


def _run_simulation_stages(
    o: SimulationRunOptions, params: SimulationParams, compute_backend: Any, *, label: str
) -> None:
    """The clean cube, observation, imaging and export of one run, with a progress bar."""
    with tqdm(
        total=100,
        desc=label,
        unit="%",
        leave=True,
    ) as progress_bar:
        logger = _progress_logger(progress_bar)
        is_sync_backend = o.backend == "sync"

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
                robust=o.robust,
                terminal_logger=logger,
                interferometer_progress_callback=_progress_callback(progress_bar, 35, 55),
            ),
        )
        logger(f"Dirty cube shape: {simulation_results['dirty_cube'].shape}")
        logger(f"UV mask cube shape: {simulation_results['uv_mask_cube'].shape}")

        # The image-domain products (interferometric, total-power and
        # combined image cubes), as run_simulation makes them.
        simulation_results = _run_stage_with_feedback(
            progress_bar,
            label="imaging",
            work=lambda: image_products(simulation_results),
        )
        _update_progress(progress_bar, 90)

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
    if o.save_mode != "memory":
        for key in sorted(exported_results):
            if key in {"ml_dataset_path", "antenna_config_paths"}:
                continue
            if key.endswith("_path"):
                typer.echo(f"{key}: {exported_results[key]}")
            elif key.endswith("_paths"):
                for path in exported_results[key]:
                    typer.echo(f"{key}: {path}")
