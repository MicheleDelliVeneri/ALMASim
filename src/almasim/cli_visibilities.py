"""Visibility processing commands for the ALMASim CLI."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer

from . import cli_image, cli_products
from .cli_shared import default_output_path

visibilities_app = typer.Typer(
    help="Visibility processing commands.",
    no_args_is_help=True,
)


@visibilities_app.command("extract")
def extract(
    source_root: Path = typer.Option(
        default_output_path("downloads"),
        "--source-root",
        help="Directory containing downloaded archive files.",
    ),
    destination: Optional[Path] = typer.Option(
        None,
        "--destination",
        help="Extraction destination (defaults to --source-root).",
    ),
    recursive: bool = typer.Option(
        True,
        "--recursive/--no-recursive",
        help="Recursively search --source-root for .tar/.tgz archives.",
    ),
    delete_archives: bool = typer.Option(
        False,
        "--delete-archives",
        help="Delete each archive after successful extraction.",
    ),
    postprocess_backend: str = typer.Option(
        "sync",
        "--postprocess-backend",
        help="Backend for extraction stage. Choices: sync, slurm.",
        case_sensitive=False,
    ),
    slurm_queue: str = typer.Option("normal", "--slurm-queue", help="Slurm queue/partition."),
    slurm_project: Optional[str] = typer.Option(
        None,
        "--slurm-project",
        help="Optional Slurm project/account.",
    ),
    slurm_walltime: str = typer.Option(
        "01:00:00",
        "--slurm-walltime",
        help="Slurm walltime per worker job (HH:MM:SS).",
    ),
    slurm_cores: int = typer.Option(
        1,
        "--slurm-cores",
        min=1,
        help="Cores per Slurm worker.",
    ),
    slurm_memory: str = typer.Option("4GB", "--slurm-memory", help="Memory per Slurm worker."),
    slurm_workers: int = typer.Option(
        4,
        "--slurm-workers",
        min=0,
        help="Number of Slurm workers. Pass 0 to spawn one worker per archive.",
    ),
    slurm_scheduler_host: Optional[str] = typer.Option(
        None,
        "--slurm-scheduler-host",
        help=(
            "IP or hostname that Slurm workers use to reach the Dask scheduler. "
            "Set this to an internal/HPC network address when the public hostname "
            "is not reachable from compute nodes (e.g. 10.20.25.44)."
        ),
    ),
) -> None:
    """Extract ALMA archive tarballs as a standalone step."""
    cli_products.products_extract(
        source_root=source_root,
        destination=destination,
        recursive=recursive,
        delete_archives=delete_archives,
        postprocess_backend=postprocess_backend,
        slurm_queue=slurm_queue,
        slurm_project=slurm_project,
        slurm_walltime=slurm_walltime,
        slurm_cores=slurm_cores,
        slurm_memory=slurm_memory,
        slurm_workers=slurm_workers,
        slurm_scheduler_host=slurm_scheduler_host,
    )


@visibilities_app.command("unpack")
def unpack(
    input_root: Path = typer.Option(
        default_output_path("downloads"),
        "--input-root",
        help="Directory containing extracted ASDM directories.",
    ),
    output_root: Path = typer.Option(
        default_output_path("downloads") / "archive_ms" / "raw_ms",
        "--output-root",
        help="Directory where raw MeasurementSets are written.",
    ),
    asdm_uid: Optional[List[str]] = typer.Option(
        None,
        "--asdm-uid",
        help="Optional ASDM UID(s) to process. Repeat or pass comma-separated values.",
    ),
    casa_data_root: Optional[Path] = typer.Option(
        None,
        "--casa-data-root",
        help="Optional CASA runtime data directory.",
    ),
    skip_casa_data_update: bool = typer.Option(
        False,
        "--skip-casa-data-update",
        help="Do not auto-download CASA runtime data if missing.",
    ),
    postprocess_backend: str = typer.Option(
        "sync",
        "--postprocess-backend",
        help="Backend for unpack stage. Choices: sync, slurm.",
        case_sensitive=False,
    ),
    slurm_queue: str = typer.Option("normal", "--slurm-queue", help="Slurm queue/partition."),
    slurm_project: Optional[str] = typer.Option(
        None,
        "--slurm-project",
        help="Optional Slurm project/account.",
    ),
    slurm_walltime: str = typer.Option(
        "02:00:00",
        "--slurm-walltime",
        help="Slurm walltime per worker job (HH:MM:SS).",
    ),
    slurm_cores: int = typer.Option(
        1,
        "--slurm-cores",
        min=1,
        help="Cores per Slurm worker.",
    ),
    slurm_memory: str = typer.Option("4GB", "--slurm-memory", help="Memory per Slurm worker."),
    slurm_workers: int = typer.Option(
        4,
        "--slurm-workers",
        min=1,
        help="Number of Slurm workers for post-processing.",
    ),
    slurm_scheduler_host: Optional[str] = typer.Option(
        None,
        "--slurm-scheduler-host",
        help=(
            "IP or hostname that Slurm workers use to reach the Dask scheduler. "
            "Set this to an internal/HPC network address when the public hostname "
            "is not reachable from compute nodes (e.g. 10.20.25.44)."
        ),
    ),
    overwrite_outputs: bool = typer.Option(
        False,
        "--overwrite-outputs",
        help="Overwrite existing raw MS outputs.",
    ),
) -> None:
    """Import ASDM directories into raw MeasurementSets as a standalone step."""
    cli_products.products_unpack(
        input_root=input_root,
        output_root=output_root,
        asdm_uid=asdm_uid,
        casa_data_root=casa_data_root,
        skip_casa_data_update=skip_casa_data_update,
        postprocess_backend=postprocess_backend,
        slurm_queue=slurm_queue,
        slurm_project=slurm_project,
        slurm_walltime=slurm_walltime,
        slurm_cores=slurm_cores,
        slurm_memory=slurm_memory,
        slurm_workers=slurm_workers,
        slurm_scheduler_host=slurm_scheduler_host,
        overwrite_outputs=overwrite_outputs,
    )


@visibilities_app.command("calibrate")
def calibrate(
    input_root: Path = typer.Option(
        default_output_path("downloads"),
        "--input-root",
        help="ALMA delivery root containing calibration products.",
    ),
    raw_ms_root: Path = typer.Option(
        default_output_path("downloads") / "archive_ms" / "raw_ms",
        "--raw-ms-root",
        help="Directory containing raw MeasurementSets.",
    ),
    output_root: Path = typer.Option(
        default_output_path("downloads") / "archive_ms" / "calibrated_ms",
        "--output-root",
        help="Directory where calibrated MeasurementSets are written.",
    ),
    asdm_uid: Optional[List[str]] = typer.Option(
        None,
        "--asdm-uid",
        help="Optional UID(s) to calibrate. Repeat or pass comma-separated values.",
    ),
    casa_data_root: Optional[Path] = typer.Option(
        None,
        "--casa-data-root",
        help="Optional CASA runtime data directory.",
    ),
    skip_casa_data_update: bool = typer.Option(
        False,
        "--skip-casa-data-update",
        help="Do not auto-download CASA runtime data if missing.",
    ),
    postprocess_backend: str = typer.Option(
        "sync",
        "--postprocess-backend",
        help="Backend for calibration stage. Choices: sync, slurm.",
        case_sensitive=False,
    ),
    slurm_queue: str = typer.Option("normal", "--slurm-queue", help="Slurm queue/partition."),
    slurm_project: Optional[str] = typer.Option(
        None,
        "--slurm-project",
        help="Optional Slurm project/account.",
    ),
    slurm_walltime: str = typer.Option(
        "02:00:00",
        "--slurm-walltime",
        help="Slurm walltime per worker job (HH:MM:SS).",
    ),
    slurm_cores: int = typer.Option(
        1,
        "--slurm-cores",
        min=1,
        help="Cores per Slurm worker.",
    ),
    slurm_memory: str = typer.Option("4GB", "--slurm-memory", help="Memory per Slurm worker."),
    slurm_workers: int = typer.Option(
        4,
        "--slurm-workers",
        min=1,
        help="Number of Slurm workers for post-processing.",
    ),
    slurm_scheduler_host: Optional[str] = typer.Option(
        None,
        "--slurm-scheduler-host",
        help=(
            "IP or hostname that Slurm workers use to reach the Dask scheduler. "
            "Set this to an internal/HPC network address when the public hostname "
            "is not reachable from compute nodes (e.g. 10.20.25.44)."
        ),
    ),
    overwrite_outputs: bool = typer.Option(
        False,
        "--overwrite-outputs",
        help="Overwrite existing calibrated MS outputs.",
    ),
    clean_intermediate_files: bool = typer.Option(
        False,
        "--clean-intermediate-files",
        help="Remove intermediate raw and working files after successful calibration.",
    ),
) -> None:
    """Create calibrated MeasurementSets as a standalone step."""
    cli_products.products_calibrate(
        input_root=input_root,
        raw_ms_root=raw_ms_root,
        output_root=output_root,
        asdm_uid=asdm_uid,
        casa_data_root=casa_data_root,
        skip_casa_data_update=skip_casa_data_update,
        postprocess_backend=postprocess_backend,
        slurm_queue=slurm_queue,
        slurm_project=slurm_project,
        slurm_walltime=slurm_walltime,
        slurm_cores=slurm_cores,
        slurm_memory=slurm_memory,
        slurm_workers=slurm_workers,
        slurm_scheduler_host=slurm_scheduler_host,
        overwrite_outputs=overwrite_outputs,
        clean_intermediate_files=clean_intermediate_files,
    )


@visibilities_app.command("spw-overview")
def spectral_windows_overview(
    input_ms: Path = typer.Argument(..., help="Source of the measurement set"),
) -> None:
    """Print the spectral-window summary for one MeasurementSet."""
    cli_image.derive_parameters(input_ms)
