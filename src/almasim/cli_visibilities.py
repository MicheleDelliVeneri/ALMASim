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
    skip_existing: bool = typer.Option(
        False,
        "--skip-existing",
        help="Skip archives that have already been extracted (detected via a .done marker file).",
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
        skip_existing=skip_existing,
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
        min=0,
        help="Number of Slurm workers for post-processing. Pass 0 to spawn one worker per UID.",
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
    continue_on_error: bool = typer.Option(
        True,
        "--continue-on-error/--fail-fast",
        help=(
            "Skip UIDs whose processing fails and carry on with the rest, then exit "
            "non-zero with a summary (default). --fail-fast aborts on the first failure."
        ),
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
        continue_on_error=continue_on_error,
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
        min=0,
        help="Number of Slurm workers for post-processing. Pass 0 to spawn one worker per UID.",
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
    continue_on_error: bool = typer.Option(
        True,
        "--continue-on-error/--fail-fast",
        help=(
            "Skip UIDs whose processing fails and carry on with the rest, then exit "
            "non-zero with a summary (default). --fail-fast aborts on the first failure."
        ),
    ),
    keep_working_copies: bool = typer.Option(
        False,
        "--keep-working-copies",
        help=(
            "Keep each UID's working directory (raw MS copy plus caltables, about 1.8x the "
            "raw MS) after its calibrated output is written. By default it is removed."
        ),
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
        continue_on_error=continue_on_error,
        keep_working_copies=keep_working_copies,
    )


@visibilities_app.command("spw-overview")
def spectral_windows_overview(
    input_ms: Path = typer.Argument(..., help="Source of the measurement set"),
) -> None:
    """Print the spectral-window summary for one MeasurementSet."""
    cli_image.derive_parameters(input_ms)


@visibilities_app.command("flag-summary")
def flag_summary(
    inputs: List[Path] = typer.Argument(
        ...,
        help=(
            "MeasurementSet(s) to summarise, or folders searched with --pattern. "
            "Several may be given."
        ),
    ),
    output_csv: Path = typer.Option(
        Path("flag_summary.csv"),
        "--output",
        "-o",
        help="CSV to write, one row per (MS, field, SPW).",
    ),
    pattern: str = typer.Option(
        "*.ms.split.cal", "--pattern", help="Glob for the MeasurementSets inside a folder."
    ),
    max_weight: float = typer.Option(
        1e4,
        "--max-weight",
        help="Unflagged rows with a WEIGHT above this count as high_weight_rows.",
    ),
    require_done_marker: bool = typer.Option(
        True,
        "--require-done-marker/--no-require-done-marker",
        help=(
            "In a folder, skip an MS whose <ms>.done calibration marker is missing. Only "
            "applies when the folder holds at least one such marker. MSs named directly "
            "are always summarised."
        ),
    ),
    fail_fast: bool = typer.Option(
        False,
        "--fail-fast",
        help="Abort on the first MS that cannot be read (default: record it and continue).",
    ),
) -> None:
    """Summarise flagging and visibility properties per MS, field and spectral window.

    For each (field, SPW): intents, channels, correlations, scans, antennas,
    time range and time on source, rows and visibilities flagged, median and
    largest unflagged weight, and unflagged rows above --max-weight. Reads only
    FLAG, FLAG_ROW, WEIGHT and the scalar columns, never DATA. An MS that cannot
    be read is listed in ``<output>.failed.tsv`` and the exit status is 1.
    """
    import pandas as pd
    from tqdm import tqdm

    from .services.archive.flag_summary import flag_summary as summarise

    mss: list[Path] = []
    for item in inputs:
        if (item / "table.dat").is_file():
            mss.append(item)
            continue
        if not item.is_dir():
            typer.echo(f"Not a MeasurementSet or folder: {item}", err=True)
            raise typer.Exit(code=1)
        found = sorted(p for p in item.glob(pattern) if p.is_dir())
        if require_done_marker:
            with_marker = [ms for ms in found if ms.with_name(ms.name + ".done").is_file()]
            if with_marker and len(with_marker) < len(found):
                typer.echo(
                    f"Skipping {len(found) - len(with_marker)} MS(s) in {item} without a "
                    ".done marker."
                )
                found = with_marker
        mss.extend(found)
    if not mss:
        typer.echo(f"Cannot find any MS matching {pattern} in {', '.join(map(str, inputs))}")
        raise typer.Exit(code=1)

    frames: list[pd.DataFrame] = []
    failures: list[tuple[Path, str]] = []
    for ms in tqdm(mss, disable=len(mss) == 1):
        try:
            frames.append(summarise(ms, max_weight=max_weight))
        except Exception as exc:
            if fail_fast:
                raise
            failures.append((ms, f"{type(exc).__name__}: {exc}"))
            typer.echo(f"FAILED {ms.name}: {type(exc).__name__}: {exc}", err=True)

    if frames:
        table = pd.concat(frames, ignore_index=True)
        table.to_csv(output_csv, index=False)
        rows, flagged = int(table["rows"].sum()), int(table["flagged_rows"].sum())
        high = int(table["high_weight_rows"].sum())
        with_high = int((table.groupby("uid")["high_weight_rows"].sum() > 0).sum())
        typer.echo(
            f"Wrote {len(table)} (MS, field, SPW) row(s) for {len(frames)} MS(s) to {output_csv}"
        )
        typer.echo(
            f"Rows: {rows:,}; flagged {flagged:,} ({100 * flagged / max(rows, 1):.2f} %); "
            f"unflagged with WEIGHT > {max_weight:g}: {high:,} in {with_high} MS(s)"
        )
        is_target = table["intents"].str.contains("OBSERVE_TARGET", na=False)
        for label, part in (("target", table[is_target]), ("other", table[~is_target])):
            if len(part):
                share = 100 * part["flagged_rows"].sum() / max(part["rows"].sum(), 1)
                typer.echo(f"  {label} fields: {share:.2f} % of rows flagged")
    if failures:
        failed_path = output_csv.with_name(output_csv.name + ".failed.tsv")
        failed_path.write_text("".join(f"{ms}\t{error}\n" for ms, error in failures))
        typer.echo(f"{len(failures)} MS(s) could not be read; see {failed_path}", err=True)
        raise typer.Exit(code=1)
