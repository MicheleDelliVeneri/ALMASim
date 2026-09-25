"""Batch imaging commands for the ALMASim CLI."""

from __future__ import annotations

import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from time import sleep, time
from typing import Any, Optional, cast

import numpy as np
import pandas as pd
import typer
from astropy.time import Time
from tqdm import tqdm

ALMA_FOV_FACTOR = 1.12  # Standard primary-beam/FOV approximation: 1.12 * λ / D
RAD_TO_ARCSEC = 180 / np.pi * 3600
MIN_IMAGE_PIXELS = 16
SPEED_OF_LIGHT_M_S = 299_792_458.0
DAY_IN_SECONDS = 3600 * 24
TARGET_INTENT = "OBSERVE_TARGET"
_IMAGING_HEARTBEAT_S = 60.0


def import_casacore_tables() -> Any:
    try:
        from casacore.tables import table

        return table
    except ImportError:
        typer.echo(
            "Missing optional dependency `python-casacore` required to read metadata from "
            "the measurement set. Install it with the `ms-casacore` extra.",
            err=True,
        )
        raise typer.Exit(code=1)


image_app = typer.Typer(
    help="Visibility-to-image and image-to-visibility commands.",
    no_args_is_help=True,
)


def _iter_ms_inputs(input_path: Path) -> list[Path]:
    if input_path.is_dir() and input_path.name.endswith(".ms"):
        return [input_path]
    if input_path.is_dir():
        return sorted(path for path in input_path.rglob("*.ms") if path.is_dir())
    raise typer.BadParameter(f"Input path is not a MeasurementSet or MS folder: {input_path}")


def _discover_models_for_ms(input_ms: Path, output_directory: Path) -> list[Path]:
    model_root = output_directory / input_ms.stem
    return sorted(model_root.glob("SPW-*/wsclean-model.fits"))


def _predict_all_models_for_ms(input_ms: Path, output_directory: Path) -> list[Path]:
    model_paths = _discover_models_for_ms(input_ms, output_directory)
    output_mss: list[Path] = []
    if not model_paths:
        typer.echo(
            f"[debug] missing model FITS, skipping MS: {input_ms}",
            err=False,
        )
        return output_mss

    for model_path in model_paths:
        output_ms = model_path.parent / f"{input_ms.name}.predicted"
        predict_from_model(input_ms=input_ms, model=model_path, output_ms=output_ms)
        output_mss.append(output_ms)

    return output_mss


def _run_commands_with_slurm_cluster(
    commands: list[tuple[str, list[str]]],
    *,
    cores_per_task: int,
    node_cores: int,
    queue: str,
    project: str | None,
    walltime: str,
    memory: str,
    n_jobs: int,
    scheduler_host: str | None,
    scheduler_interface: str | None,
    task_timeout: float | None,
) -> None:
    from almasim.services.compute import create_backend

    if not commands:
        return

    with create_backend(
        "slurm",
        queue=queue,
        node_cores=node_cores,
        memory=memory,
        walltime=walltime,
        n_workers=n_jobs,
        project=project,
        scheduler_host=scheduler_host,
        scheduler_interface=scheduler_interface,
    ) as backend:
        futures: list[tuple[str, Any]] = []
        for label, cmd in commands:
            future = backend.submit_subcommand(
                command=cmd,
                cores=cores_per_task,
                timeout=task_timeout,
            )
            futures.append((label, future))

        for label, future in tqdm(futures, total=len(futures), desc="SLURM tasks"):
            result = future.result()
            if result.returncode != 0:
                typer.echo(f"Task failed: {label}", err=True)
                typer.echo(result.stderr.rstrip(), err=True)
                raise typer.Exit(code=result.returncode)


def science_selection(input_ms: Path) -> tuple[dict[int, int], list[int]]:
    """Which spectral windows carry visibilities, and which fields are the science target.

    An ALMA MeasurementSet lists every spectral window the correlator was set
    up with (WVR, channel-average, pointing …), but after ``split`` only the
    science windows have rows; imaging the others is wasted work. The main
    table also holds the calibrator scans, so the target fields are those
    observed with the ``OBSERVE_TARGET`` intent.

    Returns ``({spw_id: n_rows}, [target field ids])``.
    """
    casacore_table = import_casacore_tables()
    data_description = casacore_table(f"{input_ms}::DATA_DESCRIPTION", ack=False)
    spw_of_dd = np.asarray(data_description.getcol("SPECTRAL_WINDOW_ID"))
    states = casacore_table(f"{input_ms}::STATE", ack=False)
    obs_modes = list(states.getcol("OBS_MODE")) if states.nrows() else []
    main = casacore_table(str(input_ms), ack=False)
    dd_ids = np.asarray(main.getcol("DATA_DESC_ID"))
    unique_dd, counts = np.unique(dd_ids, return_counts=True)
    rows_per_spw: dict[int, int] = {}
    for dd_id, count in zip(unique_dd, counts):
        spw = int(spw_of_dd[int(dd_id)])
        rows_per_spw[spw] = rows_per_spw.get(spw, 0) + int(count)

    target_states = {i for i, mode in enumerate(obs_modes) if TARGET_INTENT in str(mode)}
    target_fields: list[int] = []
    if target_states:
        state_ids = np.asarray(main.getcol("STATE_ID"))
        field_ids = np.asarray(main.getcol("FIELD_ID"))
        mask = np.isin(state_ids, list(target_states))
        target_fields = sorted(int(f) for f in np.unique(field_ids[mask]))
    return rows_per_spw, target_fields


def compute_imaging_parameters(input_ms: Path, science_only: bool = True) -> pd.DataFrame:
    """One row per spectral window with the WSClean geometry for ``input_ms``.

    With ``science_only`` (the default) only spectral windows that actually
    hold visibilities are listed, and ``target_field_ids`` names the fields
    observed with the science intent so imaging can leave the calibrators out.
    """
    casacore_table = import_casacore_tables()
    spectral_windows = casacore_table(f"{input_ms}::SPECTRAL_WINDOW", ack=False)
    observation = casacore_table(f"{input_ms}::OBSERVATION", ack=False)
    start_mjd, end_mjd = observation[0]["TIME_RANGE"]
    start_datetime = Time(start_mjd / DAY_IN_SECONDS, format="mjd").to_datetime()
    end_datetime = Time(end_mjd / DAY_IN_SECONDS, format="mjd").to_datetime()
    antennas = casacore_table(f"{input_ms}::ANTENNA", ack=False)
    reference_frequencies = spectral_windows.getcol("REF_FREQUENCY")
    min_dish_diameter = np.min(antennas.getcol("DISH_DIAMETER"))
    antenna_pos = antennas.getcol("POSITION")
    i, j = np.triu_indices(antenna_pos.shape[0], k=1)
    distance = np.linalg.norm(antenna_pos[j, :] - antenna_pos[i, :], axis=1)
    max_baseline_size = max(distance)
    fov_per_frequency = (
        ALMA_FOV_FACTOR
        * SPEED_OF_LIGHT_M_S
        * RAD_TO_ARCSEC
        / reference_frequencies
        / min_dish_diameter
    )
    synthetized_beam_size = (
        SPEED_OF_LIGHT_M_S * RAD_TO_ARCSEC / reference_frequencies / max_baseline_size
    )
    spectral_window_id = np.arange(reference_frequencies.size, dtype=int)

    rows_per_spw: dict[int, int] = {}
    target_fields: list[int] = []
    if science_only:
        rows_per_spw, target_fields = science_selection(input_ms)
        keep = np.array([int(spw) in rows_per_spw for spw in spectral_window_id], dtype=bool)
    else:
        keep = np.ones(reference_frequencies.size, dtype=bool)
    n_rows = int(keep.sum())

    derived_parameters = pd.DataFrame(
        {
            "filename": [str(input_ms.resolve())] * n_rows,
            "spectral_window_id": spectral_window_id[keep],
            "reference_frequency": reference_frequencies[keep],
            "fov_per_frequency": fov_per_frequency[keep],
            "max_baseline_size": [max_baseline_size] * n_rows,
            "synthetized_beam_size": synthetized_beam_size[keep],
            "start_mjd_s": start_mjd,
            "end_mjd_s": end_mjd,
            "duration_s": end_mjd - start_mjd,
            "start_datetime": start_datetime.isoformat(),
            "end_datetime": end_datetime.isoformat(),
            "n_visibility_rows": [
                rows_per_spw.get(int(spw), -1) for spw in spectral_window_id[keep]
            ],
            "target_field_ids": [",".join(str(f) for f in target_fields)] * n_rows,
        }
    )
    return derived_parameters


def imaging_parameter_to_command_arg(
    imaging_parameters: pd.Series,
    fov_fraction: float,
    beam_sampling: float,
    auto_threshold: float | None = None,
    auto_mask: float | None = None,
) -> list[str]:
    """WSClean geometry and deconvolution flags for one spectral window.

    ``auto_threshold`` / ``auto_mask`` (in units of the residual noise) give
    CLEAN a stopping point; without them ``-niter`` is the only limit and every
    task cleans into the noise until the iteration cap.
    """
    spw = imaging_parameters["spectral_window_id"]
    fov = imaging_parameters["fov_per_frequency"]
    synthetized_beam_size = imaging_parameters["synthetized_beam_size"]
    synthetized_beam_size /= beam_sampling
    fov *= fov_fraction
    n_pixels = max(MIN_IMAGE_PIXELS, int(math.ceil(fov / synthetized_beam_size)))
    cmd_args = [
        "-scale",
        f"{synthetized_beam_size}asec",
        "-size",
        str(n_pixels),
        str(n_pixels),
        "-spws",
        str(spw),
        "-mgain",
        "0.85",
        "-niter",
        "100000",
        "-pol",
        "I",
        "-make-psf",
        "-weight",
        "briggs",
        "0.5",
    ]
    if auto_mask is not None and auto_mask > 0:
        cmd_args += ["-auto-mask", str(auto_mask)]
    if auto_threshold is not None and auto_threshold > 0:
        cmd_args += ["-auto-threshold", str(auto_threshold)]
    return cmd_args


@image_app.command("ms-overview", hidden=True)
def derive_parameters(
    input_ms: Path = typer.Argument(
        ...,
        help="Source of the measurement set",
    ),
):

    derived_parameters = compute_imaging_parameters(input_ms)

    typer.echo(derived_parameters.to_string(index=False))


@image_app.command("compute-parameters")
def compute_parameters(
    archive_folder: Path = typer.Argument(
        default=Path("."),
        help="Processed MSs folder (default: current directory)",
    ),
    output_metadata_file: Path = typer.Argument(
        default=Path("imaging_parameters.csv"),
        help="Parameters CSV file (default: imaging_parameters.csv)",
    ),
    pattern: str = typer.Option(
        "*.cal",
        "--pattern",
        help="Glob for the MeasurementSets inside the folder.",
    ),
    science_only: bool = typer.Option(
        True,
        "--science-only/--all-spws",
        help=(
            "List only spectral windows that hold visibilities and record the "
            "OBSERVE_TARGET field ids (default). --all-spws lists every window."
        ),
    ),
    require_done_marker: bool = typer.Option(
        True,
        "--require-done-marker/--no-require-done-marker",
        help=(
            "Skip an MS whose <ms>.done calibration marker is missing. Only applies when "
            "the folder holds at least one such marker, so plain folders still work."
        ),
    ),
    fail_fast: bool = typer.Option(
        False,
        "--fail-fast",
        help="Abort on the first MS that cannot be read (default: record it and continue).",
    ),
):
    """Derive per-spectral-window imaging geometry for every MS in a folder.

    An MS that cannot be read (a truncated table, say) is reported and listed
    in ``<output>.failed.tsv`` instead of aborting the whole scan; the command
    then exits with status 1 once every MS has been attempted.
    """
    mss = sorted(archive_folder.glob(pattern))
    if len(mss) == 0:
        typer.echo(f"Cannot find any MS in {archive_folder}")
        raise typer.Exit(code=1)

    if require_done_marker:
        with_marker = [ms for ms in mss if ms.with_name(ms.name + ".done").is_file()]
        if with_marker and len(with_marker) < len(mss):
            typer.echo(
                f"Skipping {len(mss) - len(with_marker)} MS(s) without a .done calibration marker."
            )
            mss = with_marker

    frames: list[pd.DataFrame] = []
    failures: list[tuple[Path, str]] = []
    for ms in tqdm(mss):
        try:
            frames.append(compute_imaging_parameters(ms, science_only=science_only))
        except Exception as exc:
            if fail_fast:
                raise
            failures.append((ms, f"{type(exc).__name__}: {exc}"))
            typer.echo(f"FAILED {ms.name}: {type(exc).__name__}: {exc}", err=True)

    if frames:
        main_output = pd.concat(frames, axis=0, ignore_index=True)
        main_output.to_csv(output_metadata_file, index=False)
        typer.echo(
            f"Wrote {len(main_output)} spectral-window row(s) for {len(frames)} MS(s) "
            f"to {output_metadata_file}"
        )
    if failures:
        failed_path = output_metadata_file.with_name(output_metadata_file.name + ".failed.tsv")
        failed_path.write_text(
            "".join(f"{ms}\t{error}\n" for ms, error in failures), encoding="utf-8"
        )
        typer.echo(f"{len(failures)} MS(s) could not be read; listed in {failed_path}", err=True)
        raise typer.Exit(code=1)


@dataclass(frozen=True)
class ImagingTask:
    """One WSClean run: one MeasurementSet, one spectral window."""

    label: str
    ms_path: Path
    spw: int
    output_dir: Path
    command: list[str]
    field_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class ImagingFailure:
    label: str
    error: str
    log_path: Optional[str] = None


def _error_headline(error: str, limit: int = 240) -> str:
    for line in str(error).splitlines():
        stripped = line.strip()
        if stripped:
            return stripped if len(stripped) <= limit else stripped[: limit - 1] + "…"
    return str(error)[:limit]


def _field_arguments(row: pd.Series) -> list[str]:
    """``-field`` selection from the ``target_field_ids`` column, if the CSV has one."""
    value = row.get("target_field_ids") if hasattr(row, "get") else None
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    if text.endswith(".0"):  # a single id that pandas read as a float
        text = text[:-2]
    return ["-field", text]


def build_imaging_tasks(
    parameters: pd.DataFrame,
    output_directory: Path,
    *,
    fov_fraction: float,
    beam_sampling: float,
    num_cores: int,
    max_cores_per_node: int,
    wsclean_bin: str = "wsclean",
    overwrite_outputs: bool = False,
    trust_existing_images: bool = False,
    auto_threshold: float | None = None,
    auto_mask: float | None = None,
    update_model: bool = False,
    task_memory_gb: float | None = None,
    scratch_dir: str | None = None,
    single_window: bool = True,
) -> tuple[list[ImagingTask], int]:
    """Turn parameter rows into WSClean tasks, skipping the ones already done.

    Returns ``(tasks, skipped)``. A task is done when its ``SPW-<n>.done``
    marker and ``wsclean-image.fits`` both exist; ``trust_existing_images``
    also accepts an image produced before markers existed and writes its
    marker. ``overwrite_outputs`` redoes everything.
    """
    from .services.imaging.archive_imaging import (
        IMAGE_FILENAME,
        SINGLE_WINDOW_PLACEHOLDER,
        is_imaging_complete,
        write_imaging_marker,
    )

    tasks: list[ImagingTask] = []
    skipped = 0
    # WSClean's -mem is a PERCENTAGE of system memory. Passing the fraction
    # (0.1 for 10 cores of 96) limited every task to 0.1 % ≈ 0.4 GB and made
    # WSClean segfault on the first gridding pass.
    mem_percent = min(100.0, 100.0 * num_cores / max_cores_per_node)
    for _, row in parameters.iterrows():
        input_filename = Path(row["filename"])
        spw = int(row["spectral_window_id"])
        outdir = output_directory / input_filename.stem / f"SPW-{spw}"
        if not overwrite_outputs:
            if is_imaging_complete(outdir):
                skipped += 1
                continue
            if trust_existing_images and (outdir / IMAGE_FILENAME).is_file():
                write_imaging_marker(outdir, ms_path=str(input_filename), spw=spw)
                skipped += 1
                continue
        command_args = imaging_parameter_to_command_arg(
            row, fov_fraction, beam_sampling, auto_threshold=auto_threshold, auto_mask=auto_mask
        )
        if single_window:
            # The wrapper hands WSClean a per-task MS holding only this window
            # and the target fields, so no -spws/-field selection and no
            # reordering (WSClean 3.7's reordering reads out of bounds on
            # multi-window ALMA sets; see extract_single_window_ms).
            spws_at = command_args.index("-spws")
            command_args = command_args[:spws_at] + command_args[spws_at + 2 :]
        command = [
            wsclean_bin,
            "-name",
            str(outdir / "wsclean"),
            "-j",
            str(num_cores),
            *command_args,
            # An absolute limit when the caller knows the worker's allocation
            # (WSClean's -mem is a share of the *node*, which can be far more
            # than the Slurm job may use); otherwise the node share.
            *(
                ["-abs-mem", f"{task_memory_gb:g}"]
                if task_memory_gb
                else ["-mem", f"{mem_percent:g}"]
            ),
            # Reordered visibilities go next to the task output (or into a
            # per-task scratch directory), not next to the MS: several
            # spectral windows of one MS run at the same time and would
            # otherwise clobber each other's <ms>-part*.tmp files.
            "-temp-dir",
            "__SCRATCH__" if scratch_dir else str(outdir),
            # Writing the model back into MODEL_DATA makes concurrent tasks
            # write the same MS from different nodes and grows every
            # calibrated MS; the model image is on disk anyway.
            "-update-model-required" if update_model else "-no-update-model-required",
            # -field is needed on the single-window MS too: WSClean images
            # field 0 by default and the extracted MS holds the target fields
            # under their original ids.
            *_field_arguments(row),
            *(["-no-reorder"] if single_window else []),
            SINGLE_WINDOW_PLACEHOLDER if single_window else str(input_filename),
        ]
        field_args = _field_arguments(row)
        field_ids = tuple(int(f) for f in field_args[1].split(",")) if field_args else ()
        tasks.append(
            ImagingTask(
                label=f"{input_filename.stem}_{spw}",
                ms_path=input_filename,
                spw=spw,
                output_dir=outdir,
                command=command,
                field_ids=field_ids,
            )
        )
    return tasks, skipped


def _tail_line(path: Path, modified_since: float, max_bytes: int = 2048) -> Optional[str]:
    try:
        if path.stat().st_mtime < modified_since:
            return None
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            chunk = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    lines = [line.strip() for line in chunk.splitlines() if line.strip()]
    return lines[-1] if lines else None


def _record_imaging_failure(task: ImagingTask, error: str, failures: list[ImagingFailure]) -> None:
    """Record a failed task and make sure a ``.failed`` marker exists for it.

    The worker-side wrapper writes the marker itself; this covers the cases
    where it never ran or was lost with its worker (walltime, node failure).
    """
    from .services.imaging.archive_imaging import (
        imaging_failure_marker_path,
        imaging_log_path,
        imaging_marker_path,
        write_imaging_failure_marker,
    )

    log_path = imaging_log_path(task.output_dir)
    if (
        not imaging_failure_marker_path(task.output_dir).is_file()
        and not imaging_marker_path(task.output_dir).is_file()
    ):
        try:
            write_imaging_failure_marker(
                task.output_dir,
                ms_path=str(task.ms_path),
                spw=task.spw,
                error=error,
                log_path=log_path if log_path.is_file() else None,
            )
        except OSError:
            pass
    failures.append(
        ImagingFailure(task.label, error, str(log_path) if log_path.is_file() else None)
    )


def run_imaging_tasks(
    tasks: list[ImagingTask],
    *,
    backend_kind: str,
    cores_per_task: int,
    node_cores: int,
    queue: str,
    project: str | None,
    walltime: str,
    memory: str,
    n_jobs: int,
    scheduler_host: str | None,
    scheduler_interface: str | None,
    task_timeout: float | None,
    continue_on_error: bool = True,
    failures: Optional[list[ImagingFailure]] = None,
    heartbeat_interval: float = _IMAGING_HEARTBEAT_S,
    scratch_root: str | None = None,
    task_retries: int = 3,
    single_window: bool = True,
    max_weight: float | None = None,
) -> list[dict[str, Any]]:
    """Run WSClean tasks, one subprocess each, and never let one failure end the run.

    Every task ends with a ``.done`` or ``.failed`` marker next to its output
    directory. With ``continue_on_error`` (the default) failed tasks are
    collected in ``failures`` and the rest of the batch keeps going; otherwise
    the first failure stops the run.
    """
    from .services.imaging.archive_imaging import imaging_log_path, run_wsclean_task

    if failures is None:
        failures = []
    if not tasks:
        return []

    results: list[dict[str, Any]] = []
    if backend_kind == "sync":
        for index, task in enumerate(tasks, start=1):
            log_path = imaging_log_path(task.output_dir)
            typer.echo(f"Image [{index}/{len(tasks)}] {task.label} (log: {log_path})")
            try:
                results.append(
                    run_wsclean_task(
                        command=task.command,
                        output_dir=str(task.output_dir),
                        ms_path=str(task.ms_path),
                        spw=task.spw,
                        threads=cores_per_task,
                        timeout=task_timeout,
                        scratch_root=scratch_root,
                        retries=task_retries,
                        single_window=single_window,
                        field_ids=list(task.field_ids),
                        max_weight=max_weight,
                    )
                )
            except Exception as exc:
                _record_imaging_failure(task, str(exc), failures)
                typer.echo(f"  FAILED {task.label}: {_error_headline(str(exc))}", err=True)
                if not continue_on_error:
                    raise typer.Exit(code=1)
        return results

    from almasim.services.compute import create_backend

    stage_label = "Slurm image"
    started_at = time()
    with create_backend(
        "slurm",
        queue=queue,
        node_cores=node_cores,
        memory=memory,
        walltime=walltime,
        n_workers=n_jobs,
        project=project,
        scheduler_host=scheduler_host,
        scheduler_interface=scheduler_interface,
    ) as backend:
        futures = [
            backend.submit_callable(
                run_wsclean_task,
                cores=cores_per_task,
                command=task.command,
                output_dir=str(task.output_dir),
                ms_path=str(task.ms_path),
                spw=task.spw,
                threads=cores_per_task,
                timeout=task_timeout,
                scratch_root=scratch_root,
                retries=task_retries,
                single_window=single_window,
                field_ids=list(task.field_ids),
                max_weight=max_weight,
            )
            for task in tasks
        ]
        pending = dict(enumerate(futures))
        last_heartbeat = started_at
        with tqdm(total=len(futures), desc=stage_label, unit="task", leave=True) as bar:
            while pending:
                progressed = False
                for index, future in list(pending.items()):
                    if not future.done():
                        continue
                    del pending[index]
                    progressed = True
                    bar.update(1)
                    task = tasks[index]
                    error: Optional[str] = None
                    try:
                        exc = future.exception()
                        error = str(exc) if exc is not None else None
                    except Exception as future_error:  # lost/cancelled future
                        error = str(future_error) or "task lost"
                    if error is None:
                        try:
                            results.append(future.result())
                        except Exception as result_error:
                            error = str(result_error)
                    if error is None:
                        bar.write(f"{stage_label} completed {task.label}")
                        continue
                    _record_imaging_failure(task, error, failures)
                    bar.write(f"  {task.label} FAILED: {_error_headline(error)}")
                    if not continue_on_error:
                        for other in pending.values():
                            cancel = getattr(other, "cancel", None)
                            if callable(cancel):
                                cancel()
                        raise typer.Exit(code=1)
                bar.set_postfix_str(
                    f"completed {len(futures) - len(pending)}/{len(futures)} failed {len(failures)}"
                )
                if not pending:
                    break
                now = time()
                if now - last_heartbeat >= heartbeat_interval:
                    last_heartbeat = now
                    running = 0
                    for index in pending:
                        last_line = _tail_line(
                            imaging_log_path(tasks[index].output_dir), started_at - 1.0
                        )
                        if last_line is None:
                            continue
                        running += 1
                        bar.write(f"  [{tasks[index].label}] {last_line}")
                    bar.write(
                        f"{stage_label}: {running} running, {len(pending) - running} waiting, "
                        f"{len(futures) - len(pending)} done, {len(failures)} failed"
                    )
                if not progressed:
                    sleep(0.5)
    return results


def _report_imaging_failures(failures: list[ImagingFailure]) -> None:
    if not failures:
        return
    typer.echo(f"Image: {len(failures)} task(s) failed and were skipped:", err=True)
    for failure in failures:
        typer.echo(f"  {failure.label}: {_error_headline(failure.error)}", err=True)
        if failure.log_path:
            typer.echo(f"    log: {failure.log_path}", err=True)


@image_app.command("image-from-ms")
def image_from_ms(
    imaging_parameters: Path = typer.Argument(help="Imaging parameter file"),
    output_directory: Path = typer.Argument(help="Output directory path"),
    fov_fraction: float = typer.Option(
        help="Fraction of the FOV to image",
        default=1.5,
        min=1e-6,
    ),
    beam_sampling: float = typer.Option(
        help="Number of pixels to use to sample the synthetized beam. Could be fractional.",
        default=8,
        min=1e-6,
    ),
    num_cores: int = typer.Option(help="Number of cores per imaging task", default=10, min=1),
    max_cores_per_node: int = typer.Option(
        help="Number of cores per node. [Used to scale the memory usage of wsclean]",
        default=95,
        min=1,
    ),
    wsclean_bin: str = typer.Option(
        "wsclean",
        "--wsclean-bin",
        help="Path or executable name of the WSClean binary the workers run.",
    ),
    auto_threshold: float = typer.Option(
        3.0,
        "--auto-threshold",
        min=0.0,
        help=(
            "Stop cleaning when the residual peak drops below this many sigma of the "
            "residual noise (WSClean -auto-threshold). 0 disables it and cleaning runs to "
            "-niter."
        ),
    ),
    auto_mask: float = typer.Option(
        0.0,
        "--auto-mask",
        min=0.0,
        help="WSClean -auto-mask level in sigma; 0 (default) disables auto-masking.",
    ),
    task_memory_gb: float = typer.Option(
        0.0,
        "--task-memory-gb",
        min=0.0,
        help=(
            "Absolute memory limit per WSClean task in GB (WSClean -abs-mem). Use the "
            "worker's Slurm allocation divided by the tasks per node. 0 (default) falls back "
            "to -mem <num-cores/max-cores-per-node> percent of the node."
        ),
    ),
    single_window: bool = typer.Option(
        True,
        "--single-window-ms/--no-single-window-ms",
        help=(
            "Extract each task's spectral window (and target fields) into a per-task MS on "
            "the worker and image it with -no-reorder (default). WSClean 3.7's reordering "
            "reads out of bounds on multi-window ALMA sets and crashes about half the tasks; "
            "--no-single-window-ms images the original MS with -spws/-field instead."
        ),
    ),
    task_retries: int = typer.Option(
        3,
        "--task-retries",
        min=0,
        help=(
            "Retry a task whose WSClean died by a signal (segfault) this many times before "
            "marking it failed. WSClean 3.7's multi-threaded reordering crashes "
            "non-deterministically on some inputs; a crash costs seconds."
        ),
    ),
    max_weight: float = typer.Option(
        10000.0,
        "--max-weight",
        min=0.0,
        help=(
            "Leave visibility rows whose WEIGHT exceeds this out of the single-window MS "
            "(0 keeps every row). Calibrated ALMA rows the pipeline had flagged but that "
            "were never re-flagged carry WEIGHT ~1e5-1e8 with garbage amplitudes and would "
            "dominate the image; physical weights stay below ~1e3."
        ),
    ),
    scratch_dir: Optional[Path] = typer.Option(
        None,
        "--scratch-dir",
        help=(
            "Directory for WSClean's reordered visibilities (one sub-directory per task, "
            "removed afterwards). Default: the task's output directory."
        ),
    ),
    update_model: bool = typer.Option(
        False,
        "--update-model/--no-update-model",
        help=(
            "Write the CLEAN model back into the MS MODEL_DATA column (WSClean "
            "-update-model-required). Off by default: concurrent spectral-window tasks "
            "would write the same MS and every calibrated MS would grow."
        ),
    ),
    postprocess_backend: str = typer.Option(
        "slurm",
        "--postprocess-backend",
        help="Where the WSClean tasks run: slurm (default) or sync (this process, one at a time).",
        case_sensitive=False,
    ),
    slurm_queue: str = typer.Option(default="normal", help="SLURM queue/partition"),
    slurm_project: str | None = typer.Option(default=None, help="SLURM project/account"),
    slurm_walltime: str = typer.Option(default="02:00:00", help="SLURM walltime HH:MM:SS"),
    slurm_memory: str = typer.Option(default="16GB", help="SLURM memory per worker"),
    slurm_n_jobs: int = typer.Option(default=1, min=1, help="Number of SLURM workers/jobs"),
    scheduler_host: str | None = typer.Option(
        default=None,
        help="Scheduler host advertised to workers (defaults to submit HOSTNAME)",
    ),
    scheduler_interface: str | None = typer.Option(
        default=None,
        help="Scheduler/worker network interface (e.g. ib0, eth0)",
    ),
    task_timeout: float = typer.Option(
        default=3600,
        min=1,
        help="Timeout in seconds for each worker-side command",
    ),
    skip_existing: bool = typer.Option(
        False,
        "--skip-existing",
        help=(
            "Also trust an existing wsclean-image.fits that has no .done marker (an output "
            "from before markers existed) and write its marker. Tasks with a marker are "
            "always skipped."
        ),
    ),
    overwrite_outputs: bool = typer.Option(
        False,
        "--overwrite-outputs",
        help="Redo every task, including the ones with a .done marker.",
    ),
    continue_on_error: bool = typer.Option(
        True,
        "--continue-on-error/--fail-fast",
        help=(
            "Keep imaging the rest of the batch when a task fails (default); the failed "
            "tasks are listed at the end and the command exits 1. --fail-fast stops at the "
            "first failure."
        ),
    ),
):
    """Image every (MS, spectral window) row of a parameter CSV with WSClean.

    Each task leaves ``SPW-<n>.done`` or ``SPW-<n>.failed`` (with the error and
    the log path) next to its output directory, and ``SPW-<n>.log`` with the
    full WSClean output. A rerun skips the tasks with a ``.done`` marker and
    retries the rest, so an interrupted batch can simply be resubmitted.
    """
    backend_kind = postprocess_backend.lower()
    if backend_kind not in {"sync", "slurm"}:
        typer.echo("--postprocess-backend must be one of: sync, slurm.", err=True)
        raise typer.Exit(code=2)

    parameters = pd.read_csv(str(imaging_parameters))
    tasks, skipped = build_imaging_tasks(
        parameters,
        output_directory,
        fov_fraction=fov_fraction,
        beam_sampling=beam_sampling,
        num_cores=num_cores,
        max_cores_per_node=max_cores_per_node,
        wsclean_bin=wsclean_bin,
        overwrite_outputs=overwrite_outputs,
        trust_existing_images=skip_existing,
        auto_threshold=auto_threshold if auto_threshold > 0 else None,
        auto_mask=auto_mask if auto_mask > 0 else None,
        update_model=update_model,
        task_memory_gb=task_memory_gb if task_memory_gb > 0 else None,
        scratch_dir=str(scratch_dir) if scratch_dir is not None else None,
        single_window=single_window,
    )
    if skipped:
        typer.echo(f"Skipped {skipped} already-imaged SPW(s).")
    if not tasks:
        typer.echo("All requested tasks are already imaged; nothing to do.")
        return
    typer.echo(f"Imaging {len(tasks)} (MS, SPW) task(s); per-task logs: <output>/<ms>/SPW-<n>.log")

    failures: list[ImagingFailure] = []
    run_imaging_tasks(
        tasks,
        backend_kind=backend_kind,
        cores_per_task=num_cores,
        node_cores=max_cores_per_node,
        queue=slurm_queue,
        project=slurm_project,
        walltime=slurm_walltime,
        memory=slurm_memory,
        n_jobs=slurm_n_jobs,
        scheduler_host=scheduler_host,
        scheduler_interface=scheduler_interface,
        task_timeout=task_timeout,
        continue_on_error=continue_on_error,
        failures=failures,
        scratch_root=str(scratch_dir) if scratch_dir is not None else None,
        task_retries=task_retries,
        single_window=single_window,
        max_weight=max_weight if max_weight > 0 else None,
    )
    typer.echo(f"Imaged {len(tasks) - len(failures)}/{len(tasks)} task(s), {len(failures)} failed.")
    if failures:
        _report_imaging_failures(failures)
        raise typer.Exit(code=1)


@image_app.command("batch-image", hidden=True)
def image_set(
    imaging_parameters: Path = typer.Argument(help="Imaging parameter file"),
    output_directory: Path = typer.Argument(help="Output directory path"),
    fov_fraction: float = typer.Option(
        help="Fraction of the FOV to image",
        default=1.5,
        min=1e-6,
    ),
    beam_sampling: float = typer.Option(
        help="Number of pixels to use to sample the synthetized beam. Could be fractional.",
        default=8,
        min=1e-6,
    ),
    num_cores: int = typer.Option(help="Number of cores per imaging task", default=10, min=1),
    max_cores_per_node: int = typer.Option(
        help="Number of cores per node. [Used to scale the memory usage of wsclean]",
        default=95,
        min=1,
    ),
    wsclean_bin: str = typer.Option("wsclean", "--wsclean-bin"),
    auto_threshold: float = typer.Option(3.0, "--auto-threshold", min=0.0),
    auto_mask: float = typer.Option(0.0, "--auto-mask", min=0.0),
    update_model: bool = typer.Option(False, "--update-model/--no-update-model"),
    task_memory_gb: float = typer.Option(0.0, "--task-memory-gb", min=0.0),
    task_retries: int = typer.Option(3, "--task-retries", min=0),
    single_window: bool = typer.Option(True, "--single-window-ms/--no-single-window-ms"),
    max_weight: float = typer.Option(10000.0, "--max-weight", min=0.0),
    scratch_dir: Optional[Path] = typer.Option(None, "--scratch-dir"),
    postprocess_backend: str = typer.Option("slurm", "--postprocess-backend", case_sensitive=False),
    slurm_queue: str = typer.Option(default="normal", help="SLURM queue/partition"),
    slurm_project: str | None = typer.Option(default=None, help="SLURM project/account"),
    slurm_walltime: str = typer.Option(default="02:00:00", help="SLURM walltime HH:MM:SS"),
    slurm_memory: str = typer.Option(default="16GB", help="SLURM memory per worker"),
    slurm_n_jobs: int = typer.Option(default=1, min=1, help="Number of SLURM workers/jobs"),
    scheduler_host: str | None = typer.Option(default=None),
    scheduler_interface: str | None = typer.Option(default=None),
    task_timeout: float = typer.Option(default=3600, min=1),
    skip_existing: bool = typer.Option(False, "--skip-existing"),
    overwrite_outputs: bool = typer.Option(False, "--overwrite-outputs"),
    continue_on_error: bool = typer.Option(True, "--continue-on-error/--fail-fast"),
):
    """Alias of ``image-from-ms``."""
    image_from_ms(
        imaging_parameters=imaging_parameters,
        output_directory=output_directory,
        fov_fraction=fov_fraction,
        beam_sampling=beam_sampling,
        num_cores=num_cores,
        max_cores_per_node=max_cores_per_node,
        wsclean_bin=wsclean_bin,
        auto_threshold=auto_threshold,
        auto_mask=auto_mask,
        update_model=update_model,
        task_memory_gb=task_memory_gb,
        task_retries=task_retries,
        single_window=single_window,
        max_weight=max_weight,
        scratch_dir=scratch_dir,
        postprocess_backend=postprocess_backend,
        slurm_queue=slurm_queue,
        slurm_project=slurm_project,
        slurm_walltime=slurm_walltime,
        slurm_memory=slurm_memory,
        slurm_n_jobs=slurm_n_jobs,
        scheduler_host=scheduler_host,
        scheduler_interface=scheduler_interface,
        task_timeout=task_timeout,
        skip_existing=skip_existing,
        overwrite_outputs=overwrite_outputs,
        continue_on_error=continue_on_error,
    )


@image_app.command("ms-from-image", hidden=True)
def ms_from_image(
    input_ms_or_folder: Path = typer.Argument(help="Single MS directory or folder of MSs"),
    output_directory: Path = typer.Argument(help="Directory containing model FITS products"),
    use_slurm: bool = typer.Option(help="Whether or not to use slurm or not", default=True),
    num_cores: int = typer.Option(help="Number of cores per predict task", default=1, min=1),
    max_cores_per_node: int = typer.Option(
        help="Number of cores per node for SLURM worker resource accounting",
        default=95,
        min=1,
    ),
    slurm_queue: str = typer.Option(default="normal", help="SLURM queue/partition"),
    slurm_project: str | None = typer.Option(default=None, help="SLURM project/account"),
    slurm_walltime: str = typer.Option(default="02:00:00", help="SLURM walltime HH:MM:SS"),
    slurm_memory: str = typer.Option(default="16GB", help="SLURM memory per worker"),
    slurm_n_jobs: int = typer.Option(default=1, min=1, help="Number of SLURM workers/jobs"),
    scheduler_host: str | None = typer.Option(
        default=None,
        help="Scheduler host advertised to workers (defaults to submit HOSTNAME)",
    ),
    scheduler_interface: str | None = typer.Option(
        default=None,
        help="Scheduler/worker network interface (e.g. ib0, eth0)",
    ),
    task_timeout: float = typer.Option(
        default=3600,
        min=1,
        help="Timeout in seconds for each worker-side command",
    ),
):
    from .cli_predict import ms_from_image as _predict_ms_from_image

    _predict_ms_from_image(
        input_ms_or_folder=input_ms_or_folder,
        output_directory=output_directory,
        use_slurm=use_slurm,
        num_cores=num_cores,
        max_cores_per_node=max_cores_per_node,
        slurm_queue=slurm_queue,
        slurm_project=slurm_project,
        slurm_walltime=slurm_walltime,
        slurm_memory=slurm_memory,
        slurm_n_jobs=slurm_n_jobs,
        scheduler_host=scheduler_host,
        scheduler_interface=scheduler_interface,
        task_timeout=task_timeout,
    )


@image_app.command("predict-single", hidden=True)
def predict_single(
    input_ms: Path = typer.Argument(..., help="Input MS"),
    model: Path = typer.Argument(..., help="FITS model path"),
    output_ms: Path = typer.Argument(..., help="Output MS"),
):
    predict_from_model(input_ms=input_ms, model=model, output_ms=output_ms)


def predict_from_model(input_ms: Path, model: Path, output_ms: Path) -> None:
    from astropy.io import fits
    from ducc0.wgridder import dirty2vis

    typer.echo(f"Predicting visibilities from model: {model}")
    typer.echo(f"Copying measurement set: {input_ms} -> {output_ms}")

    if output_ms.exists():
        typer.echo(f"Output MS already exists: {output_ms}", err=True)
        raise typer.Exit(code=1)

    casacore_table = import_casacore_tables()
    shutil.copytree(input_ms, output_ms)
    main_table = casacore_table(str(output_ms), readonly=False)
    try:
        nthreads = int(os.environ.get("SLURM_CPUS_PER_TASK") or (os.cpu_count() or 1))
        uvw = main_table.getcol("UVW")  # (nrows, 3) metres
        with fits.open(model) as hdul:
            main_hdu = cast(fits.PrimaryHDU, hdul[0])
            header = main_hdu.header

            # Pixel sizes: FITS CDELT is in degrees, ducc0 expects radians
            deg2rad = np.pi / 180.0
            pixsize_x = abs(cast(float, header["CDELT1"])) * deg2rad
            pixsize_y = abs(cast(float, header["CDELT2"])) * deg2rad

            # Channel frequencies from the FITS WCS axis 3 (CRVAL3/CDELT3/CRPIX3/NAXIS3)
            nchans = int(cast(int, header["NAXIS3"]))
            crval3 = cast(float, header["CRVAL3"])  # reference frequency [Hz]
            cdelt3 = cast(float, header["CDELT3"])  # frequency increment [Hz]
            crpix3 = cast(float, header["CRPIX3"])  # reference pixel (1-based)
            freq = (crval3 + (np.arange(nchans) - (crpix3 - 1)) * cdelt3).astype(np.float64)
            typer.echo(f"Computed frequency grid from FITS header: nchans={nchans}")

            # Model image: FITS axes are [stokes, freq, y, x]; take first stokes & channel
            assert main_hdu.data is not None, "FITS primary HDU contains no image data"
            dirty = np.ascontiguousarray(main_hdu.data[0, 0, :, :].astype(np.float64))
            typer.echo(f"Input arrays: uvw={uvw.shape}, dirty={dirty.shape}, freq={freq.shape}")

            # Predict visibilities from the model image → shape (nrows, nchan)
            model_vis = dirty2vis(
                uvw=uvw.astype(np.float64),
                dirty=dirty,
                freq=freq,
                pixsize_x=pixsize_x,
                pixsize_y=pixsize_y,
                do_wgridding=True,
                epsilon=1e-4,
                nthreads=nthreads,
            )
            typer.echo(f"Predicted visibilities with shape: {model_vis.shape}")

            # Write predicted visibilities to MODEL_DATA (nrows, 4, nchan): XX, XY, YX, YY
            zeros = np.zeros_like(model_vis)
            half = model_vis / 2
            model_vis_col = np.stack(
                [half, zeros, zeros, half],
                axis=1,
            )  # XX=I/2, XY=0, YX=0, YY=I/2
            main_table.putcol("MODEL_DATA", model_vis_col)
    finally:
        main_table.close()

    typer.echo(f"Wrote MODEL_DATA to: {output_ms}")


@image_app.command("predict-batch", hidden=True)
def predict_batch(
    imaging_parameters: Path = typer.Argument(..., help="Imaging parameter file"),
    output_directory: Path = typer.Argument(..., help="Output directory path"),
    num_cores: int = typer.Option(help="Number of cores per predict task", default=1, min=1),
    use_slurm: bool = typer.Option(help="Whether or not to use slurm or not", default=True),
    max_cores_per_node: int = typer.Option(
        help="Number of cores per node for SLURM worker resource accounting",
        default=95,
        min=1,
    ),
    slurm_queue: str = typer.Option(default="normal", help="SLURM queue/partition"),
    slurm_project: str | None = typer.Option(default=None, help="SLURM project/account"),
    slurm_walltime: str = typer.Option(default="02:00:00", help="SLURM walltime HH:MM:SS"),
    slurm_memory: str = typer.Option(default="16GB", help="SLURM memory per worker"),
    slurm_n_jobs: int = typer.Option(default=1, min=1, help="Number of SLURM workers/jobs"),
    scheduler_host: str | None = typer.Option(
        default=None,
        help="Scheduler host advertised to workers (defaults to submit HOSTNAME)",
    ),
    scheduler_interface: str | None = typer.Option(
        default=None,
        help="Scheduler/worker network interface (e.g. ib0, eth0)",
    ),
    task_timeout: float = typer.Option(
        default=3600,
        min=1,
        help="Timeout in seconds for each worker-side command",
    ),
):
    parameters = pd.read_csv(str(imaging_parameters))
    slurm_commands: list[tuple[str, list[str]]] = []
    for _, dset_parameter in tqdm(parameters.iterrows(), total=len(parameters)):
        input_filename = Path(dset_parameter["filename"])
        spw_id = int(dset_parameter["spectral_window_id"])
        model_path = output_directory / input_filename.stem / f"SPW-{spw_id}" / "wsclean-model.fits"
        if not model_path.exists():
            typer.echo(
                "[debug] missing model FITS, skipping row:"
                "ms={input_filename} spw={spw_id} path={model_path}",
                err=False,
            )
            continue

        output_ms = model_path.parent / f"{input_filename.name}.predicted"
        predict_cmd = [
            "almasim",
            "image",
            "predict-single",
            str(input_filename.absolute()),
            str(model_path),
            str(output_ms),
        ]
        if use_slurm:
            label = input_filename.stem + f"_{spw_id}_predict"
            slurm_commands.append((label, predict_cmd))
        else:
            # Set up environment to prioritize system libraries for compatibility.
            cmd_env = os.environ.copy()
            ld_library_path = "/lib64:/usr/lib64:/usr/local/lib64:/lib:/usr/lib:/usr/local/lib"
            existing_ld = cmd_env.get("LD_LIBRARY_PATH", "")
            if existing_ld:
                ld_library_path = f"{ld_library_path}:{existing_ld}"
            cmd_env["LD_LIBRARY_PATH"] = ld_library_path
            subprocess.run(predict_cmd, check=True, env=cmd_env)

    if use_slurm:
        _run_commands_with_slurm_cluster(
            slurm_commands,
            cores_per_task=num_cores,
            node_cores=max_cores_per_node,
            queue=slurm_queue,
            project=slurm_project,
            walltime=slurm_walltime,
            memory=slurm_memory,
            n_jobs=slurm_n_jobs,
            scheduler_host=scheduler_host,
            scheduler_interface=scheduler_interface,
            task_timeout=task_timeout,
        )
