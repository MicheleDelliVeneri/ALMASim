"""Batch imaging commands for the ALMASim CLI."""

from __future__ import annotations

import math
import os
import shutil
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from time import sleep, time
from typing import Any, Optional, Sequence, cast

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
# How long the Slurm driver waits for its workers before it starts anyway.
_WORKER_WAIT_S = 1800.0
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
    rows_per_spw, target_fields, _ = _field_scan(input_ms)
    return rows_per_spw, target_fields


def intent_matches(obs_mode: str, intents: Sequence[str]) -> bool:
    """Whether a STATE ``OBS_MODE`` carries one of ``intents``.

    ``OBS_MODE`` is a comma-separated list such as
    ``CALIBRATE_PHASE#ON_SOURCE,CALIBRATE_WVR#ON_SOURCE``. An intent with a
    ``#`` sub-intent must match a token exactly; one without (``OBSERVE_TARGET``)
    matches any of its sub-intents.
    """
    tokens = [token.strip() for token in str(obs_mode).split(",") if token.strip()]
    for intent in intents:
        if "#" in intent:
            if intent in tokens:
                return True
        elif any(token.split("#", 1)[0] == intent for token in tokens):
            return True
    return False


def _field_scan(
    input_ms: Path,
    intents: Sequence[str] = (TARGET_INTENT,),
) -> tuple[dict[int, int], list[int], dict[tuple[int, int], int]]:
    """``science_selection`` plus ``{(spw_id, field_id): n_rows}`` for every field.

    The selected fields are those with rows under one of ``intents``
    (``OBSERVE_TARGET`` by default).

    The per-field counts cover all of a field's rows in a window, whatever the
    intent, which is what the single-window extraction copies for it.
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

    field_ids = np.asarray(main.getcol("FIELD_ID"))
    rows_per_spw_field: dict[tuple[int, int], int] = {}
    pairs, pair_counts = np.unique(np.stack([dd_ids, field_ids]), axis=1, return_counts=True)
    for (dd_id, field_id), count in zip(pairs.T, pair_counts):
        key = (int(spw_of_dd[int(dd_id)]), int(field_id))
        rows_per_spw_field[key] = rows_per_spw_field.get(key, 0) + int(count)

    target_states = {i for i, mode in enumerate(obs_modes) if intent_matches(mode, intents)}
    target_fields: list[int] = []
    if target_states:
        state_ids = np.asarray(main.getcol("STATE_ID"))
        mask = np.isin(state_ids, list(target_states))
        target_fields = sorted(int(f) for f in np.unique(field_ids[mask]))
    return rows_per_spw, target_fields, rows_per_spw_field


SPECTRAL_COLUMNS = (
    "n_channels",
    "channel_width_hz",
    "spectral_resolution_hz",
    "total_bandwidth_hz",
)


def spectral_window_properties(spectral_windows: Any) -> dict[str, np.ndarray]:
    """Channelisation of every row of an open SPECTRAL_WINDOW table.

    ``channel_width_hz`` and ``spectral_resolution_hz`` are the median of the
    per-channel CHAN_WIDTH and RESOLUTION (absolute: ALMA lower-sideband
    windows have negative widths). ALMA's online Hanning smoothing makes the
    resolution of an unaveraged FDM window twice its channel width (3840 x
    488 kHz channels, 977 kHz resolution); an online-averaged window has a
    resolution about equal to its width.
    """
    n_rows = spectral_windows.nrows()
    widths = np.empty(n_rows)
    resolutions = np.empty(n_rows)
    for row in range(n_rows):
        widths[row] = np.median(np.abs(spectral_windows.getcell("CHAN_WIDTH", row)))
        resolutions[row] = np.median(np.abs(spectral_windows.getcell("RESOLUTION", row)))
    return {
        "n_channels": np.asarray(spectral_windows.getcol("NUM_CHAN"), dtype=int),
        "channel_width_hz": widths,
        "spectral_resolution_hz": resolutions,
        "total_bandwidth_hz": np.abs(np.asarray(spectral_windows.getcol("TOTAL_BANDWIDTH"))),
    }


def add_spectral_columns(parameters: pd.DataFrame) -> pd.DataFrame:
    """Fill ``SPECTRAL_COLUMNS`` from each MS when a CSV predates them.

    Reads only the SPECTRAL_WINDOW subtables, so an existing parameter CSV
    can be imaged as cubes without rerunning ``compute-parameters``.
    """
    if all(column in parameters.columns for column in SPECTRAL_COLUMNS):
        return parameters
    casacore_table = import_casacore_tables()
    filled = parameters.copy()
    for column in SPECTRAL_COLUMNS:
        if column not in filled.columns:
            filled[column] = np.nan
    for filename, rows in tqdm(filled.groupby("filename").groups.items(), desc="Spectral windows"):
        spectral_windows = casacore_table(f"{filename}::SPECTRAL_WINDOW", ack=False)
        try:
            spectral = spectral_window_properties(spectral_windows)
        finally:
            close = getattr(spectral_windows, "close", None)
            if callable(close):
                close()
        spws = filled.loc[rows, "spectral_window_id"].astype(int).to_numpy()
        for column, values in spectral.items():
            filled.loc[rows, column] = values[spws]
    filled["n_channels"] = filled["n_channels"].astype(int)
    return filled


@dataclass(frozen=True)
class CubeOptions:
    """How ``cube_channels`` picks the number of output channels.

    * ``spectral_sampling``: output channels per spectral resolution element,
      the spectral counterpart of ``beam_sampling``. 2 (Nyquist) keeps the
      native channels of a Hanning-smoothed window.
    * ``velocity_resolution_km_s``: never make channels narrower than this
      (0: no limit).
    * ``max_cube_gb``: largest cube, in GB of 32-bit pixels, per product
      (image, residual ...); wide images get fewer channels.
    """

    spectral_sampling: float = 2.0
    velocity_resolution_km_s: float = 0.0
    max_cube_gb: float = 4.0


def _largest_divisor_at_most(n: int, limit: int) -> int:
    """The largest divisor of ``n`` not above ``limit`` (at least 1)."""
    for candidate in range(max(min(int(limit), int(n)), 1), 0, -1):
        if n % candidate == 0:
            return candidate
    return 1


def cube_channels(row: pd.Series, n_pixels: int, options: CubeOptions) -> int:
    """Output channels (WSClean ``-channels-out``) for one spectral window.

    The channel width is the widest of: the native channel, the spectral
    resolution divided by ``spectral_sampling``, and ``velocity_resolution``
    at the window's reference frequency. The count it gives is then capped so
    that an ``n_pixels`` x ``n_pixels`` cube stays below ``max_cube_gb``, and
    rounded down to a divisor of the native channel count, so every output
    channel averages the same number of native channels and the cube has one
    channel width.
    """
    missing = [column for column in SPECTRAL_COLUMNS if column not in row.index]
    if missing:
        raise ValueError(
            f"Cube imaging needs the spectral columns {', '.join(missing)}; "
            "rerun compute-parameters or use add_spectral_columns"
        )
    n_native = int(row["n_channels"])
    width = float(row["channel_width_hz"])
    resolution = float(row["spectral_resolution_hz"]) or width
    target_width = max(width, resolution / options.spectral_sampling)
    if options.velocity_resolution_km_s > 0:
        velocity_width = (options.velocity_resolution_km_s * 1e3 / SPEED_OF_LIGHT_M_S) * float(
            row["reference_frequency"]
        )
        target_width = max(target_width, velocity_width)
    wanted = max(int(math.floor(n_native * width / target_width + 1e-9)), 1)
    plane_bytes = 4 * n_pixels * n_pixels
    affordable = (
        wanted
        if math.isinf(options.max_cube_gb)
        else max(int(options.max_cube_gb * 1e9 // plane_bytes), 1)
    )
    return _largest_divisor_at_most(n_native, min(wanted, affordable))


def image_pixels(
    imaging_parameters: pd.Series,
    fov_fraction: float,
    beam_sampling: float,
    min_beams_across: float = 0.0,
) -> int:
    """Image side in pixels: the FOV over the cell (beam / ``beam_sampling``).

    ``min_beams_across`` widens images narrower than that many synthesized
    beams: a compact configuration's primary beam spans few of them.
    """
    cell = imaging_parameters["synthetized_beam_size"] / beam_sampling
    fov = imaging_parameters["fov_per_frequency"] * fov_fraction
    return max(
        MIN_IMAGE_PIXELS,
        int(math.ceil(fov / cell)),
        int(math.ceil(min_beams_across * beam_sampling)),
    )


def compute_imaging_parameters(
    input_ms: Path,
    science_only: bool = True,
    intents: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """WSClean geometry for ``input_ms``: one row per spectral window and field.

    Every row images one field: ``target_field_ids`` holds its single id and
    ``field_name`` its name. WSClean's ``-field`` needs all listed fields to
    share a phase centre, and multi-target observations do not: ``-field 2,4``
    on HOPS-358 and HOPS-373, 12.4' apart, gridded both onto one image
    (uid___A002_X10d9399_X32fd, Cycle 11).

    Only (window, field) pairs that hold visibilities get a row. With
    ``science_only`` (the default) the fields are those observed with one of
    ``intents`` (``OBSERVE_TARGET`` unless given, e.g.
    ``CALIBRATE_PHASE#ON_SOURCE`` for the phase calibrators); an MS without
    such a field gives no rows. Otherwise every field is listed, calibrators
    included, and ``intents`` must not be given.
    """
    if intents and not science_only:
        raise ValueError("intents select fields; they cannot be combined with all fields")
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
    spectral = spectral_window_properties(spectral_windows)

    _, selected_fields, rows_per_spw_field = _field_scan(
        input_ms, tuple(intents) if intents else (TARGET_INTENT,)
    )
    # (index into the spectral-window arrays, field id, n rows): only pairs
    # with visibilities, so every row is something WSClean can image.
    entries: list[tuple[int, int, int]] = []
    for index, spw in enumerate(int(s) for s in spectral_window_id):
        fields = sorted(
            f
            for (s, f), rows in rows_per_spw_field.items()
            if s == spw and rows > 0 and (not science_only or f in selected_fields)
        )
        entries.extend((index, f, rows_per_spw_field[(spw, f)]) for f in fields)
    field_names: list[str] = []
    if entries:
        field_table = casacore_table(f"{input_ms}::FIELD", ack=False)
        field_names = [str(name) for name in field_table.getcol("NAME")]
    keep = np.array([index for index, _, _ in entries], dtype=int)
    n_rows = len(entries)

    derived_parameters = pd.DataFrame(
        {
            "filename": [str(input_ms.resolve())] * n_rows,
            "spectral_window_id": spectral_window_id[keep],
            "reference_frequency": reference_frequencies[keep],
            "fov_per_frequency": fov_per_frequency[keep],
            "max_baseline_size": [max_baseline_size] * n_rows,
            "synthetized_beam_size": synthetized_beam_size[keep],
            **{column: values[keep] for column, values in spectral.items()},
            "start_mjd_s": start_mjd,
            "end_mjd_s": end_mjd,
            "duration_s": end_mjd - start_mjd,
            "start_datetime": start_datetime.isoformat(),
            "end_datetime": end_datetime.isoformat(),
            "n_visibility_rows": [rows for _, _, rows in entries],
            "target_field_ids": [str(f) for _, f, _ in entries],
            "field_name": [field_names[f] for _, f, _ in entries],
        }
    )
    return derived_parameters


def imaging_parameter_to_command_arg(
    imaging_parameters: pd.Series,
    fov_fraction: float,
    beam_sampling: float,
    auto_threshold: float | None = None,
    auto_mask: float | None = None,
    cube: CubeOptions | None = None,
    min_beams_across: float = 0.0,
    multiscale: bool = False,
    join_channels: bool = False,
) -> list[str]:
    """WSClean geometry and deconvolution flags for one spectral window.

    ``auto_threshold`` / ``auto_mask`` (in units of the residual noise) give
    CLEAN a stopping point; without them ``-niter`` is the only limit and every
    task cleans into the noise until the iteration cap.

    With ``cube`` the window is imaged as a spectral cube of ``cube_channels``
    channels (``-channels-out``, each channel cleaned on its own); without it,
    as one continuum plane. ``multiscale`` cleans with WSClean's multi-scale
    algorithm, which models extended emission that delta components leave
    in the residual. ``join_channels`` makes a cube find its components on
    the sum of the channels (WSClean ``-join-channels``) and fit them per
    channel, so emission too faint to clean in one channel is still cleaned.
    """
    spw = imaging_parameters["spectral_window_id"]
    synthetized_beam_size = imaging_parameters["synthetized_beam_size"] / beam_sampling
    n_pixels = image_pixels(imaging_parameters, fov_fraction, beam_sampling, min_beams_across)
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
    if multiscale:
        cmd_args.append("-multiscale")
    if cube is not None:
        channels = cube_channels(imaging_parameters, n_pixels, cube)
        if channels > 1:
            cmd_args += ["-channels-out", str(channels)]
            if join_channels:
                cmd_args.append("-join-channels")
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
            "One row per spectral window and per field observed with the selected intent "
            "(OBSERVE_TARGET by default). --all-spws gives a row per field, calibrators "
            "included. Only (window, field) pairs with visibilities are listed."
        ),
    ),
    intent: Optional[list[str]] = typer.Option(
        None,
        "--intent",
        help=(
            "Scan intent selecting the fields, e.g. CALIBRATE_PHASE#ON_SOURCE, or "
            "CALIBRATE_PHASE for any sub-intent. Repeat for several. Default: "
            "OBSERVE_TARGET. Not with --all-spws."
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
    if intent and not science_only:
        raise typer.BadParameter("--intent selects fields; it cannot be combined with --all-spws")
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
            frames.append(compute_imaging_parameters(ms, science_only=science_only, intents=intent))
        except Exception as exc:
            if fail_fast:
                raise
            failures.append((ms, f"{type(exc).__name__}: {exc}"))
            typer.echo(f"FAILED {ms.name}: {type(exc).__name__}: {exc}", err=True)

    if frames:
        main_output = pd.concat(frames, axis=0, ignore_index=True)
        main_output.to_csv(output_metadata_file, index=False)
        typer.echo(
            f"Wrote {len(main_output)} (spectral window, field) row(s) for {len(frames)} MS(s) "
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
    """One imaging task: one MeasurementSet, one spectral window (and field).

    ``command`` makes the continuum image (empty when only a cube is made);
    ``cube_command``, when set, the spectral cube.
    """

    label: str
    ms_path: Path
    spw: int
    output_dir: Path
    command: list[str]
    field_ids: tuple[int, ...] = ()
    cube_command: Optional[list[str]] = None
    dish_diameter: Optional[float] = None


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


def row_dish_diameter(row: pd.Series) -> Optional[float]:
    """The dish diameter (m) behind a parameter row's ``fov_per_frequency``.

    ``compute_imaging_parameters`` sets it to ALMA_FOV_FACTOR lambda / D of the
    smallest dish, so D comes back exactly (12 or 7 m for ALMA). None when
    the row lacks the field of view or the frequency.
    """
    from .services.imaging.primary_beam import dish_diameter_from_fov

    try:
        diameter = dish_diameter_from_fov(
            float(row["fov_per_frequency"]), float(row["reference_frequency"]), ALMA_FOV_FACTOR
        )
    except (KeyError, TypeError, ValueError):
        return None
    return round(diameter, 2) if math.isfinite(diameter) else None


def task_directory(output_directory: Path, row: pd.Series) -> tuple[Path, tuple[int, ...]]:
    """A parameter row's task directory under ``output_directory`` and its field ids.

    One field per row gets its own directory and markers, so the fields of
    one window never share an image; rows without a field, or with a list
    (CSVs from before per-field rows), keep SPW-<n>.
    """
    input_filename = Path(row["filename"]).expanduser().absolute()
    spw = int(row["spectral_window_id"])
    field_args = _field_arguments(row)
    field_ids = tuple(int(f) for f in field_args[1].split(",")) if field_args else ()
    single_field = field_ids[0] if len(field_ids) == 1 else None
    task_name = f"SPW-{spw}" if single_field is None else f"SPW-{spw}_FIELD-{single_field}"
    return Path(output_directory) / input_filename.stem / task_name, field_ids


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
    continuum: bool = True,
    cube: CubeOptions | None = None,
    cube_products: Sequence[str] | None = None,
    cube_reorder: bool = True,
    min_beams_across: float = 0.0,
    multiscale: bool = False,
    join_channels: bool = False,
) -> tuple[list[ImagingTask], int]:
    """Turn parameter rows into WSClean tasks, skipping the ones already done.

    A task makes the continuum image (``continuum``, ``wsclean-*.fits``)
    and/or, with ``cube``, a spectral cube of ``cube_channels`` channels
    (``wsclean-cube-*.fits``); both runs share the task's single-window MS.
    The cube gets a dirty image only when ``cube_products`` asks for it.

    The cube run lets WSClean reorder (``cube_reorder``) even on a
    single-window MS: with ``-no-reorder`` it reads the whole MS about four
    times per output channel, and a 480-channel Band 1 cube
    (uid___A002_X1142854_X18e99 SPW 25) took about 2 h against 4.5 min
    reordered, with the same images (2026-10-13). The 3.7 reordering crash
    needs several windows in the MS; the continuum run keeps ``-no-reorder``.

    Returns ``(tasks, skipped)``. A task is done when its ``SPW-<n>.done``
    marker and every image it makes exist, so a cube run over a folder of
    continuum images (``continuum=False``) redoes only the missing cubes;
    ``trust_existing_images`` also accepts a continuum image produced before
    markers existed and writes its marker. ``overwrite_outputs`` redoes
    everything.
    """
    from .services.imaging.archive_imaging import (
        CUBE_IMAGE_FILENAME,
        CUBE_NAME,
        IMAGE_FILENAME,
        SINGLE_WINDOW_PLACEHOLDER,
        is_imaging_complete,
        write_imaging_marker,
    )

    if not continuum and cube is None:
        raise ValueError("Nothing to image: neither the continuum nor a cube was requested")
    required = (IMAGE_FILENAME,) * continuum + (CUBE_IMAGE_FILENAME,) * (cube is not None)
    tasks: list[ImagingTask] = []
    skipped = 0
    # WSClean's -mem is a PERCENTAGE of system memory. Passing the fraction
    # (0.1 for 10 cores of 96) limited every task to 0.1 % ≈ 0.4 GB and made
    # WSClean segfault on the first gridding pass.
    mem_percent = min(100.0, 100.0 * num_cores / max_cores_per_node)
    # WSClean runs with its task directory as cwd, so a relative path in the
    # command resolves against that directory, not the caller's: a run started
    # with output "ms-imaged-flagged" looked for its single-window MS under
    # .../SPW-5/ms-imaged-flagged/... and every task failed (2026-10-09).
    output_directory = Path(output_directory).expanduser().absolute()
    for _, row in parameters.iterrows():
        input_filename = Path(row["filename"]).expanduser().absolute()
        spw = int(row["spectral_window_id"])
        outdir, field_ids = task_directory(output_directory, row)
        single_field = field_ids[0] if len(field_ids) == 1 else None
        if not overwrite_outputs:
            if is_imaging_complete(outdir, required):
                skipped += 1
                continue
            if (
                trust_existing_images
                and required == (IMAGE_FILENAME,)
                and (outdir / IMAGE_FILENAME).is_file()
            ):
                write_imaging_marker(outdir, ms_path=str(input_filename), spw=spw)
                skipped += 1
                continue

        def wsclean_command(name: str, with_cube: bool) -> list[str]:
            command_args = imaging_parameter_to_command_arg(
                row,
                fov_fraction,
                beam_sampling,
                auto_threshold=auto_threshold,
                auto_mask=auto_mask,
                cube=cube if with_cube else None,
                min_beams_across=min_beams_across,
                multiscale=multiscale,
                join_channels=join_channels,
            )
            if with_cube and cube_products is not None and "dirty" not in cube_products:
                command_args.append("-no-dirty")
            if single_window:
                # The wrapper hands WSClean a per-task MS holding only this
                # window and the target fields, so no -spws/-field selection
                # and no reordering (WSClean 3.7's reordering reads out of
                # bounds on multi-window ALMA sets; see extract_single_window_ms).
                spws_at = command_args.index("-spws")
                command_args = command_args[:spws_at] + command_args[spws_at + 2 :]
            return [
                wsclean_bin,
                "-name",
                str(outdir / name),
                "-j",
                str(num_cores),
                *command_args,
                # An absolute limit when the caller knows the worker's
                # allocation (WSClean's -mem is a share of the *node*, which
                # can be far more than the Slurm job may use); otherwise the
                # node share.
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
                # field 0 by default and the extracted MS holds the target
                # fields under their original ids.
                *_field_arguments(row),
                *(["-no-reorder"] if single_window and not (with_cube and cube_reorder) else []),
                SINGLE_WINDOW_PLACEHOLDER if single_window else str(input_filename),
            ]

        tasks.append(
            ImagingTask(
                label=f"{input_filename.stem}_{spw}"
                + ("" if single_field is None else f"_f{single_field}"),
                ms_path=input_filename,
                spw=spw,
                output_dir=outdir,
                command=wsclean_command("wsclean", False) if continuum else [],
                field_ids=field_ids,
                cube_command=wsclean_command(CUBE_NAME, True) if cube is not None else None,
                dish_diameter=row_dish_diameter(row),
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
    cube_products: Sequence[str] | None = None,
    contsub_order: int | None = None,
    uv_contsub_order: int | None = None,
    selfcal_solints: Sequence[str] | None = None,
    selfcal_min_snr: float = 20.0,
    casa_data: str | None = None,
    continuum_restart: bool = False,
    cube_line_mask: bool = False,
    noise_products: bool = False,
    pbcor: bool = False,
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
                        cube_command=task.cube_command,
                        cube_products=cube_products,
                        contsub_order=contsub_order,
                        uv_contsub_order=uv_contsub_order,
                        selfcal_solints=selfcal_solints,
                        selfcal_min_snr=selfcal_min_snr,
                        casa_data=casa_data,
                        continuum_restart=continuum_restart,
                        cube_line_mask=cube_line_mask,
                        noise_products=noise_products,
                        pb_dish_diameter=task.dish_diameter if pbcor else None,
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
        # Dask assigns a resource-restricted task to a worker the moment it is
        # submitted and does not move it later (tasks last hours, so work
        # stealing never sees a reason to). Submitting everything as soon as
        # the first worker registered put all 46 tasks of job 3879 on one
        # worker while four sat idle. So: wait for the workers, then keep only
        # as many tasks in flight as they have slots, topping up as each ends;
        # the scheduler then places every new task on a worker with room.
        wait_for_workers = getattr(backend, "wait_for_workers", None)
        if callable(wait_for_workers):
            typer.echo(f"Waiting up to {_WORKER_WAIT_S / 60:.0f} min for {n_jobs} worker(s)...")
            try:
                ready = wait_for_workers(n_jobs, timeout=_WORKER_WAIT_S)
            except RuntimeError as exc:
                typer.echo(str(exc), err=True)
                raise typer.Exit(code=1)
            typer.echo(f"{ready} worker(s) registered; starting.")
        slots = max(int(n_jobs), 1) * max(int(node_cores) // max(int(cores_per_task), 1), 1)

        def _submit(task: ImagingTask) -> Any:
            return backend.submit_callable(
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
                cube_command=task.cube_command,
                cube_products=cube_products,
                contsub_order=contsub_order,
                uv_contsub_order=uv_contsub_order,
                selfcal_solints=selfcal_solints,
                selfcal_min_snr=selfcal_min_snr,
                casa_data=casa_data,
                continuum_restart=continuum_restart,
                cube_line_mask=cube_line_mask,
                noise_products=noise_products,
                pb_dish_diameter=task.dish_diameter if pbcor else None,
            )

        queued = list(range(len(tasks)))
        pending: dict[int, Any] = {}
        while queued and len(pending) < slots:
            index = queued.pop(0)
            pending[index] = _submit(tasks[index])
        total = len(tasks)
        last_heartbeat = started_at
        with tqdm(total=total, desc=stage_label, unit="task", leave=True) as bar:
            while pending:
                progressed = False
                for index, future in list(pending.items()):
                    if not future.done():
                        continue
                    del pending[index]
                    progressed = True
                    bar.update(1)
                    if queued:
                        next_index = queued.pop(0)
                        pending[next_index] = _submit(tasks[next_index])
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
                done = total - len(pending) - len(queued)
                bar.set_postfix_str(f"completed {done}/{total} failed {len(failures)}")
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
                        f"{stage_label}: {running} running, "
                        f"{len(pending) - running + len(queued)} waiting, "
                        f"{total - len(pending) - len(queued)} done, {len(failures)} failed"
                    )
                if not progressed:
                    sleep(0.5)
    return results


CUBE_PRODUCTS = ("image", "residual", "psf", "model", "dirty")


def cube_plan(
    parameters: pd.DataFrame,
    fov_fraction: float,
    beam_sampling: float,
    options: CubeOptions,
    min_beams_across: float = 0.0,
) -> pd.DataFrame:
    """Per row: image side, native and cube channels, and one cube's size in GB.

    ``size_limited`` marks the rows that ``max_cube_gb`` gave fewer channels
    than the spectral sampling asked for.
    """
    unlimited = replace(options, max_cube_gb=math.inf)
    pixels = parameters.apply(
        lambda row: image_pixels(row, fov_fraction, beam_sampling, min_beams_across), axis=1
    )
    channels, wanted = [], []
    for (_, row), n in zip(parameters.iterrows(), pixels):
        channels.append(cube_channels(row, int(n), options))
        wanted.append(cube_channels(row, int(n), unlimited))
    plan = pd.DataFrame(
        {
            "n_pixels": pixels.astype(int).to_numpy(),
            "n_channels": parameters["n_channels"].astype(int).to_numpy(),
            "cube_channels": channels,
            "size_limited": np.array(channels) < np.array(wanted),
        },
        index=parameters.index,
    )
    plan["cube_gb"] = 4 * plan["n_pixels"] ** 2 * plan["cube_channels"] / 1e9
    return plan


def _echo_cube_plan(
    parameters: pd.DataFrame,
    fov_fraction: float,
    beam_sampling: float,
    options: CubeOptions,
    products: Sequence[str],
    min_beams_across: float = 0.0,
) -> None:
    plan = cube_plan(parameters, fov_fraction, beam_sampling, options, min_beams_across)
    if plan.empty:
        return
    averaged = int((plan["cube_channels"] < plan["n_channels"]).sum())
    capped = plan["size_limited"]
    typer.echo(
        f"Cube plan for {len(plan)} task(s): channels per cube "
        f"min {plan['cube_channels'].min()}, median {int(plan['cube_channels'].median())}, "
        f"max {plan['cube_channels'].max()}; {averaged} window(s) averaged below their "
        f"native channels; {int(capped.sum())} given fewer channels by the "
        f"{options.max_cube_gb:g} GB cap."
    )
    typer.echo(
        f"Projected volume: {plan['cube_gb'].sum() / 1e3:.1f} TB per product, "
        f"{len(products) * plan['cube_gb'].sum() / 1e3:.1f} TB for {', '.join(products)}."
    )


def _report_imaging_failures(failures: list[ImagingFailure]) -> None:
    if not failures:
        return
    typer.echo(f"Image: {len(failures)} task(s) failed and were skipped:", err=True)
    for failure in failures:
        typer.echo(f"  {failure.label}: {_error_headline(failure.error)}", err=True)
        if failure.log_path:
            typer.echo(f"    log: {failure.log_path}", err=True)


def _prepare_selfcal_casa_data(output_directory: Path, casa_data_root: Path | None) -> Path:
    """CASA runtime data for the workers' gaincal, populated here once if missing.

    The calibration stage's preflight: the tasks would otherwise each try to
    download it, or fail to import casatools without it.
    """
    from .services.archive.unpack_ms import (
        configure_casa_environment,
        ensure_casa_runtime_data,
    )

    output_directory = Path(output_directory).expanduser().absolute()
    casa_data = (
        Path(casa_data_root).expanduser().absolute()
        if casa_data_root is not None
        else output_directory / ".casa-data"
    )
    workspace = output_directory / ".casa-work"
    configure_casa_environment(workspace, casa_data, workspace_root=workspace)
    typer.echo(f"Self-calibration CASA runtime data: {casa_data}")
    ensure_casa_runtime_data(casa_data, logger_fn=typer.echo)
    return casa_data


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
    # Compact-configuration images were ~12 beams wide: the source's sidelobes
    # filled them, inflated the noise WSClean cleaned to and left more source
    # in the residual. At 32 beams the line-free residual share of four
    # cycle11 cubes fell from 7-25 % to 6-21 % and their S/N rose 1.4-4x, for
    # ~13 % more run time; multi-scale changed nothing on these compact
    # sources and diverged on a 6.6 Jy one in a 94 px image (2026-10-17).
    min_beams_across: float = typer.Option(
        32.0,
        "--min-beams-across",
        min=0.0,
        help=(
            "Make every image at least this many synthesized beams wide, whatever "
            "--fov-fraction gives (default 32; 0: no minimum)."
        ),
    ),
    multiscale: bool = typer.Option(
        False,
        "--multiscale/--no-multiscale",
        help=(
            "Clean with WSClean's multi-scale algorithm (-multiscale) for extended emission. "
            "Off by default: it can diverge on bright sources in small images."
        ),
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
        1.0,
        "--auto-threshold",
        min=0.0,
        help=(
            "Stop cleaning when the residual peak drops below this many sigma of the "
            "residual noise (WSClean -auto-threshold; default 1, inside the --auto-mask "
            "mask). 0 disables it and cleaning runs to -niter."
        ),
    ),
    auto_mask: float = typer.Option(
        3.0,
        "--auto-mask",
        min=0.0,
        help=(
            "WSClean -auto-mask level in sigma (default 3): clean to this level everywhere, "
            "then only inside the mask it found down to --auto-threshold. Must be above "
            "--auto-threshold; 0 disables masking."
        ),
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
    continuum_restart: bool = typer.Option(
        True,
        "--continuum-restart/--no-continuum-restart",
        help=(
            "Clean the continuum in two passes, the second (-continue) from the true residual "
            "with a mask rebuilt at the true noise: the first pass's auto-mask, built while "
            "a bright source's sidelobes inflate the noise, misses its fainter parts "
            "(default; only with --single-window-ms, whose per-task copy gets MODEL_DATA)."
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
    continuum: bool = typer.Option(
        True,
        "--continuum/--no-continuum",
        help=(
            "Make the continuum image of every window (wsclean-image.fits; default). "
            "--no-continuum with --cube adds cubes next to existing continuum images."
        ),
    ),
    cube: bool = typer.Option(
        False,
        "--cube/--no-cube",
        help=(
            "Also image every window as a spectral cube (WSClean -channels-out, the channel "
            "count chosen per window by --spectral-sampling, --velocity-resolution and "
            "--max-cube-gb), stacked into wsclean-cube-<product>.fits."
        ),
    ),
    contsub: bool = typer.Option(
        False,
        "--contsub/--no-contsub",
        help=(
            "Also write the image cube continuum-subtracted (wsclean-cube-image.contsub.fits): "
            "a polynomial fitted along frequency per pixel, line channels found by 3-sigma "
            "clipping. Off by default. Needs --cube."
        ),
    ),
    selfcal: str = typer.Option(
        "",
        "--selfcal",
        help=(
            "Self-calibrate each window with CASA gaincal before imaging, one round per "
            "comma-separated solution interval: inf (per scan), int (per integration) or "
            "<seconds>s for phase; ap:<interval> for amplitude and phase on top of the "
            "phase solution, e.g. 'inf,30s,ap:inf'. A round is kept only if the continuum "
            "S/N rises. Empty (default): off. Needs --continuum and --single-window-ms."
        ),
    ),
    selfcal_min_snr: float = typer.Option(
        20.0,
        "--selfcal-min-snr",
        min=0.0,
        help="Skip self-calibration when the initial continuum S/N is below this.",
    ),
    noise_products: bool = typer.Option(
        True,
        "--noise-products/--no-noise-products",
        help=(
            "Where self-calibration was applied, also image what it removed and keep it: "
            "wsclean[-cube]-selfcal-correction.fits, wsclean[-cube]-residual-total.fits "
            "(residual + correction: all the data the model does not explain) and the gain "
            "tables in selfcal_gains/ (default)."
        ),
    ),
    pbcor: bool = typer.Option(
        False,
        "--pbcor/--no-pbcor",
        help=(
            "Also primary-beam correct the continuum image and the image cube: "
            "wsclean[-cube]-pb.fits (the beam, CASA's ALMA Airy model, per channel) and "
            "wsclean[-cube]-image.pbcor.fits (blanked below 0.2). Off by default; "
            "'image pb-correct' adds them to an existing run."
        ),
    ),
    casa_data_root: Optional[Path] = typer.Option(
        None,
        "--casa-data-root",
        help=(
            "CASA runtime data for --selfcal (default <output>/.casa-data, downloaded once "
            "by the driver when missing, as the calibration stage does)."
        ),
    ),
    uv_contsub: bool = typer.Option(
        False,
        "--uv-contsub/--no-uv-contsub",
        help=(
            "Subtract the continuum from the visibilities before imaging (CASA uvcontsub): "
            "line channels are found on the source spectrum, the continuum is fitted per "
            "visibility over the others. The continuum image is made from the fitted "
            "continuum and every cube product holds only the lines. Needs --cube and "
            "--single-window-ms; replaces --contsub."
        ),
    ),
    contsub_order: int = typer.Option(
        1,
        "--contsub-order",
        min=0,
        max=1,
        help=(
            "Continuum polynomial order for --contsub and --uv-contsub: 0 (constant) or 1 "
            "(linear, default)."
        ),
    ),
    spectral_sampling: float = typer.Option(
        2.0,
        "--spectral-sampling",
        min=1e-6,
        help=(
            "Cube channels per spectral resolution element (the spectral counterpart of "
            "--beam-sampling). 2 keeps the native channels of ALMA's Hanning-smoothed "
            "windows; 1 halves them."
        ),
    ),
    velocity_resolution: float = typer.Option(
        0.0,
        "--velocity-resolution",
        min=0.0,
        help="Narrowest cube channel in km/s at the window's reference frequency (0: no limit).",
    ),
    max_cube_gb: float = typer.Option(
        4.0,
        "--max-cube-gb",
        min=1e-3,
        help=(
            "Largest cube per product in GB (32-bit pixels): windows imaged on large grids "
            "get fewer, wider channels. Channel counts are divisors of the native count, so "
            "every cube channel averages the same number of native channels."
        ),
    ),
    cube_products: str = typer.Option(
        ",".join(CUBE_PRODUCTS),
        "--cube-products",
        help=(
            "Comma-separated cube products to keep: image, residual, psf, model, dirty "
            "(default: all five). The others are removed (dirty is not even made)."
        ),
    ),
    # On uid___A002_X10ded83_X6af4 SPW 16 joining cut the stacked line-free
    # residual from 12 % to 8 % of the image but raised the line channels'
    # from 11 % to 30 % (2026-10-17), so it stays opt-in.
    join_channels: bool = typer.Option(
        False,
        "--join-channels/--no-join-channels",
        help=(
            "Clean the cube on the sum of its channels (WSClean -join-channels): "
            "continuum too faint to clean channel by channel is cleaned too, but narrow "
            "lines are cleaned worse. Off by default. Needs --cube."
        ),
    ),
    cube_line_mask: bool = typer.Option(
        True,
        "--cube-line-mask/--no-cube-line-mask",
        help=(
            "Clean the cube a second time (-continue) inside the moment-0 mask of the first "
            "pass: a line's faint parts, below each channel's own auto-mask but clear in the "
            "sum, are cleaned too (default; with --cube and --single-window-ms; ~2x the cube "
            "time)."
        ),
    ),
    cube_reorder: bool = typer.Option(
        True,
        "--cube-reorder/--cube-no-reorder",
        help=(
            "Let WSClean reorder the single-window MS per channel for the cube (default). "
            "Without it WSClean re-reads the whole MS for every cube channel: ~25x slower."
        ),
    ),
    plan_only: bool = typer.Option(
        False,
        "--plan-only",
        help="Print the channel plan and the projected cube volume, then exit without imaging.",
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

    # A 3-sigma stop without a mask left ~3 sigma of source in every channel's
    # residual: stacked over the 3537 line-free channels of uid___A002_X10ded83_X6af4
    # SPW 16 that was 7.9 sigma and 39 % of the image; masking to 3 and cleaning
    # to 1 sigma left 4.1 sigma and 16 % (2026-10-17).
    if 0 < auto_mask <= auto_threshold:
        raise typer.BadParameter("--auto-mask must be above --auto-threshold (or 0 to disable it)")
    parameters = pd.read_csv(str(imaging_parameters))
    cube_options: CubeOptions | None = None
    products: tuple[str, ...] | None = None
    if cube:
        products = tuple(p.strip().lower() for p in cube_products.split(",") if p.strip())
        unknown = sorted(set(products) - set(CUBE_PRODUCTS))
        if "image" not in products or unknown:
            raise typer.BadParameter(
                f"--cube-products must include image and only name {', '.join(CUBE_PRODUCTS)}"
                + (f" (unknown: {', '.join(unknown)})" if unknown else "")
            )
        cube_options = CubeOptions(
            spectral_sampling=spectral_sampling,
            velocity_resolution_km_s=velocity_resolution,
            max_cube_gb=max_cube_gb,
        )
        parameters = add_spectral_columns(parameters)
        _echo_cube_plan(
            parameters, fov_fraction, beam_sampling, cube_options, products, min_beams_across
        )
    elif plan_only or contsub or join_channels or uv_contsub:
        raise typer.BadParameter(
            "Options that need --cube: --plan-only, --contsub, --uv-contsub, --join-channels"
        )
    if uv_contsub and contsub:
        raise typer.BadParameter("--uv-contsub already removes the continuum: drop --contsub")
    from .services.imaging.archive_imaging import parse_selfcal_rounds

    solints = [part.strip() for part in selfcal.split(",") if part.strip()]
    try:
        parse_selfcal_rounds(solints)
    except ValueError as exc:
        raise typer.BadParameter(f"--selfcal: {exc}") from exc
    if solints and not (continuum and single_window):
        raise typer.BadParameter("--selfcal needs --continuum and --single-window-ms")
    if uv_contsub and not single_window:
        raise typer.BadParameter("--uv-contsub needs --single-window-ms: it edits the copy")
    if not continuum and not cube:
        raise typer.BadParameter("--no-continuum needs --cube: there would be nothing to image")
    if plan_only:
        return
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
        continuum=continuum,
        cube=cube_options,
        cube_products=products,
        cube_reorder=cube_reorder,
        min_beams_across=min_beams_across,
        multiscale=multiscale,
        join_channels=join_channels,
    )
    if skipped:
        typer.echo(f"Skipped {skipped} already-imaged SPW(s).")
    if not tasks:
        typer.echo("All requested tasks are already imaged; nothing to do.")
        return
    casa_data: str | None = None
    if solints:
        casa_data = str(_prepare_selfcal_casa_data(output_directory, casa_data_root))
    typer.echo(
        f"Imaging {len(tasks)} (MS, SPW, field) task(s); "
        "per-task logs: <output>/<ms>/SPW-<n>[_FIELD-<id>].log"
    )

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
        cube_products=products,
        contsub_order=contsub_order if contsub else None,
        uv_contsub_order=contsub_order if uv_contsub else None,
        selfcal_solints=solints or None,
        selfcal_min_snr=selfcal_min_snr,
        casa_data=casa_data,
        continuum_restart=continuum_restart and single_window,
        cube_line_mask=cube_line_mask and cube and single_window,
        noise_products=noise_products,
        pbcor=pbcor,
    )
    typer.echo(f"Imaged {len(tasks) - len(failures)}/{len(tasks)} task(s), {len(failures)} failed.")
    if failures:
        _report_imaging_failures(failures)
        raise typer.Exit(code=1)


@image_app.command("clean-quality")
def clean_quality(
    output_root: Path = typer.Argument(
        help="An image-from-ms output directory (or one task directory) to assess."
    ),
    moments: bool = typer.Option(
        True,
        "--moments/--no-moments",
        help="Write the cube moment maps (quality/cube-*.fits) next to each task's images.",
    ),
    summary: Optional[Path] = typer.Option(
        None,
        "--summary",
        help="Summary CSV path (default <output_root>/clean-quality.csv).",
    ),
):
    """Measure how much source signal the clean left in the residuals, task by task.

    For every task under OUTPUT_ROOT writes quality.json (continuum and cube
    statistics: S/N, the residual's share and profile at the source, the
    noise WSClean measured when it built its mask, line and line-free
    residuals, model fluxes) and, for cubes, moment maps of the line and of
    the residual; prints one row per task and writes them to a CSV. See
    almasim.services.imaging.quality for each statistic.
    """
    from .services.imaging.quality import (
        SUMMARY_FILENAME,
        assess_task,
        find_task_dirs,
        summary_rows,
        write_summary,
    )

    root = Path(output_root).expanduser().absolute()
    tasks = find_task_dirs(root)
    if not tasks:
        typer.echo(f"No imaged tasks (wsclean-image.fits or wsclean-cube-image.fits) under {root}")
        raise typer.Exit(code=1)
    results = []
    for task in tqdm(tasks, desc="clean quality", unit="task"):
        try:
            results.append(assess_task(task, write_moments=moments))
        except Exception as exc:  # one broken task must not hide the others
            typer.echo(f"  {task}: {type(exc).__name__}: {exc}", err=True)
    if not results:
        typer.echo(f"None of the {len(tasks)} task(s) under {root} could be assessed", err=True)
        raise typer.Exit(code=1)
    rows = summary_rows(results, root)
    summary_path = summary or root / SUMMARY_FILENAME
    write_summary(rows, summary_path)
    frame = pd.DataFrame(rows).set_index("task")
    frame.columns = [column.replace("continuum.", "cont.") for column in frame.columns]
    with pd.option_context("display.width", 250, "display.max_columns", None):
        typer.echo(frame.to_string(float_format=lambda value: f"{value:.3g}"))
    typer.echo(f"Wrote {len(results)} quality.json file(s) and {summary_path}")


@image_app.command("pb-correct")
def pb_correct(
    imaging_parameters: Path = typer.Argument(help="Parameter CSV the outputs were imaged from"),
    output_directory: Path = typer.Argument(help="The image-from-ms output directory"),
    cutoff: float = typer.Option(
        0.2,
        "--cutoff",
        min=1e-3,
        max=0.999,
        help="Blank the corrected images where the beam's response is below this.",
    ),
):
    """Primary-beam correct the images of an existing image-from-ms run.

    For every row of IMAGING_PARAMETERS whose task finished (a .done marker)
    writes wsclean[-cube]-pb.fits and wsclean[-cube]-image.pbcor.fits next to
    its continuum image and image cube, as image-from-ms --pbcor does, without
    imaging again. The dish diameter comes from the row's fov_per_frequency.
    """
    from .services.imaging.archive_imaging import (
        imaging_log_path,
        imaging_marker_path,
        pb_correct_task,
        pb_note,
    )

    parameters = pd.read_csv(str(imaging_parameters))
    root = Path(output_directory).expanduser().absolute()
    corrected = failed = skipped = 0
    for _, row in tqdm(list(parameters.iterrows()), desc="pb-correct", unit="task"):
        outdir, _ = task_directory(root, row)
        if not imaging_marker_path(outdir).is_file():
            skipped += 1
            continue
        dish_diameter = row_dish_diameter(row)
        try:
            if dish_diameter is None:
                raise ValueError("the row has no fov_per_frequency/reference_frequency")
            done = pb_correct_task(outdir, dish_diameter, cutoff)
        except Exception as exc:  # one broken task must not stop the others
            failed += 1
            typer.echo(f"  {outdir}: {type(exc).__name__}: {exc}", err=True)
            continue
        with imaging_log_path(outdir).open("a", encoding="utf-8") as log:
            for name, info in done.items():
                log.write(f"# primary-beam corrected {name} (pb-correct): {pb_note(info)}\n")
        corrected += 1
    typer.echo(
        f"Primary-beam corrected {corrected} task(s); {skipped} not imaged, {failed} failed."
    )
    if failed or not corrected:
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
    auto_threshold: float = typer.Option(1.0, "--auto-threshold", min=0.0),
    auto_mask: float = typer.Option(3.0, "--auto-mask", min=0.0),
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
        min_beams_across=32.0,
        multiscale=False,
        num_cores=num_cores,
        max_cores_per_node=max_cores_per_node,
        wsclean_bin=wsclean_bin,
        auto_threshold=auto_threshold,
        auto_mask=auto_mask,
        update_model=update_model,
        continuum_restart=True,
        task_memory_gb=task_memory_gb,
        task_retries=task_retries,
        single_window=single_window,
        max_weight=max_weight,
        scratch_dir=scratch_dir,
        continuum=True,
        cube=False,
        contsub=False,
        uv_contsub=False,
        selfcal="",
        selfcal_min_snr=20.0,
        noise_products=True,
        pbcor=False,
        casa_data_root=None,
        contsub_order=1,
        spectral_sampling=2.0,
        velocity_resolution=0.0,
        max_cube_gb=4.0,
        cube_products=",".join(CUBE_PRODUCTS),
        join_channels=False,
        cube_line_mask=True,
        cube_reorder=True,
        plan_only=False,
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
