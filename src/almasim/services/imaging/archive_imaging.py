"""Per-task bookkeeping for imaging archive MeasurementSets with WSClean.

One imaging task is one (MeasurementSet, spectral window) pair and produces
``<output_root>/<ms stem>/SPW-<n>/wsclean-*.fits``. Exactly as for the unpack
and calibrate stages, a task leaves a marker next to its output directory:

* ``SPW-<n>.done``   -- the task finished and ``wsclean-image.fits`` exists;
* ``SPW-<n>.failed`` -- the task failed, with the error and the log path;
* ``SPW-<n>.log``    -- everything WSClean printed, written live on the shared
  filesystem so a run on Slurm can be followed with ``tail -f``.

Directory existence alone means nothing: a killed WSClean leaves a partial
output tree that looks finished. Reruns skip tasks with a ``.done`` marker and
retry everything else.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from almasim.services.imaging.continuum import (  # noqa: F401  (moved; re-exported)
    _channel_ranges,
    _fit_continuum,
    _line_free_from_ranges,
    fit_uv_continuum,
    line_free_channels,
    subtract_continuum,
)
from almasim.services.imaging.cube import (  # noqa: F401  (moved; re-exported)
    _CHANNEL_FILE,
    _line_mask_pass,
    _zero_non_finite_models,
    line_mask_from_channels,
    stack_channel_images,
)
from almasim.services.imaging.ms.chunks import (  # noqa: F401  (moved; re-exported)
    _CHUNK_SAMPLES,
    _default_rows_per_chunk,
    _has_spectrum_weight,
    _parallel_hands,
    _read_rows,
    _source_spectrum,
    rows_for_chunk,
)
from almasim.services.imaging.ms.single_window import (  # noqa: F401  (moved; re-exported)
    _LOCK_FILE_ATTEMPTS,
    _LOCK_FILE_ERROR,
    _LOCK_FILE_RETRY_SECONDS,
    DEFAULT_MAX_WEIGHT,
    _extract_single_window_ms_once,
    extract_single_window_ms,
    extract_single_window_ms_subprocess,
)
from almasim.services.imaging.products import (  # noqa: F401  (moved; re-exported)
    CONTINUUM_COLUMN,
    CONTSUB_FILENAME,
    CUBE_IMAGE_FILENAME,
    CUBE_NAME,
    CUBE_PB_FILENAME,
    CUBE_PBCOR_FILENAME,
    CUBE_SELFCAL_CORRECTION_FILENAME,
    CUBE_TOTAL_RESIDUAL_FILENAME,
    IMAGE_FILENAME,
    LINE_COLUMN,
    LINE_MASK_FILENAME,
    PB_FILENAME,
    PBCOR_FILENAME,
    SELFCAL_COLUMN,
    SELFCAL_CONTINUUM_COLUMN,
    SELFCAL_CORRECTION_FILENAME,
    SELFCAL_GAINS_DIRNAME,
    SELFCAL_LINE_COLUMN,
    SINGLE_WINDOW_PLACEHOLDER,
    TOTAL_RESIDUAL_FILENAME,
)
from almasim.services.imaging.selfcal.casa_ops import (  # noqa: F401  (moved; re-exported)
    _configure_casa,
    adopt_corrected_data,
    casa_selfcal_apply,
    casa_selfcal_solve,
    export_gain_table,
    selfcal_prepare,
)
from almasim.services.imaging.selfcal.noise_products import (  # noqa: F401  (moved; re-exported)
    _single_image,
    add_images,
    selfcal_correction_columns,
)
from almasim.services.imaging.uv_contsub import (  # noqa: F401  (moved; re-exported)
    drop_corrected_data,
    subtract_uv_continuum,
    subtract_uv_continuum_subprocess,
)
from almasim.services.imaging.wsclean.command import (  # noqa: F401  (moved; re-exported)
    _dirty_command,
    _selfcal_command,
    _with_data_column,
    continuum_restart_commands,
)
from almasim.services.imaging.wsclean.runner import (  # noqa: F401  (moved; re-exported)
    _error_line,
    _remove_reorder_files,
    _run_once,
    _run_with_retries,
)
from almasim.services.runtime.child import call_in_child
from almasim.services.runtime.env import (  # noqa: F401  (moved; re-exported)
    _SYSTEM_LIBRARY_DIRS,
    casa_environment,
    wsclean_environment,
)

from ..markers import StageMarkers

__all__ = [
    "CONTINUUM_COLUMN",
    "CUBE_SELFCAL_CORRECTION_FILENAME",
    "CUBE_TOTAL_RESIDUAL_FILENAME",
    "SELFCAL_CORRECTION_FILENAME",
    "SELFCAL_GAINS_DIRNAME",
    "TOTAL_RESIDUAL_FILENAME",
    "add_images",
    "export_gain_table",
    "selfcal_correction_columns",
    "CONTSUB_FILENAME",
    "CUBE_IMAGE_FILENAME",
    "CUBE_NAME",
    "DEFAULT_MAX_WEIGHT",
    "IMAGE_FILENAME",
    "LINE_COLUMN",
    "LINE_MASK_FILENAME",
    "SINGLE_WINDOW_PLACEHOLDER",
    "extract_single_window_ms",
    "extract_single_window_ms_subprocess",
    "imaging_failure_marker_path",
    "imaging_log_path",
    "imaging_marker_path",
    "line_mask_from_channels",
    "is_imaging_complete",
    "run_wsclean_task",
    "stack_channel_images",
    "fit_uv_continuum",
    "line_free_channels",
    "continuum_restart_commands",
    "subtract_continuum",
    "subtract_uv_continuum",
    "subtract_uv_continuum_subprocess",
    "self_calibrate",
    "casa_selfcal_apply",
    "casa_selfcal_solve",
    "drop_corrected_data",
    "adopt_corrected_data",
    "parse_selfcal_rounds",
    "selfcal_prepare",
    "rows_for_chunk",
    "write_imaging_failure_marker",
    "write_imaging_marker",
    "wsclean_environment",
]


def imaging_marker_path(output_dir: str | os.PathLike[str]) -> Path:
    """``<root>/<ms>/SPW-3`` -> ``<root>/<ms>/SPW-3.done``."""
    out = Path(output_dir)
    return out.with_name(out.name + ".done")


def imaging_failure_marker_path(output_dir: str | os.PathLike[str]) -> Path:
    out = Path(output_dir)
    return out.with_name(out.name + ".failed")


def _imaging_markers(output_dir: str | os.PathLike[str]) -> StageMarkers:
    return StageMarkers(imaging_marker_path(output_dir), imaging_failure_marker_path(output_dir))


def imaging_log_path(output_dir: str | os.PathLike[str]) -> Path:
    out = Path(output_dir)
    return out.with_name(out.name + ".log")


def is_imaging_complete(
    output_dir: str | os.PathLike[str], required: Sequence[str] = (IMAGE_FILENAME,)
) -> bool:
    """True when the task has its marker and every image in ``required``."""
    out = Path(output_dir)
    return imaging_marker_path(out).is_file() and all((out / name).is_file() for name in required)


def write_imaging_marker(
    output_dir: str | os.PathLike[str],
    *,
    ms_path: str,
    spw: int,
    command: Sequence[str] | None = None,
    seconds: float | None = None,
    commands: Sequence[Sequence[str]] | None = None,
    images: Sequence[str] | None = None,
) -> Path:
    out = Path(output_dir)
    payload: dict[str, Any] = {
        "ms": ms_path,
        "spw": int(spw),
        "output": str(out),
        "image": str(out / IMAGE_FILENAME),
    }
    if images:
        payload["image"] = str(out / images[0])
        payload["images"] = [str(out / name) for name in images]
    if command is not None:
        payload["command"] = shlex.join(str(part) for part in command)
    if commands is not None:
        payload["commands"] = [shlex.join(str(part) for part in c) for c in commands]
    if seconds is not None:
        payload["seconds"] = round(float(seconds), 1)
    return _imaging_markers(out).write_done(**payload)


def write_imaging_failure_marker(
    output_dir: str | os.PathLike[str],
    *,
    ms_path: str,
    spw: int,
    error: str,
    log_path: str | os.PathLike[str] | None = None,
    returncode: int | None = None,
) -> Path:
    """Record why a task failed. ``error`` is kept verbatim for triage."""
    out = Path(output_dir)
    markers = _imaging_markers(out)
    # Unlike unpack and calibrate, a failed image task also drops its old .done.
    markers.done.unlink(missing_ok=True)
    fields: dict[str, Any] = {"ms": ms_path, "spw": int(spw), "output": str(out)}
    if returncode is not None:
        fields["returncode"] = int(returncode)
    return markers.write_failed(error, stage="image", log_path=log_path, **fields)


def run_wsclean_task(
    *,
    command: Sequence[str],
    output_dir: str,
    ms_path: str,
    spw: int,
    threads: int,
    timeout: float | None = None,
    scratch_root: str | None = None,
    retries: int = 3,
    single_window: bool = False,
    field_ids: Sequence[int] | None = None,
    max_weight: float | None = DEFAULT_MAX_WEIGHT,
    cube_command: Sequence[str] | None = None,
    cube_products: Sequence[str] | None = None,
    contsub_order: int | None = None,
    uv_contsub_order: int | None = None,
    selfcal_solints: Sequence[str] | None = None,
    selfcal_min_snr: float = 20.0,
    casa_data: str | None = None,
    continuum_restart: bool = False,
    cube_line_mask: bool = False,
    noise_products: bool = False,
    pb_dish_diameter: float | None = None,
) -> dict[str, object]:
    """Run one WSClean task in a subprocess and leave a marker whatever happens.

    Runs on the worker. Output is streamed to ``SPW-<n>.log`` as it arrives.
    A non-zero exit, a signal, a timeout, a binary that cannot start, or a
    clean exit that produced no ``wsclean-image.fits`` all end in a ``.failed``
    marker and a ``RuntimeError`` so the driver counts the task as failed;
    success ends in a ``.done`` marker.

    A death by signal (a segfault, say) is retried up to ``retries`` times
    before the task is declared failed: WSClean 3.7's multi-threaded
    reordering crashes non-deterministically on some inputs and passes on the
    next run, and a crash costs seconds while the task costs minutes. Exits
    with an error status are WSClean's own diagnostics and are not retried.

    ``command`` makes the continuum image (empty: none); ``cube_command``,
    run after it on the same single-window MS, the spectral cube. The cube
    run leaves one FITS per channel; they are stacked into
    ``wsclean-cube-<product>.fits`` (only ``cube_products`` kept, all when
    None) and, with ``contsub_order``, the image cube is also written
    continuum-subtracted (``subtract_continuum``) before the task counts as
    done. Only the images this task makes are removed beforehand, so a cube
    task leaves an existing continuum image alone.

    With ``uv_contsub_order`` (needs ``single_window`` and a cube) the
    continuum is subtracted from the visibilities of the single-window MS
    first (``subtract_uv_continuum``): the continuum image is made from the
    fitted continuum and the cube from what is left, the lines. When too few
    channels are line-free the subtraction is skipped and noted in the log.

    With ``selfcal_solints`` (needs ``single_window`` and a continuum
    command) the single-window MS is phase self-calibrated first, one round
    per solution interval (``self_calibrate``); everything after it, the
    continuum subtraction included, reads the corrected data.

    With ``noise_products``, where self-calibration was applied, what it
    removed is imaged too (``_selfcal_noise_products``): the correction and
    the total residual (kept residual + correction) of the continuum and the
    cube; its gain tables are kept in ``selfcal_gains/``.

    With ``pb_dish_diameter`` (the antennas' diameter in metres) the
    continuum image and the image cube are primary-beam corrected
    (``pb_correct_task``): ``wsclean[-cube]-pb.fits`` and
    ``wsclean[-cube]-image.pbcor.fits``.
    """
    out = Path(output_dir)
    log_path = imaging_log_path(out)
    commands = [[str(part) for part in c] for c in (command, cube_command or []) if c]
    if not commands:
        raise ValueError("run_wsclean_task needs a continuum or a cube command")
    if uv_contsub_order is not None and not (single_window and cube_command):
        raise ValueError("uv_contsub_order needs single_window and a cube_command")
    if selfcal_solints and not (single_window and command and casa_data):
        raise ValueError("selfcal_solints needs single_window, a continuum command and casa_data")
    if continuum_restart and command:
        if not single_window:
            raise ValueError("continuum_restart writes MODEL_DATA: it needs single_window")
        commands[:1] = continuum_restart_commands(commands[0])
    if cube_line_mask and cube_command:
        if not single_window:
            raise ValueError("cube_line_mask writes MODEL_DATA: it needs single_window")
        # The second pass continues from this one's model visibilities.
        commands[-1] = [
            "-update-model-required" if part == "-no-update-model-required" else part
            for part in commands[-1]
        ]
    required = [IMAGE_FILENAME] * bool(command) + [CUBE_IMAGE_FILENAME] * bool(cube_command)
    if cube_command and contsub_order is not None:
        required.append(CONTSUB_FILENAME)
    # Only after the arguments are checked, so a rejected call leaves no directory.
    out.mkdir(parents=True, exist_ok=True)
    imaging_marker_path(out).unlink(missing_ok=True)
    imaging_failure_marker_path(out).unlink(missing_ok=True)
    for name in required:
        (out / name).unlink(missing_ok=True)
    # The noise products of what this task regenerates: a rerun whose
    # self-calibration is skipped or rejected must not leave the previous
    # run's beside the new images. Self-calibration runs with the continuum.
    stale_products = [SELFCAL_CORRECTION_FILENAME, TOTAL_RESIDUAL_FILENAME] * bool(command)
    stale_products += [CUBE_SELFCAL_CORRECTION_FILENAME, CUBE_TOTAL_RESIDUAL_FILENAME] * bool(
        cube_command
    )
    stale_products += [PB_FILENAME, PBCOR_FILENAME] * bool(command)
    stale_products += [CUBE_PB_FILENAME, CUBE_PBCOR_FILENAME] * bool(cube_command)
    for name in stale_products:
        (out / name).unlink(missing_ok=True)
    if command:
        shutil.rmtree(out / SELFCAL_GAINS_DIRNAME, ignore_errors=True)
    log_path.write_text("", encoding="utf-8")
    # A previous attempt that crashed or was killed with its worker leaves
    # WSClean's reorder files (hundreds of GB across a run) and the
    # single-window copy behind; nothing else ever removes them.
    _remove_reorder_files(out)
    for stale in out.glob("spw*.single.ms"):
        shutil.rmtree(stale, ignore_errors=True)
    for stale in out.glob(f"{CUBE_NAME}-[0-9][0-9][0-9][0-9]-*.fits"):
        stale.unlink(missing_ok=True)

    single_window_ms: Path | None = None
    if single_window:
        # Extract the window (and fields) into a per-task MS so WSClean never
        # reorders; ``SINGLE_WINDOW_PLACEHOLDER`` in the command is replaced.
        single_window_ms = out / f"spw{int(spw)}.single.ms"
        try:
            extracted = extract_single_window_ms_subprocess(
                ms_path,
                spw,
                field_ids,
                single_window_ms,
                timeout=timeout,
                log_path=log_path,
                max_weight=max_weight,
            )
        except Exception as exc:
            error = f"Could not extract spectral window {spw} into a single-window MS: {exc}"
            write_imaging_failure_marker(out, ms_path=ms_path, spw=spw, error=error)
            raise RuntimeError(error) from exc
        fields_note = list(field_ids) if field_ids else "all"
        size_gb = float(extracted.get("bytes", 0) or 0) / 1e9
        kept = int(extracted.get("rows") or 0)
        dropped = int(extracted.get("dropped_rows") or 0)
        drop_note = (
            f", dropped {dropped} rows ({100.0 * dropped / max(kept + dropped, 1):.1f} %) "
            f"with WEIGHT > {float(extracted.get('max_weight') or 0):g}"
            if extracted.get("max_weight")
            else ""
        )
        with log_path.open("a", encoding="utf-8") as log:
            log.write(
                f"# extracted spectral window {int(spw)} fields {fields_note} into "
                f"{single_window_ms.name}: {kept} rows, {size_gb:.1f} GB, "
                f"{extracted.get('seconds')} s{drop_note}\n"
            )
        commands = [
            [str(single_window_ms) if part == SINGLE_WINDOW_PLACEHOLDER else part for part in c]
            for c in commands
        ]
    scratch_dir: Path | None = None
    if scratch_root:
        # Per-task scratch for WSClean's reordered visibilities, created here
        # on the worker and removed whatever happens. ``__SCRATCH__`` in the
        # command (the -temp-dir value) is replaced by it.
        import tempfile

        Path(scratch_root).mkdir(parents=True, exist_ok=True)
        scratch_dir = Path(tempfile.mkdtemp(prefix=f"{out.name}-", dir=scratch_root))
        commands = [
            [str(scratch_dir) if part == "__SCRATCH__" else part for part in c] for c in commands
        ]
    env = wsclean_environment(threads)
    tail: deque[str] = deque(maxlen=60)
    started = time.time()

    def _fail(error: str, returncode: int | None = None) -> RuntimeError:
        write_imaging_failure_marker(
            out,
            ms_path=ms_path,
            spw=spw,
            error=error,
            log_path=log_path,
            returncode=returncode,
        )
        return RuntimeError(f"{error}\nFull log: {log_path}")

    try:
        selfcal: dict[str, object] = {}
        fit: dict[str, object] | None = None
        if single_window_ms is not None and (selfcal_solints or uv_contsub_order is not None):
            with log_path.open("a", encoding="utf-8") as log:
                if selfcal_solints:
                    selfcal = self_calibrate(
                        single_window_ms,
                        commands[0],
                        out=out,
                        log=log,
                        solints=selfcal_solints,
                        min_snr=selfcal_min_snr,
                        run=lambda cmd: _run_with_retries(
                            cmd, out, log, env, timeout, tail, _fail, max(int(retries), 0)
                        ),
                        fail=_fail,
                        casa_data=str(casa_data),
                        timeout=timeout,
                        restart=continuum_restart,
                        keep_gains=noise_products,
                    )
                if uv_contsub_order is not None:
                    commands, fit = _subtract_uv_continuum_step(
                        single_window_ms,
                        commands,
                        has_continuum=bool(command),
                        order=uv_contsub_order,
                        log=log,
                        fail=_fail,
                        timeout=timeout,
                    )
        after_images = None
        if noise_products and selfcal.get("applied") and single_window_ms is not None:
            sw_ms, final_commands, final_fit = single_window_ms, list(commands), fit

            def after_images() -> None:
                with log_path.open("a", encoding="utf-8") as log:
                    _selfcal_noise_products(
                        sw_ms,
                        continuum_command=final_commands[0] if command else None,
                        cube_command=final_commands[-1] if cube_command else None,
                        fit=final_fit,
                        out=out,
                        log=log,
                        run=lambda cmd: _run_with_retries(
                            cmd, out, log, env, timeout, tail, _fail, max(int(retries), 0)
                        ),
                        fail=_fail,
                        timeout=timeout,
                    )

        return _run_and_mark(
            commands=commands,
            required=required,
            out=out,
            log_path=log_path,
            env=env,
            ms_path=ms_path,
            spw=spw,
            timeout=timeout,
            tail=tail,
            started=started,
            fail=_fail,
            retries=max(int(retries), 0),
            cube_products=cube_products,
            contsub_order=contsub_order,
            cube_line_mask=bool(cube_line_mask and cube_command),
            after_images=after_images,
            pb_dish_diameter=pb_dish_diameter,
        )
    finally:
        if scratch_dir is not None:
            shutil.rmtree(scratch_dir, ignore_errors=True)
        if single_window_ms is not None:
            shutil.rmtree(single_window_ms, ignore_errors=True)


def _run_and_mark(
    *,
    commands: list[list[str]],
    required: Sequence[str],
    out: Path,
    log_path: Path,
    env: dict[str, str],
    ms_path: str,
    spw: int,
    timeout: float | None,
    tail: deque[str],
    started: float,
    fail: Callable[..., RuntimeError],
    retries: int = 0,
    cube_products: Sequence[str] | None = None,
    contsub_order: int | None = None,
    cube_line_mask: bool = False,
    after_images: Callable[[], None] | None = None,
    pb_dish_diameter: float | None = None,
) -> dict[str, object]:
    attempts = 0
    commands = list(commands)
    # run_wsclean_task truncated the log; for a single-window task it already
    # holds the extraction notes, so append.
    with log_path.open("a", encoding="utf-8") as log:
        for cmd in list(commands):
            attempts = _run_with_retries(cmd, out, log, env, timeout, tail, fail, retries)
        if cube_line_mask:
            second = _line_mask_pass(commands[-1], out, log)
            if second:
                commands.append(second)
                attempts = _run_with_retries(second, out, log, env, timeout, tail, fail, retries)
    cube_channels: dict[str, int] = {}
    if CUBE_IMAGE_FILENAME in required:
        try:
            cube_channels = stack_channel_images(out, prefix=CUBE_NAME, keep=cube_products)
        except Exception as exc:
            raise fail(f"Could not stack the channel images into cubes: {exc}")
        # WSClean also writes a <name>-MFS-<product>.fits for a -channels-out
        # run: a copy of the continuum the continuum run already made.
        for duplicate in out.glob(f"{CUBE_NAME}-MFS-*.fits"):
            duplicate.unlink(missing_ok=True)
        notes = ", ".join(f"{product} ({n} channels)" for product, n in cube_channels.items())
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"# stacked channel images into cubes: {notes or 'none'}\n")
        if contsub_order is not None and (out / CUBE_IMAGE_FILENAME).is_file():
            try:
                fit = subtract_continuum(
                    out / CUBE_IMAGE_FILENAME, out / CONTSUB_FILENAME, order=contsub_order
                )
            except Exception as exc:
                raise fail(f"Could not subtract the continuum from the cube: {exc}")
            with log_path.open("a", encoding="utf-8") as log:
                log.write(
                    f"# continuum-subtracted cube (order {contsub_order}): "
                    f"{fit['line_fraction']:.1%} of the fitted pixels' channels clipped as line\n"
                )
    if after_images is not None:
        after_images()
    missing = [name for name in required if not (out / name).is_file()]
    if missing:
        raise fail(f"WSClean exited 0 but wrote no {', '.join(missing)} under {out}", 0)
    if pb_dish_diameter:
        try:
            corrected = pb_correct_task(out, pb_dish_diameter)
        except Exception as exc:
            raise fail(f"Could not primary-beam correct the images: {exc}")
        with log_path.open("a", encoding="utf-8") as log:
            for name, info in corrected.items():
                log.write(f"# primary-beam corrected {name}: {pb_note(info)}\n")

    elapsed = time.time() - started
    write_imaging_marker(
        out,
        ms_path=ms_path,
        spw=spw,
        command=commands[0] if len(commands) == 1 else None,
        commands=commands if len(commands) > 1 else None,
        seconds=elapsed,
        images=required,
    )
    first = out / required[0]
    return {
        "ms": ms_path,
        "spw": int(spw),
        "output": str(out),
        "image": str(first),
        "seconds": round(elapsed, 1),
        "attempts": attempts,
        **({"channels": max(cube_channels.values())} if cube_channels else {}),
    }


def pb_correct_task(
    output_dir: str | os.PathLike[str], dish_diameter: float, cutoff: float | None = None
) -> dict[str, dict[str, Any]]:
    """Primary-beam correct a task's continuum image and image cube, where present.

    Returns ``{image file name: primary_beam_correct's summary}``.
    """
    from .primary_beam import PB_CUTOFF, primary_beam_correct

    out = Path(output_dir)
    corrected: dict[str, dict[str, Any]] = {}
    for image, pb, pbcor in (
        (IMAGE_FILENAME, PB_FILENAME, PBCOR_FILENAME),
        (CUBE_IMAGE_FILENAME, CUBE_PB_FILENAME, CUBE_PBCOR_FILENAME),
    ):
        if (out / image).is_file():
            corrected[image] = primary_beam_correct(
                out / image,
                out / pb,
                out / pbcor,
                dish_diameter,
                cutoff=PB_CUTOFF if cutoff is None else cutoff,
            )
    return corrected


def pb_note(info: Mapping[str, Any]) -> str:
    return (
        f"{info['model']} ({float(info['dish_diameter']):g} m dish), FWHM "
        f'{info["fwhm_arcsec"]}" at the middle channel, {info["channels"]} channel(s), '
        f"{100 * float(info['blanked_fraction']):.1f} % blanked below {info['cutoff']}"
    )


def _call_in_child(
    function: str,
    kwargs: Mapping[str, object],
    *,
    what: str,
    timeout: float | None = None,
    log_path: str | os.PathLike[str] | None = None,
    casa: bool = False,
) -> dict[str, object]:
    """``function(**kwargs)`` of this module in a new Python process, its JSON result.

    casacore and CASA hold the GIL for the whole call, which would freeze a
    Dask worker (see ``extract_single_window_ms_subprocess``), and
    python-casacore and casatools must not share a process. The child's
    stderr is appended to ``log_path``; ``casa`` gives it ``casa_environment``.
    """
    return call_in_child(
        f"almasim.services.imaging.archive_imaging:{function}",
        kwargs,
        what=what,
        env=casa_environment() if casa else wsclean_environment(1),
        timeout=timeout,
        log_path=log_path,
    )


def image_snr(image_path: Path, residual_path: Path) -> float:
    """Peak of the restored image over the robust noise (1.4826 MAD) of the residual."""
    import numpy as np
    from astropy.io import fits

    image = fits.getdata(image_path).squeeze()
    residual = fits.getdata(residual_path).squeeze()
    finite = residual[np.isfinite(residual)]
    noise = 1.4826 * np.median(np.abs(finite - np.median(finite)))
    return float(np.nanmax(image) / noise) if noise > 0 else 0.0


def image_peak(image_path: Path) -> float:
    """Peak of the restored image."""
    import numpy as np
    from astropy.io import fits

    return float(np.nanmax(fits.getdata(image_path)))


def parse_selfcal_rounds(solints: Sequence[str]) -> list[tuple[str, str]]:
    """``(calmode, solint)`` per round: ``inf`` or ``30s`` is a phase round,
    ``ap:inf`` an amplitude-and-phase one. An amplitude round must come after
    a phase round: it is solved on top of the phase solution."""
    rounds = []
    for item in solints:
        calmode, _, solint = item.rpartition(":")
        calmode = calmode or "p"
        if calmode not in ("p", "ap") or not re.fullmatch(r"inf|int|\d+(\.\d+)?s", solint):
            raise ValueError(
                f"self-calibration rounds are [ap:]inf, [ap:]int or [ap:]<seconds>s, not {item!r}"
            )
        rounds.append((calmode, solint))
    if rounds and rounds[0][0] == "ap":
        raise ValueError("an amplitude (ap:) round needs a phase round before it")
    return rounds


def self_calibrate(
    ms_path: Path,
    continuum_command: list[str],
    *,
    out: Path,
    log,
    solints: Sequence[str],
    min_snr: float,
    run: Callable[[list[str]], int],
    fail: Callable[..., RuntimeError],
    casa_data: str,
    timeout: float | None = None,
    restart: bool = False,
    keep_gains: bool = True,
) -> dict[str, object]:
    """Self-calibrate ``ms_path`` against its own continuum, round by round.

    The continuum is imaged (``continuum_command`` into ``selfcal-0``, writing
    MODEL_DATA); below ``min_snr`` (peak over residual noise) nothing is
    done. Each round (``parse_selfcal_rounds``: ``inf``, ``30s`` for phase,
    ``ap:inf`` for amplitude and phase) then solves gains of DATA against the
    model at its solution interval with CASA ``gaincal``
    (``casa_selfcal_solve``), an amplitude round on top of the kept phase
    table, applies the kept phase table and the new one into CORRECTED_DATA
    (``casa_selfcal_apply``) and reimages.
    A round is kept only if the image S/N rose; otherwise the best earlier
    gains are restored (or CORRECTED_DATA dropped) and the rounds stop.
    The temporary images and calibration tables are removed; CASA, run with
    ``casa_data`` as its runtime data, logs to ``casa-selfcal.log`` in
    ``out``.

    With ``restart`` every image is cleaned in two passes, as the continuum
    is (``continuum_restart_commands``), so the model the gains are solved
    against is complete. With ``keep_gains`` the applied tables are kept in
    ``selfcal_gains/`` (``_keep_gain_tables``).

    ALMASim's own StefCal-style solver gave the same S/N within ~1 % on four
    cycle11 cubes and was dropped for CASA (2026-10-17).
    """
    plan = parse_selfcal_rounds(solints)
    rounds: list[dict[str, object]] = []
    casa_log = str(out / "casa-selfcal.log")
    prepared: dict[str, object] = {}

    def image(index: int) -> tuple[float, float]:
        name = f"selfcal-{index}"
        command = _selfcal_command(continuum_command, out, name)
        run(command)
        if restart:
            run([*command[:-1], "-continue", command[-1]])
        image_path = out / f"{name}-image.fits"
        return image_snr(image_path, out / f"{name}-residual.fits"), image_peak(image_path)

    def call(function: str, **kwargs: object) -> dict[str, object]:
        try:
            return _call_in_child(
                function,
                kwargs,
                what=function,
                timeout=timeout,
                log_path=log.name,
                casa=function.startswith("casa_"),
            )
        except Exception as exc:
            raise fail(f"Self-calibration failed: {exc}")

    def solve(gains: Path, calmode: str, solint: str, pre_apply: list[Path]) -> dict:
        return call(
            "casa_selfcal_solve",
            ms_path=str(ms_path),
            caltable=str(gains),
            solint=solint,
            spw=prepared["spw"],
            refant=prepared["refant"],
            casa_data=casa_data,
            log_file=casa_log,
            calmode=calmode,
            pre_apply=[str(t) for t in pre_apply],
        )

    def apply(tables: list[Path]) -> None:
        if not tables:
            call("drop_corrected_data", ms_path=str(ms_path))
        else:
            call(
                "casa_selfcal_apply",
                ms_path=str(ms_path),
                caltables=[str(t) for t in tables],
                casa_data=casa_data,
                log_file=casa_log,
            )

    try:
        if call("adopt_corrected_data", ms_path=str(ms_path)).get("adopted"):
            log.write("# self-calibration: incoming CORRECTED_DATA adopted as DATA\n")
        best_snr, best_peak = image(0)
        log.write(
            f"# self-calibration: initial continuum S/N {best_snr:.1f}, "
            f"peak {best_peak:.4g} Jy/beam\n"
        )
        if best_snr < min_snr:
            log.write(f"# self-calibration skipped: S/N below {min_snr:g}\n")
            log.flush()
            return {"applied": False, "initial_snr": best_snr, "rounds": rounds}
        prepared = call("selfcal_prepare", ms_path=str(ms_path))
        log.write(
            f"# self-calibration with CASA gaincal: refant {prepared['refant']}, "
            f"{prepared['line_free_channels']} line-free channels\n"
        )
        # The kept tables: the best phase table, then the best amplitude one.
        best_phase: Path | None = None
        best_amp: Path | None = None
        for index, (calmode, solint) in enumerate(plan, start=1):
            gains = out / f"selfcal-{index}.{'G' if calmode == 'p' else 'A'}"
            # An ap round always follows a kept phase round: parse_selfcal_rounds
            # puts a phase round first, and the rounds stop at the first rejection.
            pre_apply = [best_phase] if calmode == "ap" else []
            solved = solve(gains, calmode, solint, pre_apply)
            trial = [gains] if calmode == "p" else [best_phase, gains]
            apply(trial)
            snr, peak = image(index)
            kept = snr > best_snr
            rounds.append({**solved, "snr": round(snr, 1), "peak": peak, "kept": kept})
            detail = (
                f"median |phase| {solved['median_phase_deg']} deg"
                if calmode == "p"
                else f"median |amp - 1| {solved['median_amp_dev']}"
            )
            log.write(
                f"# self-calibration round {index} ({calmode}, solint {solint}): "
                f"{solved['solutions']}/{solved['possible']} antenna solutions, {detail}; "
                f"S/N {best_snr:.1f} -> {snr:.1f}, peak {best_peak:.4g} -> {peak:.4g} Jy/beam"
                f"{'' if kept else ', rejected'}\n"
            )
            log.flush()
            if not kept:
                apply([t for t in (best_phase, best_amp) if t])
                break
            best_snr, best_peak = snr, peak
            if calmode == "p":
                best_phase, best_amp = gains, None
            else:
                best_amp = gains
        applied = [t.name for t in (best_phase, best_amp) if t]
        outcome = f"applied ({', '.join(applied)})" if applied else "not applied"
        log.write(f"# self-calibration {outcome}: final continuum S/N {best_snr:.1f}\n")
        kept_gains = (
            _keep_gain_tables(
                [(t, kind) for t, kind in ((best_phase, "phase"), (best_amp, "amplitude")) if t],
                out,
                call,
            )
            if keep_gains
            else []
        )
        if kept_gains:
            log.write(
                f"# self-calibration gains kept in {SELFCAL_GAINS_DIRNAME}/: "
                f"{', '.join(kept_gains)}\n"
            )
        log.flush()
        return {"applied": bool(applied), "snr": best_snr, "rounds": rounds, "gains": kept_gains}
    finally:
        for leftover in out.glob("selfcal-*"):
            if leftover.is_dir():
                shutil.rmtree(leftover, ignore_errors=True)
            else:
                leftover.unlink(missing_ok=True)


def _subtract_uv_continuum_step(
    ms_path: Path,
    commands: list[list[str]],
    *,
    has_continuum: bool,
    order: int,
    log,
    fail: Callable[..., RuntimeError],
    timeout: float | None = None,
) -> tuple[list[list[str]], dict[str, object] | None]:
    """Run ``subtract_uv_continuum`` and point the commands at its columns.

    Returns the commands and the fit's result (None when it was skipped).
    """
    try:
        fit = subtract_uv_continuum_subprocess(
            ms_path, order=order, timeout=timeout, log_path=log.name
        )
    except Exception as exc:
        raise fail(f"Could not subtract the continuum from the visibilities: {exc}")
    if fit.get("skipped"):
        log.write(
            f"# continuum NOT subtracted from the visibilities: {fit['skipped']}; "
            "the cube keeps the continuum\n"
        )
        log.flush()
        return commands, None
    log.write(
        f"# continuum subtracted from {fit['source_column']} (order "
        f"{fit['order']}, {fit['seconds']} s): {fit['line_free_channels']} of "
        f"{fit['usable_channels']} channels line-free ({fit['channels']} in the window), "
        "lines in channels "
        f"{fit['line_ranges'] or 'none'}\n"
    )
    log.flush()
    # Every continuum pass (two with continuum_restart), then the cube.
    continuum = (
        [_with_data_column(c, CONTINUUM_COLUMN) for c in commands[:-1]] if has_continuum else []
    )
    return [*continuum, _with_data_column(commands[-1], LINE_COLUMN)], fit


def _keep_gain_tables(
    tables: Sequence[tuple[Path, str]], out: Path, call: Callable[..., dict[str, object]]
) -> list[str]:
    """Copy the kept calibration tables to ``selfcal_gains/`` as ``<kind>.cal``,
    each with a CSV of its solutions (``export_gain_table``); the names written."""
    if not tables:
        return []
    target = out / SELFCAL_GAINS_DIRNAME
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True)
    written = []
    for table, kind in tables:
        shutil.copytree(table, target / f"{kind}.cal")
        call("export_gain_table", caltable=str(table), csv_path=str(target / f"{kind}.csv"))
        written += [f"{kind}.cal", f"{kind}.csv"]
    return written


def _selfcal_noise_products(
    ms_path: Path,
    *,
    continuum_command: Sequence[str] | None,
    cube_command: Sequence[str] | None,
    fit: dict[str, object] | None,
    out: Path,
    log,
    run: Callable[[list[str]], int],
    fail: Callable[..., RuntimeError],
    timeout: float | None = None,
) -> list[str]:
    """Image what self-calibration removed and form the total residuals.

    The kept residual is the dirty image (same grid and weighting) of the
    self-calibrated data minus the model. The correction column
    (``selfcal_correction_columns``, split like the data when the continuum
    was subtracted in the visibilities) imaged the same way gives
    ``wsclean[-cube]-selfcal-correction.fits``; imaging with fixed weights is
    linear, so residual + correction is the dirty image of the data before
    self-calibration minus the model, everything the final model does not
    explain: ``wsclean[-cube]-residual-total.fits``. Returns the files written.
    """
    split = fit is not None
    try:
        _call_in_child(
            "selfcal_correction_columns",
            {
                "ms_path": str(ms_path),
                "line_ranges": fit["line_ranges"] if fit is not None else None,
                "order": fit["order"] if fit is not None else 1,
            },
            what="selfcal_correction_columns",
            timeout=timeout,
            log_path=log.name,
        )
    except Exception as exc:
        raise fail(f"Could not write the self-calibration correction: {exc}")
    written: list[str] = []
    products = (
        (
            continuum_command,
            "wsclean",
            SELFCAL_CONTINUUM_COLUMN,
            SELFCAL_CORRECTION_FILENAME,
            TOTAL_RESIDUAL_FILENAME,
            False,
        ),
        (
            cube_command,
            CUBE_NAME,
            SELFCAL_LINE_COLUMN,
            CUBE_SELFCAL_CORRECTION_FILENAME,
            CUBE_TOTAL_RESIDUAL_FILENAME,
            True,
        ),
    )
    for command, prefix, split_column, correction_name, total_name, is_cube in products:
        if not command:
            continue
        name = f"{prefix}-selfcal-correction"
        run(_dirty_command(command, out, name, split_column if split else SELFCAL_COLUMN))
        image: Path | None = None
        if is_cube:
            try:
                stacked = stack_channel_images(out, prefix=name, keep=("dirty", "image"))
            except Exception as exc:
                raise fail(f"Could not stack the self-calibration correction cube: {exc}")
            for duplicate in out.glob(f"{name}-MFS-*.fits"):
                duplicate.unlink(missing_ok=True)
            image = next(
                (
                    out / f"{name}-{product}.fits"
                    for product in ("dirty", "image")
                    if product in stacked
                ),
                None,
            )
        image = image or _single_image(out, name)
        if image is None:
            raise fail(f"WSClean wrote no {name} image under {out}")
        image.replace(out / correction_name)
        for leftover in out.glob(f"{name}-*.fits"):
            leftover.unlink(missing_ok=True)
        written.append(correction_name)
        residual = out / f"{prefix}-residual.fits"
        if residual.is_file():
            try:
                add_images(
                    residual,
                    out / correction_name,
                    out / total_name,
                    "ALMASim total residual: residual + self-calibration correction",
                )
            except Exception as exc:
                raise fail(f"Could not form {total_name}: {exc}")
            written.append(total_name)
    log.write(f"# self-calibration noise products: {', '.join(written) or 'none'}\n")
    log.flush()
    return written
