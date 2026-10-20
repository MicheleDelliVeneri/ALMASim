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

import json
import os
import re
import shlex
import shutil
import subprocess
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

from almasim.services.runtime.child import ChildFailed, append_to_log, call_in_child, error_line

if TYPE_CHECKING:
    import numpy as np

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

# WSClean is invoked with ``-name <output_dir>/wsclean`` and a single SPW, so
# the restored image is always this file.
IMAGE_FILENAME = "wsclean-image.fits"

# A cube run uses ``-name <output_dir>/wsclean-cube``; its per-channel images
# are stacked into ``wsclean-cube-<product>.fits`` (see stack_channel_images).
CUBE_NAME = "wsclean-cube"
CUBE_IMAGE_FILENAME = f"{CUBE_NAME}-image.fits"
CONTSUB_FILENAME = f"{CUBE_NAME}-image.contsub.fits"
# The moment-0 mask the cube's second pass cleans in (``line_mask_from_channels``).
LINE_MASK_FILENAME = f"{CUBE_NAME}-linemask.fits"
# The primary beam and the primary-beam corrected image, continuum and cube
# (``primary_beam.primary_beam_correct``).
PB_FILENAME = "wsclean-pb.fits"
PBCOR_FILENAME = "wsclean-image.pbcor.fits"
CUBE_PB_FILENAME = f"{CUBE_NAME}-pb.fits"
CUBE_PBCOR_FILENAME = f"{CUBE_NAME}-image.pbcor.fits"

# Columns ``subtract_uv_continuum`` adds to the per-task single-window MS: the
# continuum fitted per visibility over the line-free channels, and the data
# minus it (the lines). The continuum image is made from the first, the cube
# from the second.
CONTINUUM_COLUMN = "CONTINUUM_DATA"
LINE_COLUMN = "LINE_DATA"

# What self-calibration removed (DATA - CORRECTED_DATA), whole or split into
# continuum and lines like the data (``selfcal_correction_columns``), and the
# images made from it: the dirty image of the correction and the total
# residual, the kept residual plus the correction (``_selfcal_noise_products``).
SELFCAL_COLUMN = "SELFCAL_DATA"
SELFCAL_CONTINUUM_COLUMN = "SELFCAL_CONTINUUM_DATA"
SELFCAL_LINE_COLUMN = "SELFCAL_LINE_DATA"
SELFCAL_CORRECTION_FILENAME = "wsclean-selfcal-correction.fits"
TOTAL_RESIDUAL_FILENAME = "wsclean-residual-total.fits"
CUBE_SELFCAL_CORRECTION_FILENAME = f"{CUBE_NAME}-selfcal-correction.fits"
CUBE_TOTAL_RESIDUAL_FILENAME = f"{CUBE_NAME}-residual-total.fits"
SELFCAL_GAINS_DIRNAME = "selfcal_gains"

# Placeholder in a task command for the per-task single-window MS that the
# wrapper extracts on the worker (see ``extract_single_window_ms``).
SINGLE_WINDOW_PLACEHOLDER = "__SINGLE_WINDOW_MS__"

# Rows whose WEIGHT exceeds this are left out of the single-window MS. The
# Cycle-11 calibrated MSs (2026-09-25, 40/40 sampled) carry 4-58 % of
# unflagged target rows with WEIGHT ~1e5-1e8 and amplitudes 1e3 too low or
# 4-26x too high: data the ALMA pipeline had flagged (bad Tsys, antenna or
# scan) that received garbage gains because the calibration stage did not
# restore the delivered Pipeline_Final flags. Physical ALMA weights
# (1/sigma^2 per visibility, calwt) stay below ~1e3; the two populations are
# separated by more than an order of magnitude on every EB sampled.
DEFAULT_MAX_WEIGHT = 1.0e4

# Prefer the system C library over spack builds, but keep whatever the caller
# had after it (same policy as the unpack/calibrate subprocess wrappers).
_SYSTEM_LIBRARY_DIRS = "/lib64:/usr/lib64:/usr/local/lib64:/lib:/usr/lib:/usr/local/lib"


def imaging_marker_path(output_dir: str | os.PathLike[str]) -> Path:
    """``<root>/<ms>/SPW-3`` -> ``<root>/<ms>/SPW-3.done``."""
    out = Path(output_dir)
    return out.with_name(out.name + ".done")


def imaging_failure_marker_path(output_dir: str | os.PathLike[str]) -> Path:
    out = Path(output_dir)
    return out.with_name(out.name + ".failed")


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
    imaging_failure_marker_path(out).unlink(missing_ok=True)
    marker = imaging_marker_path(out)
    payload = {
        "ms": ms_path,
        "spw": int(spw),
        "output": str(out),
        "image": str(out / IMAGE_FILENAME),
        "completed_at": datetime.now(timezone.utc).isoformat(),
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
    marker.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return marker


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
    out.parent.mkdir(parents=True, exist_ok=True)
    imaging_marker_path(out).unlink(missing_ok=True)
    marker = imaging_failure_marker_path(out)
    payload = {
        "ms": ms_path,
        "spw": int(spw),
        "stage": "image",
        "output": str(out),
        "error": error,
        "failed_at": datetime.now(timezone.utc).isoformat(),
    }
    if returncode is not None:
        payload["returncode"] = int(returncode)
    if log_path is not None:
        payload["log"] = str(log_path)
    marker.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return marker


def wsclean_environment(cores: int, base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Environment for a WSClean child using ``cores`` threads.

    WSClean refuses to start when it is linked against a multi-threaded
    OpenBLAS unless ``OPENBLAS_NUM_THREADS=1``; its own ``-j`` flag carries the
    parallelism. The other threading knobs follow the requested core count.
    """
    env = dict(os.environ if base is None else base)
    env["OPENBLAS_NUM_THREADS"] = "1"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = str(max(int(cores), 1))
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = (
        f"{_SYSTEM_LIBRARY_DIRS}:{existing}" if existing else _SYSTEM_LIBRARY_DIRS
    )
    return env


def casa_environment(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Environment for a CASA child: one thread each, and no system library
    directories in ``LD_LIBRARY_PATH``.

    ``wsclean_environment`` and the Slurm worker's job script put /lib64 and
    friends first for WSClean's spack build; casatools then loads the
    system PMIx under its bundled Open MPI and fails to import
    (``libopen-pal.so.80: undefined symbol: pmix_framework_names``, every
    task of job 5299, 2026-10-17).
    """
    env = wsclean_environment(1, base)
    system = set(_SYSTEM_LIBRARY_DIRS.split(":"))
    kept = [d for d in env.get("LD_LIBRARY_PATH", "").split(":") if d and d not in system]
    if kept:
        env["LD_LIBRARY_PATH"] = ":".join(kept)
    else:
        env.pop("LD_LIBRARY_PATH", None)
    return env


_error_line = error_line


# casacore keeps a list of the processes using a table in its table.lock file
# and writes it with pwrite even when no read lock is taken. With several
# tasks of one MS reading it from different nodes over NFS that write fails
# now and then ("LockFile.cc : 300 Failed AlwaysAssert tracePWRITE", 4 of
# ~370 tasks on 2026-09-25). It is transient, so the extraction is retried,
# each attempt in a fresh child process: casacore keeps the half-written
# output of a failed copy in its process-wide table cache, and a retry in the
# same process then fails with "Cannot copy/rename; target table ... is still
# open (is in the table cache)" (job 3841, 2 tasks on 2026-09-26).
_LOCK_FILE_ERROR = "LockFile.cc"
_LOCK_FILE_ATTEMPTS = 5
_LOCK_FILE_RETRY_SECONDS = 20.0


def extract_single_window_ms(
    ms_path: str | os.PathLike[str],
    spw: int,
    field_ids: Sequence[int] | None,
    out_path: str | os.PathLike[str],
    *,
    max_weight: float | None = DEFAULT_MAX_WEIGHT,
) -> dict[str, object]:
    """Copy one spectral window (and optionally some fields) into a new MS.

    One attempt; ``extract_single_window_ms_subprocess`` retries lock-file
    failures (see ``_LOCK_FILE_ERROR``) in a new process each time.

    Rows whose WEIGHT exceeds ``max_weight`` are left out (see
    ``DEFAULT_MAX_WEIGHT``; ``None`` or 0 keeps every row); the result
    reports how many were dropped.

    WSClean 3.7 reads out of bounds in its reordering step when an MS holds
    several spectral windows and ``-spws`` selects one (an ALMA split MS
    lists 60 windows, 4 of them populated); the crash is heap-layout
    dependent, so about half the tasks die. Handing WSClean an MS with a
    single spectral window lets it use its contiguous reader (``-no-reorder``)
    and skip that code entirely. The copy is a deep copy of the selected rows
    with all subtables; DATA_DESCRIPTION and SPECTRAL_WINDOW are collapsed to
    the one window (renumbered 0) so WSClean sees exactly one band.
    """
    return _extract_single_window_ms_once(ms_path, spw, field_ids, out_path, max_weight=max_weight)


def _extract_single_window_ms_once(
    ms_path: str | os.PathLike[str],
    spw: int,
    field_ids: Sequence[int] | None,
    out_path: str | os.PathLike[str],
    *,
    max_weight: float | None,
) -> dict[str, object]:
    from casacore.tables import table, taql

    src = str(ms_path)
    out = Path(out_path)
    shutil.rmtree(out, ignore_errors=True)
    # The source MS is only read, by several tasks at once (one per spectral
    # window, on different nodes) over NFS. Read locking would make every
    # task write the shared table.lock file, which failed with
    # "LockFile.cc: Failed AlwaysAssert tracePWRITE" when three tasks of one
    # MS ran together (job 3835); no read locking touches no lock file.
    lock = {"lockoptions": "usernoread", "ack": False}
    dd_table = table(f"{src}/DATA_DESCRIPTION", **lock)
    spw_of_dd = [int(x) for x in dd_table.getcol("SPECTRAL_WINDOW_ID")]
    pol_of_dd = [int(x) for x in dd_table.getcol("POLARIZATION_ID")]
    dd_table.close()
    dd_ids = [i for i, s in enumerate(spw_of_dd) if s == int(spw)]
    if not dd_ids:
        raise RuntimeError(f"Spectral window {spw} has no DATA_DESCRIPTION row in {src}")
    where = f"DATA_DESC_ID in [{','.join(str(i) for i in dd_ids)}]"
    if field_ids:
        where += f" and FIELD_ID in [{','.join(str(int(f)) for f in field_ids)}]"
    main_in = table(src, **lock)
    n_all = int(taql(f"select from $1 where {where}", tables=[main_in]).nrows())
    if n_all == 0:
        main_in.close()
        raise RuntimeError(f"No visibilities for spectral window {spw} fields {field_ids} in {src}")
    if max_weight:
        where += f" and max(WEIGHT) <= {float(max_weight)!r}"
    selection = taql(f"select from $1 where {where}", tables=[main_in])
    n_rows = int(selection.nrows())
    if n_rows == 0:
        selection.close()
        main_in.close()
        raise RuntimeError(
            f"All {n_all} visibilities for spectral window {spw} fields {field_ids} in {src} "
            f"have WEIGHT > {max_weight:g}"
        )
    selection.copy(str(out), deep=True)
    selection.close()
    main_in.close()

    dd_out = table(f"{out}/DATA_DESCRIPTION", readonly=False, ack=False)
    dd_out.removerows([i for i in range(dd_out.nrows()) if i not in dd_ids])
    dd_out.putcol("SPECTRAL_WINDOW_ID", [0] * len(dd_ids))
    dd_out.putcol("POLARIZATION_ID", [pol_of_dd[i] for i in dd_ids])
    dd_out.close()
    spw_out = table(f"{out}/SPECTRAL_WINDOW", readonly=False, ack=False)
    spw_out.removerows([i for i in range(spw_out.nrows()) if i != int(spw)])
    spw_out.close()
    renumber = {old: new for new, old in enumerate(dd_ids)}
    main = table(str(out), readonly=False, ack=False)
    main.putcol("DATA_DESC_ID", [renumber[int(x)] for x in main.getcol("DATA_DESC_ID")])
    main.close()
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    return {
        "rows": n_rows,
        "bytes": size,
        "data_desc_ids": dd_ids,
        "dropped_rows": n_all - n_rows,
        "max_weight": float(max_weight) if max_weight else None,
    }


def extract_single_window_ms_subprocess(
    ms_path: str | os.PathLike[str],
    spw: int,
    field_ids: Sequence[int] | None,
    out_path: str | os.PathLike[str],
    *,
    timeout: float | None = None,
    log_path: str | os.PathLike[str] | None = None,
    max_weight: float | None = DEFAULT_MAX_WEIGHT,
    attempts: int = _LOCK_FILE_ATTEMPTS,
    retry_seconds: float = _LOCK_FILE_RETRY_SECONDS,
) -> dict[str, object]:
    """Run ``extract_single_window_ms`` in a child Python process.

    casacore does the deep copy in C++ without releasing the GIL, for tens of
    seconds to minutes per task. Called in a Dask worker thread that freezes
    the worker's event loop, the scheduler stops receiving heartbeats and
    drops the worker, and dask-jobqueue then cancels its Slurm job with every
    task on it (production job 3828, 2026-09-25: all five workers lost within
    two minutes of starting). A child process keeps the worker responsive,
    exactly like the WSClean subprocess. The child's stderr is appended to
    ``log_path`` when given.

    A child that dies of a casacore lock-file error (see ``_LOCK_FILE_ERROR``)
    is rerun up to ``attempts`` times, ``retry_seconds`` apart, each time in a
    new process; any other failure raises at once. ``timeout`` applies to
    each attempt.
    """
    kwargs = {
        "ms_path": str(ms_path),
        "spw": int(spw),
        "field_ids": [int(f) for f in field_ids] if field_ids else None,
        "out_path": str(out_path),
        "max_weight": float(max_weight) if max_weight else None,
    }
    started = time.time()
    for attempt in range(1, max(int(attempts), 1) + 1):
        try:
            result = call_in_child(
                "almasim.services.imaging.archive_imaging:extract_single_window_ms",
                kwargs,
                what="extraction",
                env=wsclean_environment(1),
                timeout=timeout,
                timeout_noun="task timeout",
                log_path=log_path,
            )
            break
        except ChildFailed as exc:
            if _LOCK_FILE_ERROR not in exc.stderr or attempt >= attempts:
                raise
            append_to_log(
                log_path,
                f"extraction attempt {attempt}/{attempts} hit a casacore lock-file error; "
                f"retrying in a new process in {retry_seconds:g} s\n",
            )
            time.sleep(retry_seconds)
    result["seconds"] = round(time.time() - started, 1)
    result["attempts"] = attempt
    return result


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


def _run_with_retries(
    cmd: list[str],
    out: Path,
    log,
    env: dict[str, str],
    timeout: float | None,
    tail: deque[str],
    fail: Callable[..., RuntimeError],
    retries: int,
) -> int:
    """Run one WSClean command, retrying deaths by signal; raise ``fail`` otherwise."""
    attempts = 0
    log.write("# " + shlex.join(cmd) + "\n")
    log.flush()
    tail.clear()
    while True:
        attempts += 1
        # Vary the size of the environment between attempts. The WSClean
        # 3.7 reordering crash is deterministic for a given argv+environ
        # and flips with their layout, so an identical retry never helps;
        # shifting the strings by one byte per attempt does.
        attempt_env = dict(env)
        attempt_env["ALMASIM_WSCLEAN_ATTEMPT"] = str(attempts)
        attempt_env["ALMASIM_WSCLEAN_PAD"] = "x" * (attempts - 1)
        returncode, timed_out = _run_once(cmd, out, log, attempt_env, timeout, tail, fail)
        if returncode >= 0 or timed_out or attempts > retries:
            break
        log.write(
            f"# attempt {attempts} died with signal {-returncode}; "
            f"retrying ({retries - attempts + 1} left)\n"
        )
        log.flush()
        tail.clear()
        _remove_reorder_files(out)
    if timed_out:
        raise fail(f"WSClean killed after exceeding the {timeout:.0f}s task timeout", returncode)
    if returncode != 0:
        error = f"WSClean exited with return code {returncode}"
        reason = _error_line(list(tail))
        if reason:
            error = f"{error}: {reason}"
        if attempts > 1:
            error = f"{error} (after {attempts} attempts)"
        raise fail(error, returncode)
    return attempts


_CHANNEL_FILE = re.compile(r"^(?P<prefix>.+)-(?P<channel>\d{4})-(?P<product>[a-z-]+)\.fits$")


def stack_channel_images(
    output_dir: str | os.PathLike[str],
    *,
    prefix: str = "wsclean",
    keep: Sequence[str] | None = None,
) -> dict[str, int]:
    """Stack WSClean's per-channel FITS (``-channels-out``) into one cube per product.

    WSClean writes ``<prefix>-0000-image.fits``, ``<prefix>-0001-image.fits``
    ... and no cube. Each product (image, residual, model, psf, dirty) becomes
    ``<prefix>-<product>.fits`` with a regular frequency axis, and the
    per-channel files are removed. Products not in ``keep`` (all when None)
    are removed without stacking.

    The channel beams differ (the beam scales with wavelength), so a cube
    with beam keywords also gets a CASA-style ``BEAMS`` table (one row per
    channel, BMAJ/BMIN in arcsec, BPA in degrees, ``CASAMBM = T``), which
    CASA and spectral-cube read; the primary header keeps the middle
    channel's beam.

    Returns ``{product: n_channels}`` of the cubes written.
    """
    import numpy as np
    from astropy.io import fits

    out = Path(output_dir)
    groups: dict[str, list[tuple[int, Path]]] = {}
    for path in out.iterdir():
        match = _CHANNEL_FILE.match(path.name)
        if match and match["prefix"] == prefix:
            groups.setdefault(match["product"], []).append((int(match["channel"]), path))
    written: dict[str, int] = {}
    for product, members in sorted(groups.items()):
        members.sort()
        paths = [path for _, path in members]
        if keep is not None and product not in keep:
            for path in paths:
                path.unlink(missing_ok=True)
            continue
        headers = [fits.getheader(path) for path in paths]
        first = headers[0]
        if first.get("CTYPE3", "").upper() != "FREQ":
            raise RuntimeError(f"{paths[0]} has no FREQ third axis (CTYPE3={first.get('CTYPE3')})")
        frequencies = np.array([float(h["CRVAL3"]) for h in headers])
        steps = np.diff(frequencies)
        if steps.size and not np.allclose(steps, steps[0], rtol=1e-6, atol=1e-3):
            raise RuntimeError(f"{product} channels of {out} are not evenly spaced in frequency")
        cube = np.empty((1, len(paths), int(first["NAXIS2"]), int(first["NAXIS1"])), np.float32)
        for index, path in enumerate(paths):
            cube[0, index] = fits.getdata(path).reshape(cube.shape[2:])
        header = first.copy()
        header["NAXIS3"] = len(paths)
        header["CRPIX3"] = 1.0
        header["CRVAL3"] = frequencies[0]
        if steps.size:
            header["CDELT3"] = float(steps[0])
        hdus: list[fits.hdu.base.ExtensionHDU] = []
        if all("BMAJ" in h for h in headers):
            middle = headers[len(headers) // 2]
            for key in ("BMAJ", "BMIN", "BPA"):
                header[key] = middle[key]
            header["CASAMBM"] = (True, "one beam per channel in the BEAMS table")
            beams = fits.BinTableHDU.from_columns(
                [
                    fits.Column("BMAJ", "1E", "arcsec", array=[h["BMAJ"] * 3600 for h in headers]),
                    fits.Column("BMIN", "1E", "arcsec", array=[h["BMIN"] * 3600 for h in headers]),
                    fits.Column("BPA", "1E", "deg", array=[h["BPA"] for h in headers]),
                    fits.Column("CHAN", "1J", array=np.arange(len(headers))),
                    fits.Column("POL", "1J", array=np.zeros(len(headers), dtype=int)),
                ],
                name="BEAMS",
            )
            beams.header["NCHAN"] = len(headers)
            beams.header["NPOL"] = 1
            hdus.append(beams)
        target = out / f"{prefix}-{product}.fits"
        partial = target.with_name(target.name + ".partial")
        fits.HDUList([fits.PrimaryHDU(cube, header=header), *hdus]).writeto(partial, overwrite=True)
        partial.replace(target)
        for path in paths:
            path.unlink(missing_ok=True)
        written[product] = len(paths)
    return written


def _fit_continuum(
    spectra: np.ndarray,
    usable: np.ndarray,
    order: int,
    clip_sigma: float,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sigma-clipped polynomial continuum of ``spectra`` (channels x pixels).

    Returns ``(continuum, kept)``: the fitted continuum per channel and pixel,
    and which channels the last iteration fitted (the line-free ones).
    """
    import numpy as np

    n_channels = spectra.shape[0]
    x = np.linspace(-1.0, 1.0, n_channels)[:, None] if n_channels > 1 else np.zeros((1, 1))
    y = np.where(usable, spectra, 0.0).astype(np.float64)
    kept = usable.copy()
    continuum = np.zeros_like(y)
    for _ in range(max(int(iterations), 1)):
        w = kept.astype(np.float64)
        s0 = w.sum(axis=0)
        t0 = (w * y).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(s0 > 0, t0 / s0, 0.0)
            if order >= 1:
                s1 = (w * x).sum(axis=0)
                s2 = (w * x * x).sum(axis=0)
                t1 = (w * x * y).sum(axis=0)
                det = s0 * s2 - s1 * s1
                fit = det > 1e-12 * np.maximum(s0 * s2, 1e-300)
                slope = np.where(fit, (s0 * t1 - s1 * t0) / det, 0.0)
                intercept = np.where(fit, (t0 * s2 - t1 * s1) / det, mean)
            else:
                slope, intercept = np.zeros_like(mean), mean
        continuum = intercept[None, :] + slope[None, :] * x
        residual = np.abs(y - continuum)
        sigma = 1.4826 * np.nanmedian(np.where(kept, residual, np.nan), axis=0)
        new_kept = usable & (residual <= clip_sigma * np.nan_to_num(sigma, nan=np.inf))
        # Never clip a pixel down to fewer channels than the fit needs.
        new_kept = np.where(new_kept.sum(axis=0) > order, new_kept, kept)
        if np.array_equal(new_kept, kept):
            break
        kept = new_kept
    return continuum, kept


def subtract_continuum(
    cube_path: str | os.PathLike[str],
    out_path: str | os.PathLike[str],
    *,
    order: int = 1,
    clip_sigma: float = 3.0,
    iterations: int = 10,
    rows_per_chunk: int = 64,
) -> dict[str, float]:
    """Write ``cube_path`` minus its continuum, fitted per pixel along frequency.

    The image-plane counterpart of CASA's ``imcontsub``, with the line-free
    channels found automatically: each pixel's spectrum is fitted with a
    polynomial of ``order`` (0: constant, 1: linear in frequency) by least
    squares, channels further than ``clip_sigma`` robust sigma (1.4826 x
    MAD) from the fit are dropped as line, and the fit is repeated until no
    channel changes (at most ``iterations``). Channels that are all zero or
    NaN (WSClean writes zeros for a fully flagged channel) are left out of
    the fit and stay NaN in the output.

    Emission covering most of a window (a broad line, a line forest) leaves
    too few line-free channels for the clipping to find; such cubes need
    the line-free ranges given explicitly.

    Returns ``{"line_fraction": ...}``: the share of fitted pixel-channels
    clipped as line.
    """
    import numpy as np
    from astropy.io import fits

    if order not in (0, 1):
        raise ValueError(f"continuum order must be 0 or 1, not {order}")
    with fits.open(cube_path, memmap=True) as hdus:
        header = hdus[0].header.copy()
        extensions = [hdu.copy() for hdu in hdus[1:]]
        data = np.asarray(hdus[0].data, dtype=np.float32)
    cube = data.reshape(data.shape[-3:])  # (channels, y, x); Stokes I only
    n_channels, ny, nx = cube.shape
    planes_blank = ~np.any(np.nan_to_num(cube) != 0, axis=(1, 2))
    result = np.full_like(cube, np.nan)
    clipped = fitted = 0
    for y0 in range(0, ny, max(int(rows_per_chunk), 1)):
        block = cube[:, y0 : y0 + rows_per_chunk, :]
        spectra = block.reshape(n_channels, -1)
        usable = np.isfinite(spectra) & ~planes_blank[:, None]
        continuum, kept = _fit_continuum(spectra, usable, order, clip_sigma, iterations)
        subtracted = np.where(usable, spectra - continuum, np.nan).astype(np.float32)
        result[:, y0 : y0 + rows_per_chunk, :] = subtracted.reshape(block.shape)
        fitted += int(usable.sum())
        clipped += int((usable & ~kept).sum())
    header["HISTORY"] = (
        f"ALMASim continuum subtracted: order {order}, {clip_sigma:g} sigma clipping per pixel"
    )
    out = Path(out_path)
    partial = out.with_name(out.name + ".partial")
    fits.HDUList([fits.PrimaryHDU(result.reshape(data.shape), header=header), *extensions]).writeto(
        partial, overwrite=True
    )
    partial.replace(out)
    return {"line_fraction": clipped / fitted if fitted else 0.0}


def line_free_channels(
    spectrum: np.ndarray,
    usable: np.ndarray,
    *,
    order: int = 1,
    clip_sigma: float = 3.0,
    widths: Sequence[int] = (1, 4, 16, 64),
    grow: int | None = None,
    iterations: int = 10,
) -> np.ndarray:
    """Line-free channels of a source spectrum: the ones a continuum fit keeps.

    A polynomial of ``order`` is fitted to the kept channels and its
    residual smoothed with a boxcar of each of ``widths`` channels; a channel
    is line wherever any smoothed residual exceeds ``clip_sigma`` times that
    scale's robust noise (1.4826 x MAD over the kept channels). The fit is
    repeated on the channels left until they stop changing. Each line is
    then widened by ``grow`` channels on both sides (default 1 % of the
    channels, at least 2) so its faint wings stay out of the fit.

    One-channel clipping alone missed the wings of the broad line of
    uid___A002_X10ded83_X6af4 SPW 16: ~0.3 sigma per channel over ~100
    channels, 2.5 sigma once smoothed over them (2026-10-17).
    """
    import numpy as np

    spectrum = np.asarray(spectrum, dtype=np.float64)
    usable = np.asarray(usable, dtype=bool) & np.isfinite(spectrum)
    n = spectrum.size
    x = np.linspace(-1.0, 1.0, n) if n > 1 else np.zeros(1)
    kept = usable.copy()
    scales = [int(k) for k in widths if 1 <= int(k) <= max(n // 8, 1)] or [1]
    # A noiseless (simulated) spectrum has zero spread: then anything above
    # rounding is a line, or the fit would absorb it.
    floor = 1e-9 * float(np.max(np.abs(spectrum[usable]))) if usable.any() else 0.0
    for _ in range(max(int(iterations), 1)):
        if kept.sum() <= order:
            break
        coef = np.polyfit(x[kept], spectrum[kept], order)
        residual = np.where(usable, spectrum - np.polyval(coef, x), 0.0)
        line = np.zeros(n, dtype=bool)
        for k in scales:
            box = np.ones(k)
            counts = np.convolve(usable.astype(float), box, mode="same")
            smooth = np.convolve(residual, box, mode="same") / np.maximum(counts, 1.0)
            # Deviations from the median, not from zero: the first fit still
            # includes the lines, which lifts it and offsets every residual.
            centre = np.median(smooth[kept])
            spread = 1.4826 * np.median(np.abs(smooth[kept] - centre))
            if spread > 0:
                line |= usable & (np.abs(smooth - centre) > clip_sigma * spread)
            elif floor > 0:
                line |= usable & (np.abs(smooth - centre) > floor)
        new_kept = usable & ~line
        if new_kept.sum() <= order or np.array_equal(new_kept, kept):
            break
        kept = new_kept
    line = usable & ~kept
    if grow is None:
        grow = max(2, n // 100)
    if grow > 0 and line.any():
        # "full" and sliced: "same" returns the longer input, which a short
        # spectrum (fewer than 2 * grow + 1 channels) is not.
        line = np.convolve(line, np.ones(2 * grow + 1), mode="full")[grow : grow + n] > 0
    return usable & ~line


def fit_uv_continuum(
    data: np.ndarray,
    weight: np.ndarray,
    flag: np.ndarray,
    line_free: np.ndarray,
    order: int = 1,
) -> np.ndarray:
    """Continuum of each visibility row, fitted over the line-free channels.

    ``data``, ``weight`` and ``flag`` are ``(rows, channels, correlations)``;
    every row and correlation gets a weighted least-squares polynomial of
    ``order`` (0 or 1) in channel, from its unflagged ``line_free`` channels,
    evaluated at every channel: CASA ``uvcontsub``. A row with no usable
    line-free channel gets zero continuum; one whose channels cannot
    constrain a slope gets a constant.
    """
    import numpy as np

    if order not in (0, 1):
        raise ValueError(f"continuum order must be 0 or 1, not {order}")
    n_channels = data.shape[1]
    x = np.linspace(-1.0, 1.0, n_channels)[None, :, None] if n_channels > 1 else np.zeros((1, 1, 1))
    w = np.where(flag | ~np.asarray(line_free, dtype=bool)[None, :, None], 0.0, weight)
    w = np.where(np.isfinite(data), w, 0.0)
    y = np.where(w > 0, data, 0.0)
    s0 = w.sum(axis=1)
    t0 = (w * y).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(s0 > 0, t0 / np.where(s0 > 0, s0, 1.0), 0.0)
        slope = np.zeros_like(mean)
        intercept = mean
        if order == 1:
            s1 = (w * x).sum(axis=1)
            s2 = (w * x * x).sum(axis=1)
            t1 = (w * x * y).sum(axis=1)
            det = s0 * s2 - s1 * s1
            fit = det > 1e-9 * np.maximum(s0 * s2, 1e-300)
            safe = np.where(fit, det, 1.0)
            slope = np.where(fit, (s0 * t1 - s1 * t0) / safe, 0.0)
            intercept = np.where(fit, (t0 * s2 - t1 * s1) / safe, mean)
    return (intercept[:, None, :] + slope[:, None, :] * x).astype(data.dtype)


def _channel_ranges(mask: np.ndarray) -> str:
    """``0~10;20~30``: the runs of True in ``mask``, CASA channel-selection style."""
    import numpy as np

    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(int), [0]))))
    return ";".join(f"{a}~{b - 1}" for a, b in zip(edges[::2], edges[1::2]))


# Samples (rows x channels x correlations) per chunk read from an MS: with
# complex64 data, float64 weights, flags and the fit's temporaries about
# 1 GB. A fixed 20000 rows was over 5 GB for a 3840-channel window.
_CHUNK_SAMPLES = 20_000_000


def rows_for_chunk(n_channels: int, n_correlations: int, samples: int | None = None) -> int:
    """Rows per chunk keeping it near ``samples`` (default ``_CHUNK_SAMPLES``) samples, >= 1."""
    budget = _CHUNK_SAMPLES if samples is None else samples
    return max(int(budget // max(int(n_channels) * int(n_correlations), 1)), 1)


def _has_spectrum_weight(ms) -> bool:
    return (
        "WEIGHT_SPECTRUM" in ms.colnames()
        and ms.nrows() > 0
        and (ms.iscelldefined("WEIGHT_SPECTRUM", 0))
    )


def _read_rows(
    ms, column: str, start: int, n: int, spectrum_weight: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``column``, its weights (per channel) and flags for ``n`` rows from ``start``.

    The flags combine FLAG with FLAG_ROW: a row flagged as a whole counts as
    flagged in every sample.
    """
    import numpy as np

    data = ms.getcol(column, startrow=start, nrow=n)
    flag = ms.getcol("FLAG", startrow=start, nrow=n)
    if "FLAG_ROW" in ms.colnames():
        flag = flag | ms.getcol("FLAG_ROW", startrow=start, nrow=n)[:, None, None]
    if spectrum_weight:
        weight = ms.getcol("WEIGHT_SPECTRUM", startrow=start, nrow=n)
    else:
        weight = np.broadcast_to(
            ms.getcol("WEIGHT", startrow=start, nrow=n)[:, None, :], data.shape
        )
    return data, np.asarray(weight, dtype=np.float64), flag


def _default_rows_per_chunk(ms, column: str) -> int:
    """``rows_for_chunk`` for the shape of ``column``'s cells (one row for an empty MS)."""
    shape = ms.getcell(column, 0).shape if ms.nrows() else (1, 1)
    return rows_for_chunk(shape[0], shape[-1])


def _parallel_hands(n_corr: int) -> list[int]:
    """XX and YY (or RR and LL): the first and last correlation."""
    return [0, n_corr - 1] if n_corr > 1 else [0]


def _source_spectrum(ms, column: str, rows_per_chunk: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Amplitude of the weighted vector average of every row, per channel, and
    which channels have any unflagged data; None for an empty MS."""
    import numpy as np

    n_rows = ms.nrows()
    spectrum_weight = _has_spectrum_weight(ms)
    num = den = None
    for start in range(0, n_rows, rows_per_chunk):
        data, weight, flag = _read_rows(
            ms, column, start, min(rows_per_chunk, n_rows - start), spectrum_weight
        )
        corrs = _parallel_hands(data.shape[2])
        w = np.where(flag[..., corrs] | ~np.isfinite(data[..., corrs]), 0.0, weight[..., corrs])
        d = np.where(w > 0, data[..., corrs], 0.0)
        if num is None:
            num = np.zeros(data.shape[1], dtype=np.complex128)
            den = np.zeros(data.shape[1])
        num += (w * d).sum(axis=(0, 2))
        den += w.sum(axis=(0, 2))
    if num is None:
        return None
    usable = den > 0
    return np.abs(np.where(usable, num / np.where(usable, den, 1.0), 0.0)), usable


def subtract_uv_continuum(
    ms_path: str | os.PathLike[str],
    *,
    order: int = 1,
    clip_sigma: float = 3.0,
    rows_per_chunk: int | None = None,
    min_line_free_fraction: float = 0.2,
) -> dict[str, object]:
    """Split a single-window MS into continuum and line visibilities.

    The line channels are found on the source spectrum: the visibilities of
    every row averaged (vector, weighted, XX and YY) per channel, its
    amplitude fitted by ``line_free_channels``. Each row is then fitted by
    ``fit_uv_continuum`` over the line-free channels, and the fit and the
    data minus the fit are written to ``CONTINUUM_COLUMN`` and
    ``LINE_COLUMN``. The data are CORRECTED_DATA when the MS has it, DATA
    otherwise (WSClean's own choice).

    Channel-by-channel cleaning stops at the threshold in each channel and
    so leaves a faint source's continuum in the residual (uid___A002_
    X10ded83_X6af4 SPW 16: S/N 8 per channel, 12 % of the line-free image
    left); imaged from the fitted continuum, all line-free channels are
    cleaned together, and the cube holds only the lines.

    Nothing is written when fewer than ``min_line_free_fraction`` of the
    usable channels are line-free (a line forest or a window filled by one
    line); the result then carries ``skipped`` with the reason.

    Rows are read ``rows_per_chunk`` at a time, by default as many as keep a
    chunk near ``_CHUNK_SAMPLES`` samples; rows flagged with FLAG_ROW are
    left out like flagged samples.
    """
    from casacore.tables import makecoldesc, table

    started = time.time()
    ms = table(str(ms_path), readonly=False, ack=False)
    try:
        source = "CORRECTED_DATA" if "CORRECTED_DATA" in ms.colnames() else "DATA"
        n_rows = ms.nrows()
        spectrum_weight = _has_spectrum_weight(ms)
        if rows_per_chunk is None:
            rows_per_chunk = _default_rows_per_chunk(ms, source)
        measured = _source_spectrum(ms, source, rows_per_chunk)
        if measured is None:
            return {"skipped": "the MS has no rows", "source_column": source}
        spectrum, usable = measured
        line_free = line_free_channels(spectrum, usable, order=order, clip_sigma=clip_sigma)
        result: dict[str, object] = {
            "source_column": source,
            "channels": int(usable.size),
            "usable_channels": int(usable.sum()),
            "line_free_channels": int(line_free.sum()),
            "line_channels": int((usable & ~line_free).sum()),
            "line_ranges": _channel_ranges(usable & ~line_free),
            "order": int(order),
        }
        if line_free.sum() < max(order + 2, min_line_free_fraction * usable.sum()):
            result["skipped"] = (
                f"only {int(line_free.sum())} of {int(usable.sum())} usable channels are line-free"
            )
            return result
        for name in (CONTINUUM_COLUMN, LINE_COLUMN):
            if name in ms.colnames():
                ms.removecols(name)
            ms.addcols(makecoldesc(name, ms.getcoldesc(source)))
        for start in range(0, n_rows, rows_per_chunk):
            n = min(rows_per_chunk, n_rows - start)
            data, weight, flag = _read_rows(ms, source, start, n, spectrum_weight)
            continuum = fit_uv_continuum(data, weight, flag, line_free, order)
            ms.putcol(CONTINUUM_COLUMN, continuum, startrow=start, nrow=n)
            ms.putcol(LINE_COLUMN, data - continuum, startrow=start, nrow=n)
        result["seconds"] = round(time.time() - started, 1)
        return result
    finally:
        ms.close()


def subtract_uv_continuum_subprocess(
    ms_path: str | os.PathLike[str],
    *,
    order: int = 1,
    timeout: float | None = None,
    log_path: str | os.PathLike[str] | None = None,
) -> dict[str, object]:
    """Run ``subtract_uv_continuum`` in a child process, for the reason
    ``extract_single_window_ms_subprocess`` gives: casacore holds the GIL."""
    return call_in_child(
        "almasim.services.imaging.archive_imaging:subtract_uv_continuum",
        {"ms_path": str(ms_path), "order": int(order)},
        what="continuum subtraction",
        env=wsclean_environment(1),
        timeout=timeout,
        timeout_noun="task timeout",
        log_path=log_path,
    )


def drop_corrected_data(ms_path: str) -> dict[str, object]:
    """Remove CORRECTED_DATA, so imaging reads DATA again (an undone self-calibration)."""
    from casacore.tables import table

    ms = table(str(ms_path), readonly=False, ack=False)
    try:
        dropped = "CORRECTED_DATA" in ms.colnames()
        if dropped:
            ms.removecols("CORRECTED_DATA")
        return {"dropped": dropped}
    finally:
        ms.close()


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


def selfcal_prepare(ms_path: str, rows_per_chunk: int | None = None) -> dict[str, object]:
    """CASA selections for self-calibrating ``ms_path``: the line-free channels of
    DATA's source spectrum (``spw``) and the antenna with the most
    cross-correlation rows holding an unflagged line-free parallel-hand sample
    (``refant``); a row unflagged as a whole can still have every such sample
    flagged."""
    import numpy as np
    from casacore.tables import table

    ms = table(str(ms_path), ack=False)
    try:
        if rows_per_chunk is None:
            rows_per_chunk = _default_rows_per_chunk(ms, "DATA")
        measured = _source_spectrum(ms, "DATA", rows_per_chunk)
        if measured is None:
            raise RuntimeError("the MS has no rows")
        spectrum, usable = measured
        line_free = line_free_channels(spectrum, usable)
        if not line_free.any():
            line_free = usable
        ant1, ant2 = ms.getcol("ANTENNA1"), ms.getcol("ANTENNA2")
        has_data = np.zeros(ms.nrows(), dtype=bool)
        spectrum_weight = _has_spectrum_weight(ms)
        for start in range(0, ms.nrows(), rows_per_chunk):
            n = min(rows_per_chunk, ms.nrows() - start)
            _, _, flag = _read_rows(ms, "DATA", start, n, spectrum_weight)
            corrs = _parallel_hands(flag.shape[2])
            has_data[start : start + n] = (~flag[:, line_free][..., corrs]).any(axis=(1, 2))
        good = has_data & (ant1 != ant2)
    finally:
        ms.close()
    names = table(f"{ms_path}/ANTENNA", ack=False).getcol("NAME")
    counts = np.bincount(ant1[good], minlength=len(names)) + np.bincount(
        ant2[good], minlength=len(names)
    )
    return {
        "spw": "0:" + _channel_ranges(line_free),
        "refant": str(names[int(np.argmax(counts))]),
        "line_free_channels": int(line_free.sum()),
    }


def adopt_corrected_data(ms_path: str, rows_per_chunk: int | None = None) -> dict[str, object]:
    """Make an incoming CORRECTED_DATA the task copy's DATA, and drop it.

    Self-calibration solves from DATA, writes CORRECTED_DATA and, when a
    round is rejected, drops it: an MS that arrived with its calibration in
    CORRECTED_DATA would then be solved from, and fall back to, the
    uncalibrated DATA. The single-window MS is the task's own copy.
    """
    from casacore.tables import table

    ms = table(str(ms_path), readonly=False, ack=False)
    try:
        if "CORRECTED_DATA" not in ms.colnames():
            return {"adopted": False}
        if rows_per_chunk is None:
            rows_per_chunk = _default_rows_per_chunk(ms, "CORRECTED_DATA")
        for start in range(0, ms.nrows(), rows_per_chunk):
            n = min(rows_per_chunk, ms.nrows() - start)
            ms.putcol(
                "DATA", ms.getcol("CORRECTED_DATA", startrow=start, nrow=n), startrow=start, nrow=n
            )
        ms.removecols("CORRECTED_DATA")
        return {"adopted": True}
    finally:
        ms.close()


def _configure_casa(casa_data: str, log_file: str) -> Path:
    """Point CASA at ``casa_data`` and ``log_file`` with a private workspace (removed at exit)."""
    import atexit
    import tempfile

    from almasim.services.archive.unpack_ms import configure_casa_environment

    workspace = Path(tempfile.mkdtemp(prefix="almasim-selfcal-casa-"))
    atexit.register(shutil.rmtree, workspace, True)
    configure_casa_environment(workspace, casa_data, workspace_root=workspace, log_file=log_file)
    return workspace


def casa_selfcal_solve(
    ms_path: str,
    caltable: str,
    *,
    solint: str,
    spw: str,
    refant: str,
    casa_data: str,
    log_file: str,
    min_snr: float = 3.0,
    calmode: str = "p",
    pre_apply: Sequence[str] = (),
) -> dict[str, object]:
    """CASA ``gaincal`` of DATA against MODEL_DATA: one solution per antenna for
    both polarizations (gaintype T), over ``spw`` (line-free).

    ``calmode`` is ``p`` (phase) or ``ap`` (amplitude and phase); an ``ap``
    solve is made on top of the ``pre_apply`` tables (the kept phase
    solution) and normalised (``solnorm``) so the flux scale stays put.
    """
    import numpy as np

    _configure_casa(casa_data, log_file)
    from casatasks import gaincal
    from casatools import table

    started = time.time()
    gaincal(
        vis=ms_path,
        caltable=caltable,
        solint=solint,
        spw=spw,
        refant=refant,
        calmode=calmode,
        gaintype="T",
        minsnr=min_snr,
        minblperant=4,
        solnorm=calmode == "ap",
        gaintable=list(pre_apply),
        interp=["linear"] * len(pre_apply),
    )
    tb = table()
    tb.open(caltable)
    try:
        flags = tb.getcol("FLAG")
        gains = tb.getcol("CPARAM")
        times = tb.getcol("TIME")
    finally:
        tb.close()
    good = ~flags
    phases = np.degrees(np.abs(np.angle(gains[good]))) if good.any() else np.zeros(1)
    amplitudes = np.abs(gains[good]) if good.any() else np.ones(1)
    return {
        "calmode": calmode,
        "median_amp_dev": round(float(np.median(np.abs(amplitudes - 1.0))), 3),
        "solint": solint,
        "intervals": int(np.unique(times).size),
        "solutions": int(good.sum()),
        "possible": int(flags.size),
        "median_phase_deg": round(float(np.median(phases)), 1),
        "seconds": round(time.time() - started, 1),
    }


def casa_selfcal_apply(
    ms_path: str, caltables: Sequence[str], *, casa_data: str, log_file: str
) -> dict:
    """CASA ``applycal`` of ``caltables`` into CORRECTED_DATA. ``calonly``: data
    whose solution failed stay as they were, unflagged; weights unchanged."""
    _configure_casa(casa_data, log_file)
    from casatasks import applycal

    applycal(
        vis=ms_path,
        gaintable=list(caltables),
        interp=["linear"] * len(caltables),
        calwt=[False] * len(caltables),
        applymode="calonly",
        flagbackup=False,
    )
    return {"applied": True}


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


def _selfcal_command(cmd: list[str], out: Path, name: str) -> list[str]:
    """The continuum command imaging into ``out/name`` and writing MODEL_DATA."""
    result = list(cmd)
    result[result.index("-name") + 1] = str(out / name)
    return [
        "-update-model-required" if part == "-no-update-model-required" else part for part in result
    ]


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


def export_gain_table(caltable: str, csv_path: str) -> dict[str, object]:
    """A CASA gain table as CSV: one row per solution (time, antenna, polarization),
    with amplitude, phase in degrees and the flag. Read with casacore, no CASA."""
    import csv

    import numpy as np
    from casacore.tables import table

    tb = table(str(caltable), ack=False)
    try:
        times = tb.getcol("TIME")
        antennas = tb.getcol("ANTENNA1")
        gains = tb.getcol("CPARAM")  # (rows, channels, polarizations)
        flags = tb.getcol("FLAG")
    finally:
        tb.close()
    names = table(f"{caltable}/ANTENNA", ack=False).getcol("NAME")
    rows = 0
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["time_mjd_s", "antenna", "polarization", "amplitude", "phase_deg", "flagged"]
        )
        for row in range(len(times)):
            antenna = int(antennas[row])
            for pol in range(gains.shape[2]):
                gain = complex(gains[row, 0, pol])
                writer.writerow(
                    [
                        f"{times[row]:.3f}",
                        str(names[antenna]) if antenna < len(names) else antenna,
                        pol,
                        f"{abs(gain):.6g}",
                        f"{np.degrees(np.angle(gain)):.4g}",
                        bool(flags[row, 0, pol]),
                    ]
                )
                rows += 1
    return {"rows": rows}


def _line_free_from_ranges(ranges: str, n_channels: int) -> np.ndarray:
    """Line-free channels from ``_channel_ranges`` text ("" when there are no lines)."""
    import numpy as np

    line_free = np.ones(n_channels, dtype=bool)
    for part in filter(None, ranges.split(";")):
        start, end = map(int, part.split("~"))
        line_free[start : end + 1] = False
    return line_free


def selfcal_correction_columns(
    ms_path: str,
    *,
    line_ranges: str | None = None,
    order: int = 1,
    rows_per_chunk: int | None = None,
) -> dict[str, object]:
    """Write what self-calibration removed, DATA - CORRECTED_DATA, to new columns.

    Without ``line_ranges`` it goes whole to ``SELFCAL_COLUMN``. With them (the
    visibility-plane continuum subtraction's line channels) it is split the
    way the data were: the per-row continuum fit over the same line-free
    channels and order to ``SELFCAL_CONTINUUM_COLUMN``, the rest to
    ``SELFCAL_LINE_COLUMN``. The fit is linear in the data, so each part of
    the correction plus the matching part of the self-calibrated data is that
    part of DATA.
    """
    from casacore.tables import makecoldesc, table

    ms = table(str(ms_path), readonly=False, ack=False)
    try:
        if "CORRECTED_DATA" not in ms.colnames():
            raise RuntimeError("the MS has no CORRECTED_DATA: nothing was self-calibrated")
        n_rows = ms.nrows()
        if rows_per_chunk is None:
            rows_per_chunk = _default_rows_per_chunk(ms, "DATA")
        split = line_ranges is not None
        names = [SELFCAL_CONTINUUM_COLUMN, SELFCAL_LINE_COLUMN] if split else [SELFCAL_COLUMN]
        for name in names:
            if name in ms.colnames():
                ms.removecols(name)
            ms.addcols(makecoldesc(name, ms.getcoldesc("DATA")))
        spectrum_weight = _has_spectrum_weight(ms)
        line_free = None
        for start in range(0, n_rows, rows_per_chunk):
            n = min(rows_per_chunk, n_rows - start)
            data, weight, flag = _read_rows(ms, "DATA", start, n, spectrum_weight)
            correction = (data - ms.getcol("CORRECTED_DATA", startrow=start, nrow=n)).astype(
                data.dtype
            )
            if not split:
                ms.putcol(SELFCAL_COLUMN, correction, startrow=start, nrow=n)
                continue
            if line_free is None:
                line_free = _line_free_from_ranges(str(line_ranges), data.shape[1])
            continuum = fit_uv_continuum(correction, weight, flag, line_free, order)
            ms.putcol(SELFCAL_CONTINUUM_COLUMN, continuum, startrow=start, nrow=n)
            ms.putcol(
                SELFCAL_LINE_COLUMN,
                (correction - continuum).astype(data.dtype),
                startrow=start,
                nrow=n,
            )
        return {"columns": names, "rows": int(n_rows)}
    finally:
        ms.close()


def _dirty_command(cmd: Sequence[str], out: Path, name: str, column: str) -> list[str]:
    """``cmd`` turned into a dirty image of ``column`` into ``out/name``: same grid,
    weighting and channels, no cleaning, no PSF, the model column left alone."""
    parts = [str(part) for part in cmd]
    with_value = {"-auto-mask", "-auto-threshold", "-fits-mask", "-data-column", "-name", "-niter"}
    alone = {"-continue", "-make-psf", "-update-model-required", "-no-update-model-required"}
    kept: list[str] = []
    index = 1
    while index < len(parts) - 1:
        if parts[index] in with_value:
            index += 2
            continue
        if parts[index] not in alone:
            kept.append(parts[index])
        index += 1
    return [
        parts[0],
        "-name",
        str(out / name),
        *kept,
        "-niter",
        "0",
        "-no-update-model-required",
        "-data-column",
        column,
        parts[-1],
    ]


def add_images(first: Path, second: Path, target: Path, history: str) -> None:
    """Write ``first + second`` to ``target`` plane by plane, with ``first``'s
    header and extensions (a cube's BEAMS table included)."""
    import numpy as np
    from astropy.io import fits

    partial = target.with_name(target.name + ".partial")
    shutil.copyfile(first, partial)
    with (
        fits.open(partial, mode="update", memmap=True) as hdus,
        fits.open(second, memmap=True) as others,
    ):
        data, other = hdus[0].data, others[0].data
        if data.shape != other.shape:
            raise RuntimeError(f"{first.name} {data.shape} and {second.name} {other.shape} differ")
        planes = data.reshape(-1, *data.shape[-2:])
        other_planes = other.reshape(-1, *other.shape[-2:])
        for index in range(planes.shape[0]):
            planes[index] = np.asarray(planes[index], dtype=np.float64) + other_planes[index]
        hdus[0].header["HISTORY"] = history
    partial.replace(target)


def _single_image(out: Path, name: str) -> Path | None:
    """The image a ``-niter 0`` WSClean run wrote: ``-dirty`` or ``-image``."""
    return next(
        (
            path
            for path in (out / f"{name}-dirty.fits", out / f"{name}-image.fits")
            if path.is_file()
        ),
        None,
    )


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


def _with_data_column(cmd: list[str], column: str) -> list[str]:
    """``cmd`` reading ``column``: ``-data-column`` goes before the MS, its last argument."""
    return [*cmd[:-1], "-data-column", column, cmd[-1]]


def line_mask_from_channels(
    output_dir: str | os.PathLike[str],
    mask_path: str | os.PathLike[str],
    *,
    prefix: str = CUBE_NAME,
    sigma: float = 3.0,
) -> int:
    """Write the cube's moment-0 mask from WSClean's per-channel images; its pixel count.

    The channel images ``<prefix>-NNNN-image.fits`` are summed, the sum is
    thresholded at ``sigma`` times its robust noise (1.4826 x MAD) and the
    result grown by one beam (the median BMAJ of the channels). 0 (and no
    file) when the run left no channel images or nothing is above it.
    """
    import numpy as np
    from astropy.io import fits

    out = Path(output_dir)
    # A mask left by an earlier run must not survive a run that builds none.
    Path(mask_path).unlink(missing_ok=True)
    paths = sorted(
        path
        for path in out.glob(f"{prefix}-[0-9][0-9][0-9][0-9]-image.fits")
        if _CHANNEL_FILE.match(path.name)
    )
    if not paths:
        return 0
    moment = None
    beams = []
    for path in paths:
        with fits.open(path) as hdus:
            plane = np.nan_to_num(np.asarray(hdus[0].data, dtype=np.float64).squeeze())
            beams.append(float(hdus[0].header.get("BMAJ", 0.0)))
            header = hdus[0].header.copy()
        moment = plane if moment is None else moment + plane
    assert moment is not None
    noise = 1.4826 * np.median(np.abs(moment - np.median(moment)))
    mask = moment > sigma * noise if noise > 0 else np.zeros(moment.shape, bool)
    if not mask.any():
        return 0
    cell = abs(float(header.get("CDELT2", 0.0)))
    beam_pixels = float(np.median(beams)) / cell if cell else 0.0
    if beam_pixels > 0:
        radius = int(np.ceil(beam_pixels))
        yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
        disk = (np.hypot(yy, xx) <= beam_pixels).astype(float)
        grown = np.fft.irfft2(
            np.fft.rfft2(mask.astype(float), s=[n + 2 * radius for n in mask.shape])
            * np.fft.rfft2(disk, s=[n + 2 * radius for n in mask.shape]),
            s=[n + 2 * radius for n in mask.shape],
        )
        mask = grown[radius : radius + mask.shape[0], radius : radius + mask.shape[1]] > 0.5
    fits.PrimaryHDU(mask.astype(np.float32)[None, None], header=header).writeto(
        mask_path, overwrite=True
    )
    return int(mask.sum())


def _line_mask_pass(cube_cmd: Sequence[str], out: Path, log) -> list[str] | None:
    """The cube's second pass: ``-continue`` inside the moment-0 mask of the first.

    Each channel's ``-auto-mask`` is built from that channel alone, so the
    faint parts of a line, below 3 sigma in every channel but clear in their
    sum, are never cleaned. uid___A002_X10ded83_X6af4 SPW 16 (1121 channels
    of its line): 34 % of the line within its moment-0 mask left in the
    residual; continuing inside that mask down to the same 1 sigma left 12 %,
    with 23 % more model flux and the per-channel noise down from 135 to 107
    mJy; going below 1 sigma there diverged (2026-10-18).
    """
    mask_path = out / LINE_MASK_FILENAME
    pixels = line_mask_from_channels(out, mask_path)
    if pixels:
        blanked = _zero_non_finite_models(out)
        if blanked:
            log.write(f"# cube line mask: {blanked} channel model(s) were NaN (no data); zeroed\n")
    if not pixels:
        log.write("# cube line mask: nothing above 3 sigma in the moment-0 map; one pass\n")
        log.flush()
        return None
    log.write(f"# cube line mask: {pixels} pixels; second pass inside it\n")
    log.flush()
    second: list[str] = []
    parts = list(cube_cmd)
    index = 0
    while index < len(parts):
        if parts[index] == "-auto-mask":
            index += 2  # the fixed mask replaces the per-channel one
            continue
        second.append(parts[index])
        index += 1
    return [*second[:-1], "-fits-mask", str(mask_path), "-continue", second[-1]]


def _zero_non_finite_models(out: Path, prefix: str = CUBE_NAME) -> int:
    """Set the non-finite pixels of the per-channel model images to 0; how many files changed.

    WSClean writes a NaN model for a channel with no unflagged data, and
    ``-continue`` refuses to predict from it ("The input image contains
    non-finite values"): uid___A002_X11aaa01_X32ce SPW 16 has 128 such
    channels (2026-10-18).
    """
    import numpy as np
    from astropy.io import fits

    changed = 0
    for path in sorted(out.glob(f"{prefix}-[0-9][0-9][0-9][0-9]-model.fits")):
        with fits.open(path, mode="update") as hdus:
            data = hdus[0].data
            bad = ~np.isfinite(data)
            if bad.any():
                data[bad] = 0.0
                changed += 1
    return changed


def continuum_restart_commands(cmd: Sequence[str]) -> list[list[str]]:
    """The continuum ``cmd`` as two passes: one writing MODEL_DATA, then ``-continue``.

    WSClean's auto-mask is built in the first major cycle, at ``-auto-mask``
    times the noise it measures then; a bright source's sidelobes inflate that
    noise, so the mask misses its fainter parts and later cycles, which clean
    only inside it, never reach them. uid___A002_X10ded83_X6af4 SPW 16: noise
    91 mJy at mask creation against 26 mJy at the end, a 167 mJy feature 0.7
    beam off the peak left uncleaned, 16 % of the flux within a beam in the
    residual. The second pass starts from the true residual, measures the
    noise again and builds a new mask: residual 3 %, model flux +20 %
    (2026-10-18). The model visibilities go into the task's own copy of the MS.
    """
    first = [
        "-update-model-required" if part == "-no-update-model-required" else str(part)
        for part in cmd
    ]
    if "-update-model-required" not in first:
        first.insert(len(first) - 1, "-update-model-required")
    return [first, [*first[:-1], "-continue", first[-1]]]


def _remove_reorder_files(out: Path) -> None:
    """Drop WSClean's ``*-part*.tmp`` reorder files a crashed attempt left behind."""
    for leftover in out.glob("*.tmp"):
        try:
            leftover.unlink()
        except OSError:
            pass


def _run_once(
    cmd: list[str],
    out: Path,
    log,
    env: dict[str, str],
    timeout: float | None,
    tail: deque[str],
    fail: Callable[..., RuntimeError],
) -> tuple[int, bool]:
    """Run WSClean once, streaming to ``log``. Returns ``(returncode, timed_out)``."""
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
            cwd=str(out),
            env=env,
        )
    except OSError as exc:
        message = f"Could not start {cmd[0]}: {exc}"
        log.write(message + "\n")
        raise fail(message) from exc

    timed_out = threading.Event()
    killer: threading.Timer | None = None
    if timeout is not None and timeout > 0:

        def _kill() -> None:
            timed_out.set()
            process.kill()

        killer = threading.Timer(timeout, _kill)
        killer.daemon = True
        killer.start()

    assert process.stdout is not None
    try:
        for line in process.stdout:
            log.write(line)
            log.flush()
            tail.append(line.rstrip("\n"))
        returncode = process.wait()
    finally:
        if killer is not None:
            killer.cancel()
    return returncode, timed_out.is_set()
