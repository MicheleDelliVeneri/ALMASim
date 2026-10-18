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
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Mapping, Sequence

if TYPE_CHECKING:
    import numpy as np

__all__ = [
    "CONTSUB_FILENAME",
    "CUBE_IMAGE_FILENAME",
    "CUBE_NAME",
    "DEFAULT_MAX_WEIGHT",
    "IMAGE_FILENAME",
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
    "continuum_restart_commands",
    "subtract_continuum",
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


def _error_line(lines: Sequence[str]) -> str:
    """The most recent line that looks like the reason WSClean died."""
    for line in reversed(lines):
        lower = line.lower()
        if "what():" in line or "exception" in lower or "error" in lower:
            return line.strip()
    return ""


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


_EXTRACT_CHILD = (
    "import json, sys\n"
    "from almasim.services.imaging.archive_imaging import extract_single_window_ms\n"
    "args = json.loads(sys.argv[1])\n"
    "result = extract_single_window_ms(args['ms'], args['spw'], args['fields'], args['out'], "
    "max_weight=args.get('max_weight'))\n"
    "sys.stdout.write(json.dumps(result))\n"
)


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
    args = json.dumps(
        {
            "ms": str(ms_path),
            "spw": int(spw),
            "fields": [int(f) for f in field_ids] if field_ids else None,
            "out": str(out_path),
            "max_weight": float(max_weight) if max_weight else None,
        }
    )
    started = time.time()
    for attempt in range(1, max(int(attempts), 1) + 1):
        try:
            proc = subprocess.run(
                [sys.executable, "-c", _EXTRACT_CHILD, args],
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout,
                env=wsclean_environment(1),
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"extraction killed after exceeding the {timeout:.0f}s task timeout"
            ) from exc
        if log_path is not None and proc.stderr.strip():
            with Path(log_path).open("a", encoding="utf-8") as log:
                log.write(proc.stderr if proc.stderr.endswith("\n") else proc.stderr + "\n")
        if proc.returncode == 0:
            break
        if _LOCK_FILE_ERROR in proc.stderr and attempt < attempts:
            message = (
                f"extraction attempt {attempt}/{attempts} hit a casacore lock-file error; "
                f"retrying in a new process in {retry_seconds:g} s\n"
            )
            if log_path is not None:
                with Path(log_path).open("a", encoding="utf-8") as log:
                    log.write(message)
            time.sleep(retry_seconds)
            continue
        reason = _error_line([line for line in proc.stderr.splitlines() if line.strip()])
        detail = f": {reason}" if reason else ""
        raise RuntimeError(f"extraction exited with return code {proc.returncode}{detail}")
    try:
        result = json.loads(proc.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError) as exc:
        raise RuntimeError("extraction wrote no result") from exc
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
    continuum_restart: bool = False,
    cube_line_mask: bool = False,
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
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = imaging_log_path(out)
    commands = [[str(part) for part in c] for c in (command, cube_command or []) if c]
    if not commands:
        raise ValueError("run_wsclean_task needs a continuum or a cube command")
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
    imaging_marker_path(out).unlink(missing_ok=True)
    imaging_failure_marker_path(out).unlink(missing_ok=True)
    for name in required:
        (out / name).unlink(missing_ok=True)
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
    missing = [name for name in required if not (out / name).is_file()]
    if missing:
        raise fail(f"WSClean exited 0 but wrote no {', '.join(missing)} under {out}", 0)

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
