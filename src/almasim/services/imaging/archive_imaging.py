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
import shlex
import subprocess
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

__all__ = [
    "IMAGE_FILENAME",
    "SINGLE_WINDOW_PLACEHOLDER",
    "extract_single_window_ms",
    "imaging_failure_marker_path",
    "imaging_log_path",
    "imaging_marker_path",
    "is_imaging_complete",
    "run_wsclean_task",
    "write_imaging_failure_marker",
    "write_imaging_marker",
    "wsclean_environment",
]

# WSClean is invoked with ``-name <output_dir>/wsclean`` and a single SPW, so
# the restored image is always this file.
IMAGE_FILENAME = "wsclean-image.fits"

# Placeholder in a task command for the per-task single-window MS that the
# wrapper extracts on the worker (see ``extract_single_window_ms``).
SINGLE_WINDOW_PLACEHOLDER = "__SINGLE_WINDOW_MS__"

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


def is_imaging_complete(output_dir: str | os.PathLike[str]) -> bool:
    """True when the task has both its marker and the restored image."""
    out = Path(output_dir)
    return imaging_marker_path(out).is_file() and (out / IMAGE_FILENAME).is_file()


def write_imaging_marker(
    output_dir: str | os.PathLike[str],
    *,
    ms_path: str,
    spw: int,
    command: Sequence[str] | None = None,
    seconds: float | None = None,
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
    if command is not None:
        payload["command"] = shlex.join(str(part) for part in command)
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


def extract_single_window_ms(
    ms_path: str | os.PathLike[str],
    spw: int,
    field_ids: Sequence[int] | None,
    out_path: str | os.PathLike[str],
) -> dict[str, object]:
    """Copy one spectral window (and optionally some fields) into a new MS.

    WSClean 3.7 reads out of bounds in its reordering step when an MS holds
    several spectral windows and ``-spws`` selects one (an ALMA split MS
    lists 60 windows, 4 of them populated); the crash is heap-layout
    dependent, so about half the tasks die. Handing WSClean an MS with a
    single spectral window lets it use its contiguous reader (``-no-reorder``)
    and skip that code entirely. The copy is a deep copy of the selected rows
    with all subtables; DATA_DESCRIPTION and SPECTRAL_WINDOW are collapsed to
    the one window (renumbered 0) so WSClean sees exactly one band.
    """
    import shutil

    from casacore.tables import table, taql

    src = str(ms_path)
    out = Path(out_path)
    shutil.rmtree(out, ignore_errors=True)
    dd_table = table(f"{src}/DATA_DESCRIPTION", ack=False)
    spw_of_dd = [int(x) for x in dd_table.getcol("SPECTRAL_WINDOW_ID")]
    pol_of_dd = [int(x) for x in dd_table.getcol("POLARIZATION_ID")]
    dd_table.close()
    dd_ids = [i for i, s in enumerate(spw_of_dd) if s == int(spw)]
    if not dd_ids:
        raise RuntimeError(f"Spectral window {spw} has no DATA_DESCRIPTION row in {src}")
    where = f"DATA_DESC_ID in [{','.join(str(i) for i in dd_ids)}]"
    if field_ids:
        where += f" and FIELD_ID in [{','.join(str(int(f)) for f in field_ids)}]"
    selection = taql(f"select from {src} where {where}")
    n_rows = int(selection.nrows())
    if n_rows == 0:
        selection.close()
        raise RuntimeError(f"No visibilities for spectral window {spw} fields {field_ids} in {src}")
    selection.copy(str(out), deep=True)
    selection.close()

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
    return {"rows": n_rows, "bytes": size, "data_desc_ids": dd_ids}


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
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = imaging_log_path(out)
    imaging_marker_path(out).unlink(missing_ok=True)
    imaging_failure_marker_path(out).unlink(missing_ok=True)
    (out / IMAGE_FILENAME).unlink(missing_ok=True)

    cmd = [str(part) for part in command]
    single_window_ms: Path | None = None
    if single_window:
        # Extract the window (and fields) into a per-task MS so WSClean never
        # reorders; ``SINGLE_WINDOW_PLACEHOLDER`` in the command is replaced.
        single_window_ms = out / f"spw{int(spw)}.single.ms"
        try:
            extract_single_window_ms(ms_path, spw, field_ids, single_window_ms)
        except Exception as exc:
            error = f"Could not extract spectral window {spw} into a single-window MS: {exc}"
            write_imaging_failure_marker(out, ms_path=ms_path, spw=spw, error=error)
            raise RuntimeError(error) from exc
        cmd = [str(single_window_ms) if part == SINGLE_WINDOW_PLACEHOLDER else part for part in cmd]
    scratch_dir: Path | None = None
    if scratch_root:
        # Per-task scratch for WSClean's reordered visibilities, created here
        # on the worker and removed whatever happens. ``__SCRATCH__`` in the
        # command (the -temp-dir value) is replaced by it.
        import tempfile

        Path(scratch_root).mkdir(parents=True, exist_ok=True)
        scratch_dir = Path(tempfile.mkdtemp(prefix=f"{out.name}-", dir=scratch_root))
        cmd = [str(scratch_dir) if part == "__SCRATCH__" else part for part in cmd]
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
            cmd=cmd,
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
        )
    finally:
        import shutil

        if scratch_dir is not None:
            shutil.rmtree(scratch_dir, ignore_errors=True)
        if single_window_ms is not None:
            shutil.rmtree(single_window_ms, ignore_errors=True)


def _run_and_mark(
    *,
    cmd: list[str],
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
) -> dict[str, object]:
    attempts = 0
    with log_path.open("w", encoding="utf-8") as log:
        log.write("# " + shlex.join(cmd) + "\n")
        log.flush()
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

    elapsed = time.time() - started
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
    image = out / IMAGE_FILENAME
    if not image.is_file():
        raise fail(f"WSClean exited 0 but wrote no {IMAGE_FILENAME} under {out}", returncode)

    write_imaging_marker(out, ms_path=ms_path, spw=spw, command=cmd, seconds=elapsed)
    return {
        "ms": ms_path,
        "spw": int(spw),
        "output": str(out),
        "image": str(image),
        "seconds": round(elapsed, 1),
        "attempts": attempts,
    }


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
