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
from typing import Mapping, Sequence

__all__ = [
    "IMAGE_FILENAME",
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


def run_wsclean_task(
    *,
    command: Sequence[str],
    output_dir: str,
    ms_path: str,
    spw: int,
    threads: int,
    timeout: float | None = None,
) -> dict[str, object]:
    """Run one WSClean task in a subprocess and leave a marker whatever happens.

    Runs on the worker. Output is streamed to ``SPW-<n>.log`` as it arrives.
    A non-zero exit, a signal, a timeout, a binary that cannot start, or a
    clean exit that produced no ``wsclean-image.fits`` all end in a ``.failed``
    marker and a ``RuntimeError`` so the driver counts the task as failed;
    success ends in a ``.done`` marker.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = imaging_log_path(out)
    imaging_marker_path(out).unlink(missing_ok=True)
    imaging_failure_marker_path(out).unlink(missing_ok=True)
    (out / IMAGE_FILENAME).unlink(missing_ok=True)

    cmd = [str(part) for part in command]
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

    with log_path.open("w", encoding="utf-8") as log:
        log.write("# " + shlex.join(cmd) + "\n")
        log.flush()
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
            raise _fail(message) from exc

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

    elapsed = time.time() - started
    if timed_out.is_set():
        raise _fail(f"WSClean killed after exceeding the {timeout:.0f}s task timeout", returncode)
    if returncode != 0:
        error = f"WSClean exited with return code {returncode}"
        reason = _error_line(list(tail))
        if reason:
            error = f"{error}: {reason}"
        raise _fail(error, returncode)
    image = out / IMAGE_FILENAME
    if not image.is_file():
        raise _fail(f"WSClean exited 0 but wrote no {IMAGE_FILENAME} under {out}", returncode)

    write_imaging_marker(out, ms_path=ms_path, spw=spw, command=cmd, seconds=elapsed)
    return {
        "ms": ms_path,
        "spw": int(spw),
        "output": str(out),
        "image": str(image),
        "seconds": round(elapsed, 1),
    }
