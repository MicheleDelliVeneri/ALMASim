"""Run WSClean with retries, a live log and failure capture.

Moved out of archive_imaging.py (MADIV-28 module split).
"""

from __future__ import annotations

import shlex
import subprocess
import threading
from collections import deque
from pathlib import Path
from typing import Callable

from almasim.services.runtime.child import error_line

_error_line = error_line


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


def _remove_reorder_files(out: Path) -> None:
    """Drop WSClean's ``*-part*.tmp`` reorder files a crashed attempt left behind."""
    for leftover in out.glob("*.tmp"):
        try:
            leftover.unlink()
        except OSError:
            pass
