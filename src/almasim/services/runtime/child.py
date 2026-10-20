"""Run a library function in a fresh Python process (MADIV-28 T2.4).

casacore and CASA hold the GIL for the whole of a long call, which freezes a
Dask worker's event loop until the scheduler drops it, and python-casacore and
casatools must not share a process. So those calls run in a child: the
function is named as ``"package.module:function"``, its keyword arguments and
its result travel as JSON, and its stderr is appended to the task log.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

_CHILD = (
    "import importlib, json, sys\n"
    "args = json.loads(sys.argv[1])\n"
    "module, _, name = args['target'].partition(':')\n"
    "result = getattr(importlib.import_module(module), name)(**args['kwargs'])\n"
    "sys.stdout.write(json.dumps(result))\n"
)


class ChildFailed(RuntimeError):
    """The child exited with a non-zero code; ``stderr`` is what it printed."""

    def __init__(self, message: str, *, returncode: int, stderr: str) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stderr = stderr


def error_line(lines: Sequence[str]) -> str:
    """The most recent line that looks like the reason a child process died."""
    for line in reversed(lines):
        lower = line.lower()
        if "what():" in line or "exception" in lower or "error" in lower:
            return line.strip()
    return ""


def append_to_log(log_path: str | os.PathLike[str] | None, text: str) -> None:
    """Append ``text`` to ``log_path`` (if any), ending it with a newline."""
    if log_path is None or not text.strip():
        return
    with Path(log_path).open("a", encoding="utf-8") as log:
        log.write(text if text.endswith("\n") else text + "\n")


def call_in_child(
    target: str,
    kwargs: Mapping[str, object],
    *,
    what: str,
    env: Mapping[str, str],
    timeout: float | None = None,
    timeout_noun: str = "timeout",
    log_path: str | os.PathLike[str] | None = None,
) -> Any:
    """``target(**kwargs)`` in a new Python process; returns its JSON result.

    ``what`` names the step in error messages. A non-zero exit raises
    :class:`ChildFailed` (with the child's stderr, so a caller can decide to
    retry); a timeout or a missing result raises ``RuntimeError``.
    """
    args = json.dumps({"target": target, "kwargs": dict(kwargs)})
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _CHILD, args],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            env=dict(env),
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"{what} killed after exceeding the {timeout:.0f}s {timeout_noun}"
        ) from exc
    append_to_log(log_path, proc.stderr)
    if proc.returncode != 0:
        reason = error_line([line for line in proc.stderr.splitlines() if line.strip()])
        raise ChildFailed(
            f"{what} exited with return code {proc.returncode}" + (f": {reason}" if reason else ""),
            returncode=proc.returncode,
            stderr=proc.stderr,
        )
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError) as exc:
        raise RuntimeError(f"{what} wrote no result") from exc


def run_stage_subprocess(
    cmd: Sequence[str],
    log_path: str | os.PathLike[str],
    *,
    cwd: str | os.PathLike[str],
    env: Mapping[str, str],
    tail: int = 200,
) -> tuple[int, list[str]]:
    """Run a pipeline stage's command, streaming its output to the console and a log.

    Every line the child prints is echoed and written to ``log_path`` (after a
    ``# <command>`` header) as it arrives, so long CASA logs never pile up in
    memory and the run can be followed with ``tail -f``, also on Slurm.
    Returns the exit code and the last ``tail`` lines.
    """
    from collections import deque

    process = subprocess.Popen(
        list(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        bufsize=1,
        cwd=str(cwd),
        env=dict(env),
    )
    tail_lines: deque[str] = deque(maxlen=tail)
    if process.stdout is None:  # not reached: stdout=PIPE
        raise RuntimeError(f"the output of {cmd[0]} is not captured")
    with Path(log_path).open("w", encoding="utf-8") as log_file:
        log_file.write("# " + " ".join(cmd) + "\n")
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
            tail_lines.append(line.rstrip("\n"))
    return process.wait(), list(tail_lines)
