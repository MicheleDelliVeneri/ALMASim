"""Run jobs as processes beside the server (web console plan, J1.5).

For the commands that need the internet (metadata, products resolve, the
transfer part of products download): zwicky's compute nodes have none. They
run on the node that hosts ``almasim serve``, so they must stay polite:

- at most ``max_running`` at a time; the rest wait as queued and
  ``start_queued`` starts them as slots free up;
- under ``nice`` and ``ionice`` (best effort, lowest priority);
- each in its own session, so stopping or restarting the server does not
  stop them. ``server.pid`` (pid and process start time, against pid reuse)
  lets a restarted server find them again.

The process is the wrapper (J1.2), as in a Slurm job, so the run folder
looks the same whichever executor ran it.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ..model import Job, JobState
from ..profile import Profile
from ..rundir import RunStatus, current_state, read_manifest, read_status, write_manifest

PID_NAME = "server.pid"
SERVER_OUT_NAME = "server.out"  # the wrapper's own output; the command's is in stdout/stderr.log
DEFAULT_MAX_RUNNING = 2


class ServerExecutorError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProcessRef:
    pid: int
    start_time: int  # /proc/<pid>/stat field 22, in clock ticks since boot


def _start_time(pid: int) -> int | None:
    """The process's start time, or None if it is gone or a zombie (ended, not reaped)."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    # The command name (field 2) may hold spaces and parentheses: split after it.
    fields = stat[stat.rindex(")") + 2 :].split()
    if fields[0] in ("Z", "X"):  # field 3: zombie or dead
        return None
    return int(fields[19])  # field 22 overall


def read_process(run_dir: str | os.PathLike[str]) -> ProcessRef | None:
    try:
        data = json.loads((Path(run_dir) / PID_NAME).read_text(encoding="utf-8"))
        return ProcessRef(int(data["pid"]), int(data["start_time"]))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def is_alive(ref: ProcessRef | None) -> bool:
    """Whether the process ``ref`` names still runs (a reused pid does not count)."""
    return ref is not None and _start_time(ref.pid) == ref.start_time


def _polite_prefix() -> list[str]:
    prefix = []
    if shutil.which("nice"):
        prefix += ["nice", "-n", "10"]
    if shutil.which("ionice"):
        prefix += ["ionice", "-c", "2", "-n", "7"]
    return prefix


def start(job: Job, run_dir: str | os.PathLike[str], profile: Profile) -> ProcessRef:
    """Start ``job``'s wrapper now; the caller has checked the concurrency cap."""
    if job.executor != "server":
        raise ServerExecutorError(f"run {job.id} is a {job.executor} job, not a server job")
    run_dir = Path(run_dir)
    cmd = [*_polite_prefix(), profile.python, "-m", "almasim.jobs.wrapper", str(run_dir)]
    with (run_dir / SERVER_OUT_NAME).open("ab") as out:
        process = subprocess.Popen(
            cmd,
            cwd=run_dir,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    # Reap it when it ends, so it does not linger as a zombie of the server.
    threading.Thread(target=process.wait, name=f"reap-{job.id}", daemon=True).start()
    start_time = _start_time(process.pid)
    ref = ProcessRef(process.pid, start_time if start_time is not None else -1)
    (run_dir / PID_NAME).write_text(
        json.dumps({"pid": ref.pid, "start_time": ref.start_time}) + "\n", encoding="utf-8"
    )
    job.state = JobState.RUNNING
    write_manifest(run_dir, job)
    return ref


def state_of(job: Job, run_dir: str | os.PathLike[str]) -> tuple[JobState, str]:
    """The state of a server job: the wrapper's report, else whether its process lives."""
    status: RunStatus = read_status(run_dir)
    if status.finished_at is not None:
        return current_state(job, status), ""
    if job.state is JobState.CANCELLED:
        return JobState.CANCELLED, ""
    ref = read_process(run_dir)
    if ref is None:
        return (
            (JobState.QUEUED, "waiting for a free slot")
            if job.state is JobState.QUEUED
            else (
                job.state,
                "",
            )
        )
    if is_alive(ref):
        return JobState.RUNNING, ""
    return JobState.LOST, "its process ended without a report"


def running(run_dirs: Iterable[Path]) -> list[Path]:
    """The run folders among ``run_dirs`` whose process still runs."""
    return [d for d in run_dirs if read_status(d).finished_at is None and is_alive(read_process(d))]


def start_queued(
    queued: Iterable[Path],
    active: Iterable[Path],
    profile: Profile,
    max_running: int = DEFAULT_MAX_RUNNING,
) -> list[Path]:
    """Start queued server jobs, oldest first, while fewer than ``max_running`` run.

    ``queued`` are run folders whose job waits; ``active`` the ones started
    before. Returns the folders started now.
    """
    slots = max_running - len(running(active))
    started: list[Path] = []
    for run_dir in sorted(queued, key=lambda d: d.name):  # job ids sort by time
        if slots <= 0:
            break
        job = read_manifest(run_dir)
        if job.state is not JobState.QUEUED or read_process(run_dir) is not None:
            continue
        start(job, run_dir, profile)
        started.append(run_dir)
        slots -= 1
    return started


def queue(job: Job, run_dir: str | os.PathLike[str]) -> None:
    """Mark ``job`` as waiting for a slot; ``start_queued`` starts it."""
    job.state = JobState.QUEUED
    write_manifest(run_dir, job)


def cancel(job: Job, run_dir: str | os.PathLike[str]) -> None:
    """Stop a running server job (its wrapper records the SIGTERM) or drop a queued one."""
    ref = read_process(run_dir)
    if ref is not None and is_alive(ref):
        try:
            os.killpg(ref.pid, signal.SIGTERM)  # the wrapper leads its own session
        except ProcessLookupError:
            pass
    elif read_status(run_dir).finished_at is not None:
        return  # already over: nothing to cancel
    # Recorded too, so a run stopped before its wrapper started reads as
    # cancelled rather than lost.
    job.state = JobState.CANCELLED
    write_manifest(run_dir, job)
