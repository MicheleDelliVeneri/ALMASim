"""The server executor (web console plan, J1.5)."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

from almasim.jobs import Job, JobState, create_run_dir, new_job_id, read_manifest, write_manifest
from almasim.jobs.executors import server
from almasim.jobs.executors.server import (
    PID_NAME,
    ProcessRef,
    ServerExecutorError,
    cancel,
    is_alive,
    queue,
    read_process,
    start,
    start_queued,
    state_of,
)
from almasim.jobs.profile import Profile

pytestmark = pytest.mark.unit

PROFILE = Profile(python=sys.executable)


def _run_dir(root: Path, argv=("--help",), executor="server") -> tuple[Job, Path]:
    job = Job(id=new_job_id(), owner="astro", argv=list(argv), executor=executor)
    run_dir = create_run_dir(root, job)
    write_manifest(run_dir, job)
    return job, run_dir


def _wait(predicate, timeout=60.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.1)


def _slow_python(tmp_path: Path, seconds: int) -> Profile:
    """A 'python' that waits before becoming the real one: a long first step to cancel."""
    script = tmp_path / "slow-python"
    script.write_text(f'#!/bin/sh\nsleep {seconds}\nexec {sys.executable} "$@"\n')
    script.chmod(0o755)
    return Profile(python=str(script))


def test_a_server_job_runs_its_wrapper_and_succeeds(tmp_path):
    job, run_dir = _run_dir(tmp_path)
    ref = start(job, run_dir, PROFILE)
    assert read_process(run_dir) == ref
    assert read_manifest(run_dir).state is JobState.RUNNING
    _wait(lambda: state_of(read_manifest(run_dir), run_dir)[0].finished)
    assert state_of(read_manifest(run_dir), run_dir) == (JobState.SUCCEEDED, "")
    assert "Usage" in (run_dir / "stdout.log").read_text()
    _wait(lambda: not is_alive(ref))  # the wrapper exits just after its last report


def test_only_server_jobs_start_here(tmp_path):
    job, run_dir = _run_dir(tmp_path, executor="slurm")
    with pytest.raises(ServerExecutorError):
        start(job, run_dir, PROFILE)


def test_a_reused_pid_does_not_count_as_alive(tmp_path):
    assert is_alive(ProcessRef(os.getpid(), server._start_time(os.getpid())))
    assert not is_alive(ProcessRef(os.getpid(), -5))
    assert not is_alive(None)
    (tmp_path / PID_NAME).write_text("not json")
    assert read_process(tmp_path) is None


def test_queued_jobs_start_oldest_first_within_the_cap(tmp_path, monkeypatch):
    started = []
    monkeypatch.setattr(server, "start", lambda job, run_dir, profile: started.append(run_dir))
    monkeypatch.setattr(server, "running", lambda dirs: list(dirs))
    dirs = []
    for _ in range(3):
        job, run_dir = _run_dir(tmp_path)
        queue(job, run_dir)
        dirs.append(run_dir)
    dirs.sort(key=lambda d: d.name)  # ids made in the same millisecond sort by their random part
    already = [tmp_path / "busy"]

    assert start_queued(reversed(dirs), already, PROFILE, max_running=2) == dirs[:1]
    assert started == dirs[:1]
    assert start_queued(dirs, [], PROFILE, max_running=2) == dirs[:2]


def test_cancelling_a_running_job_stops_it(tmp_path):
    job, run_dir = _run_dir(tmp_path)
    ref = start(job, run_dir, _slow_python(tmp_path, 120))
    _wait(lambda: is_alive(ref))
    cancel(read_manifest(run_dir), run_dir)
    _wait(lambda: not is_alive(ref))
    assert state_of(read_manifest(run_dir), run_dir)[0] is JobState.CANCELLED


def test_cancelling_a_queued_job_drops_it(tmp_path):
    job, run_dir = _run_dir(tmp_path)
    queue(job, run_dir)
    assert state_of(read_manifest(run_dir), run_dir) == (JobState.QUEUED, "waiting for a free slot")
    cancel(read_manifest(run_dir), run_dir)
    assert state_of(read_manifest(run_dir), run_dir)[0] is JobState.CANCELLED
    assert start_queued([run_dir], [], PROFILE) == []


def test_a_process_that_vanished_without_a_report_is_lost(tmp_path):
    job, run_dir = _run_dir(tmp_path)
    job.state = JobState.RUNNING
    write_manifest(run_dir, job)
    (run_dir / PID_NAME).write_text('{"pid": 999999999, "start_time": 1}')
    assert state_of(read_manifest(run_dir), run_dir) == (
        JobState.LOST,
        "its process ended without a report",
    )


def test_without_proc_the_start_time_comes_from_ps():
    """macOS has no /proc: ps gives the state and the start time."""
    started = server._start_time_from_ps(os.getpid())
    assert started is not None and abs(started - time.time()) < 86400
    assert server._start_time_from_ps(os.getpid()) == started
    assert server._start_time_from_ps(999999999) is None
