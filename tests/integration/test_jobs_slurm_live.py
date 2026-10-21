"""The Slurm executor against a real cluster (web console plan, J1.4).

Opt-in: set ALMASIM_LIVE_SLURM=1 on a machine with Slurm, and
ALMASIM_LIVE_SLURM_ROOT to a folder the compute nodes can see (a local /tmp
is not). ALMASIM_LIVE_SLURM_PARTITION picks the partition. Never runs in CI.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

from almasim.jobs import (
    Job,
    JobState,
    Resources,
    create_run_dir,
    new_job_id,
    read_manifest,
    read_status,
    write_manifest,
)
from almasim.jobs.executors.slurm import cancel, poll, reconcile, submit
from almasim.jobs.profile import Profile

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("ALMASIM_LIVE_SLURM") != "1", reason="set ALMASIM_LIVE_SLURM=1"
    ),
]


@pytest.fixture
def root() -> Path:
    path = Path(os.environ.get("ALMASIM_LIVE_SLURM_ROOT", "")).expanduser()
    if not path.is_absolute():
        pytest.skip("set ALMASIM_LIVE_SLURM_ROOT to a folder the compute nodes see")
    return path


def _submit(root: Path, argv: list[str], profile: Profile) -> Job:
    job = Job(
        id=new_job_id(),
        owner=os.environ.get("USER", "test"),
        argv=argv,
        executor="slurm",
        resources=Resources(
            partition=os.environ.get("ALMASIM_LIVE_SLURM_PARTITION"),
            cpus=1,
            memory="2G",
            time="00:10:00",
        ),
    )
    run_dir = create_run_dir(root, job)
    write_manifest(run_dir, job)
    submit(job, run_dir, profile)
    return job


def _wait(job: Job, run_dir: Path, until: set[JobState], timeout: float = 600) -> JobState:
    deadline = time.monotonic() + timeout
    while True:
        state, reason = reconcile(
            read_manifest(run_dir),
            read_status(run_dir),
            poll([job.slurm_job_id]).get(job.slurm_job_id),
        )
        if state in until:
            return state
        assert time.monotonic() < deadline, f"still {state} {reason}"
        time.sleep(3)


def test_a_real_job_runs_the_cli_and_succeeds(root):
    job = _submit(root, ["--help"], Profile(python=sys.executable))
    run_dir = next(root.glob(f"*/{job.id}"))
    assert _wait(job, run_dir, {s for s in JobState if s.finished}) is JobState.SUCCEEDED
    assert "Usage" in (run_dir / "stdout.log").read_text()
    assert read_status(run_dir).host


def test_a_real_job_can_be_cancelled(root):
    # The setup line keeps the job busy before the wrapper starts.
    job = _submit(root, ["--help"], Profile(python=sys.executable, setup=("sleep 300",)))
    run_dir = next(root.glob(f"*/{job.id}"))
    _wait(job, run_dir, {JobState.RUNNING} | {s for s in JobState if s.finished})
    cancel(job)
    assert _wait(job, run_dir, {s for s in JobState if s.finished}) is JobState.CANCELLED
