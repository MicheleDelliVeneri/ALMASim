"""Submit, list and control runs: what ``almasim jobs`` and the server share (J1.7).

Everything here works from the run folders under the profile's runs root,
so the terminal and the web console see the same runs, and a teammate's
runs in a shared root show up too (read only: only the owner may cancel).
"""

from __future__ import annotations

import getpass
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .executors import server as server_exec
from .executors import slurm as slurm_exec
from .model import Job, JobState, new_job_id
from .policy import plan_job
from .profile import Profile
from .rundir import (
    ManifestError,
    RunStatus,
    create_run_dir,
    current_state,
    iter_runs,
    read_manifest,
    read_status,
    write_manifest,
)


class RunError(ValueError):
    """A run that cannot be found, or an action this user may not take on it."""


@dataclass(frozen=True)
class RunView:
    """One run as listed: its manifest, folder, wrapper status and current state."""

    job: Job
    run_dir: Path
    status: RunStatus
    state: JobState
    reason: str = ""


def submit_command(
    argv: list[str],
    profile: Profile,
    *,
    resources: Mapping[str, Any] | None = None,
    params: Mapping[str, Any] | None = None,
    title: str = "",
    owner: str | None = None,
) -> RunView:
    """Plan, record and submit ``almasim <argv>``; the new run."""
    plan = plan_job(argv, profile, resources)
    job = Job(
        id=new_job_id(),
        owner=owner or getpass.getuser(),
        argv=plan.argv,
        executor=plan.executor,
        resources=plan.resources,
        params=dict(params or {}),
        title=title,
    )
    run_dir = create_run_dir(profile.runs_root, job)
    write_manifest(run_dir, job)
    if job.executor == "slurm":
        slurm_exec.submit(job, run_dir, profile)
    else:
        server_exec.queue(job, run_dir)
        start_queued_server_jobs(profile)
    return view(run_dir)


# view()'s default: Slurm was not asked, which is not the same as Slurm not
# knowing the job (that makes a submitted job lost).
_UNASKED: Any = object()


def view(run_dir: Path, slurm: slurm_exec.SlurmState | None = _UNASKED) -> RunView:
    """``run_dir`` with its state as the folder (and Slurm's answer, if given) shows it."""
    job = read_manifest(run_dir)
    status = read_status(run_dir)
    if job.executor == "server":
        state, reason = server_exec.state_of(job, run_dir)
    elif job.slurm_job_id is None or slurm is _UNASKED:
        state, reason = current_state(job, status), ""
    else:
        state, reason = slurm_exec.reconcile(job, status, slurm)
    return RunView(job, run_dir, status, state, reason)


def list_runs(
    profile: Profile,
    *,
    owner: str | None = None,
    limit: int | None = None,
    ask_slurm: bool = True,
) -> list[RunView]:
    """Runs under the profile's runs root, newest first; ``owner`` None means everyone's.

    Slurm is asked once about all the listed runs that are not finished.
    """
    dirs: list[Path] = []
    jobs: list[Job] = []
    for run_dir in iter_runs(profile.runs_root):
        try:
            job = read_manifest(run_dir)
        except ManifestError:
            continue  # half-written or foreign folder: skip, never fail the listing
        if owner is not None and job.owner != owner:
            continue
        dirs.append(run_dir)
        jobs.append(job)
        if limit is not None and len(dirs) >= limit:
            break
    states: dict[str, slurm_exec.SlurmState] = {}
    if ask_slurm:
        active = [
            j.slurm_job_id
            for j, d in zip(jobs, dirs, strict=True)
            if j.slurm_job_id and read_status(d).finished_at is None
        ]
        states = slurm_exec.poll([i for i in active if i])
    return [view(d, states.get(j.slurm_job_id or "")) for j, d in zip(jobs, dirs, strict=True)]


def find_run(profile: Profile, ref: str) -> Path:
    """The run whose id is ``ref``, starts with it, or ends with it (``short_id``)."""
    ref = ref.strip().upper()
    if len(ref) < 4:
        raise RunError("give at least 4 characters of the run id")
    matches = [
        d for d in iter_runs(profile.runs_root) if d.name.startswith(ref) or d.name.endswith(ref)
    ]
    if not matches:
        raise RunError(f"no run {ref} under {profile.runs_root}")
    if len(matches) > 1:
        raise RunError(f"{ref} matches {len(matches)} runs; give more of the id")
    return matches[0]


def _own(job: Job, user: str | None) -> None:
    me = user or getpass.getuser()
    if job.owner != me:
        raise RunError(f"run {job.id} belongs to {job.owner}; only they can change it")


def cancel_run(run_dir: Path, *, user: str | None = None) -> RunView:
    job = read_manifest(run_dir)
    _own(job, user)
    if read_status(run_dir).finished_at is not None:
        raise RunError(f"run {job.id} has already finished")
    if job.executor == "server":
        server_exec.cancel(job, run_dir)
    elif job.slurm_job_id:
        slurm_exec.cancel(job)
    else:  # recorded but never submitted
        job.state = JobState.CANCELLED
        write_manifest(run_dir, job)
    return view(run_dir)


def resubmit(run_dir: Path, profile: Profile, *, user: str | None = None) -> RunView:
    """A new run with the same command line, resources and form values."""
    old = read_manifest(run_dir)
    return submit_command(
        old.argv,
        profile,
        resources=old.resources.to_dict(),
        params=old.params,
        title=old.title,
        owner=user,
    )


def start_queued_server_jobs(profile: Profile, max_running: int | None = None) -> list[Path]:
    """Start waiting server jobs while slots are free (the CLI and the server both call it)."""
    queued: list[Path] = []
    active: list[Path] = []
    me = getpass.getuser()
    for run_dir in iter_runs(profile.runs_root):
        try:
            job = read_manifest(run_dir)
        except ManifestError:
            continue
        if job.executor != "server" or job.owner != me:
            continue
        if job.state is JobState.QUEUED and server_exec.read_process(run_dir) is None:
            queued.append(run_dir)
        elif job.state is JobState.RUNNING:
            active.append(run_dir)
    return server_exec.start_queued(
        queued,
        active,
        profile,
        max_running if max_running is not None else server_exec.DEFAULT_MAX_RUNNING,
    )


def wait(run_dir: Path, *, poll_s: float = 5.0, timeout: float | None = None) -> RunView:
    """Block until the run finishes; its final view."""
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        job = read_manifest(run_dir)
        slurm = slurm_exec.poll([job.slurm_job_id or ""]).get(job.slurm_job_id or "")
        current = view(run_dir, slurm)
        if current.state.finished:
            return current
        if deadline is not None and time.monotonic() > deadline:
            return current
        time.sleep(poll_s)


def short_id(job_id: str) -> str:
    """The last 10 characters: random, so distinct even among runs of the same second."""
    return job_id[-10:]


def unfinished(runs: Iterable[RunView]) -> list[RunView]:
    return [r for r in runs if not r.state.finished]
