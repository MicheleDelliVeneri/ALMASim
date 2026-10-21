"""Run jobs as Slurm batch jobs (web console plan, J1.4).

``submit`` writes ``job.sbatch`` into the run folder and passes it to
``sbatch``; the script does nothing but start the wrapper (J1.2), so it can be
read, and rerun by hand, as it is. ``poll`` asks Slurm about every active job
with one ``squeue`` and one ``sacct`` call, however many there are, and
``reconcile`` merges Slurm's answer with what the wrapper wrote: the wrapper
knows how the command ended, Slurm knows why a job never started or died
without the wrapper noticing (a time limit, a node failure, the memory
limit).
"""

from __future__ import annotations

import re
import shlex
import subprocess
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from ..model import Job, JobState
from ..profile import Profile
from ..rundir import RunStatus, current_state, write_manifest

SCRIPT_NAME = "job.sbatch"
SLURM_OUT_NAME = "slurm.out"  # sbatch's own output; the command's is in stdout/stderr.log

Runner = Callable[[list[str]], subprocess.CompletedProcess[str]]


class SlurmError(RuntimeError):
    """A Slurm command failed; the message carries its error output."""


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)


def _job_name(job: Job) -> str:
    words = re.sub(r"[^A-Za-z0-9_.-]+", "-", job.command).strip("-")
    return f"almasim-{words or 'job'}"[:64]


def render_script(job: Job, run_dir: Path, profile: Profile) -> str:
    """The batch script for ``job``. Every value in it was validated by Resources."""
    r = job.resources
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={_job_name(job)}",
        f"#SBATCH --chdir={run_dir}",
        f"#SBATCH --output={run_dir / SLURM_OUT_NAME}",
        f"#SBATCH --cpus-per-task={r.cpus}",
        f"#SBATCH --mem={r.memory}",
        f"#SBATCH --time={r.time}",
    ]
    if r.partition:
        lines.append(f"#SBATCH --partition={r.partition}")
    if r.account:
        lines.append(f"#SBATCH --account={r.account}")
    lines += [
        "",
        f"# almasim {shlex.join(job.argv)}",
        f"# run {job.id}, submitted by {job.owner}",
        *profile.setup,
        f"exec {shlex.quote(profile.python)} -m almasim.jobs.wrapper {shlex.quote(str(run_dir))}",
    ]
    return "\n".join(lines) + "\n"


def submit(
    job: Job,
    run_dir: str | Path,
    profile: Profile,
    *,
    after: Iterable[str] = (),
    run: Runner = _run,
) -> str:
    """Submit ``job``; its Slurm job id. ``after``: Slurm ids it must wait for (afterok).

    The manifest is updated with the id and the queued state.
    """
    run_dir = Path(run_dir)
    script = run_dir / SCRIPT_NAME
    script.write_text(render_script(job, run_dir, profile), encoding="utf-8")
    cmd = ["sbatch", "--parsable"]
    parents = [p for p in after if p]
    if parents:
        cmd.append("--dependency=afterok:" + ":".join(parents))
    result = run([*cmd, str(script)])
    if result.returncode != 0:
        raise SlurmError(f"sbatch failed: {result.stderr.strip() or result.stdout.strip()}")
    slurm_id = result.stdout.strip().split(";")[0]  # "12345" or "12345;cluster"
    if not slurm_id.isdigit():
        raise SlurmError(f"sbatch printed {result.stdout.strip()!r}, not a job id")
    job.slurm_job_id = slurm_id
    job.state = JobState.QUEUED
    write_manifest(run_dir, job)
    return slurm_id


def cancel(job: Job, *, run: Runner = _run) -> None:
    """Ask Slurm to stop ``job``; the wrapper records the SIGTERM it then receives."""
    if not job.slurm_job_id:
        raise SlurmError(f"run {job.id} has no Slurm job id")
    result = run(["scancel", job.slurm_job_id])
    if result.returncode != 0:
        raise SlurmError(f"scancel {job.slurm_job_id}: {result.stderr.strip()}")


@dataclass(frozen=True)
class SlurmState:
    """Slurm's view of one job: its state word, and a reason when there is one."""

    state: str  # PENDING, RUNNING, COMPLETED, FAILED, CANCELLED, TIMEOUT, ...
    reason: str = ""


def poll(slurm_ids: Iterable[str], *, run: Runner = _run) -> dict[str, SlurmState]:
    """Slurm's state for each id, from one squeue and (for the rest) one sacct call.

    Ids neither command knows are left out: the caller treats them as lost
    once the wrapper has not reported either.
    """
    ids = sorted({i for i in slurm_ids if i})
    if not ids:
        return {}
    found: dict[str, SlurmState] = {}
    queue = run(["squeue", "-h", "-j", ",".join(ids), "-o", "%i|%T|%r"])
    if queue.returncode == 0:
        for line in queue.stdout.splitlines():
            parts = line.strip().split("|")
            if len(parts) == 3 and parts[0] in ids:
                reason = "" if parts[2] in ("None", "(null)") else parts[2]
                found[parts[0]] = SlurmState(parts[1], reason)
    rest = [i for i in ids if i not in found]
    if rest:
        acct = run(["sacct", "-n", "-X", "-P", "-j", ",".join(rest), "-o", "JobID,State,ExitCode"])
        if acct.returncode == 0:
            for line in acct.stdout.splitlines():
                parts = line.strip().split("|")
                if len(parts) == 3 and parts[0] in rest:
                    # "CANCELLED by 1234" -> CANCELLED
                    word = parts[1].split()[0] if parts[1] else ""
                    found[parts[0]] = SlurmState(word, f"exit code {parts[2]}")
    return found


# Slurm ended the job for a reason the wrapper cannot see (it may have
# recorded only the SIGTERM Slurm sent first).
_SLURM_ENDED: Mapping[str, tuple[JobState, str]] = {
    "TIMEOUT": (JobState.FAILED, "reached its time limit"),
    "OUT_OF_MEMORY": (JobState.FAILED, "ran out of its memory limit"),
    "NODE_FAIL": (JobState.LOST, "its node failed"),
    "BOOT_FAIL": (JobState.LOST, "its node failed to boot"),
    "DEADLINE": (JobState.FAILED, "missed its deadline"),
    "PREEMPTED": (JobState.LOST, "was preempted"),
}
_WAITING = {"PENDING", "CONFIGURING", "REQUEUED", "RESIZING", "SUSPENDED"}


def reconcile(job: Job, status: RunStatus, slurm: SlurmState | None) -> tuple[JobState, str]:
    """The job's state and, when Slurm knows one, the reason.

    Slurm's end reasons win (a time limit reaches the wrapper as a SIGTERM,
    which alone would read as cancelled); otherwise the wrapper's report
    wins; otherwise Slurm's queue state. A job Slurm has finished but the
    wrapper never reported on is lost.
    """
    if slurm is not None and slurm.state in _SLURM_ENDED:
        return _SLURM_ENDED[slurm.state]
    from_wrapper = current_state(job, status)
    if status.finished_at is not None:
        return from_wrapper, ""
    if slurm is None:
        if job.state is JobState.QUEUED or status.started_at is not None:
            return JobState.LOST, "Slurm no longer knows this job"
        return from_wrapper, ""
    if slurm.state in _WAITING:
        return JobState.QUEUED, slurm.reason
    if slurm.state in ("RUNNING", "COMPLETING"):
        return JobState.RUNNING, ""
    if slurm.state == "CANCELLED":
        return JobState.CANCELLED, ""
    # COMPLETED or FAILED, yet the wrapper never wrote its end.
    return JobState.LOST, f"Slurm reports {slurm.state} ({slurm.reason}) without a report"
