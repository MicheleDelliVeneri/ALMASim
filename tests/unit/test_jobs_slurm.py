"""The Slurm executor, with Slurm's commands faked (web console plan, J1.4)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from almasim.jobs import (
    Job,
    JobState,
    Resources,
    RunStatus,
    create_run_dir,
    new_job_id,
    read_manifest,
    write_manifest,
)
from almasim.jobs.executors.slurm import (
    SCRIPT_NAME,
    SlurmError,
    SlurmState,
    cancel,
    poll,
    reconcile,
    render_script,
    submit,
)
from almasim.jobs.profile import Profile

pytestmark = pytest.mark.unit

PROFILE = Profile(python="/opt/py/bin/python", setup=("export OPENBLAS_NUM_THREADS=1",))


def _job(**overrides) -> Job:
    fields = {
        "id": new_job_id(),
        "owner": "astro",
        "argv": ["image", "image-from-ms", "/data/params.csv", "/data/out dir", "--cube"],
        "executor": "slurm",
        "resources": Resources(partition="zwickyA-CPU", account="ska", cpus=4, memory="16G"),
    }
    fields.update(overrides)
    return Job(**fields)


class FakeSlurm:
    """Records commands and answers each by its program name."""

    def __init__(self, **answers: tuple[int, str, str]):
        self.answers = answers
        self.calls: list[list[str]] = []

    def __call__(self, cmd: list[str]) -> subprocess.CompletedProcess[str]:
        self.calls.append(cmd)
        code, out, err = self.answers.get(cmd[0], (0, "", ""))
        return subprocess.CompletedProcess(cmd, code, out, err)


def test_the_script_only_starts_the_wrapper(tmp_path):
    job = _job()
    script = render_script(job, tmp_path, PROFILE)
    lines = script.splitlines()
    assert lines[0] == "#!/bin/bash"
    assert "#SBATCH --job-name=almasim-image-image-from-ms" in lines
    assert f"#SBATCH --chdir={tmp_path}" in lines
    assert "#SBATCH --cpus-per-task=4" in lines and "#SBATCH --mem=16G" in lines
    assert "#SBATCH --partition=zwickyA-CPU" in lines and "#SBATCH --account=ska" in lines
    assert "export OPENBLAS_NUM_THREADS=1" in lines
    assert lines[-1] == f"exec /opt/py/bin/python -m almasim.jobs.wrapper {tmp_path}"
    # The command line appears only as a comment, quoted.
    assert "# almasim image image-from-ms /data/params.csv '/data/out dir' --cube" in lines


def test_a_job_without_partition_or_account_leaves_them_to_slurm(tmp_path):
    script = render_script(_job(resources=Resources()), tmp_path, PROFILE)
    assert "--partition" not in script and "--account" not in script


def test_submit_records_the_slurm_id_and_queues_the_job(tmp_path):
    job = _job()
    run_dir = create_run_dir(tmp_path, job)
    write_manifest(run_dir, job)
    slurm = FakeSlurm(sbatch=(0, "6402;zwicky-cluster\n", ""))

    assert submit(job, run_dir, PROFILE, after=["6400", "", "6401"], run=slurm) == "6402"
    assert slurm.calls == [
        ["sbatch", "--parsable", "--dependency=afterok:6400:6401", str(run_dir / SCRIPT_NAME)]
    ]
    assert (run_dir / SCRIPT_NAME).read_text().startswith("#!/bin/bash")
    saved = read_manifest(run_dir)
    assert saved.slurm_job_id == "6402" and saved.state is JobState.QUEUED


@pytest.mark.parametrize(
    "answer",
    [(1, "", "sbatch: error: invalid partition specified: nope"), (0, "Submitted\n", "")],
)
def test_submit_failures_raise(tmp_path, answer):
    job = _job()
    run_dir = create_run_dir(tmp_path, job)
    with pytest.raises(SlurmError):
        submit(job, run_dir, PROFILE, run=FakeSlurm(sbatch=answer))


def test_cancel_uses_scancel(tmp_path):
    slurm = FakeSlurm()
    cancel(_job(slurm_job_id="6402"), run=slurm)
    assert slurm.calls == [["scancel", "6402"]]
    with pytest.raises(SlurmError):
        cancel(_job(), run=slurm)
    with pytest.raises(SlurmError, match="Invalid job id"):
        cancel(_job(slurm_job_id="6402"), run=FakeSlurm(scancel=(1, "", "Invalid job id")))


def test_poll_asks_squeue_then_sacct_once_each():
    slurm = FakeSlurm(
        squeue=(0, "6402|RUNNING|None\n6403|PENDING|Priority\n", ""),
        sacct=(0, "6399|COMPLETED|0:0\n6400|CANCELLED by 1234|0:15\n", ""),
    )
    states = poll(["6402", "6403", "6399", "6400", "6400", ""], run=slurm)
    assert states == {
        "6402": SlurmState("RUNNING"),
        "6403": SlurmState("PENDING", "Priority"),
        "6399": SlurmState("COMPLETED", "exit code 0:0"),
        "6400": SlurmState("CANCELLED", "exit code 0:15"),
    }
    assert [c[0] for c in slurm.calls] == ["squeue", "sacct"]
    assert slurm.calls[1][slurm.calls[1].index("-j") + 1] == "6399,6400"
    assert poll([], run=slurm) == {}


QUEUED = _job(slurm_job_id="6402", state=JobState.QUEUED)
STARTED = RunStatus(started_at=1.0, host="compute-node47")
DONE = RunStatus(started_at=1.0, finished_at=2.0, exit_code=0)
TERMED = RunStatus(started_at=1.0, finished_at=2.0, exit_code=143, signal="SIGTERM")


@pytest.mark.parametrize(
    ("status", "slurm", "expected"),
    [
        (RunStatus(), SlurmState("PENDING", "Priority"), (JobState.QUEUED, "Priority")),
        (RunStatus(), SlurmState("RUNNING"), (JobState.RUNNING, "")),
        (STARTED, SlurmState("RUNNING"), (JobState.RUNNING, "")),
        (DONE, SlurmState("COMPLETED", "exit code 0:0"), (JobState.SUCCEEDED, "")),
        (DONE, None, (JobState.SUCCEEDED, "")),  # aged out of sacct
        (TERMED, SlurmState("CANCELLED"), (JobState.CANCELLED, "")),
        (TERMED, SlurmState("TIMEOUT"), (JobState.FAILED, "reached its time limit")),
        (STARTED, SlurmState("OUT_OF_MEMORY"), (JobState.FAILED, "ran out of its memory limit")),
        (STARTED, SlurmState("NODE_FAIL"), (JobState.LOST, "its node failed")),
        (RunStatus(), SlurmState("CANCELLED"), (JobState.CANCELLED, "")),  # before it started
        (STARTED, None, (JobState.LOST, "Slurm no longer knows this job")),
        (RunStatus(), None, (JobState.LOST, "Slurm no longer knows this job")),
    ],
)
def test_reconcile(status, slurm, expected):
    assert reconcile(QUEUED, status, slurm) == expected


def test_a_job_that_completed_without_a_report_is_lost():
    state, reason = reconcile(QUEUED, STARTED, SlurmState("COMPLETED", "exit code 0:0"))
    assert state is JobState.LOST and "COMPLETED" in reason


def test_a_pending_job_not_yet_submitted_stays_pending():
    assert reconcile(_job(), RunStatus(), None) == (JobState.PENDING, "")


def test_the_script_names_the_profile_files_by_absolute_path(tmp_path):
    profile = Profile(site_file=Path("/etc/almasim/site.toml"), user_file=Path("/home/me/a b.toml"))
    script = render_script(_job(), tmp_path, profile)
    assert "export ALMASIM_SITE_PROFILE=/etc/almasim/site.toml" in script
    assert "export ALMASIM_PROFILE='/home/me/a b.toml'" in script
