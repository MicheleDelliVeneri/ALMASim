"""almasim submit / almasim jobs and the service they share (web console plan, J1.7)."""

from __future__ import annotations

import getpass
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from almasim import cli
from almasim.jobs import JobState, read_manifest, write_manifest
from almasim.jobs import service as service_mod
from almasim.jobs.executors import slurm as slurm_exec
from almasim.jobs.executors.slurm import SlurmState
from almasim.jobs.profile import CommandPolicy, Profile
from almasim.jobs.service import (
    RunError,
    cancel_run,
    find_run,
    list_runs,
    resubmit,
    short_id,
    submit_command,
    wait,
)

pytestmark = pytest.mark.unit

ME = getpass.getuser()


class FakeSlurm:
    def __init__(self):
        self.next_id = 4242
        self.states: dict[str, SlurmState] = {}
        self.cancelled: list[str] = []

    def submit(self, job, run_dir, profile, **kwargs):
        job.slurm_job_id = str(self.next_id)
        self.next_id += 1
        job.state = JobState.QUEUED
        write_manifest(run_dir, job)
        self.states[job.slurm_job_id] = SlurmState("PENDING", "Priority")
        return job.slurm_job_id

    def poll(self, ids, **kwargs):
        return {i: self.states[i] for i in ids if i in self.states}

    def cancel(self, job, **kwargs):
        self.cancelled.append(job.slurm_job_id)


@pytest.fixture
def slurm(monkeypatch):
    fake = FakeSlurm()
    monkeypatch.setattr(slurm_exec, "submit", fake.submit)
    monkeypatch.setattr(slurm_exec, "poll", fake.poll)
    monkeypatch.setattr(slurm_exec, "cancel", fake.cancel)
    return fake


@pytest.fixture
def profile(tmp_path) -> Profile:
    # `doctor --help` stands in for a network command: it runs here, offline.
    return Profile(
        runs_root=tmp_path / "runs",
        python=sys.executable,
        commands={"doctor": CommandPolicy(executor="server")},
    )


def test_a_slurm_run_is_recorded_submitted_and_listed(profile, slurm):
    run = submit_command(["image", "clean-quality", "/out"], profile, resources={"cpus": 8})
    assert run.job.executor == "slurm" and run.job.slurm_job_id == "4242"
    assert run.job.resources.cpus == 8 and run.job.owner == ME
    assert read_manifest(run.run_dir).state is JobState.QUEUED

    [listed] = list_runs(profile)
    assert listed.state is JobState.QUEUED and listed.reason == "Priority"
    slurm.states["4242"] = SlurmState("RUNNING")
    assert list_runs(profile)[0].state is JobState.RUNNING


def test_a_server_run_starts_here_and_succeeds(profile, slurm):
    run = submit_command(["doctor", "--help"], profile)
    assert run.job.executor == "server"
    final = wait(run.run_dir, poll_s=0.2, timeout=60)
    assert final.state is JobState.SUCCEEDED
    assert "Usage" in (run.run_dir / "stdout.log").read_text()


def test_runs_are_found_by_the_start_or_end_of_their_id(profile, slurm):
    a = submit_command(["image", "clean-quality", "/a"], profile)
    b = submit_command(["image", "clean-quality", "/b"], profile)
    assert find_run(profile, short_id(a.job.id).lower()) == a.run_dir
    assert find_run(profile, b.job.id) == b.run_dir
    with pytest.raises(RunError, match="matches 2 runs"):
        find_run(profile, a.job.id[:6])  # same second: same time prefix
    with pytest.raises(RunError, match="at least 4"):
        find_run(profile, "AB")
    with pytest.raises(RunError, match="no run"):
        find_run(profile, "ZZZZZZ")


def test_only_the_owner_cancels_and_only_unfinished_runs(profile, slurm):
    run = submit_command(["image", "clean-quality", "/out"], profile)
    with pytest.raises(RunError, match="belongs to"):
        cancel_run(run.run_dir, user="someone-else")
    cancel_run(run.run_dir)
    assert slurm.cancelled == ["4242"]

    (run.run_dir / "status.json").write_text('{"started_at": 1, "finished_at": 2, "exit_code": 0}')
    with pytest.raises(RunError, match="already finished"):
        cancel_run(run.run_dir)


def test_resubmit_repeats_the_command_and_resources(profile, slurm):
    first = submit_command(
        ["image", "clean-quality", "/out"], profile, resources={"cpus": 6}, params={"x": 1}
    )
    again = resubmit(first.run_dir, profile)
    assert again.job.id != first.job.id
    assert again.job.argv == first.job.argv
    assert again.job.resources == first.job.resources
    assert again.job.params == {"x": 1}


def test_listing_skips_broken_folders_and_filters_by_owner(profile, slurm):
    run = submit_command(["image", "clean-quality", "/out"], profile)
    broken = run.run_dir.parent / ("0" * 26)
    broken.mkdir()
    (broken / "job.json").write_text("{half")
    assert [r.run_dir for r in list_runs(profile)] == [run.run_dir]
    assert list_runs(profile, owner="nobody") == []


def _cli_env(monkeypatch, tmp_path: Path) -> None:
    user = tmp_path / "almasim.toml"
    user.write_text(f'runs_root = "{tmp_path / "runs"}"\n\n[commands.doctor]\nexecutor = "slurm"\n')
    monkeypatch.setenv("ALMASIM_PROFILE", str(user))
    monkeypatch.delenv("ALMASIM_SITE_PROFILE", raising=False)


def test_the_cli_submits_lists_shows_and_cancels(tmp_path, monkeypatch, slurm):
    _cli_env(monkeypatch, tmp_path)
    runner = CliRunner()

    result = runner.invoke(cli.app, ["submit", "--cpus", "2", "doctor", "--slurm"])
    assert result.exit_code == 0, result.output
    assert "on Slurm job 4242: almasim doctor --slurm" in result.output
    [run_dir] = list((tmp_path / "runs").glob("*/*"))
    assert read_manifest(run_dir).resources.cpus == 2

    listing = runner.invoke(cli.app, ["jobs", "list"])
    assert listing.exit_code == 0, listing.output
    assert short_id(run_dir.name) in listing.output and "queued" in listing.output

    shown = runner.invoke(cli.app, ["jobs", "show", run_dir.name[-6:]])
    assert "Slurm job 4242" in shown.output and "almasim doctor --slurm" in shown.output

    assert runner.invoke(cli.app, ["jobs", "cancel", run_dir.name]).exit_code == 0
    assert slurm.cancelled == ["4242"]


def test_the_cli_refuses_unknown_commands(tmp_path, monkeypatch, slurm):
    _cli_env(monkeypatch, tmp_path)
    result = CliRunner().invoke(cli.app, ["submit", "rm", "-rf", "/"])
    assert result.exit_code == 1
    assert "unknown command" in result.output


def test_logs_print_the_run_output(tmp_path, monkeypatch, slurm):
    _cli_env(monkeypatch, tmp_path)
    CliRunner().invoke(cli.app, ["submit", "doctor"])
    [run_dir] = list((tmp_path / "runs").glob("*/*"))
    (run_dir / "stdout.log").write_text("hello from the job\n")
    result = CliRunner().invoke(cli.app, ["jobs", "logs", run_dir.name])
    assert result.output == "hello from the job\n"


def test_service_module_exports(monkeypatch):
    assert service_mod.unfinished([]) == []
