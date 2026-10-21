"""The job wrapper runs a command and reports it (web console plan, J1.2)."""

from __future__ import annotations

import signal
import subprocess
import sys
import time

import pytest

from almasim.jobs import (
    Job,
    JobState,
    create_run_dir,
    current_state,
    new_job_id,
    read_manifest,
    read_status,
    write_manifest,
)
from almasim.jobs.wrapper import STDERR_NAME, STDOUT_NAME, main, run_job

pytestmark = pytest.mark.unit


def _run_dir(tmp_path, argv=("metadata", "--band", "3")):
    job = Job(id=new_job_id(), owner="astro", argv=list(argv), executor="slurm")
    run_dir = create_run_dir(tmp_path, job)
    write_manifest(run_dir, job)
    return run_dir


def _python(code: str) -> list[str]:
    """A command prefix that runs ``code`` with the job's argv as sys.argv[1:]."""
    return [sys.executable, "-c", code]


def test_a_finished_command_is_reported_with_its_output(tmp_path):
    run_dir = _run_dir(tmp_path)
    code = run_job(
        run_dir,
        _python(
            "import os, sys; print('args', *sys.argv[1:]); print('cwd', os.getcwd());"
            " print('a warning', file=sys.stderr)"
        ),
    )
    assert code == 0
    out = (run_dir / STDOUT_NAME).read_text()
    assert "args metadata --band 3" in out
    assert f"cwd {run_dir}" in out
    assert (run_dir / STDERR_NAME).read_text() == "a warning\n"

    status = read_status(run_dir)
    assert status.host and status.started_at <= status.finished_at
    assert status.exit_code == 0 and status.signal is None
    assert current_state(read_manifest(run_dir), status) is JobState.SUCCEEDED


def test_run_dir_in_an_argument_becomes_the_run_folder(tmp_path):
    run_dir = _run_dir(tmp_path, ("metadata", "--save-csv", "{run_dir}/results.csv"))
    assert run_job(run_dir, _python("import sys; print(*sys.argv[1:])")) == 0
    assert f"--save-csv {run_dir.absolute()}/results.csv" in (run_dir / STDOUT_NAME).read_text()
    # The manifest keeps the placeholder, so running it again uses the new folder.
    assert read_manifest(run_dir).argv[-1] == "{run_dir}/results.csv"


def test_a_failing_command_reads_as_failed(tmp_path):
    run_dir = _run_dir(tmp_path)
    assert run_job(run_dir, _python("import sys; sys.exit(3)")) == 3
    status = read_status(run_dir)
    assert status.exit_code == 3
    assert current_state(read_manifest(run_dir), status) is JobState.FAILED


def test_a_command_killed_by_a_signal_reports_it(tmp_path):
    run_dir = _run_dir(tmp_path)
    code = run_job(run_dir, _python("import os, signal; os.kill(os.getpid(), signal.SIGKILL)"))
    assert code == 128 + signal.SIGKILL
    status = read_status(run_dir)
    assert status.signal == "SIGKILL"
    assert current_state(read_manifest(run_dir), status) is JobState.FAILED


def test_scancel_reaches_the_command_and_reads_as_cancelled(tmp_path):
    """scancel sends SIGTERM to the batch script, which is the wrapper."""
    run_dir = _run_dir(tmp_path)
    wrapper = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys; from almasim.jobs.wrapper import run_job;"
            " sys.exit(run_job(sys.argv[1], [sys.executable, '-c',"
            " 'import time; time.sleep(120)']))",
            str(run_dir),
        ]
    )
    deadline = time.monotonic() + 30
    while read_status(run_dir).started_at is None:
        assert time.monotonic() < deadline, "the wrapper never reported its start"
        time.sleep(0.05)
    time.sleep(0.3)  # let the child start
    wrapper.send_signal(signal.SIGTERM)
    assert wrapper.wait(timeout=30) == 128 + signal.SIGTERM

    status = read_status(run_dir)
    assert status.signal == "SIGTERM"
    assert status.finished_at is not None
    assert current_state(read_manifest(run_dir), status) is JobState.CANCELLED


def test_main_needs_exactly_one_run_dir(capsys):
    assert main([]) == 2
    assert "usage" in capsys.readouterr().err


def test_by_default_the_job_runs_the_almasim_cli(tmp_path):
    run_dir = _run_dir(tmp_path, argv=("--help",))
    assert run_job(run_dir) == 0
    assert "Usage" in (run_dir / STDOUT_NAME).read_text()
