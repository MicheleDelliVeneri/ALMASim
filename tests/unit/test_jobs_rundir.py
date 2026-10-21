"""Run folders (web console plan, J1.1)."""

from __future__ import annotations

import calendar
import json
import stat

import pytest

from almasim.jobs import (
    MANIFEST_NAME,
    Job,
    JobState,
    ManifestError,
    RunStatus,
    create_run_dir,
    current_state,
    iter_runs,
    new_job_id,
    read_manifest,
    read_status,
    write_manifest,
    write_status,
)
from almasim.jobs.rundir import is_group_shared

pytestmark = pytest.mark.unit

DAY = calendar.timegm((2026, 10, 21, 12, 0, 0))  # noon UTC


def _job(created_at: float = DAY, ms: int | None = None) -> Job:
    job_id = new_job_id(now_ms=ms) if ms is not None else new_job_id()
    return Job(
        id=job_id, owner="astro", argv=["metadata"], executor="server", created_at=created_at
    )


def test_a_run_folder_is_dated_named_after_the_job_and_group_shared(tmp_path):
    job = _job()
    run_dir = create_run_dir(tmp_path / "runs", job)
    assert run_dir == tmp_path / "runs" / "2026-10-21" / job.id
    assert is_group_shared(run_dir) and is_group_shared(run_dir.parent)
    assert stat.S_IMODE(run_dir.stat().st_mode) & 0o070 == 0o070
    with pytest.raises(FileExistsError):
        create_run_dir(tmp_path / "runs", job)


def test_the_manifest_round_trips_and_is_written_whole(tmp_path):
    job = _job()
    run_dir = create_run_dir(tmp_path, job)
    path = write_manifest(run_dir, job)
    assert path.name == MANIFEST_NAME
    assert read_manifest(run_dir) == job
    assert [p.name for p in run_dir.iterdir()] == [MANIFEST_NAME]  # no temporary left
    assert stat.S_IMODE(path.stat().st_mode) == 0o664

    job.state = JobState.QUEUED
    job.slurm_job_id = "6402"
    write_manifest(run_dir, job)
    assert read_manifest(run_dir).slurm_job_id == "6402"


def test_unreadable_manifests_raise_one_error(tmp_path):
    with pytest.raises(ManifestError, match="no manifest"):
        read_manifest(tmp_path)
    (tmp_path / MANIFEST_NAME).write_text("{not json")
    with pytest.raises(ManifestError):
        read_manifest(tmp_path)
    (tmp_path / MANIFEST_NAME).write_text(json.dumps({"id": "x"}))
    with pytest.raises(ManifestError):
        read_manifest(tmp_path)


def test_the_status_decides_the_state_once_the_wrapper_reports(tmp_path):
    job = _job()
    assert read_status(tmp_path) == RunStatus()
    assert current_state(job, read_status(tmp_path)) is JobState.PENDING

    write_status(tmp_path, RunStatus(started_at=1.0, host="compute-node47"))
    assert current_state(job, read_status(tmp_path)) is JobState.RUNNING

    cases = [
        (RunStatus(started_at=1.0, finished_at=2.0, exit_code=0), JobState.SUCCEEDED),
        (RunStatus(started_at=1.0, finished_at=2.0, exit_code=2), JobState.FAILED),
        (
            RunStatus(started_at=1.0, finished_at=2.0, exit_code=143, signal="SIGTERM"),
            JobState.CANCELLED,
        ),
    ]
    for status, expected in cases:
        write_status(tmp_path, status)
        assert read_status(tmp_path) == status
        assert current_state(job, read_status(tmp_path)) is expected


def test_runs_are_listed_newest_first_and_other_folders_skipped(tmp_path):
    older = _job(created_at=DAY - 86400, ms=1_000)
    newer_a = _job(ms=2_000)
    newer_b = _job(ms=3_000)
    for job in (older, newer_a, newer_b):
        create_run_dir(tmp_path, job)
    (tmp_path / "notes").mkdir()
    (tmp_path / "2026-10-21" / "scratch").mkdir()
    (tmp_path / "README").write_text("team runs")

    assert [p.name for p in iter_runs(tmp_path)] == [newer_b.id, newer_a.id, older.id]
    assert [p.name for p in iter_runs(tmp_path, newest_first=False)] == [
        older.id,
        newer_a.id,
        newer_b.id,
    ]
    assert list(iter_runs(tmp_path / "missing")) == []
