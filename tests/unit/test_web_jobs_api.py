"""API v2: commands, jobs, logs and live events (web console plan, S2.3)."""

from __future__ import annotations

import asyncio
import getpass
import subprocess
import sys
import time

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from almasim.jobs import JobState, read_manifest, write_manifest
from almasim.jobs.executors import slurm as slurm_exec
from almasim.jobs.executors.slurm import SlurmState
from almasim.jobs.profile import CommandPolicy, Profile
from almasim.web.app import COOKIE, create_app
from almasim.web.events import EventHub, stream

pytestmark = pytest.mark.unit

TOKEN = "api-test-token"


class FakeSlurm:
    def __init__(self):
        self.next_id = 9100
        self.states: dict[str, SlurmState] = {}
        self.cancelled: list[str] = []

    def submit(self, job, run_dir, profile, *, after=(), **kwargs):
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
    return Profile(
        runs_root=tmp_path / "runs",
        python=sys.executable,
        partition="zwickyA-CPU",
        commands={"doctor": CommandPolicy(executor="server")},
    )


@pytest.fixture
def client(profile, slurm, tmp_path):
    app = create_app(TOKEN, static_dir=tmp_path / "none", profile=profile, events_interval=0.2)
    with TestClient(app) as c:
        c.cookies.set(COOKIE, TOKEN)
        yield c


def test_the_catalogue_is_served(client):
    names = [c["name"] for c in client.get("/api/v2/commands").json()]
    assert "image image-from-ms" in names and "metadata" in names


def test_validation_returns_the_clis_message_or_the_plan(client):
    bad = client.post(
        "/api/v2/commands/validate", json={"argv": ["image", "clean-quality", "/o", "--nope"]}
    ).json()
    assert not bad["ok"] and "--nope" in bad["problems"][0]
    good = client.post(
        "/api/v2/commands/validate",
        json={"argv": ["image", "clean-quality", "/o"], "resources": {"cpus": 8}},
    ).json()
    assert good["ok"] and good["plan"]["executor"] == "slurm"
    assert good["plan"]["resources"]["cpus"] == 8
    assert good["plan"]["resources"]["partition"] == "zwickyA-CPU"


def test_a_job_is_submitted_listed_shown_cancelled_and_resubmitted(client, slurm):
    refused = client.post("/api/v2/jobs", json={"argv": ["image", "clean-quality", "--nope"]})
    assert refused.status_code == 422 and refused.json()["detail"]["problems"]

    created = client.post(
        "/api/v2/jobs", json={"argv": ["image", "clean-quality", "/out"], "title": "check"}
    )
    assert created.status_code == 201, created.text
    run = created.json()
    assert run["slurm_job_id"] == "9100" and run["state"] == "queued" and run["title"] == "check"

    listed = client.get("/api/v2/jobs").json()
    assert [r["id"] for r in listed["runs"]] == [run["id"]]
    assert listed["runs"][0]["reason"] == "Priority"
    assert client.get("/api/v2/jobs", params={"state": "running"}).json()["runs"] == []

    detail = client.get(f"/api/v2/jobs/{run['short_id']}").json()
    assert "job.json" in detail["files"] and detail["id"] == run["id"]

    assert client.post(f"/api/v2/jobs/{run['id']}/cancel").status_code == 200
    assert slurm.cancelled == ["9100"]

    again = client.post(f"/api/v2/jobs/{run['id']}/resubmit")
    assert again.status_code == 201 and again.json()["id"] != run["id"]
    assert client.get("/api/v2/jobs/ZZZZZZZZ").status_code == 404


def test_a_slurm_runs_accounting_comes_from_sacct(client, monkeypatch):
    run = client.post("/api/v2/jobs", json={"argv": ["image", "clean-quality", "/out"]}).json()
    asked = []

    def fake_accounting(slurm_id):
        asked.append(slurm_id)
        return [{"JobID": slurm_id, "State": "COMPLETED", "MaxRSS": ""}]

    monkeypatch.setattr(slurm_exec, "accounting", fake_accounting)
    got = client.get(f"/api/v2/jobs/{run['id']}/accounting").json()
    assert asked == ["9100"]
    assert got == {
        "slurm_job_id": "9100",
        "rows": [{"JobID": "9100", "State": "COMPLETED", "MaxRSS": ""}],
    }

    def broken(slurm_id):
        raise slurm_exec.SlurmError("sacct: slurmdbd unreachable")

    monkeypatch.setattr(slurm_exec, "accounting", broken)
    failed = client.get(f"/api/v2/jobs/{run['id']}/accounting")
    assert failed.status_code == 502 and "slurmdbd" in failed.json()["detail"]


def test_a_server_run_has_no_accounting(client):
    run = client.post("/api/v2/jobs", json={"argv": ["doctor", "--help"]}).json()
    got = client.get(f"/api/v2/jobs/{run['id']}/accounting").json()
    assert got == {"slurm_job_id": None, "rows": []}


def test_sacct_rows_become_records():
    out = (
        "6402|almasim|COMPLETED|0:0|00:03:12|2026-10-21T10:00:00|2026-10-21T10:03:12"
        "|node47|4|16G||00:05:00\n"
        "6402.batch|batch|COMPLETED|0:0|00:03:12|s|e|node47|4||2100M|00:05:00\n"
        "short|line\n"
    )
    rows = slurm_exec.accounting(
        "6402", run=lambda cmd: subprocess.CompletedProcess(cmd, 0, out, "")
    )
    assert [r["JobID"] for r in rows] == ["6402", "6402.batch"]
    assert rows[1]["MaxRSS"] == "2100M" and rows[0]["NodeList"] == "node47"
    with pytest.raises(slurm_exec.SlurmError, match="not a Slurm job id"):
        slurm_exec.accounting("6402; rm -rf /")
    with pytest.raises(slurm_exec.SlurmError, match="no db"):
        slurm_exec.accounting(
            "6402", run=lambda cmd: subprocess.CompletedProcess(cmd, 1, "", "no db")
        )


def test_only_the_owner_can_cancel(client, profile):
    run = client.post("/api/v2/jobs", json={"argv": ["image", "clean-quality", "/out"]}).json()
    run_dir = next(profile.runs_root.glob(f"*/{run['id']}"))
    job = read_manifest(run_dir)
    job.owner = "someone-else"
    write_manifest(run_dir, job)
    response = client.post(f"/api/v2/jobs/{run['id']}/cancel")
    assert response.status_code == 409 and "someone-else" in response.json()["detail"]


def test_a_server_job_runs_and_its_log_can_be_followed(client):
    run = client.post("/api/v2/jobs", json={"argv": ["doctor", "--help"]}).json()
    assert run["executor"] == "server"
    deadline = time.monotonic() + 60
    while not client.get(f"/api/v2/jobs/{run['id']}").json()["finished"]:
        assert time.monotonic() < deadline
        time.sleep(0.2)
    assert client.get(f"/api/v2/jobs/{run['id']}").json()["state"] == "succeeded"

    first = client.get(f"/api/v2/jobs/{run['id']}/log", params={"limit": 10}).json()
    assert first["next_offset"] == 10 and len(first["text"]) == 10
    rest = client.get(f"/api/v2/jobs/{run['id']}/log", params={"offset": 10}).json()
    assert "Usage" in first["text"] + rest["text"]
    assert rest["next_offset"] == rest["size"]
    empty = client.get(f"/api/v2/jobs/{run['id']}/log", params={"stream": "stderr"}).json()
    assert empty["text"] == ""


def test_log_reads_never_split_a_character(client, profile):
    run = client.post("/api/v2/jobs", json={"argv": ["image", "clean-quality", "/out"]}).json()
    run_dir = next(profile.runs_root.glob(f"*/{run['id']}"))
    (run_dir / "stdout.log").write_text("ab€cd", encoding="utf-8")  # € is 3 bytes
    part = client.get(f"/api/v2/jobs/{run['id']}/log", params={"limit": 3}).json()
    assert part["text"] == "ab" and part["next_offset"] == 2
    rest = client.get(f"/api/v2/jobs/{run['id']}/log", params={"offset": 2}).json()
    assert rest["text"] == "€cd"


def test_the_hub_reports_each_change_once(profile, slurm):
    from almasim.jobs.service import submit_command

    hub = EventHub(profile, interval=60)
    run = submit_command(["image", "clean-quality", "/out"], profile)
    [event] = hub.tick()
    assert event["run"]["state"] == "queued"
    assert hub.tick() == []
    slurm.states[run.job.slurm_job_id] = SlurmState("RUNNING")
    [changed] = hub.tick()
    assert changed["run"]["state"] == "running"
    assert [n for n, _ in hub.since(1)] == [2]
    assert hub.since(0) is not None


def test_only_a_finished_run_of_yours_can_be_deleted(client, profile, slurm):
    run = client.post("/api/v2/jobs", json={"argv": ["image", "clean-quality", "/out"]}).json()
    run_dir = next(profile.runs_root.glob(f"*/{run['id']}"))
    unfinished = client.delete(f"/api/v2/jobs/{run['id']}")
    assert unfinished.status_code == 409 and "cancel it first" in unfinished.json()["detail"]

    slurm.states[run["slurm_job_id"]] = SlurmState("COMPLETED", "exit code 0:0")
    job = read_manifest(run_dir)
    job.owner = "someone-else"
    write_manifest(run_dir, job)
    theirs = client.delete(f"/api/v2/jobs/{run['id']}")
    assert theirs.status_code == 409 and "someone-else" in theirs.json()["detail"]

    assert run_dir.exists()  # the refusal changed nothing
    job.owner = getpass.getuser()
    write_manifest(run_dir, job)
    assert client.delete(f"/api/v2/jobs/{run['id']}").status_code == 204
    assert not run_dir.exists()
    assert client.get(f"/api/v2/jobs/{run['id']}").status_code == 404


def test_the_hub_says_when_a_run_is_gone(profile, slurm):
    import shutil

    from almasim.jobs.service import submit_command

    hub = EventHub(profile, interval=60)
    run = submit_command(["image", "clean-quality", "/out"], profile)
    hub.tick()
    shutil.rmtree(run.run_dir)
    [gone] = hub.tick()
    assert gone == {"type": "gone", "id": run.job.id}
    assert hub.tick() == []


def test_the_stream_starts_with_a_snapshot_and_resumes_from_an_id(profile, slurm):
    from almasim.jobs.service import submit_command

    hub = EventHub(profile, interval=60)
    submit_command(["image", "clean-quality", "/out"], profile)
    hub.tick()

    async def first(last_id):
        gen = stream(hub, last_id, poll=0.01)
        try:
            return await gen.__anext__()
        finally:
            await gen.aclose()

    snapshot = asyncio.run(first(None))
    assert snapshot.startswith("id: 1\nevent: snapshot\n") and '"state":"queued"' in snapshot
    resumed = asyncio.run(first(0))
    assert resumed.startswith("id: 1\nevent: run\n")


def test_events_need_the_token(profile, slurm, tmp_path):
    app = create_app(TOKEN, static_dir=tmp_path / "none", profile=profile, events_interval=60)
    with TestClient(app) as c:
        assert c.get("/api/v2/events").status_code == 401
        assert c.get("/api/v2/jobs").status_code == 401
