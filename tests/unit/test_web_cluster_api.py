"""API v2 cluster: partitions, load and queue (web console plan, S2.5)."""

from __future__ import annotations

import subprocess

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from almasim.jobs.profile import Partition, Profile
from almasim.web.api import cluster as cluster_api
from almasim.web.api.cluster import ClusterView, parse_sinfo, parse_squeue
from almasim.web.app import COOKIE, create_app

pytestmark = pytest.mark.unit

SINFO = """zwickyA-GPU|up|infinite|9|64+|770000+|18/2062/128/2208
zwickyA-CPU*|up|infinite|6|96+|770000+|58/646/1120/1824
zwickyB-CPU|drain|2-00:00:00|8|96|380000|0/0/2048/2048
"""
SQUEUE = """6454|wrap|RUNNING|0:01|5:00|zwickyA-CPU|None|1|2
6439|golden-83e127d7|PENDING|0:00|12:00:00|zwickyA-CPU|Priority|1|12
"""


def test_sinfo_lines_become_partitions():
    gpu, cpu, drained = parse_sinfo(SINFO + "garbage line\n")
    assert cpu["name"] == "zwickyA-CPU" and cpu["default"] and cpu["up"]
    assert cpu["cpus_per_node"] == 96 and cpu["memory_mb_per_node"] == 770000
    assert cpu["time_limit"] is None
    # 58 of the 704 usable CPUs (1824 total, 1120 down or drained) are busy.
    assert cpu["load"] == round(58 / 704, 3)
    assert not drained["up"] and drained["time_limit"] == "2-00:00:00" and drained["load"] is None
    assert gpu["cpus"] == {"allocated": 18, "idle": 2062, "unavailable": 128, "total": 2208}


def test_squeue_lines_become_jobs():
    running, pending = parse_squeue(SQUEUE)
    assert running["id"] == "6454" and running["reason"] == "" and running["cpus"] == 2
    assert pending["reason"] == "Priority"


def test_the_view_asks_slurm_once_per_cache_period(monkeypatch):
    monkeypatch.setattr(cluster_api.shutil, "which", lambda tool: "/usr/bin/" + tool)
    calls = []

    def run(cmd):
        calls.append(cmd[0])
        out = SINFO if cmd[0] == "sinfo" else SQUEUE
        return subprocess.CompletedProcess(cmd, 0, out, "")

    profile = Profile(
        partition="zwickyA-CPU",
        account="ska",
        partitions={"zwickyA-CPU": Partition("zwickyA-CPU", cpus=96, memory="770G")},
    )
    view = ClusterView(profile, run=run, cache_s=60)
    first = view.get()
    assert view.get() is first and calls == ["sinfo", "squeue"]
    assert first["available"] and first["defaults"] == {
        "partition": "zwickyA-CPU",
        "account": "ska",
    }
    assert first["configured"]["zwickyA-CPU"]["memory"] == "770G"
    assert [p["name"] for p in first["partitions"]][:2] == ["zwickyA-GPU", "zwickyA-CPU"]


def test_without_slurm_the_view_says_so(monkeypatch, tmp_path):
    monkeypatch.setattr(cluster_api.shutil, "which", lambda tool: None)
    app = create_app(
        "t", static_dir=tmp_path / "none", profile=Profile(runs_root=tmp_path), events_interval=60
    )
    client = TestClient(app)
    client.cookies.set(COOKIE, "t")
    body = client.get("/api/v2/cluster").json()
    assert body["available"] is False and body["partitions"] == [] and body["queue"] == []
