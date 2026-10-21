"""The cluster as the console shows it: partitions, their load, your queue (S2.5).

One ``sinfo`` and one ``squeue`` call, cached for ``CACHE_S`` seconds, so a
page that refreshes often costs Slurm little. The resource panel takes its
limits from here (CPUs and memory per node, from Slurm and the profile).
"""

from __future__ import annotations

import getpass
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel

from almasim.jobs.profile import Profile

router = APIRouter(prefix="/api/v2", tags=["cluster"])

CACHE_S = 30.0

Runner = Callable[[list[str]], subprocess.CompletedProcess[str]]


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)


def _number(text: str) -> int | None:
    digits = text.rstrip("+")
    return int(digits) if digits.isdigit() else None


def parse_sinfo(text: str) -> list[dict[str, Any]]:
    """``sinfo -h -o "%P|%a|%l|%D|%c|%m|%C"`` lines as partitions."""
    partitions = []
    for line in text.splitlines():
        parts = line.strip().split("|")
        if len(parts) != 7:
            continue
        name, avail, limit, nodes, cpus, memory, cpu_states = parts
        alloc, idle, other, total = (
            _number(x) or 0 for x in (cpu_states.split("/") + ["0"] * 4)[:4]
        )
        usable = total - other  # drained or down CPUs do not count
        partitions.append(
            {
                "name": name.rstrip("*"),
                "default": name.endswith("*"),
                "up": avail == "up",
                "time_limit": None if limit in ("infinite", "UNLIMITED") else limit,
                "nodes": _number(nodes),
                "cpus_per_node": _number(cpus),
                "memory_mb_per_node": _number(memory),
                "cpus": {"allocated": alloc, "idle": idle, "unavailable": other, "total": total},
                "load": round(alloc / usable, 3) if usable > 0 else None,
            }
        )
    return partitions


def parse_squeue(text: str) -> list[dict[str, Any]]:
    """``squeue -h -o "%i|%j|%T|%M|%l|%P|%r|%D|%C"`` lines as jobs."""
    jobs = []
    for line in text.splitlines():
        parts = line.strip().split("|")
        if len(parts) != 9:
            continue
        job_id, name, state, elapsed, limit, partition, reason, nodes, cpus = parts
        jobs.append(
            {
                "id": job_id,
                "name": name,
                "state": state,
                "elapsed": elapsed,
                "time_limit": limit,
                "partition": partition,
                "reason": "" if reason in ("None", "(null)") else reason,
                "nodes": _number(nodes),
                "cpus": _number(cpus),
            }
        )
    return jobs


class ClusterView:
    def __init__(self, profile: Profile, run: Runner = _run, cache_s: float = CACHE_S) -> None:
        self.profile = profile
        self.run = run
        self.cache_s = cache_s
        self._lock = threading.Lock()
        self._at = -1e9
        self._value: dict[str, Any] = {}

    def forget(self) -> None:
        """Read Slurm again next time (the profile changed)."""
        with self._lock:
            self._at = -1e9

    def get(self) -> dict[str, Any]:
        with self._lock:
            if time.monotonic() - self._at > self.cache_s:
                self._value = self._read()
                self._at = time.monotonic()
            return self._value

    def _read(self) -> dict[str, Any]:
        profile = self.profile
        configured = {
            name: {"cpus": p.cpus, "memory": p.memory, "max_time": p.max_time}
            for name, p in profile.partitions.items()
        }
        base: dict[str, Any] = {
            "defaults": {"partition": profile.partition, "account": profile.account},
            "configured": configured,
            "checked_at": time.time(),
        }
        if not shutil.which("sinfo"):
            return {**base, "available": False, "partitions": [], "queue": []}
        info = self.run(["sinfo", "-h", "-o", "%P|%a|%l|%D|%c|%m|%C"])
        queue = self.run(
            ["squeue", "-u", getpass.getuser(), "-h", "-o", "%i|%j|%T|%M|%l|%P|%r|%D|%C"]
        )
        return {
            **base,
            "available": info.returncode == 0,
            "error": info.stderr.strip() if info.returncode else "",
            "partitions": parse_sinfo(info.stdout) if info.returncode == 0 else [],
            "queue": parse_squeue(queue.stdout) if queue.returncode == 0 else [],
        }


class CpuCounts(BaseModel):
    allocated: int
    idle: int
    unavailable: int
    total: int


class Partition(BaseModel):
    name: str
    default: bool
    up: bool
    time_limit: str | None
    nodes: int | None
    cpus_per_node: int | None
    memory_mb_per_node: int | None
    cpus: CpuCounts
    load: float | None


class QueuedJob(BaseModel):
    id: str
    name: str
    state: str
    elapsed: str
    time_limit: str
    partition: str
    reason: str
    nodes: int | None
    cpus: int | None


class Defaults(BaseModel):
    partition: str | None
    account: str | None


class Configured(BaseModel):
    """A partition's limits as the site profile states them."""

    cpus: int | None
    memory: str | None
    max_time: str | None


class Cluster(BaseModel):
    defaults: Defaults
    configured: dict[str, Configured]
    checked_at: float
    available: bool
    error: str = ""
    partitions: list[Partition]
    queue: list[QueuedJob]


@router.get("/cluster", response_model=Cluster)
def cluster(request: Request) -> dict[str, Any]:
    """Partitions with their CPU load, the profile's defaults and limits, and your queue."""
    return request.app.state.cluster.get()
