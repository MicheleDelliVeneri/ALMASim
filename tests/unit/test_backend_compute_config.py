"""Compute-backend settings from requests are an allowlist (MADIV-28 W0.4)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.core.compute_config import validate_compute_backend_config  # noqa: E402
from fastapi import HTTPException  # noqa: E402

pytestmark = pytest.mark.unit


def _detail(backend: str, config: dict) -> str:
    with pytest.raises(HTTPException) as info:
        validate_compute_backend_config(backend, config)
    assert info.value.status_code == 400
    return str(info.value.detail)


def test_slurm_takes_the_documented_settings():
    config = {
        "queue": "zwickyA-CPU",
        "project": "ska.madiv",
        "walltime": "1-12:00:00",
        "cores": 4,
        "memory": "16GB",
        "n_workers": 0,
    }
    assert validate_compute_backend_config("slurm", config) == config


@pytest.mark.parametrize(
    "key", ["job_extra_directives", "job_script_prologue", "python", "worker_extra_args"]
)
def test_slurm_refuses_settings_that_reach_the_job_script(key):
    detail = _detail("slurm", {key: ["--x"]})
    assert key in detail
    assert "queue" in detail  # names what is accepted


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("queue", "normal\n#SBATCH --uid=0"),
        ("queue", "zwickyA-CPU --exclusive"),
        ("project", "a;rm -rf ~"),
        ("walltime", "two hours"),
        ("memory", "16GB\n"),
        ("cores", "4"),
        ("cores", True),
        ("cores", 0),
        ("n_workers", 5000),
        ("scheduler_host", "host name"),
    ],
)
def test_slurm_values_are_checked(key, value):
    assert f"compute_backend_config.{key}" in _detail("slurm", {key: value})


def test_cleared_form_fields_mean_not_set():
    assert validate_compute_backend_config("slurm", {"queue": "", "project": None}) == {}


def test_local_and_dask_take_only_their_own_settings():
    assert validate_compute_backend_config("local", {"n_workers": 2}) == {"n_workers": 2}
    assert "queue" in _detail("local", {"queue": "normal"})
    assert "max_workers" in _detail("dask", {"max_workers": 2})


def test_kubernetes_resources_are_cpu_and_memory_quantities():
    good = {"requests": {"cpu": "500m", "memory": "2Gi"}, "limits": {"cpu": "2"}}
    assert validate_compute_backend_config("kubernetes", {"resources": good}) == {"resources": good}
    assert "resources" in _detail("kubernetes", {"resources": {"requests": {"gpu": "1"}}})
    assert "resources" in _detail("kubernetes", {"resources": {"limits": {"cpu": "lots"}}})


def test_unknown_backend_is_refused():
    assert "Unknown compute backend" in _detail("mpi", {})
