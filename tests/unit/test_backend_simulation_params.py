"""The API request becomes the simulator's SimulationParams (MADIV-28)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def service_module(monkeypatch):
    """app.services.simulation_service, with the database package stubbed for this test."""
    monkeypatch.syspath_prepend(str(REPO_ROOT / "src"))
    monkeypatch.syspath_prepend(str(REPO_ROOT / "backend"))
    stub = MagicMock()
    for name in ("app.db", "app.db.config", "app.db.models", "app.db.service"):
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, stub)
    return importlib.import_module("app.services.simulation_service")


def _request(**overrides):
    from app.schemas.simulation import SimulationParamsCreate

    values = dict(
        idx=0,
        source_name="J0842+1218",
        member_ouid="uid://A001/X1/X1",
        project_name="test",
        ra=130.6,
        dec=12.3,
        band=6,
        ang_res=1.0,
        vel_res=10.0,
        fov=25.0,
        obs_date="2017-03-14",
        pwv=2.1,
        int_time=450.0,
        bandwidth=1.875,
        freq=269.7,
        freq_support="[251.05..252.92GHz,1938.48kHz]",
        cont_sens=0.045,
        antenna_array="A001 A002",
    )
    values.update(overrides)
    return SimulationParamsCreate(**values)


def test_the_seeds_reach_the_simulator(service_module, tmp_path):
    service = service_module.SimulationService(
        main_dir=tmp_path,
        output_dir=tmp_path,
        tng_dir=tmp_path,
        galaxy_zoo_dir=tmp_path,
        hubble_dir=tmp_path,
    )
    params = service._simulation_params(_request(seed=11, background_seed=7))
    assert params.seed == 11 and params.background_seed == 7
    assert params.source_name == "J0842+1218" and params.output_dir == str(tmp_path.resolve())


def test_no_seed_means_fresh_noise(service_module, tmp_path):
    service = service_module.SimulationService(
        main_dir=tmp_path,
        output_dir=tmp_path,
        tng_dir=tmp_path,
        galaxy_zoo_dir=tmp_path,
        hubble_dir=tmp_path,
    )
    assert service._simulation_params(_request()).seed is None


def test_to_sim_params_uses_the_server_directories(service_module, tmp_path):
    request = _request(main_dir="/elsewhere", tng_dir="/elsewhere", seed=3)
    params = service_module.to_sim_params(
        request,
        main_dir=tmp_path / "main",
        output_dir=tmp_path / "out",
        tng_dir=tmp_path / "tng",
        galaxy_zoo_dir=tmp_path / "gz",
        hubble_dir=tmp_path / "hubble",
    )
    assert params.main_dir == str((tmp_path / "main").resolve())
    assert params.tng_dir == str((tmp_path / "tng").resolve())
    assert params.seed == 3
    assert params.remote is False
