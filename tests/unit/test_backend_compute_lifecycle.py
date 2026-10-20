"""The backend's compute backends: one default, per-job ones closed (MADIV-28 T6.5)."""

from __future__ import annotations

import importlib
import logging
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def dependencies(backend_db, monkeypatch):
    module = importlib.import_module("app.core.dependencies")
    monkeypatch.setattr(module, "_default_backend", None)
    created = []

    def fake_create(kind, **config):
        backend = MagicMock(name=f"{kind}-backend")
        backend.kind = kind
        created.append(backend)
        return backend

    monkeypatch.setattr(module, "create_backend", fake_create)
    module.created = created
    return module


def test_the_default_backend_is_created_once_and_closed_at_shutdown(dependencies, monkeypatch):
    monkeypatch.setattr(dependencies.settings, "COMPUTE_BACKEND", "local")
    first = dependencies.get_compute_backend()
    assert dependencies.get_compute_backend() is first
    assert len(dependencies.created) == 1

    dependencies.close_compute_backend()
    first.close.assert_called_once()
    assert dependencies.get_compute_backend() is not first


def test_fallbacks_to_the_local_backend_are_logged(dependencies, monkeypatch, caplog):
    monkeypatch.setattr(dependencies.settings, "COMPUTE_BACKEND", "nonsense")
    with caplog.at_level(logging.WARNING, logger=dependencies.__name__):
        assert dependencies.get_compute_backend().kind == "local"
    assert "Unknown COMPUTE_BACKEND" in caplog.text

    dependencies.close_compute_backend()
    caplog.clear()

    def failing(kind, **config):
        if kind == "dask":
            raise OSError("no scheduler at tcp://nowhere:8786")
        return MagicMock(kind=kind)

    monkeypatch.setattr(dependencies, "create_backend", failing)
    monkeypatch.setattr(dependencies.settings, "COMPUTE_BACKEND", "dask")
    with caplog.at_level(logging.WARNING, logger=dependencies.__name__):
        assert dependencies.get_compute_backend().kind == "local"
    assert "no scheduler" in caplog.text


@pytest.mark.parametrize("owns", [True, False])
def test_a_per_job_backend_is_closed_when_its_job_ends(backend_db, monkeypatch, tmp_path, owns):
    service_module = importlib.import_module("app.services.simulation_service")
    backend = MagicMock()
    service = service_module.SimulationService(
        main_dir=tmp_path,
        output_dir=tmp_path,
        tng_dir=tmp_path,
        galaxy_zoo_dir=tmp_path,
        hubble_dir=tmp_path,
        compute_backend=backend,
        owns_backend=owns,
    )

    def failing_params(params):
        raise RuntimeError("bad parameters")

    monkeypatch.setattr(service, "_simulation_params", failing_params)
    monkeypatch.setattr(service, "_persist_status", lambda *a, **k: None)
    monkeypatch.setattr(service, "_persist_log", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="bad parameters"):
        service.run_simulation("sim-1", MagicMock())
    assert backend.close.called is owns
