"""Directory settings mean the same thing everywhere (MADIV-28 W0.3)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.core import config  # noqa: E402

import almasim  # noqa: E402

pytestmark = pytest.mark.unit


def test_relative_directories_are_taken_from_the_backend_directory(monkeypatch):
    monkeypatch.setenv("DATA_DIR", "data")
    monkeypatch.setenv("OUTPUT_DIR", "outputs")
    settings = config.Settings(_env_file=None)
    assert settings.DATA_DIR == config.BACKEND_DIR / "data"
    assert settings.OUTPUT_DIR == config.BACKEND_DIR / "outputs"


def test_absolute_directories_are_kept(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    assert config.Settings(_env_file=None).DATA_DIR == tmp_path


def test_main_dir_defaults_to_the_installed_package(monkeypatch):
    monkeypatch.delenv("MAIN_DIR", raising=False)
    assert config.Settings(_env_file=None).MAIN_DIR == Path(almasim.__file__).resolve().parent


def test_compute_backend_defaults_to_the_servers():
    """Leaving compute_backend out selects COMPUTE_BACKEND, not a new local pool."""
    from app.schemas.simulation import SimulationParamsBase

    assert SimulationParamsBase.model_fields["compute_backend"].default is None
