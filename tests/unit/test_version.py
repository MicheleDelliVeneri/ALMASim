"""almasim.__version__ comes from the package metadata (MADIV-28 T0.3)."""

import importlib
import importlib.metadata

import pytest

# The package re-exports the string under the same name, so import the module itself.
version_module = importlib.import_module("almasim.__version__")

pytestmark = pytest.mark.unit


def test_version_matches_installed_metadata():
    assert version_module.__version__ == importlib.metadata.version("almasim")


def test_version_falls_back_when_not_installed(monkeypatch):
    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing)
    try:
        assert importlib.reload(version_module).__version__ == "0+unknown"
    finally:
        monkeypatch.undo()
        importlib.reload(version_module)
