"""Fixtures shared by the unit tests."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_SWAPPED = ("app.db", "app.services.metadata_service")


def _swapped(name: str) -> bool:
    return any(name == prefix or name.startswith(prefix + ".") for prefix in _SWAPPED)


@pytest.fixture
def backend_db(monkeypatch):
    """The backend's real database modules on an in-memory SQLite database.

    Several router tests replace the ``database`` package with stubs when they
    are imported, so the real modules are swapped in for the test and the stubs
    put back afterwards. Only tables whose columns SQLite can store are created.
    """
    pytest.importorskip("sqlalchemy")
    monkeypatch.syspath_prepend(str(REPO_ROOT / "src"))
    monkeypatch.syspath_prepend(str(REPO_ROOT / "backend"))
    saved = {name: module for name, module in sys.modules.items() if _swapped(name)}
    for name in saved:
        del sys.modules[name]
    try:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        models = importlib.import_module("app.db.models")
        service = importlib.import_module("app.db.service")
        engine = create_engine("sqlite://")
        models.Base.metadata.create_all(
            engine,
            tables=[
                models.Observation.__table__,
                models.ScienceKeyword.__table__,
                models.ScientificCategory.__table__,
                models.observation_keywords,
                models.QueryResult.__table__,
            ],
        )
        session = sessionmaker(bind=engine)()
        yield SimpleNamespace(
            models=models,
            service=service,
            session=session,
            db=service.DatabaseService(session),
        )
        session.close()
    finally:
        for name in [name for name in sys.modules if _swapped(name)]:
            del sys.modules[name]
        sys.modules.update(saved)
