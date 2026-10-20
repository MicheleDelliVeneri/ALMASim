"""The backend database package after the move to app.db (MADIV-28 T6.2).

The session helpers, the startup initialisation and the callers that open
their own sessions run here on SQLite. The download store's session lookup
used to name ``SessionLocal``, which the lazy engine replaced.
"""

from __future__ import annotations

import importlib
import uuid
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def sqlite_sessions(backend_db, monkeypatch):
    """app.db.config.new_session bound to the fixture's SQLite engine, with the download tables."""
    from sqlalchemy.orm import sessionmaker

    models = backend_db.models
    engine = backend_db.session.get_bind()
    models.Base.metadata.create_all(
        engine,
        tables=[models.DownloadJobRecord.__table__, models.DownloadFileRecord.__table__],
    )
    config = importlib.import_module("app.db.config")
    monkeypatch.setattr(config, "new_session", sessionmaker(bind=engine))
    return config, engine


def test_download_store_persists_jobs_in_the_database(sqlite_sessions):
    from app.services.download_service import DownloadJob, DownloadStore

    store = DownloadStore()
    job_id = uuid.uuid4()  # SQLite needs the UUID object; PostgreSQL also takes the string
    store.create(DownloadJob(job_id=job_id, destination="/data/run"))  # type: ignore[arg-type]

    assert [record.destination for record in store.list_all()] == ["/data/run"]
    assert store.update_in_db(job_id, status="completed") is True
    assert store.get_from_db(job_id).status == "completed"
    assert store.delete_from_db(job_id) is True
    assert store.list_all() == []


def test_get_db_rolls_back_and_closes_on_error(monkeypatch):
    config = importlib.import_module("app.db.config")
    session = MagicMock()
    monkeypatch.setattr(config, "new_session", lambda: session)

    sessions = config.get_db()
    assert next(sessions) is session
    with pytest.raises(RuntimeError):
        sessions.throw(RuntimeError("request failed"))
    session.rollback.assert_called_once()
    session.close.assert_called_once()


def test_get_db_context_commits_on_success(monkeypatch):
    config = importlib.import_module("app.db.config")
    session = MagicMock()
    monkeypatch.setattr(config, "new_session", lambda: session)

    with config.get_db_context() as db:
        assert db is session
    session.commit.assert_called_once()
    session.close.assert_called_once()


def test_startup_creates_tables_and_imports_csvs_only_into_an_empty_database(
    backend_db, monkeypatch, tmp_path
):
    from sqlalchemy.orm import sessionmaker

    startup = importlib.import_module("app.db.startup")
    engine = backend_db.session.get_bind()
    monkeypatch.setattr(startup, "get_engine", lambda: engine)
    monkeypatch.setattr(startup, "new_session", sessionmaker(bind=engine))
    monkeypatch.setattr(startup, "init_db", lambda: None)  # the fixture created the tables
    imported = []

    def fake_import(db, data_dir):
        imported.append(data_dir)
        return {"cycle11.csv": 2, "empty.csv": 0}

    monkeypatch.setattr(startup, "initialize_database_from_csv", fake_import)

    # An empty database and no data directory: tables only.
    startup.initialize_database_on_startup(tmp_path / "missing")
    assert imported == [] and startup.check_database_initialized() is False

    # An empty database with a data directory: the CSVs are imported.
    startup.initialize_database_on_startup(tmp_path)
    assert imported == [tmp_path]

    # A populated database is left alone.
    backend_db.session.add(backend_db.models.Observation(member_ous_uid="uid://A001/X1/X1"))
    backend_db.session.commit()
    assert startup.check_database_initialized() is True
    startup.initialize_database_on_startup(tmp_path)
    assert imported == [tmp_path]


def test_science_types_from_tap_are_cached_in_the_database(backend_db, monkeypatch):
    metadata_service = importlib.import_module("app.services.metadata_service")
    monkeypatch.setattr(
        metadata_service, "query_science_types", lambda: (["Disks"], ["Star formation"])
    )

    service = metadata_service.MetadataService(db=backend_db.session)
    assert service.get_science_types() == (["Disks"], ["Star formation"])
    assert backend_db.db.get_all_science_keywords() == ["Disks"]
    assert backend_db.db.get_all_scientific_categories() == ["Star formation"]


def test_background_query_runs_in_its_own_session(backend_db, monkeypatch):
    metadata_service = importlib.import_module("app.services.metadata_service")
    config = importlib.import_module("app.db.config")
    status_store = importlib.import_module("app.services.status_store")
    seen = []

    class Context:
        def __enter__(self):
            return backend_db.session

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(config, "get_db_context", Context)
    monkeypatch.setattr(
        metadata_service.MetadataService,
        "_run_background_query_impl",
        lambda self, query_id, params, store: seen.append((self.db, query_id, store)),
    )

    metadata_service.MetadataService().run_background_query("q1", MagicMock())
    assert seen == [(backend_db.session, "q1", status_store.query_store)]
