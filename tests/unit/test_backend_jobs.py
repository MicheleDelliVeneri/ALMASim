"""One job store, stale-job reconciliation and one UTC clock (MADIV-28 T6.4)."""

from __future__ import annotations

import importlib
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

import pytest

pytestmark = pytest.mark.unit


def test_clock_helpers(backend_db):
    clock = importlib.import_module("app.core.clock")
    now = clock.utc_now()
    assert now.tzinfo is UTC
    assert clock.db_now().tzinfo is None
    naive = datetime(2026, 10, 21, 3, 0)
    assert clock.as_utc(naive) == naive.replace(tzinfo=UTC)
    assert clock.as_utc(now) is now and clock.as_utc(None) is None


def test_live_progress_stamps_updated_at_when_editing(backend_db):
    live = importlib.import_module("app.services.live_progress")

    @dataclass
    class Job:
        value: int = 0
        updated_at: datetime = field(default_factory=lambda: datetime(2000, 1, 1, tzinfo=UTC))

    store = live.LiveProgress()
    store.put("a", Job())
    with store.editing("a") as job:
        job.value = 3
    assert store.get("a").value == 3 and store.get("a").updated_at.year > 2000
    with store.editing("missing") as job:
        assert job is None
    assert [job.value for job in store.values()] == [3]
    assert store.pop("a") is True and store.pop("a") is False


def test_simulation_and_query_stores_use_utc(backend_db):
    status_store = importlib.import_module("app.services.status_store")
    sims = status_store.StatusStore()
    created = sims.create("s1")
    assert created.created_at.tzinfo is UTC
    sims.update("s1", status="running", log="started")
    assert sims.get("s1").status == "running" and "+00:00" in sims.get("s1").logs[0]
    assert sims.cancel("s1") and sims.is_cancelled("s1") and not sims.cancel("nope")
    assert [s.simulation_id for s in sims.list_all()] == ["s1"] and sims.delete("s1")

    queries = status_store.QueryStore()
    queries.create("q1")
    queries.append_rows("q1", [{"a": 1}])
    queries.complete("q1")
    page = queries.get_page("q1", 0, 10)
    assert page["done"] and page["total_fetched"] == 1
    assert queries.cancel("q1") and queries.is_cancelled("q1") and queries.delete("q1")


@pytest.fixture
def download_tables(backend_db):
    models = backend_db.models
    engine = backend_db.session.get_bind()
    models.Base.metadata.create_all(
        engine, tables=[models.DownloadJobRecord.__table__, models.DownloadFileRecord.__table__]
    )
    return backend_db


def test_job_repository_reconciles_interrupted_jobs(download_tables, monkeypatch):
    jobs = importlib.import_module("app.db.jobs")
    models = download_tables.models
    # SQLite cannot hold SimulationJob (a PostgreSQL ARRAY column): reconcile downloads only.
    assert jobs._UNFINISHED[models.SimulationJob] == ("queued", "running")
    monkeypatch.setattr(jobs, "_UNFINISHED", {models.DownloadJobRecord: ("pending", "running")})
    ids = {}
    for status in ("pending", "running", "completed", "failed"):
        ids[status] = uuid.uuid4()
        download_tables.session.add(
            models.DownloadJobRecord(job_id=ids[status], destination="/d", status=status)
        )
    download_tables.session.commit()

    repository = jobs.JobRepository(download_tables.session)
    assert repository.reconcile_stale_jobs() == 2
    statuses = {
        status: repository.get(models.DownloadJobRecord, job_id).status
        for status, job_id in ids.items()
    }
    assert statuses == {
        "pending": "failed",
        "running": "failed",
        "completed": "completed",
        "failed": "failed",
    }
    assert repository.get(models.DownloadJobRecord, ids["running"]).error == jobs.INTERRUPTED
    assert repository.reconcile_stale_jobs() == 0


def test_job_repository_update_list_delete(download_tables):
    jobs = importlib.import_module("app.db.jobs")
    record = download_tables.models.DownloadJobRecord
    repository = jobs.JobRepository(download_tables.session)
    job_id = uuid.uuid4()
    download_tables.session.add(record(job_id=job_id, destination="/d"))
    download_tables.session.commit()

    assert repository.update(record, job_id, status="running") is True
    assert [job.status for job in repository.list(record, status="running")] == ["running"]
    assert repository.update(record, uuid.uuid4(), status="x") is False
    assert repository.delete(record, job_id) is True and repository.list(record) == []
    assert repository.delete(record, job_id) is False


def test_the_simulation_list_sorts_database_and_live_jobs_together(backend_db, monkeypatch):
    """A naive (database, UTC) time next to an aware (live) one used to raise in the sort."""
    router = importlib.import_module("app.api.v1.routers.simulation")
    stored = type(
        "Row",
        (),
        {
            "simulation_id": uuid.uuid4(),
            "status": "completed",
            "progress": 1.0,
            "message": "",
            "created_at": datetime(2026, 1, 1),
            "updated_at": datetime(2026, 1, 1),
            "error": None,
            "output_path": None,
        },
    )()

    class FakeService:
        def __init__(self, db):
            pass

        def list_simulations(self, limit):
            return [stored]

    store = importlib.import_module("app.services.status_store").StatusStore()
    store.create("live")
    monkeypatch.setattr(router, "DatabaseService", FakeService)
    monkeypatch.setattr(router, "status_store", store)

    listed = router.list_simulations(db=None)
    assert [s.simulation_id for s in listed.simulations] == ["live", str(stored.simulation_id)]
    assert all(s.updated_at.tzinfo is not None for s in listed.simulations)
