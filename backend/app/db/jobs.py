"""One repository for the persisted simulation and download jobs (MADIV-28 T6.4)."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.core.clock import db_now

from .models import DownloadJobRecord, SimulationJob

logger = logging.getLogger(__name__)

# SimulationJob or DownloadJobRecord.
JobModel = Any

# The statuses of a job that was still going: a job in one of them when the
# server starts was interrupted by the previous shutdown.
_UNFINISHED: dict[JobModel, tuple[str, ...]] = {
    SimulationJob: ("queued", "running"),
    DownloadJobRecord: ("pending", "running"),
}
INTERRUPTED = "interrupted: the server stopped while the job was running"


def _key(model: JobModel):
    return model.simulation_id if model is SimulationJob else model.job_id


def as_uuid(job_id: Any) -> uuid.UUID | None:
    """``job_id`` as the UUID the id columns hold, or None if it is not one.

    The routers carry ids as strings. PostgreSQL's UUID type converts them,
    but on SQLite SQLAlchemy calls ``.hex`` on the value, so a string id
    failed there (MADIV-28 W0.3).
    """
    if isinstance(job_id, uuid.UUID):
        return job_id
    try:
        return uuid.UUID(str(job_id))
    except ValueError:
        return None


class JobRepository:
    """Get, list, update and reconcile both kinds of job the same way."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def get(self, model: JobModel, job_id: Any) -> Any | None:
        key = as_uuid(job_id)
        if key is None:
            return None
        return self.db.execute(select(model).where(_key(model) == key)).scalar_one_or_none()

    def list(self, model: JobModel, *, status: str | None = None, limit: int = 100) -> list:
        stmt = select(model).order_by(model.created_at.desc())
        if status:
            stmt = stmt.where(model.status == status)
        return list(self.db.execute(stmt.limit(limit)).scalars().all())

    def update(self, model: JobModel, job_id: Any, **fields: Any) -> bool:
        """Set ``fields`` on the job; whether it exists."""
        job = self.get(model, job_id)
        if job is None:
            return False
        for name, value in fields.items():
            setattr(job, name, value)
        job.updated_at = db_now()
        self.db.commit()
        return True

    def delete(self, model: JobModel, job_id: Any) -> bool:
        job = self.get(model, job_id)
        if job is None:
            return False
        self.db.delete(job)
        self.db.commit()
        return True

    def reconcile_stale_jobs(self) -> int:
        """Mark the jobs a previous server left unfinished as failed; how many there were.

        Their worker threads died with that process, so without this they
        stayed "running" for ever.
        """
        count = 0
        for model, statuses in _UNFINISHED.items():
            result = self.db.execute(
                update(model)
                .where(model.status.in_(statuses))
                .values(status="failed", error=INTERRUPTED, updated_at=db_now())
            )
            count += getattr(result, "rowcount", 0) or 0
        self.db.commit()
        if count:
            logger.warning("Marked %d interrupted job(s) as failed", count)
        return count
