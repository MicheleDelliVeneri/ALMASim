"""Simulation jobs and their logs."""

import uuid
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.clock import db_now

from ..jobs import JobRepository
from ..models import SimulationJob, SimulationLog


class SimulationRepository:
    """Simulation jobs: one row per run, with its log lines."""

    def __init__(self, db: Session):
        self.db = db

    def create_simulation_job(
        self, source_name: str, observation_id: Optional[int] = None, **kwargs
    ) -> SimulationJob:
        """Create a new simulation job."""
        simulation = SimulationJob(source_name=source_name, observation_id=observation_id, **kwargs)
        self.db.add(simulation)
        self.db.commit()
        self.db.refresh(simulation)
        return simulation

    def get_simulation_by_id(self, simulation_id: uuid.UUID) -> Optional[SimulationJob]:
        """Get a simulation job by its UUID."""
        return JobRepository(self.db).get(SimulationJob, simulation_id)

    def update_simulation_status(
        self,
        simulation_id: uuid.UUID,
        status: str,
        progress: Optional[float] = None,
        current_step: Optional[str] = None,
        message: Optional[str] = None,
        error: Optional[str] = None,
    ) -> Optional[SimulationJob]:
        """Update simulation job status."""
        simulation = self.get_simulation_by_id(simulation_id)
        if not simulation:
            return None

        simulation.status = status
        if progress is not None:
            simulation.progress = progress
        if current_step is not None:
            simulation.current_step = current_step
        if message is not None:
            simulation.message = message
        if error is not None:
            simulation.error = error

        # Update timestamps
        if status == "running" and not simulation.started_at:
            simulation.started_at = db_now()
        elif status in ("completed", "failed", "cancelled"):
            simulation.completed_at = db_now()

        self.db.commit()
        self.db.refresh(simulation)
        return simulation

    def add_simulation_log(
        self,
        simulation_id: uuid.UUID,
        message: str,
        level: str = "INFO",
    ) -> SimulationLog:
        """Add a log entry to a simulation."""
        log = SimulationLog(
            simulation_id=simulation_id,
            message=message,
            level=level,
        )
        self.db.add(log)
        self.db.commit()
        return log

    def get_simulation_logs(
        self,
        simulation_id: uuid.UUID,
        limit: int = 1000,
    ) -> List[SimulationLog]:
        """Get logs for a simulation."""
        stmt = (
            select(SimulationLog)
            .where(SimulationLog.simulation_id == simulation_id)
            .order_by(SimulationLog.timestamp.desc())
            .limit(limit)
        )
        return list(self.db.execute(stmt).scalars().all())

    def list_simulations(
        self,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[SimulationJob]:
        """List simulation jobs with optional status filter."""
        if offset:
            stmt = select(SimulationJob).order_by(SimulationJob.created_at.desc())
            if status:
                stmt = stmt.where(SimulationJob.status == status)
            return list(self.db.execute(stmt.offset(offset).limit(limit)).scalars().all())
        return JobRepository(self.db).list(SimulationJob, status=status, limit=limit)
