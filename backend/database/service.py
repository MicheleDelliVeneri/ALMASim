"""Database service layer for querying and managing data."""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session, joinedload

from .models import (
    Observation,
    QueryResult,
    ScienceKeyword,
    ScientificCategory,
    SimulationJob,
    SimulationLog,
)


class DatabaseService:
    """Service layer for database operations."""

    def __init__(self, db: Session):
        self.db = db

    # Science Keywords and Categories

    def get_all_science_keywords(self) -> List[str]:
        """Get all unique science keywords."""
        stmt = select(ScienceKeyword.keyword).order_by(ScienceKeyword.keyword)
        return [row[0] for row in self.db.execute(stmt).all()]

    def get_all_scientific_categories(self) -> List[str]:
        """Get all unique scientific categories."""
        stmt = select(ScientificCategory.category).order_by(ScientificCategory.category)
        return [row[0] for row in self.db.execute(stmt).all()]

    # Observations

    def query_observations(
        self,
        science_keywords: Optional[Sequence[str]] = None,
        scientific_categories: Optional[Sequence[str]] = None,
        bands: Optional[Sequence[int]] = None,
        fov_range: Optional[tuple[float, float]] = None,
        time_resolution_range: Optional[tuple[float, float]] = None,
        frequency_range: Optional[tuple[float, float]] = None,
        target_name: Optional[str] = None,
        member_ous_uid: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = 0,
    ) -> List[Observation]:
        """
        Query observations with various filters.

        Args:
            science_keywords: List of science keywords (OR logic)
            scientific_categories: List of scientific categories (OR logic)
            bands: List of bands to filter by
            fov_range: Tuple of (min_fov, max_fov) in arcsec
            time_resolution_range: Tuple of (min_time, max_time)
            frequency_range: Tuple of (min_freq, max_freq) in GHz
            target_name: Filter by target name (partial match)
            member_ous_uid: Exact match on member ous uid
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of Observation objects
        """
        stmt = select(Observation).options(
            joinedload(Observation.scientific_category),
            joinedload(Observation.science_keywords),
        )

        # Build filters
        filters = []

        if member_ous_uid:
            filters.append(Observation.member_ous_uid == member_ous_uid)

        if target_name:
            filters.append(Observation.target_name.ilike(f"%{target_name}%"))

        if bands:
            filters.append(Observation.band.in_(bands))

        if fov_range:
            min_fov, max_fov = fov_range
            if min_fov is not None:
                filters.append(Observation.s_fov >= min_fov)
            if max_fov is not None:
                filters.append(Observation.s_fov <= max_fov)

        if time_resolution_range:
            min_time, max_time = time_resolution_range
            if min_time is not None:
                filters.append(Observation.t_resolution >= min_time)
            if max_time is not None:
                filters.append(Observation.t_resolution <= max_time)

        if frequency_range:
            min_freq, max_freq = frequency_range
            if min_freq is not None:
                filters.append(Observation.frequency >= min_freq)
            if max_freq is not None:
                filters.append(Observation.frequency <= max_freq)

        if scientific_categories:
            # Join with categories table for filtering
            stmt = stmt.join(Observation.scientific_category)
            filters.append(ScientificCategory.category.in_(scientific_categories))

        if science_keywords:
            # Join with keywords table for filtering
            stmt = stmt.join(Observation.science_keywords)
            filters.append(ScienceKeyword.keyword.in_(science_keywords))

        # Apply all filters
        if filters:
            stmt = stmt.where(and_(*filters))

        # Apply ordering
        stmt = stmt.order_by(Observation.created_at.desc())

        # Apply pagination
        if offset:
            stmt = stmt.offset(offset)
        if limit:
            stmt = stmt.limit(limit)

        return list(self.db.execute(stmt).unique().scalars().all())

    def get_observation_by_member_uid(
        self, member_ous_uid: str
    ) -> Optional[Observation]:
        """Get a single observation by member OUS UID."""
        stmt = select(Observation).where(Observation.member_ous_uid == member_ous_uid)
        return self.db.execute(stmt).scalar_one_or_none()

    def count_observations(
        self,
        science_keywords: Optional[Sequence[str]] = None,
        scientific_categories: Optional[Sequence[str]] = None,
        bands: Optional[Sequence[int]] = None,
    ) -> int:
        """Count observations matching the given filters."""
        stmt = select(func.count(Observation.id))

        filters = []

        if bands:
            filters.append(Observation.band.in_(bands))

        if scientific_categories:
            stmt = stmt.join(Observation.scientific_category)
            filters.append(ScientificCategory.category.in_(scientific_categories))

        if science_keywords:
            stmt = stmt.join(Observation.science_keywords)
            filters.append(ScienceKeyword.keyword.in_(science_keywords))

        if filters:
            stmt = stmt.where(and_(*filters))

        return self.db.execute(stmt).scalar_one()

    def create_observation(self, observation: Observation) -> Observation:
        """Create a new observation."""
        self.db.add(observation)
        self.db.commit()
        self.db.refresh(observation)
        return observation

    def bulk_create_observations(self, observations: List[Observation]) -> int:
        """
        Bulk create observations.

        Returns:
            Number of observations created
        """
        self.db.add_all(observations)
        self.db.commit()
        return len(observations)

    # Query Results

    def save_query_result(
        self,
        query_name: str,
        observation_ids: List[int],
        query_params: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
    ) -> QueryResult:
        """Save a query result for later retrieval."""
        query_result = QueryResult(
            query_name=query_name,
            query_params=query_params,
            result_count=len(observation_ids),
            observation_ids=observation_ids,
            description=description,
        )
        self.db.add(query_result)
        self.db.commit()
        self.db.refresh(query_result)
        return query_result

    def get_query_result(self, query_name: str) -> Optional[QueryResult]:
        """Get a saved query result by name."""
        stmt = (
            select(QueryResult)
            .where(QueryResult.query_name == query_name)
            .order_by(QueryResult.created_at.desc())
        )
        return self.db.execute(stmt).scalar_one_or_none()

    def list_query_results(self, limit: int = 50) -> List[QueryResult]:
        """List all saved query results."""
        stmt = select(QueryResult).order_by(QueryResult.created_at.desc()).limit(limit)
        return list(self.db.execute(stmt).scalars().all())

    # Simulation Jobs

    def create_simulation_job(
        self, source_name: str, observation_id: Optional[int] = None, **kwargs
    ) -> SimulationJob:
        """Create a new simulation job."""
        simulation = SimulationJob(
            source_name=source_name, observation_id=observation_id, **kwargs
        )
        self.db.add(simulation)
        self.db.commit()
        self.db.refresh(simulation)
        return simulation

    def get_simulation_by_id(self, simulation_id: uuid.UUID) -> Optional[SimulationJob]:
        """Get a simulation job by its UUID."""
        stmt = select(SimulationJob).where(SimulationJob.simulation_id == simulation_id)
        return self.db.execute(stmt).scalar_one_or_none()

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
            simulation.started_at = datetime.utcnow()
        elif status in ("completed", "failed"):
            simulation.completed_at = datetime.utcnow()

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
        stmt = select(SimulationJob).order_by(SimulationJob.created_at.desc())

        if status:
            stmt = stmt.where(SimulationJob.status == status)

        stmt = stmt.offset(offset).limit(limit)

        return list(self.db.execute(stmt).scalars().all())

    def count_simulations(self, status: Optional[str] = None) -> int:
        """Count simulation jobs."""
        stmt = select(func.count(SimulationJob.id))

        if status:
            stmt = stmt.where(SimulationJob.status == status)

        return self.db.execute(stmt).scalar_one()
