"""Database service layer for querying and managing data."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
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

# Antenna prefix → array type mapping (mirrors the TAP service constant)
_DB_ARRAY_PREFIX_MAP: Dict[str, List[str]] = {
    "12m": ["%DA%", "%DV%"],
    "7m": ["%CM%"],
    "TP": ["%PM%"],
}


@dataclass
class ObservationQueryParams:
    """Inclusion parameters for database observation queries."""

    science_keywords: Optional[List[str]] = None
    scientific_categories: Optional[List[str]] = None
    bands: Optional[List[int]] = None
    antenna_arrays: Optional[str] = None
    array_type: Optional[List[str]] = None
    array_configuration: Optional[List[str]] = None
    angular_resolution_range: Optional[tuple] = None
    observation_date_range: Optional[tuple] = None
    qa2_status: Optional[List[str]] = None
    obs_type: Optional[List[str]] = None
    fov_range: Optional[tuple] = None
    time_resolution_range: Optional[tuple] = None
    frequency_range: Optional[tuple] = None
    target_name: Optional[str] = None
    member_ous_uid: Optional[str] = None
    proposal_id_prefix: Optional[List[str]] = None


@dataclass
class ObservationExclusionParams:
    """Exclusion parameters for database observation queries."""

    science_keywords: Optional[List[str]] = None
    scientific_categories: Optional[List[str]] = None
    source_names: Optional[List[str]] = None
    obs_types: Optional[List[str]] = None
    solar: bool = False


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
        include: Optional[ObservationQueryParams] = None,
        exclude: Optional[ObservationExclusionParams] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = 0,
        # Legacy flat kwargs kept for backwards compatibility with direct callers
        **kwargs,
    ) -> List[Observation]:
        """Query observations with inclusion and exclusion filter dataclasses.

        Accepts ``include`` (ObservationQueryParams) and ``exclude``
        (ObservationExclusionParams).  Any extra keyword arguments are merged
        into an ObservationQueryParams so existing call-sites continue working.
        """
        if include is None:
            include = ObservationQueryParams(**{
                k: v for k, v in kwargs.items()
                if k in ObservationQueryParams.__dataclass_fields__
            })
        if exclude is None:
            exclude = ObservationExclusionParams(**{
                k: v for k, v in kwargs.items()
                if k in ObservationExclusionParams.__dataclass_fields__
            })

        stmt = select(Observation).options(
            joinedload(Observation.scientific_category),
            joinedload(Observation.science_keywords),
        )
        stmt, filters = self._apply_inclusion_filters(stmt, include)
        filters.extend(self._build_exclusion_filters(exclude))

        if filters:
            stmt = stmt.where(and_(*filters))
        stmt = stmt.order_by(Observation.created_at.desc())
        if offset:
            stmt = stmt.offset(offset)
        if limit:
            stmt = stmt.limit(limit)
        return list(self.db.execute(stmt).unique().scalars().all())

    # -- private filter helpers -----------------------------------------------

    @staticmethod
    def _apply_inclusion_filters(stmt, p: ObservationQueryParams):
        """Return (stmt, filters) with all positive filter clauses applied."""
        filters: list = []
        filters.extend(DatabaseService._direct_filters(p))
        filters.extend(DatabaseService._array_filters(p))
        filters.extend(DatabaseService._range_filters(p))
        stmt, taxonomy_filters = DatabaseService._taxonomy_filters(stmt, p)
        filters.extend(taxonomy_filters)
        return stmt, filters

    @staticmethod
    def _direct_filters(p: ObservationQueryParams) -> list:
        """Simple equality / ilike filters that don't require extra joins."""
        filters = []
        if p.member_ous_uid:
            filters.append(Observation.member_ous_uid == p.member_ous_uid)
        if p.target_name:
            filters.append(Observation.target_name.ilike(f"%{p.target_name}%"))
        if p.bands:
            filters.append(Observation.band.in_(p.bands))
        if p.qa2_status:
            filters.append(Observation.qa2_passed.in_(p.qa2_status))
        if p.obs_type:
            filters.append(or_(*(Observation.obs_type.ilike(f"%{t}%") for t in p.obs_type)))
        if p.antenna_arrays:
            filters.append(Observation.antenna_arrays.ilike(f"%{p.antenna_arrays}%"))
        if p.proposal_id_prefix:
            filters.append(or_(*(Observation.proposal_id.ilike(f"{prefix}%") for prefix in p.proposal_id_prefix)))
        return filters

    @staticmethod
    def _array_filters(p: ObservationQueryParams) -> list:
        """Filters for array_type (antenna prefix) and array_configuration (schedblock_name)."""
        filters = []
        if p.array_type:
            clauses = [
                Observation.antenna_arrays.ilike(pat)
                for atype in p.array_type
                for pat in _DB_ARRAY_PREFIX_MAP.get(atype, [f"%{atype}%"])
            ]
            if clauses:
                filters.append(or_(*clauses))
        if p.array_configuration:
            filters.append(or_(*(
                Observation.schedblock_name.ilike(f"%{c}%") for c in p.array_configuration
            )))
        return filters

    @staticmethod
    def _range_filters(p: ObservationQueryParams) -> list:
        """Numeric and date range filters."""
        filters = []
        for col, rng in (
            (Observation.s_fov, p.fov_range),
            (Observation.t_resolution, p.time_resolution_range),
            (Observation.frequency, p.frequency_range),
            (Observation.spatial_resolution, p.angular_resolution_range),
        ):
            if rng:
                lo, hi = rng
                if lo is not None:
                    filters.append(col >= lo)
                if hi is not None:
                    filters.append(col <= hi)
        if p.observation_date_range:
            lo_str, hi_str = p.observation_date_range
            if lo_str:
                filters.append(Observation.obs_release_date >= datetime.fromisoformat(lo_str))
            if hi_str:
                filters.append(Observation.obs_release_date <= datetime.fromisoformat(hi_str))
        return filters

    @staticmethod
    def _taxonomy_filters(stmt, p: ObservationQueryParams):
        """Filters that require joining the keywords / categories association tables."""
        filters = []
        if p.scientific_categories:
            stmt = stmt.join(Observation.scientific_category)
            filters.append(ScientificCategory.category.in_(p.scientific_categories))
        if p.science_keywords:
            stmt = stmt.join(Observation.science_keywords)
            filters.append(ScienceKeyword.keyword.in_(p.science_keywords))
        return stmt, filters

    @staticmethod
    def _build_exclusion_filters(e: ObservationExclusionParams) -> list:
        """Return a list of NOT conditions for the exclusion parameters."""
        from .models import observation_keywords

        filters = []

        if e.source_names:
            for name in e.source_names:
                filters.append(~Observation.target_name.ilike(f"%{name}%"))

        if e.obs_types:
            for t in e.obs_types:
                filters.append(~Observation.obs_type.ilike(f"%{t}%"))

        if e.solar:
            filters.append(~Observation.target_name.ilike("%sun%"))

        if e.science_keywords:
            for kw in e.science_keywords:
                kw_subq = (
                    select(observation_keywords.c.observation_id)
                    .join(ScienceKeyword, ScienceKeyword.id == observation_keywords.c.keyword_id)
                    .where(ScienceKeyword.keyword.ilike(f"%{kw}%"))
                )
                filters.append(~Observation.id.in_(kw_subq))

        if e.scientific_categories:
            for cat in e.scientific_categories:
                cat_subq = (
                    select(Observation.id)
                    .join(Observation.scientific_category)
                    .where(ScientificCategory.category.ilike(f"%{cat}%"))
                )
                filters.append(~Observation.id.in_(cat_subq))

        return filters

    def get_observation_by_member_uid(
        self, member_ous_uid: str
    ) -> Optional[Observation]:
        """Get a single observation by member OUS UID."""
        stmt = select(Observation).where(Observation.member_ous_uid == member_ous_uid)
        return self.db.execute(stmt).scalar_one_or_none()

    def count_observations_with_keywords(self) -> int:
        """Count observations that have at least one science keyword associated."""
        from sqlalchemy import exists
        from .models import observation_keywords
        stmt = select(func.count(Observation.id)).where(
            exists().where(observation_keywords.c.observation_id == Observation.id)
        )
        return self.db.execute(stmt).scalar_one()

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
        elif status in ("completed", "failed", "cancelled"):
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
