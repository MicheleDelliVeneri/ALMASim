"""Database service layer for querying and managing data."""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session, joinedload

from almasim.services.metadata.filters import (
    ARRAY_TYPE_ANTENNA_PATTERNS,
    ExclusionFilters,
    InclusionFilters,
    qa2_codes,
)

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
        include: Optional[InclusionFilters] = None,
        exclude: Optional[ExclusionFilters] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = 0,
    ) -> List[Observation]:
        """The cached observations matching the same filters a TAP query takes.

        ``public_only``, ``science_only`` and ``exclude_mosaic`` are TAP-side
        selections: the cache only ever holds rows a TAP query returned.
        """
        include = include or InclusionFilters()
        exclude = exclude or ExclusionFilters()

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
    def _apply_inclusion_filters(stmt, p: InclusionFilters):
        """Return (stmt, filters) with all positive filter clauses applied."""
        filters: list = []
        filters.extend(DatabaseService._direct_filters(p))
        filters.extend(DatabaseService._array_filters(p))
        filters.extend(DatabaseService._range_filters(p))
        stmt, taxonomy_filters = DatabaseService._taxonomy_filters(stmt, p)
        filters.extend(taxonomy_filters)
        return stmt, filters

    @staticmethod
    def _direct_filters(p: InclusionFilters) -> list:
        """Simple equality / ilike filters that don't require extra joins."""
        filters = []
        if p.source_name:
            filters.append(Observation.target_name.ilike(f"%{p.source_name}%"))
        if p.band:
            filters.append(Observation.band.in_(p.band))
        if p.qa2_status:
            filters.append(Observation.qa2_passed.in_(qa2_codes(p.qa2_status)))
        if p.obs_type:
            filters.append(or_(*(Observation.obs_type.ilike(f"%{t}%") for t in p.obs_type)))
        if p.antenna_arrays:
            filters.append(Observation.antenna_arrays.ilike(f"%{p.antenna_arrays}%"))
        if p.proposal_id_prefix:
            filters.append(
                or_(
                    *(
                        Observation.proposal_id.ilike(f"{prefix}%")
                        for prefix in p.proposal_id_prefix
                    )
                )
            )
        if p.proposal_id:
            filters.append(Observation.proposal_id.in_(p.proposal_id))
        return filters

    @staticmethod
    def _array_filters(p: InclusionFilters) -> list:
        """Filters for array_type (antenna prefix) and array_configuration."""
        filters = []
        if p.array_type:
            clauses = [
                Observation.antenna_arrays.ilike(pat)
                for atype in p.array_type
                for pat in ARRAY_TYPE_ANTENNA_PATTERNS.get(atype, [f"%{atype}%"])
            ]
            if clauses:
                filters.append(or_(*clauses))
        if p.array_configuration:
            filters.append(
                or_(*(Observation.schedblock_name.ilike(f"%{c}%") for c in p.array_configuration))
            )
        return filters

    @staticmethod
    def _range_filters(p: InclusionFilters) -> list:
        """Numeric and date range filters."""
        filters = []
        # Int.Time sits in t_max on rows cached before MADIV-28 T1.12.
        integration_time = func.coalesce(Observation.t_resolution, Observation.t_max)
        for col, rng in (
            (Observation.s_fov, p.fov_range),
            (integration_time, p.time_resolution_range),
            (Observation.frequency, p.frequency_range),
            (Observation.spatial_resolution, p.angular_resolution_range),
        ):
            if rng:
                lo, hi = rng
                if lo is not None:
                    filters.append(col >= lo)
                if hi is not None:
                    filters.append(col <= hi)
        # The cache holds only the release date. TAP filters on the observation's
        # end time (t_max), so MetadataService does not answer dated queries from here.
        if p.observation_date_range:
            lo_str, hi_str = p.observation_date_range
            if lo_str:
                filters.append(Observation.obs_release_date >= datetime.fromisoformat(lo_str))
            if hi_str:
                filters.append(Observation.obs_release_date <= datetime.fromisoformat(hi_str))
        return filters

    @staticmethod
    def _taxonomy_filters(stmt, p: InclusionFilters):
        """Filters that require joining the keywords / categories association tables."""
        filters = []
        if p.scientific_category:
            stmt = stmt.join(Observation.scientific_category)
            filters.append(ScientificCategory.category.in_(p.scientific_category))
        if p.science_keyword:
            stmt = stmt.join(Observation.science_keywords)
            filters.append(ScienceKeyword.keyword.in_(p.science_keyword))
        return stmt, filters

    @staticmethod
    def _build_exclusion_filters(e: ExclusionFilters) -> list:
        """Return a list of NOT conditions for the exclusion parameters."""
        from .models import observation_keywords

        filters = []

        if e.source_name:
            for name in e.source_name:
                filters.append(~Observation.target_name.ilike(f"%{name}%"))

        if e.obs_type:
            for t in e.obs_type:
                filters.append(~Observation.obs_type.ilike(f"%{t}%"))

        # Solar exclusion matches TAP: the target name, a science keyword or the
        # category mentioning the Sun excludes the observation.
        if e.solar:
            filters.append(~Observation.target_name.ilike("%sun%"))

        excluded_keywords = list(e.science_keyword or [])
        excluded_categories = list(e.scientific_category or [])
        if e.solar:
            excluded_keywords.append("sun")
            excluded_categories.append("sun")

        if excluded_keywords:
            for kw in excluded_keywords:
                kw_subq = (
                    select(observation_keywords.c.observation_id)
                    .join(
                        ScienceKeyword,
                        ScienceKeyword.id == observation_keywords.c.keyword_id,
                    )
                    .where(ScienceKeyword.keyword.ilike(f"%{kw}%"))
                )
                filters.append(~Observation.id.in_(kw_subq))

        if excluded_categories:
            for cat in excluded_categories:
                cat_subq = (
                    select(Observation.id)
                    .join(Observation.scientific_category)
                    .where(ScientificCategory.category.ilike(f"%{cat}%"))
                )
                filters.append(~Observation.id.in_(cat_subq))

        return filters

    def get_observation_by_member_uid(self, member_ous_uid: str) -> Optional[Observation]:
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

    # Query Results

    # Query presets live in query_results, one row per name. Rows saved before
    # MADIV-28 T1.10 may repeat a name: reads take the newest, writes fold the rest.

    def _query_results_named(self, query_name: str) -> List[QueryResult]:
        stmt = (
            select(QueryResult)
            .where(QueryResult.query_name == query_name)
            .order_by(QueryResult.created_at.desc(), QueryResult.id.desc())
        )
        return list(self.db.execute(stmt).scalars().all())

    def save_query_preset(
        self,
        query_name: str,
        filters: Dict[str, Any],
        *,
        description: Optional[str] = None,
        result_count: int = 0,
    ) -> QueryResult:
        """Create the preset, or replace the one already saved under this name."""
        rows = self._query_results_named(query_name)
        if rows:
            row, duplicates = rows[0], rows[1:]
            for duplicate in duplicates:
                self.db.delete(duplicate)
            row.created_at = datetime.utcnow()
        else:
            row = QueryResult(query_name=query_name, observation_ids=[])
            self.db.add(row)
        row.query_params = {"filters": filters}
        row.result_count = result_count
        row.description = description
        self.db.commit()
        self.db.refresh(row)
        return row

    def get_query_result(self, query_name: str) -> Optional[QueryResult]:
        """Get a saved query result by name (the newest if the name repeats)."""
        rows = self._query_results_named(query_name)
        return rows[0] if rows else None

    def list_query_results(self, limit: int = 50) -> List[QueryResult]:
        """List saved query results, newest first, one per name."""
        stmt = select(QueryResult).order_by(QueryResult.created_at.desc(), QueryResult.id.desc())
        newest: Dict[str, QueryResult] = {}
        for row in self.db.execute(stmt).scalars():
            newest.setdefault(row.query_name, row)
            if len(newest) == limit:
                break
        return list(newest.values())

    def delete_query_result(self, query_name: str) -> bool:
        """Delete every saved query result with this name. Returns True if any existed."""
        rows = self._query_results_named(query_name)
        for row in rows:
            self.db.delete(row)
        if rows:
            self.db.commit()
        return bool(rows)

    # Simulation Jobs

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
