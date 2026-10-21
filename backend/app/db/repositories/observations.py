"""Cached observations and their science keywords and categories."""

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from almasim.services.metadata.filters import ExclusionFilters, InclusionFilters

from ..cache_query import observation_query
from ..models import Observation, ScienceKeyword, ScientificCategory


class ObservationRepository:
    """The metadata cache."""

    def __init__(self, db: Session):
        self.db = db

    def get_all_science_keywords(self) -> list[str]:
        """Get all unique science keywords."""
        stmt = select(ScienceKeyword.keyword).order_by(ScienceKeyword.keyword)
        return [row[0] for row in self.db.execute(stmt).all()]

    def get_all_scientific_categories(self) -> list[str]:
        """Get all unique scientific categories."""
        stmt = select(ScientificCategory.category).order_by(ScientificCategory.category)
        return [row[0] for row in self.db.execute(stmt).all()]

    def query_observations(
        self,
        include: InclusionFilters | None = None,
        exclude: ExclusionFilters | None = None,
        limit: int | None = None,
        offset: int | None = 0,
    ) -> list[Observation]:
        """The cached observations matching the same filters a TAP query takes."""
        stmt = observation_query(include, exclude)
        if offset:
            stmt = stmt.offset(offset)
        if limit:
            stmt = stmt.limit(limit)
        return list(self.db.execute(stmt).unique().scalars().all())

    def get_observation_by_member_uid(self, member_ous_uid: str) -> Observation | None:
        """Get a single observation by member OUS UID."""
        stmt = select(Observation).where(Observation.member_ous_uid == member_ous_uid)
        return self.db.execute(stmt).scalar_one_or_none()

    def count_observations_with_keywords(self) -> int:
        """Count observations that have at least one science keyword associated."""
        from sqlalchemy import exists

        from ..models import observation_keywords

        stmt = select(func.count(Observation.id)).where(
            exists().where(observation_keywords.c.observation_id == Observation.id)
        )
        return self.db.execute(stmt).scalar_one()
