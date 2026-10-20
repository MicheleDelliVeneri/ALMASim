"""The cache's answer to a metadata query: the same filters TAP takes (MADIV-28 T6.10)."""

from datetime import datetime
from typing import Optional

from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import joinedload

from almasim.services.metadata.filters import (
    ARRAY_TYPE_ANTENNA_PATTERNS,
    ExclusionFilters,
    InclusionFilters,
    qa2_codes,
)

from .models import Observation, ScienceKeyword, ScientificCategory


def observation_query(
    include: Optional[InclusionFilters] = None, exclude: Optional[ExclusionFilters] = None
):
    """The select of cached observations matching ``include`` and not ``exclude``, newest first.

    ``public_only``, ``science_only`` and ``exclude_mosaic`` are TAP-side
    selections: the cache only ever holds rows a TAP query returned.
    """
    include = include or InclusionFilters()
    exclude = exclude or ExclusionFilters()
    stmt = select(Observation).options(
        joinedload(Observation.scientific_category),
        joinedload(Observation.science_keywords),
    )
    stmt, filters = _apply_inclusion_filters(stmt, include)
    filters.extend(_build_exclusion_filters(exclude))
    if filters:
        stmt = stmt.where(and_(*filters))
    return stmt.order_by(Observation.created_at.desc())


def _apply_inclusion_filters(stmt, p: InclusionFilters):
    """Return (stmt, filters) with all positive filter clauses applied."""
    filters: list = []
    filters.extend(_direct_filters(p))
    filters.extend(_array_filters(p))
    filters.extend(_range_filters(p))
    stmt, taxonomy_filters = _taxonomy_filters(stmt, p)
    filters.extend(taxonomy_filters)
    return stmt, filters


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
            or_(*(Observation.proposal_id.ilike(f"{prefix}%") for prefix in p.proposal_id_prefix))
        )
    if p.proposal_id:
        filters.append(Observation.proposal_id.in_(p.proposal_id))
    return filters


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
