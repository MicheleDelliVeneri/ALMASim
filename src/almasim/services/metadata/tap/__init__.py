"""TAP (Table Access Protocol) service for ALMA metadata queries."""

from .service import (
    get_tap_service,
    search_with_retry,
    get_science_types,
    query_observations,
    query_all_targets,
    query_by_science_type,
    query_products_for_members,
    InclusionFilters,
    ExclusionFilters,
)
from .queries import (
    ALL_COLUMNS,
    query_science_types,
    query_metadata_by_science,
    query_metadata_by_targets,
    query_products,
    load_metadata,
)

__all__ = [
    # Service layer
    "get_tap_service",
    "search_with_retry",
    "get_science_types",
    "query_observations",
    "query_all_targets",
    "query_by_science_type",
    "query_products_for_members",
    "InclusionFilters",
    "ExclusionFilters",
    # High-level query layer
    "ALL_COLUMNS",
    "query_science_types",
    "query_metadata_by_science",
    "query_metadata_by_targets",
    "query_products",
    "load_metadata",
]
