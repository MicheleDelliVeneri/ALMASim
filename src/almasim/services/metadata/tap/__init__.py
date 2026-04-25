"""TAP (Table Access Protocol) service for ALMA metadata queries."""

from .queries import (
    ALL_COLUMNS,
    load_metadata,
    query_metadata_by_science,
    query_metadata_by_targets,
    query_products,
    query_science_types,
)
from .service import (
    ExclusionFilters,
    InclusionFilters,
    get_science_types,
    get_tap_service,
    query_all_targets,
    query_by_science_type,
    query_observations,
    query_products_for_members,
    search_with_retry,
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
