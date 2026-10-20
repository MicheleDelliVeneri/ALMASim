"""TAP service functions for querying ALMA metadata."""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import pyvo
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from ..filters import (  # noqa: F401  (re-exported)
    ARRAY_TYPE_ANTENNA_PATTERNS,
    ExclusionFilters,
    InclusionFilters,
    qa2_codes,
)
from .adql import (  # noqa: F401  (re-exported)
    _SCIENCE_COLUMNS,
    _adql_in,
    _adql_num,
    _adql_str,
    _build_exclusion_conditions,
    _build_inclusion_conditions,
    _like_or_clause,
)
from .client import (  # noqa: F401  (re-exported)
    get_tap_service,
    reset_tap_service,
    search_with_retry,
)

logger = logging.getLogger(__name__)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_science_types():
    """Get available science types from TAP service."""
    service = get_tap_service()
    query = """
            SELECT science_keyword, scientific_category
            FROM ivoa.obscore
            WHERE science_observation = 'T'
            """
    try:
        db = service.search(query).to_table().to_pandas()
    except (
        pyvo.dal.exceptions.DALServiceError,
        requests.exceptions.RequestException,
    ) as e:
        logger.error("Error querying the TAP service: %s", e)
        reset_tap_service()
        raise
    science_keywords = db["science_keyword"].unique()
    scientific_category = db["scientific_category"].unique()
    science_keywords = list(filter(lambda x: x != "", science_keywords))
    scientific_category = list(filter(lambda x: x != "", scientific_category))

    unique_keywords = []
    for keywords_string in science_keywords:
        keywords_list = [keyword.strip() for keyword in keywords_string.split(",")]
        unique_keywords.extend(keywords_list)
    unique_keywords = sorted(set(unique_keywords))
    unique_keywords = [
        keyword
        for keyword in unique_keywords
        if (
            keyword != "Evolved stars: Shaping/physical structure"
            and keyword != "Exoplanets"
            and keyword != "Galaxy structure &evolution"
        )
    ]

    return unique_keywords, scientific_category


def query_observations(member_ous_uid, target_name):
    """Query for all science observations of given member OUS UID and target name.

    Parameters:
    member_ous_uid (str): The unique identifier for the member OUS
                                   to filter observations by.
    target_name (str): The target name to filter observations by.

    Returns:
    pandas.DataFrame: A table of query results.
    """
    service = get_tap_service()
    columns = [
        "target_name",
        "member_ous_uid",
        "group_ous_uid",
        "pwv",
        "schedblock_name",
        "velocity_resolution",
        "spatial_resolution",
        "s_ra",
        "s_dec",
        "s_fov",
        "t_resolution",
        "proposal_id",
        "cont_sensitivity_bandwidth",
        "sensitivity_10kms",
        "obs_release_date",
        "band_list",
        "bandwidth",
        "frequency",
        "frequency_support",
        "science_keyword",
        "scientific_category",
        "antenna_arrays",
        "t_max",
        "proposal_abstract",
        "qa2_passed",
        "type",
    ]
    columns_str = ", ".join(columns)
    query = f"""
            SELECT {columns_str}
            FROM ivoa.obscore
            WHERE member_ous_uid = {_adql_str(member_ous_uid)}
            AND target_name = {_adql_str(target_name)}
            AND is_mosaic = 'F'
            AND science_observation = 'T'
            """

    result = search_with_retry(service, query)
    return result


def query_all_targets(targets):
    """Query observations for all predefined targets and compile the results.

    Parameters:
    targets (list of tuples): A list of tuples ``(target_name, member_ous_uid)``.

    Returns:
    pandas.DataFrame: A DataFrame containing the results for all queried targets.
    """
    results = []

    for target_name, member_ous_uid in targets:
        result = query_observations(member_ous_uid, target_name)
        results.append(result)

    # Concatenate all DataFrames into a single DataFrame
    df = pd.concat(results, ignore_index=True)

    return df


def query_by_science_type(
    include: Optional[InclusionFilters] = None,
    exclude: Optional[ExclusionFilters] = None,
):
    """Query for all science observations filtered by inclusion and exclusion criteria.

    Parameters:
    include: InclusionFilters dataclass (positive/keep-only filters)
    exclude: ExclusionFilters dataclass (negative/remove filters)

    Returns:
    pandas.DataFrame: A table of query results.
    """
    service = get_tap_service()
    columns_str = ", ".join(_SCIENCE_COLUMNS)

    conditions = []
    if include is not None:
        conditions.extend(_build_inclusion_conditions(include))
    else:
        # defaults when no filters provided
        conditions.append("is_mosaic = 'F'")
        conditions.append("science_observation = 'T'")
    if exclude is not None:
        conditions.extend(_build_exclusion_conditions(exclude))

    where_clause = " AND ".join(conditions)
    query = f"""
            SELECT {columns_str}
            FROM ivoa.obscore
            WHERE {where_clause}
            """
    return search_with_retry(service, query)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def query_products_for_members(member_ous_uids):
    """Return available data products for one or more member OUS UIDs.

    Each row in the result corresponds to a single data product (visibility data,
    cube, continuum image, etc.) associated with the given member_ous_uid(s).
    Unlike the main science query, rows are NOT deduplicated so all products are
    returned.

    Parameters:
    member_ous_uids: A single member OUS UID string or a list of them.

    Returns:
    pandas.DataFrame with columns:
        member_ous_uid, target_name, dataproduct_type, calib_level,
        access_url, obs_publisher_did, band_list, qa2_passed, type
    """
    service = get_tap_service()
    if isinstance(member_ous_uids, str):
        member_ous_uids = [member_ous_uids]
    product_columns = [
        "member_ous_uid",
        "target_name",
        "dataproduct_type",
        "calib_level",
        "access_url",
        "obs_publisher_did",
        "band_list",
        "qa2_passed",
        "type",
    ]
    columns_str = ", ".join(product_columns)
    query = f"""
            SELECT {columns_str}
            FROM ivoa.obscore
            WHERE member_ous_uid IN {_adql_in(member_ous_uids)}
            AND science_observation = 'T'
            """
    return search_with_retry(service, query)
