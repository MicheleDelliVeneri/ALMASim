"""TAP service functions for querying ALMA metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import pyvo
import pandas as pd
from astropy.time import Time
from tenacity import retry, stop_after_attempt, wait_exponential
import requests

# Map human-friendly QA2 labels to ALMA TAP stored values
_QA2_STATUS_MAP = {
    "Pass": "T",
    "Fail": "F",
    "SemiPass": "X",
    # Pass-through for raw values already in TAP form
    "T": "T",
    "F": "F",
    "X": "X",
}

# Antenna prefix patterns used to identify array type from antenna_arrays column
# 12m arrays: DA / DV antennas; 7m arrays: CM antennas; TP: PM antennas
_ARRAY_TYPE_ANTENNA_PATTERNS = {
    "12m": ["%DA%", "%DV%"],
    "7m": ["%CM%"],
    "TP": ["%PM%"],
}


@dataclass
class InclusionFilters:
    """Positive (include-only) filters for ALMA TAP science queries."""

    science_keyword: Optional[List[str]] = None
    scientific_category: Optional[List[str]] = None
    band: Optional[List[int]] = None
    fov_range: Optional[Tuple[float, float]] = None
    time_resolution_range: Optional[Tuple[float, float]] = None
    frequency_range: Optional[Tuple[float, float]] = None
    source_name: Optional[str] = None
    antenna_arrays: Optional[str] = None
    array_type: Optional[List[str]] = None  # e.g. ['12m', '7m', 'TP']
    array_configuration: Optional[List[str]] = None  # e.g. ['C-1', 'C-2']
    angular_resolution_range: Optional[Tuple[float, float]] = None
    observation_date_range: Optional[Tuple[str, str]] = None
    qa2_status: Optional[List[str]] = None
    obs_type: Optional[List[str]] = None
    proposal_id_prefix: Optional[List[str]] = None  # e.g. ['2016.', '2017.']
    public_only: bool = True  # default: only query non-proprietary data
    science_only: bool = True  # default: only science observations
    exclude_mosaic: bool = True  # default: exclude mosaic observations


@dataclass
class ExclusionFilters:
    """Negative (exclude) filters for ALMA TAP science queries."""

    science_keyword: Optional[List[str]] = None
    scientific_category: Optional[List[str]] = None
    source_name: Optional[List[str]] = None
    obs_type: Optional[List[str]] = None
    solar: bool = False  # exclude observations related to the Sun


def get_tap_service():
    """Establishes a connection to a TAP service for astronomical data access.

    This function attempts to connect to multiple TAP service endpoints until
    a successful connection is established. It also performs a simple test query
    to verify that the service is responding.

    Returns:
        pyvo.dal.TAPService: A TAP service object if a successful connection is made,
                             otherwise None.

    Notes:
        - The function uses an infinite loop to retry connections indefinitely.
        - It prints messages to the console indicating connection status and errors.
        - If all URLs fail, it restarts the loop to try again from the beginning.

    Dependencies:
        - pyvo: The Python library for Virtual Observatory data access.
    """
    urls = [
        "https://almascience.eso.org/tap",
        "https://almascience.nao.ac.jp/tap",
        "https://almascience.nrao.edu/tap",
    ]
    while True:  # Infinite loop to keep trying until successful
        for url in urls:
            try:
                service = pyvo.dal.TAPService(url)
                # Test the connection with a simple query to ensure the service is working
                service.search("SELECT TOP 1 * FROM ivoa.obscore")
                print(f"Connected successfully to {url}")
                return service
            except Exception as e:
                print("Failed to connect to {}: {}".format(url, e))
                print("Retrying other servers...")
        print("All URLs attempted and failed, retrying...")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def search_with_retry(service, query):
    """Search TAP service with retry logic."""
    return service.search(query, response_timeout=120).to_table().to_pandas()


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
        print("Error querying TAP service: {}".format(e))
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
            WHERE member_ous_uid = '{member_ous_uid}'
            AND target_name = '{target_name}'
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


_SCIENCE_COLUMNS = [
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
    "data_rights",
]


def _like_or_clause(column: str, values: list, negate: bool = False) -> str:
    """Build a (col LIKE '%v1%' OR col LIKE '%v2%') clause."""
    op = "NOT LIKE" if negate else "LIKE"
    join_op = " AND " if negate else " OR "
    clauses = [f"{column} {op} '%{v}%'" for v in values]
    return f"({join_op.join(clauses)})"


def _build_inclusion_conditions(f: InclusionFilters) -> list:
    """Translate an InclusionFilters object into a list of ADQL condition strings."""
    conds = []

    if f.science_keyword:
        conds.append(_like_or_clause("science_keyword", f.science_keyword))

    if f.scientific_category:
        conds.append(_like_or_clause("scientific_category", f.scientific_category))

    if f.band:
        conds.append(_like_or_clause("band_list", [str(b) for b in f.band]))

    if f.source_name:
        conds.append(f"target_name LIKE '%{f.source_name}%'")

    if f.antenna_arrays:
        conds.append(f"antenna_arrays LIKE '%{f.antenna_arrays}%'")

    # Array type: 12m → DA/DV antennas, 7m → CM antennas, TP → PM antennas
    if f.array_type:
        type_clauses = [
            f"antenna_arrays LIKE '{pattern}'"
            for atype in f.array_type
            for pattern in _ARRAY_TYPE_ANTENNA_PATTERNS.get(atype, [f"%{atype}%"])
        ]
        conds.append(f"({' OR '.join(type_clauses)})")

    # Array configuration matched against schedblock_name (e.g. 'C-1', 'C-2')
    if f.array_configuration:
        conf_clauses = [f"schedblock_name LIKE '%{c}%'" for c in f.array_configuration]
        conds.append(f"({' OR '.join(conf_clauses)})")

    if f.angular_resolution_range:
        lo, hi = f.angular_resolution_range
        conds.append(f"spatial_resolution BETWEEN {lo} AND {hi}")

    if f.observation_date_range:
        mjd_min = Time(f.observation_date_range[0], format="iso", scale="utc").mjd
        mjd_max = Time(f.observation_date_range[1], format="iso", scale="utc").mjd
        conds.append(f"t_max >= {mjd_min} AND t_max <= {mjd_max}")

    # QA2 status: map human-friendly labels to TAP-stored single-char values
    if f.qa2_status:
        mapped = [_QA2_STATUS_MAP.get(s, s) for s in f.qa2_status]
        statuses = "', '".join(mapped)
        conds.append(f"qa2_passed IN ('{statuses}')")

    if f.obs_type:
        clauses = " OR ".join(f"type LIKE '%{t}%'" for t in f.obs_type)
        conds.append(f"({clauses})")

    if f.fov_range:
        conds.append(f"s_fov BETWEEN {f.fov_range[0]} AND {f.fov_range[1]}")

    if f.time_resolution_range:
        lo, hi = f.time_resolution_range
        conds.append(f"t_resolution BETWEEN {lo} AND {hi}")

    if f.frequency_range:
        conds.append(
            f"frequency BETWEEN {f.frequency_range[0]} AND {f.frequency_range[1]}"
        )

    if f.proposal_id_prefix:
        prefix_clauses = [f"proposal_id LIKE '{p}%'" for p in f.proposal_id_prefix]
        conds.append(f"({' OR '.join(prefix_clauses)})")

    if f.public_only:
        conds.append("data_rights = 'Public'")

    if f.science_only:
        conds.append("science_observation = 'T'")

    if f.exclude_mosaic:
        conds.append("is_mosaic = 'F'")

    return conds


def _build_exclusion_conditions(f: ExclusionFilters) -> list:
    """Translate an ExclusionFilters object into a list of ADQL NOT conditions."""
    conds = []

    if f.science_keyword:
        for kw in f.science_keyword:
            conds.append(f"science_keyword NOT LIKE '%{kw}%'")

    if f.scientific_category:
        for cat in f.scientific_category:
            conds.append(f"scientific_category NOT LIKE '%{cat}%'")

    if f.source_name:
        for name in f.source_name:
            conds.append(f"LOWER(target_name) NOT LIKE LOWER('%{name}%')")

    if f.obs_type:
        for t in f.obs_type:
            conds.append(f"type NOT LIKE '%{t}%'")

    # Solar exclusion: all three fields must not mention 'sun' (case-insensitive)
    if f.solar:
        conds.append("LOWER(target_name) NOT LIKE '%sun%'")
        conds.append("LOWER(science_keyword) NOT LIKE '%sun%'")
        conds.append("LOWER(scientific_category) NOT LIKE '%sun%'")

    return conds


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
    uid_list = "', '".join(member_ous_uids)
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
            WHERE member_ous_uid IN ('{uid_list}')
            AND science_observation = 'T'
            """
    return search_with_retry(service, query)
