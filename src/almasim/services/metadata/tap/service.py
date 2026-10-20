"""TAP service functions for querying ALMA metadata."""

from __future__ import annotations

import logging
import math
from typing import Optional

import pandas as pd
import pyvo
import requests
from astropy.time import Time
from tenacity import retry, stop_after_attempt, wait_exponential

from ..filters import (  # noqa: F401  (re-exported)
    ARRAY_TYPE_ANTENNA_PATTERNS,
    ExclusionFilters,
    InclusionFilters,
    qa2_codes,
)

logger = logging.getLogger(__name__)


_TAP_URLS = [
    "https://almascience.eso.org/tap",
    "https://almascience.nao.ac.jp/tap",
    "https://almascience.nrao.edu/tap",
]

_TAP_TEST_TIMEOUT = 30  # seconds for the connectivity test query

# The mirror that last answered; reused until a query on it fails.
_connected_service: pyvo.dal.TAPService | None = None


def reset_tap_service() -> None:
    """Forget the cached mirror, so the next query probes the mirrors again."""
    global _connected_service
    _connected_service = None


def get_tap_service(max_rounds: int = 2) -> pyvo.dal.TAPService:
    """Return a connected ALMA TAP service, trying each mirror in turn.

    The first mirror that answers a test query is cached and returned by
    later calls, so each query does not pay for a new connectivity probe.
    Tries each mirror URL up to *max_rounds* times total. Raises
    ``RuntimeError`` if every attempt fails so callers can handle the
    failure rather than blocking forever.
    """
    global _connected_service
    if _connected_service is not None:
        return _connected_service
    errors: list[str] = []
    for _ in range(max_rounds):
        for url in _TAP_URLS:
            try:
                service = pyvo.dal.TAPService(url)
                service.search(
                    "SELECT TOP 1 * FROM ivoa.obscore",
                    response_timeout=_TAP_TEST_TIMEOUT,
                )
                logger.info("Connected to the ALMA TAP service at %s", url)
                _connected_service = service
                return service
            except Exception as exc:
                msg = f"{url}: {exc}"
                errors.append(msg)
                logger.warning("Could not connect to the ALMA TAP service at %s", msg)
    raise RuntimeError(
        f"All ALMA TAP mirrors unreachable after {max_rounds} rounds. Errors:\n" + "\n".join(errors)
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def search_with_retry(service, query):
    """Search TAP service with retry logic."""
    try:
        return service.search(query, response_timeout=120).to_table().to_pandas()
    except Exception:
        # The next query probes the mirrors again instead of reusing this one.
        reset_tap_service()
        raise


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


def _adql_str(value: object) -> str:
    """An ADQL string literal for *value*: quoted, with embedded quotes doubled.

    User text goes into the WHERE clause, so a quote in it must not end the
    literal (``O'Brien`` broke the query; ``x' OR 'a'='a`` rewrote it).
    """
    return "'" + str(value).replace("'", "''") + "'"


def _adql_in(values) -> str:
    """An ADQL ``('a', 'b')`` list of string literals."""
    return "(" + ", ".join(_adql_str(value) for value in values) + ")"


def _adql_num(value: object) -> str:
    """*value* written into ADQL, after checking it is a finite number."""
    number = float(value)  # type: ignore[arg-type]
    if not math.isfinite(number):
        raise ValueError(f"Not a finite number: {value!r}")
    return str(value)


def _like_or_clause(column: str, values: list, negate: bool = False) -> str:
    """Build a (col LIKE '%v1%' OR col LIKE '%v2%') clause."""
    op = "NOT LIKE" if negate else "LIKE"
    join_op = " AND " if negate else " OR "
    clauses = [f"{column} {op} {_adql_str(f'%{v}%')}" for v in values]
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
        conds.append(f"target_name LIKE {_adql_str(f'%{f.source_name}%')}")

    if f.antenna_arrays:
        conds.append(f"antenna_arrays LIKE {_adql_str(f'%{f.antenna_arrays}%')}")

    # Array type: 12m → DA/DV antennas, 7m → CM antennas, TP → PM antennas
    if f.array_type:
        type_clauses = [
            f"antenna_arrays LIKE {_adql_str(pattern)}"
            for atype in f.array_type
            for pattern in ARRAY_TYPE_ANTENNA_PATTERNS.get(atype, [f"%{atype}%"])
        ]
        conds.append(f"({' OR '.join(type_clauses)})")

    # Array configuration matched against schedblock_name (e.g. 'C-1', 'C-2')
    if f.array_configuration:
        conf_clauses = [
            f"schedblock_name LIKE {_adql_str(f'%{c}%')}" for c in f.array_configuration
        ]
        conds.append(f"({' OR '.join(conf_clauses)})")

    if f.angular_resolution_range:
        lo, hi = f.angular_resolution_range
        conds.append(f"spatial_resolution BETWEEN {_adql_num(lo)} AND {_adql_num(hi)}")

    if f.observation_date_range:
        mjd_min = Time(f.observation_date_range[0], format="iso", scale="utc").mjd
        mjd_max = Time(f.observation_date_range[1], format="iso", scale="utc").mjd
        conds.append(f"t_max >= {mjd_min} AND t_max <= {mjd_max}")

    # QA2 status: map human-friendly labels to TAP-stored single-char values
    if f.qa2_status:
        mapped = qa2_codes(f.qa2_status)
        conds.append(f"qa2_passed IN {_adql_in(mapped)}")

    if f.obs_type:
        clauses = " OR ".join(f"type LIKE {_adql_str(f'%{t}%')}" for t in f.obs_type)
        conds.append(f"({clauses})")

    if f.fov_range:
        lo, hi = f.fov_range
        conds.append(f"s_fov BETWEEN {_adql_num(lo)} AND {_adql_num(hi)}")

    if f.time_resolution_range:
        lo, hi = f.time_resolution_range
        conds.append(f"t_resolution BETWEEN {_adql_num(lo)} AND {_adql_num(hi)}")

    if f.frequency_range:
        lo, hi = f.frequency_range
        conds.append(f"frequency BETWEEN {_adql_num(lo)} AND {_adql_num(hi)}")

    if f.proposal_id_prefix:
        prefix_clauses = [f"proposal_id LIKE {_adql_str(f'{p}%')}" for p in f.proposal_id_prefix]
        conds.append(f"({' OR '.join(prefix_clauses)})")

    if f.proposal_id:
        conds.append(f"proposal_id IN {_adql_in(f.proposal_id)}")

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
            conds.append(f"science_keyword NOT LIKE {_adql_str(f'%{kw}%')}")

    if f.scientific_category:
        for cat in f.scientific_category:
            conds.append(f"scientific_category NOT LIKE {_adql_str(f'%{cat}%')}")

    if f.source_name:
        for name in f.source_name:
            conds.append(f"LOWER(target_name) NOT LIKE LOWER({_adql_str(f'%{name}%')})")

    if f.obs_type:
        for t in f.obs_type:
            conds.append(f"type NOT LIKE {_adql_str(f'%{t}%')}")

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
