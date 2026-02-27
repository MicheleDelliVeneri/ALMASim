"""TAP service functions for querying ALMA metadata."""
import pyvo
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential
import requests


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
    targets (list of tuples): A list where each tuple contains (target_name, member_ous_uid).

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
    science_keyword=None,
    scientific_category=None,
    band=None,
    fov_range=None,
    time_resolution_range=None,
    frequency_range=None,
    source_name=None,
    antenna_arrays=None,
    angular_resolution_range=None,
    observation_date_range=None,
    qa2_status=None,
    obs_type=None,
):
    """Query for all science observations by science type and other filters.

    Parameters:
    science_keyword: Science keyword filter (str or list)
    scientific_category: Scientific category filter (str or list)
    band: Band filter (int or list of ints)
    fov_range: Field of view range [min, max]
    time_resolution_range: Time resolution range [min, max]
    frequency_range: Frequency range [min, max]
    source_name: Target name filter (partial match)
    antenna_arrays: Antenna array configuration filter (partial match)
    angular_resolution_range: Angular resolution range [min, max] in arcsec
    observation_date_range: Observation date range [min, max] as ISO date strings
    qa2_status: QA2 status filter (str or list, e.g. 'T', 'F')
    obs_type: Observation type filter (partial match)

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

    conditions = ["is_mosaic = 'F'", "science_observation = 'T'"]

    # Science keyword
    if science_keyword:
        if isinstance(science_keyword, list):
            if len(science_keyword) == 1:
                conditions.append(f"science_keyword LIKE '%{science_keyword[0]}%'")
            else:
                kw_clauses = " OR ".join(
                    f"science_keyword LIKE '%{kw}%'" for kw in science_keyword
                )
                conditions.append(f"({kw_clauses})")
        else:
            conditions.append(f"science_keyword LIKE '%{science_keyword}%'")

    # Scientific category
    if scientific_category:
        if isinstance(scientific_category, list):
            if len(scientific_category) == 1:
                conditions.append(
                    f"scientific_category LIKE '%{scientific_category[0]}%'"
                )
            else:
                cat_clauses = " OR ".join(
                    f"scientific_category LIKE '%{cat}%'" for cat in scientific_category
                )
                conditions.append(f"({cat_clauses})")
        else:
            conditions.append(f"scientific_category LIKE '%{scientific_category}%'")

    # Band
    if band:
        if isinstance(band, list):
            if len(band) == 1:
                conditions.append(f"band_list LIKE '%{band[0]}%'")
            else:
                band_clauses = " OR ".join(
                    f"band_list LIKE '%{b}%'" for b in band
                )
                conditions.append(f"({band_clauses})")
        else:
            conditions.append(f"band_list LIKE '%{band}%'")

    # Source name
    if source_name:
        conditions.append(f"target_name LIKE '%{source_name}%'")

    # Antenna arrays
    if antenna_arrays:
        conditions.append(f"antenna_arrays LIKE '%{antenna_arrays}%'")

    # Angular resolution range
    if angular_resolution_range:
        conditions.append(
            f"spatial_resolution BETWEEN {angular_resolution_range[0]} AND {angular_resolution_range[1]}"
        )

    # Observation date range
    if observation_date_range:
        conditions.append(
            f"obs_release_date BETWEEN '{observation_date_range[0]}' AND '{observation_date_range[1]}'"
        )

    # QA2 status
    if qa2_status:
        if isinstance(qa2_status, list):
            statuses = "', '".join(qa2_status)
            conditions.append(f"qa2_passed IN ('{statuses}')")
        else:
            conditions.append(f"qa2_passed = '{qa2_status}'")

    # Observation type
    if obs_type:
        conditions.append(f"type LIKE '%{obs_type}%'")

    # FOV range
    if fov_range:
        conditions.append(f"s_fov BETWEEN {fov_range[0]} AND {fov_range[1]}")

    # Time resolution range
    if time_resolution_range:
        conditions.append(
            f"t_resolution BETWEEN {time_resolution_range[0]} AND {time_resolution_range[1]}"
        )

    # Frequency range
    if frequency_range:
        conditions.append(
            f"frequency BETWEEN {frequency_range[0]} AND {frequency_range[1]}"
        )

    where_clause = " AND ".join(conditions)
    query = f"""
            SELECT {columns_str}
            FROM ivoa.obscore
            WHERE {where_clause}
            """
    results = search_with_retry(service, query)
    return results


