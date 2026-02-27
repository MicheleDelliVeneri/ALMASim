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
):
    """Query for all science observations by science type and other filters.

    Parameters:
    science_keyword: Science keyword filter
    scientific_category: Scientific category filter
    band: Band filter
    fov_range: Field of view range [min, max]
    time_resolution_range: Time resolution range [min, max]
    frequency_range: Frequency range [min, max]

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
    # Default values for parameters if they are None
    if science_keyword is None:
        science_keyword = ""
    if scientific_category is None:
        scientific_category = ""
    if band is None:
        band = ""

    # Build query components based on the type and content of each parameter
    science_keyword_query = f"science_keyword like '%{science_keyword}%'"
    if isinstance(science_keyword, list):
        if len(science_keyword) == 1:
            science_keyword_query = f"science_keyword like '%{science_keyword[0]}%'"
        else:
            science_keywords = "', '".join(science_keyword)
            science_keyword_query = f"science_keyword in ('{science_keywords}')"

    scientific_category_query = f"scientific_category like '%{scientific_category}%'"
    if isinstance(scientific_category, list):
        if len(scientific_category) == 1:
            scientific_category_query = (
                f"scientific_category like '%{scientific_category[0]}%'"
            )
        else:
            scientific_categories = "', '".join(scientific_category)
            scientific_category_query = (
                f"scientific_category in ('{scientific_categories}')"
            )

    band_query = f"band_list like '%{band}%'"
    if isinstance(band, list):
        if len(band) == 1:
            band_query = f"band_list like '%{band[0]}%'"
        else:
            bands = [str(x) for x in band]
            bands = "', '".join(bands)
            band_query = f"band_list in ('{bands}')"

    # Additional filtering based on ranges
    if fov_range is None:
        fov_query = ""
    else:
        fov_query = f"s_fov BETWEEN {fov_range[0]} AND {fov_range[1]}"
    if time_resolution_range is None:
        time_resolution_query = ""
    else:
        time_resolution_query = f"t_resolution BETWEEN {
            time_resolution_range[0]} AND {
            time_resolution_range[1]}"

    if frequency_range is None:
        frequency_query = ""
    else:
        frequency_query = (
            f"frequency BETWEEN {frequency_range[0]} AND {frequency_range[1]}"
        )

    # Combine all conditions into one WHERE clause
    conditions = [
        science_keyword_query,
        scientific_category_query,
        band_query,
        fov_query,
        time_resolution_query,
        frequency_query,
    ]
    conditions = [cond for cond in conditions if cond]  # Remove empty conditions
    where_clause = " AND ".join(conditions)
    where_clause = where_clause + " AND is_mosaic = 'F' AND science_observation = 'T'"

    query = f"""
            SELECT {columns_str}
            FROM ivoa.obscore
            WHERE {where_clause}
            """
    results = search_with_retry(service, query)
    return results


