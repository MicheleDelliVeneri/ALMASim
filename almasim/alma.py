from astropy.constants import c
from astropy.units import Quantity
import astropy.units as U
import math
import pyvo
import numpy as np
import pandas as pd
import os
from tenacity import retry, stop_after_attempt, wait_exponential
import requests


# -------------- Database Query Functions --------------------------- #
def get_tap_service():
    """Establishes a connection to a TAP service for astronomical data access.

    This function attempts to connect to multiple TAP service endpoints until
    a successful connection is established. It also performs a simple test query
    to verify that the service is responding.

    Returns:
        pyvo.dal.TAPService: A TAP service object if a successful connection is made,
                             otherwise None.

    Raises:
        None

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
    return service.search(query, response_timeout=120).to_table().to_pandas()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_science_types():
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
    """Query for all science observations of given member OUS UID
    and target name, selecting all columns of interest.

    Parameters:
    service (pyvo.dal.TAPService): A TAPService instance for
                                   querying the database.
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
    """Query observations for all predefined targets and compile
    the results into a single DataFrame.

    Parameters:
    service (pyvo.dal.TAPService): A TAPService instance
                                   for querying the database.
    targets (list of tuples): A list where each t
                              uple contains (target_name, member_ous_uid).

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
    """Query for all science observations of given member OUS UID and target name,
       selecting all columns of interest.

    Parameters:
    service (pyvo.dal.TAPService): A TAPService instance for querying the database.

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


# -------------------- Metadata Processing Functions ---------------------- #
def estimate_alma_beam_size(central_frequency_ghz, max_baseline_km, return_value=True):
    """
    Estimates the beam size of the Atacama Large Millimeter/submillimeter Array (ALMA)
    in arcseconds.

    This function provides an approximation based on the theoretical relationship between
    observing frequency and maximum baseline. The formula used is:
    beam_size = (speed_of_light / central_frequency) / max_baseline * (180 / pi) * 3600
    arcseconds [km]/[s] * [s] / [km] = [radians] * [arcsec /radian] * [arcseconds/degree]

    Args:
        central_frequency_ghz: Central frequency of the observing band in GHz (float).
        max_baseline_km: Maximum baseline of the antenna array in kilometers (float).

    Returns:
        Estimated beam size in arcseconds (float).

    Raises:
        ValueError: If either input argument is non-positive.
    """
    # Input validation
    if central_frequency_ghz <= 0 or max_baseline_km <= 0:
        raise ValueError(
            "Central frequency and maximum baseline must be positive values."
        )

    if not isinstance(central_frequency_ghz, Quantity):
        central_frequency_ghz = central_frequency_ghz * U.GHz
    if not isinstance(max_baseline_km, Quantity):
        max_baseline_km = max_baseline_km * U.km

    # Speed of light in meters per second
    light_speed = c.to(U.m / U.s).value

    # Convert frequency to Hz
    central_frequency_hz = central_frequency_ghz.to(U.Hz).value

    # Convert baseline to meters
    max_baseline_meters = max_baseline_km.to(U.m).value

    # Theoretical estimate of beam size (radians)
    theta_radians = (light_speed / central_frequency_hz) / max_baseline_meters

    # Convert theta from radians to arcseconds
    beam_size_arcsec = theta_radians * (180 / math.pi) * 3600 * U.arcsec
    if return_value is True:
        return beam_size_arcsec.value
    else:
        return beam_size_arcsec


def get_fov_from_band(band, antenna_diameter: int = 12, return_value=True):
    """
    This function returns the field of view of an ALMA band in arcseconds
    input:
        band number (int): the band number of the ALMA band, between 1 and 10
        antenna_diameter (int): the diameter of the antenna in meters
    output:
        fov (astropy unit): the field of view in arcseconds

    """
    light_speed = c.to(U.m / U.s).value
    if band == 1:
        central_freq = 43 * U.GHz
    elif band == 2:
        central_freq = 67 * U.GHz
    elif band == 3:
        central_freq = 100 * U.GHz
    elif band == 4:
        central_freq = 150 * U.GHz
    elif band == 5:
        central_freq = 217 * U.GHz
    elif band == 6:
        central_freq = 250 * U.GHz
    elif band == 7:
        central_freq = 353 * U.GHz
    elif band == 8:
        central_freq = 545 * U.GHz
    elif band == 9:
        central_freq = 650 * U.GHz
    elif band == 10:
        central_freq = 868.5 * U.GHz
    central_freq = central_freq.to(U.Hz).value
    central_freq_s = 1 / central_freq
    wavelength = light_speed * central_freq_s
    # this is the field of view in Radians
    fov = 1.22 * wavelength / antenna_diameter
    # fov in arcsec
    fov = fov * (180 / math.pi) * 3600 * U.arcsec
    if return_value is True:
        return fov.value
    else:
        return fov


def generate_antenna_config_file_from_antenna_array(
    antenna_array, master_path, output_dir
):
    antenna_coordinates = pd.read_csv(
        os.path.join(master_path, "antenna_config", "antenna_coordinates.csv")
    )
    obs_antennas = antenna_array.split(" ")
    obs_antennas = [antenna.split(":")[0] for antenna in obs_antennas]
    obs_coordinates = antenna_coordinates[
        antenna_coordinates["name"].isin(obs_antennas)
    ]
    intro_string = (
        "# observatory=ALMA\n# coordsys=LOC (local tangent plane)\n# x y z diam pad#\n"
    )
    with open(os.path.join(output_dir, "antenna.cfg"), "w") as f:
        f.write(intro_string)
        for i in range(len(obs_coordinates)):
            f.write(
                f"{
                    obs_coordinates['x'].values[i]} {
                    obs_coordinates['y'].values[i]} {
                    obs_coordinates['z'].values[i]} 12. {
                    obs_coordinates['name'].values[i]}\n"
            )
    f.close()


def compute_distance(x1, y1, z1, x2, y2, z2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


def get_max_baseline_from_antenna_config(update_progress, antenna_config):
    """
    takes an antenna configuration .cfg file as input and outputs
    """
    positions = []
    with open(antenna_config, "r") as f:
        lines = f.readlines()
        for line in lines:
            if not line.strip().startswith("#"):
                if "\t" in line:
                    row = [x for x in line.split("\t")][:3]
                else:
                    row = [x for x in line.split(" ")][:3]
                positions.append([float(x) for x in row])
    positions = np.array(positions)
    max_baseline = 0

    for i in range(len(positions)):
        x1, y1, z1 = positions[i]
        for j in range(i + 1, len(positions)):
            x2, y2, z2 = positions[j]
            dist = compute_distance(x1, y1, z1, x2, y2, z2) / 1000
            if dist > max_baseline:
                max_baseline = dist
        update_progress.emit((i / len(positions) * 100))

    return max_baseline


def get_max_baseline_from_antenna_array(antenna_array, master_path):
    antenna_coordinates = pd.read_csv(
        os.path.join(master_path, "antenna_config", "antenna_coordinates.csv")
    )
    obs_antennas = antenna_array.split(" ")
    obs_antennas = [antenna.split(":")[0] for antenna in obs_antennas]
    obs_coordinates = antenna_coordinates[
        antenna_coordinates["name"].isin(obs_antennas)
    ].values
    max_baseline = 0
    for i in range(len(obs_coordinates)):
        name, x1, y1, z1 = obs_coordinates[i]
        for j in range(i + 1, len(obs_coordinates)):
            name, x2, y2, z2 = obs_coordinates[j]
            dist = compute_distance(x1, y1, z1, x2, y2, z2) / 1000
            if dist > max_baseline:
                max_baseline = dist
    return max_baseline


# -------------- OLD -------------------------------- #

"""
def get_band_range(band):
    if band == 1:
        return (31, 45)
    elif band == 2:
        return (67, 90)
    elif band == 3:
        return (84, 116)
    elif band == 4:
        return (125, 163)
    elif band == 5:
        return (163, 211)
    elif band == 6:
        return (211, 275)
    elif band == 7:
        return (275, 373)
    elif band == 8:
        return (385, 500)
    elif band == 9:
        return (602, 720)
    elif band == 10:
        return (787, 950)


def get_band_central_freq(band):
    if band == 1:
        return 38
    elif band == 2:
        return 78.5
    elif band == 3:
        return 100
    elif band == 4:
        return 143
    elif band == 5:
        return 217
    elif band == 6:
        return 250
    elif band == 7:
        return 353
    elif band == 8:
        return 545
    elif band == 9:
        return 650
    elif band == 10:
        return 850


def get_antennas_distances_from_reference(antenna_config):
    f = open(antenna_config)
    lines = f.readlines()
    nlines = len(lines)
    frefant = int((nlines - 1) // 2)
    f.close()
    zx, zy, zz, zztot = [], [], [], []
    for i in range(3, nlines):
        stuff = lines[i].split()
        zx.append(float(stuff[0]))
        zy.append(float(stuff[1]))
        zz.append(float(stuff[2]))
    nant = len(zx)
    nref = int(frefant)
    for i in range(0, nant):
        zxref = zx[i] - zx[nref]
        zyref = zy[i] - zy[nref]
        zzref = zz[i] - zz[nref]
        zztot.append(np.sqrt(zxref**2 + zyref**2 + zzref**2))
    return zztot, frefant
"""
