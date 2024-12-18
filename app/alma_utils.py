from pyvo.dal import TAPService
from tenacity import retry, stop_after_attempt, wait_exponential
import pandas as pd
import logging
import numpy as np
import os
import math
from astropy import units as u
from astropy.constants import c
from astropy.units import Quantity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TAP_URLS = [
    "https://almascience.eso.org/tap",
    "https://almascience.nao.ac.jp/tap",
    "https://almascience.nrao.edu/tap",
]


def get_tap_service():
    """
    Connect to an available ALMA TAP service.
    """
    for url in TAP_URLS:
        try:
            service = TAPService(url)
            try:
                # Test Query
                service.search("SELECT TOP 1 * FROM ivoa.obscore")
            except Exception as query_error:
                logger.error(
                    f"TAP service query failed for {url}: {query_error}")
                continue
            logger.info(f"Connected to {url}")
            return service
        except Exception as e:
            logger.error(f"Failed to connect to {url}: {e}")
    raise Exception("All TAP services failed!")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def query_alma(service, query):
    """
    Perform a query to ALMA's TAP service.

    Args:
        service: TAPService instance.
        query: SQL query string.

    Returns:
        pandas.DataFrame: Query results as a DataFrame.
    """
    return service.search(query, response_timeout=120).to_table().to_pandas()


def validate_science_filters(science_filters):
    """
    Validate the science_filters dictionary.

    Args:
        science_filters (dict): Filters for science type queries.

    Raises:
        ValueError: If required keys in science_filters are missing or
                    improperly formatted.
    """
    required_keys = [
        "science_keywords",
        "scientific_categories",
        "bands",
        "frequency",
        "spatial_resolution",
        "velocity_resolution",
    ]

    for key in science_filters.keys():
        if key not in required_keys:
            raise ValueError(
                f"Invalid key '{key}' in science_filters. \
                Must be one of {required_keys}."
            )
        if key in ["frequency", "spatial_resolution", "velocity_resolution"]:
            if (
                not isinstance(science_filters[key], tuple)
                or len(science_filters[key]) != 2
            ):
                raise ValueError(
                    f"'{key}' must be a tuple with two \
                    numeric values (min, max)."
                )


def generate_query_for_targets(target_list):
    """
    Generate a SQL query for a target list.

    Args:
        target_list (list of tuples): List of targets as
        (target_name, member_ous_uid).

    Returns:
        str: SQL query string.
    """
    base_query = """
        SELECT target_name, member_ous_uid, s_ra, s_dec, frequency,
               velocity_resolution, spatial_resolution, band_list,
               science_keyword, scientific_category
        FROM ivoa.obscore
        WHERE science_observation = 'T'
    """
    target_conditions = [
        f"(target_name = '{target}' AND member_ous_uid = '{uid}')"
        for target, uid in target_list
    ]
    return base_query + " AND (" + " OR ".join(target_conditions) + ")"


def generate_query_for_science(science_filters):
    """
    Generate a SQL query based on science-related filters.

    Args:
        science_filters (dict): Filters for science type queries.

    Returns:
        str: SQL query string.
    """
    base_query = """
        SELECT target_name, member_ous_uid, s_ra, s_dec, frequency,
               velocity_resolution, spatial_resolution, band_list,
               science_keyword, scientific_category
        FROM ivoa.obscore
        WHERE science_observation = 'T'
    """
    conditions = []

    if science_filters.get("science_keywords"):
        keywords = "', '".join(science_filters["science_keywords"])
        conditions.append(f"science_keyword IN ('{keywords}')")

    if science_filters.get("scientific_categories"):
        categories = "', '".join(science_filters["scientific_categories"])
        conditions.append(f"scientific_category IN ('{categories}')")

    if science_filters.get("bands"):
        bands = "', '".join([str(band) for band in science_filters["bands"]])
        conditions.append(f"band_list IN ('{bands}')")

    if science_filters.get("frequency"):
        f_min, f_max = science_filters["frequency"]
        conditions.append(f"frequency BETWEEN {f_min} AND {f_max}")

    if science_filters.get("spatial_resolution"):
        sr_min, sr_max = science_filters["spatial_resolution"]
        conditions.append(f"spatial_resolution BETWEEN {sr_min} AND {sr_max}")

    if science_filters.get("velocity_resolution"):
        vr_min, vr_max = science_filters["velocity_resolution"]
        conditions.append(f"velocity_resolution BETWEEN {vr_min} AND {vr_max}")

    if conditions:
        base_query += " AND " + " AND ".join(conditions)

    return base_query


def fetch_science_types():
    """
    Retrieve available science keywords and categories from the
    ALMA TAP service.

    Returns:
        tuple: (list of science keywords, list of scientific categories)
    """
    service = get_tap_service()
    query = """
        SELECT DISTINCT science_keyword, scientific_category
        FROM ivoa.obscore
        WHERE science_observation = 'T'
    """
    results = query_alma(service, query)
    return (
        results["science_keyword"].dropna().unique().tolist(),
        results["scientific_category"].dropna().unique().tolist(),
    )


def query_by_targets(target_list):
    """
    Query ALMA observations based on a list of targets.

    Args:
        target_list (list of tuples): List of targets as
        (target_name, member_ous_uid).

    Returns:
        pandas.DataFrame: Query results.
    """
    service = get_tap_service()
    query = generate_query_for_targets(target_list)
    return query_alma(service, query)


def query_by_science(science_filters):
    """
    Query ALMA observations based on science-related filters.

    Args:
        science_filters (dict): Filters for science type queries.

    Returns:
        pandas.DataFrame: Query results.
    """
    validate_science_filters(science_filters)
    service = get_tap_service()
    query = generate_query_for_science(science_filters)
    return query_alma(service, query)


def estimate_alma_beam_size(central_frequency_ghz,
                            max_baseline_km, return_value=True):
    """
    Estimate ALMA beam size in arcseconds.
    """
    if central_frequency_ghz <= 0 or max_baseline_km <= 0:
        raise ValueError(
            "Central frequency and maximum baseline must be positive.")

    if not isinstance(central_frequency_ghz, Quantity):
        central_frequency_ghz = central_frequency_ghz * u.GHz
    if not isinstance(max_baseline_km, Quantity):
        max_baseline_km = max_baseline_km * u.km
    light_speed = c.to(u.m / u.s).value
    theta = (light_speed / central_frequency_ghz.to(u.Hz).value) / max_baseline_km.to(u.m).value
    beam_size_arcsec = theta * (180 / math.pi) * 3600 * u.arcsec
    return beam_size_arcsec.value if return_value else beam_size_arcsec


def get_fov_from_band(band, antenna_diameter=12, return_value=True):
    """
    Calculate field of view for an ALMA band.
    """
    band_frequencies = {
        1: 43 * u.GHz,
        2: 67 * u.GHz,
        3: 100 * u.GHz,
        4: 150 * u.GHz,
        5: 217 * u.GHz,
        6: 250 * u.GHz,
        7: 353 * u.GHz,
        8: 545 * u.GHz,
        9: 650 * u.GHz,
        10: 868.5 * u.GHz,
    }
    if band not in band_frequencies:
        raise ValueError("Invalid band number. Must be between 1 and 10.")

    wavelength = (c / band_frequencies[band]).to(u.m)
    fov = (1.22 * wavelength / antenna_diameter) * \
        (180 / math.pi) * 3600 * u.arcsec
    return fov.value if return_value else fov


def generate_antenna_config_file_from_antenna_array(
    antenna_array, master_path, output_dir
):
    """
    Generate antenna configuration file from antenna array.
    """
    antenna_coordinates = pd.read_csv(
        os.path.join(master_path, "antenna_config", "antenna_coordinates.csv")
    )
    obs_antennas = [ant.split(":")[0] for ant in antenna_array.split()]
    obs_coordinates = antenna_coordinates[
        antenna_coordinates["name"].isin(obs_antennas)
    ]

    header = "# observatory=ALMA\n# coordsys=LOC\n# x y z diam pad#\n"
    output_path = os.path.join(output_dir, "antenna.cfg")
    with open(output_path, "w") as f:
        f.write(header)
        obs_coordinates[["x", "y", "z", "name"]].to_csv(
            f, sep=" ", index=False, header=False
        )


def compute_distance(x1, y1, z1, x2, y2, z2):
    """
    Compute Euclidean distance between two points.
    """
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


def get_max_baseline_from_antenna_config(antenna_config):
    """
    Calculate maximum baseline from an antenna configuration file.
    """
    positions = np.loadtxt(
        antenna_config, comments="#", usecols=(
            0, 1, 2), dtype=float)
    distances = np.linalg.norm(
        positions[:, None, :] - positions[None, :, :], axis=-1)
    max_baseline = np.max(distances) / 1000  # Convert to km
    return max_baseline


def get_max_baseline_from_antenna_array(antenna_array, master_path):
    """
    Calculate the maximum baseline from an antenna array.
    """
    antenna_coordinates = pd.read_csv(
        os.path.join(master_path, "antenna_config", "antenna_coordinates.csv")
    )
    obs_antennas = [ant.split(":")[0] for ant in antenna_array.split()]
    obs_coordinates = antenna_coordinates[
        antenna_coordinates["name"].isin(obs_antennas)
    ][["x", "y", "z"]].to_numpy()

    distances = np.linalg.norm(
        obs_coordinates[:, None, :] - obs_coordinates[None, :, :], axis=-1
    )
    max_baseline = np.max(distances) / 1000  # Convert to km
    return max_baseline
