from astropy.constants import c
from astropy.units import Quantity
import astropy.units as U
from astropy.io import fits
import math
import pyvo
import numpy as np 
import pandas as pd
import os 
import sys
import random
from math import pi
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential

# Utility function to get a TAP service from a list of URLs, cycling through until one works
def get_tap_service():
    urls = ["https://almascience.eso.org/tap",
        "https://almascience.nao.ac.jp/tap",
        "https://almascience.nrao.edu/tap"
    ]
    while True:  # Infinite loop to keep trying until successful
        for url in urls:
            try:
                service = pyvo.dal.TAPService(url)
                # Test the connection with a simple query to ensure the service is working
                result = service.search("SELECT TOP 1 * FROM ivoa.obscore")
                print(f"Connected successfully to {url}")
                return service
            except Exception as e:
                print(f"Failed to connect to {url}: {e}")
                print("Retrying other servers...")
        print("All URLs attempted and failed, retrying...")

# Metadata related functions
def estimate_alma_beam_size(central_frequency_ghz, max_baseline_km, return_value=True):
    """
    Estimates the beam size of the Atacama Large Millimeter/submillimeter Array (ALMA) in arcseconds.

    This function provides an approximation based on the theoretical relationship between
    observing frequency and maximum baseline. The formula used is:
    beam_size = (speed_of_light / central_frequency) / max_baseline * (180 / pi) * 3600 arcseconds
    [km]/[s] * [s] / [km] = [radians] * [arcsec /radian] * [arcseconds/degree]

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
        raise ValueError("Central frequency and maximum baseline must be positive values.")
    
    if type(central_frequency_ghz) != Quantity:
        central_frequency_ghz = central_frequency_ghz * U.GHz
    if type(max_baseline_km) != Quantity:
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
    if return_value == True:
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
    if return_value == True:
        return fov.value
    else:
        return fov

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
    """
    Takes as input the band number and returns its central frequency in GHz
    """
    if band == 1:
        return 38
    elif band == 2:
        return 78.5
    elif band == 3:
        return 100
    elif band == 4:
        return  143
    elif band == 5:
        return  217
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

def generate_antenna_config_file_from_antenna_array(antenna_array, master_path, output_dir):
    antenna_coordinates = pd.read_csv(os.path.join(master_path, 'antenna_config', 'antenna_coordinates.csv'))
    obs_antennas = antenna_array.split(' ')
    obs_antennas = [antenna.split(':')[0] for antenna in obs_antennas]
    obs_coordinates = antenna_coordinates[antenna_coordinates['name'].isin(obs_antennas)]
    intro_string = "# observatory=ALMA\n# coordsys=LOC (local tangent plane)\n# x y z diam pad#\n"
    with open(os.path.join(output_dir, 'antenna.cfg'), 'w') as f:
        f.write(intro_string)
        for i in range(len(obs_coordinates)):
            f.write(f"{obs_coordinates['x'].values[i]} {obs_coordinates['y'].values[i]} {obs_coordinates['z'].values[i]} 12. {obs_coordinates['name'].values[i]}\n")
    f.close()

def compute_distance(x1, y1, z1, x2, y2, z2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

def get_max_baseline_from_antenna_config(antenna_config):
    """
    takes an antenna configuration .cfg file as input and outputs
    """
    positions = []
    with open(antenna_config, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if not line.strip().startswith('#'):
                if '\t' in line:
                    row = [x for x in line.split("\t")][:3]
                else:
                    row = [x for x in line.split(" ")][:3]
                positions.append([float(x) for x in row])  
    positions = np.array(positions)
    max_baseline = 0
    
    for i in tqdm(range(len(positions)), total=len(positions)):
        x1, y1, z1 = positions[i]
        for j in range(i + 1, len(positions)):
            x2, y2, z2 = positions[j]
            dist = compute_distance(x1, y1, z1, x2, y2, z2) / 1000
            if dist > max_baseline:
                max_baseline = dist

    return max_baseline

def get_max_baseline_from_antenna_array(antenna_array, master_path):
    antenna_coordinates = pd.read_csv(os.path.join(master_path, 'antenna_config', 'antenna_coordinates.csv'))
    obs_antennas = antenna_array.split(' ')
    obs_antennas = [antenna.split(':')[0] for antenna in obs_antennas]
    obs_coordinates = antenna_coordinates[antenna_coordinates['name'].isin(obs_antennas)].values
    max_baseline = 0
    for i in range(len(obs_coordinates)):
        name, x1, y1, z1 = obs_coordinates[i]
        for j in range(i + 1, len(obs_coordinates)):
            name, x2, y2, z2 = obs_coordinates[j]
            dist = compute_distance(x1, y1, z1, x2, y2, z2) / 1000
            if dist > max_baseline:
                max_baseline = dist
    return max_baseline

def query_observations(member_ous_uid, target_name):
    """Query for all science observations of given member OUS UID and target name, selecting all columns of interest.

    Parameters:
    service (pyvo.dal.TAPService): A TAPService instance for querying the database.
    member_ous_uid (str): The unique identifier for the member OUS to filter observations by.
    target_name (str): The target name to filter observations by.

    Returns:
    pandas.DataFrame: A table of query results.
    """

    query = f"""
            SELECT *
            FROM ivoa.obscore
            WHERE member_ous_uid = '{member_ous_uid}'
            AND target_name = '{target_name}'
            AND is_mosaic = 'F'
            AND science_observation = 'T'    
            """
    service = get_tap_service()
    result = service.search(query).to_table().to_pandas()

    return result

def query_all_targets(targets):
    """Query observations for all predefined targets and compile the results into a single DataFrame.

    Parameters:
    service (pyvo.dal.TAPService): A TAPService instance for querying the database.
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

def query_for_metadata_by_targets(targets, path):
    """Query for metadata for all predefined targets and compile the results into a single DataFrame.

    Parameters:
    service (pyvo.dal.TAPService): A TAPService instance for querying the database.
    targets (list of tuples): A list where each tuple contains (target_name, member_ous_uid).
    path (str): The path to save the results to.

    Returns:
    pandas.DataFrame: A DataFrame containing the results for all queried targets.
    """
    # Query all targets and compile the results
    df = query_all_targets(targets)
    df = df.drop_duplicates(subset='member_ous_uid')
    # Define a dictionary to map existing column names to new names with unit initials
    rename_columns = {
    'target_name': 'ALMA_source_name',
    'pwv': 'PWV',
    'schedblock_name': 'SB_name',
    'velocity_resolution': 'Vel.res.',
    'spatial_resolution': 'Ang.res.',
    's_ra': 'RA',
    's_dec': 'Dec',
    's_fov': 'FOV',
    't_resolution': 'Int.Time',
    't_max': 'Total.Time',
    'cont_sensitivity_bandwidth': 'Cont_sens_mJybeam',
    'sensitivity_10kms': 'Line_sens_10kms_mJybeam',
    'obs_release_date': 'Obs.date',
    'band_list': 'Band',
    'bandwidth': 'Bandwidth',
    'frequency': 'Freq',
    'frequency_support': 'Freq.sup.'

    }
    # Rename the columns in the DataFrame
    df.rename(columns=rename_columns, inplace=True)
    database = df[['ALMA_source_name', 'Band', 'PWV', 'SB_name', 'Vel.res.', 'Ang.res.', 'RA', 'Dec', 'FOV', 'Int.Time', 
                    'Total.Time', 'Cont_sens_mJybeam', 'Line_sens_10kms_mJybeam', 'Obs.date', 'Bandwidth', 'Freq', 
                    'Freq.sup.', 'antenna_arrays']]
    database.loc[:, 'Obs.date'] = database['Obs.date'].apply(lambda x: x.split('T')[0])
    database.to_csv(path, index=False)
    return database

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_science_types():
    service = get_tap_service()
    query = f"""  
            SELECT science_keyword, scientific_category  
            FROM ivoa.obscore  
            WHERE science_observation = 'T'    
            """
    try:
        db = service.search(query).to_table().to_pandas()
    except(pyvo.dal.exceptions.DALServiceError, requests.exceptions.RequestException) as e:
        print(f"Error querying TAP service: {e}")
        raise
    science_keywords = db['science_keyword'].unique()
    scientific_category = db['scientific_category'].unique()
    science_keywords = list(filter(lambda x: x != "", science_keywords))
    scientific_category = list(filter(lambda x: x != "", scientific_category))

    unique_keywords = []
    # Iterazione attraverso ogni stringa nella lista
    for keywords_string in science_keywords:
    # Dividi la stringa in base alla virgola e rimuovi gli spazi bianchi
        keywords_list = [keyword.strip() for keyword in keywords_string.split(',')]
    # Aggiungi le parole alla lista dei valori univoci
        unique_keywords.extend(keywords_list)
    # Utilizza il set per ottenere i valori univoci
    unique_keywords = sorted(set(unique_keywords))
    unique_keywords = [keyword for keyword in unique_keywords if (
                        keyword != 'Evolved stars: Shaping/physical structure' and
                        keyword != 'Exoplanets' and 
                        keyword != 'Galaxy structure &evolution')]
    
    return  unique_keywords, scientific_category

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))

def search_with_retry(service, query):
    return service.search(query, response_timeout=120).to_table().to_pandas()


def query_by_science_type(science_keyword=None, scientific_category=None, band=None, fov_range=None, time_resolution_range=None, total_time_range=None, frequency_range=None):
    """Query for all science observations of given member OUS UID and target name, selecting all columns of interest.

    Parameters:
    service (pyvo.dal.TAPService): A TAPService instance for querying the database.

    Returns:
    pandas.DataFrame: A table of query results.
    """
    service = get_tap_service()
    columns = [
        'target_name', 'member_ous_uid', 'pwv', 'schedblock_name',  'velocity_resolution',
        'spatial_resolution', 's_ra', 's_dec', 's_fov', 't_resolution',
        'cont_sensitivity_bandwidth', 'sensitivity_10kms', 'obs_release_date', 
        'band_list', 'bandwidth', 'frequency', 'frequency_support', 
        'science_keyword', 'scientific_category', 'antenna_arrays', 't_max'
    ]
    columns_str = ', '.join(columns)
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
            scientific_category_query = f"scientific_category like '%{scientific_category[0]}%'"
        else:
            scientific_categories = "', '".join(scientific_category)
            scientific_category_query = f"scientific_category in ('{scientific_categories}')"

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
        time_resolution_query = f"t_resolution BETWEEN {time_resolution_range[0]} AND {time_resolution_range[1]}"

    if total_time_range is None:
        total_time_query = ""
    else:    
        total_time_query = f"t_max BETWEEN {total_time_range[0]} AND {total_time_range[1]}"

    if frequency_range is None:
        frequency_query = ""
    else:
        frequency_query = f"frequency BETWEEN {frequency_range[0]} AND {frequency_range[1]}"

    # Combine all conditions into one WHERE clause
    conditions = [science_keyword_query, scientific_category_query, band_query, fov_query, time_resolution_query, total_time_query, frequency_query]
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
    

def plot_science_keywords_distributions(master_path):
    service = get_tap_service()
    plot_dir = os.path.join(master_path, "plots")

    # Check if plot directory exists
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        existing_plots = []  # Initialize as empty list if plot directory doesn't exist
    else:
        # Check if plot files already exist
        existing_plots = [f for f in os.listdir(plot_dir) if f.endswith('.png')]

    expected_plots = ['science_vs_bands.png', 'science_vs_int_time.png', 'science_vs_source_freq.png',
                      'science_vs_FoV.png', 'science_vs_beam_size.png', 'science_vs_total_time.png']

    if all(plot_file in existing_plots for plot_file in expected_plots):
        return
    else:
        print(f"Generating helping plots to guide you in the scientific query, check them in {plot_dir}.")
        # Identify missing plots
    missing_plots = [plot for plot in expected_plots if plot not in existing_plots]

    # Query only for variables associated with missing plots
    query_variables = set()
    for missing_plot in missing_plots:
        if missing_plot == 'science_vs_bands.png':
            query_variables.update(['science_keyword', 'band_list'])
        elif missing_plot == 'science_vs_int_time.png':
            query_variables.update(['science_keyword', 't_resolution'])
        elif missing_plot == 'science_vs_source_freq.png':
            query_variables.update(['science_keyword', 'frequency'])
        elif missing_plot == 'science_vs_FoV.png':
            query_variables.update(['science_keyword', 'band_list'])
        elif missing_plot == 'science_vs_beam_size.png':
            query_variables.update(['science_keyword', 'band_list', 'antenna_arrays'])
        elif missing_plot == 'science_vs_total_time.png':
            query_variables.update(['science_keyword', 't_max'])

    query = f"""  
            SELECT {', '.join(query_variables)}, member_ous_uid
            FROM ivoa.obscore  
            WHERE science_observation = 'T'
            AND is_mosaic = 'F'
            """
    
    custom_palette = sns.color_palette("tab20")
    sns.set_palette(custom_palette)
    db = service.search(query).to_table().to_pandas()
    db = db.drop_duplicates(subset='member_ous_uid')

    # Splitting the science keywords at commas
    db['science_keyword'] = db['science_keyword'].str.split(',')
    db['science_keyword'] = db['science_keyword'].apply(lambda x: [y.strip() for y in x])
    db = db.explode('science_keyword')
    db = db.drop(db[db['science_keyword'] == ''].index)
    db = db.drop(db[db['science_keyword'] == 'Exoplanets'].index)
    db = db.drop(db[db['science_keyword'] == 'Galaxy structure &evolution'].index)
    db = db.drop(db[db['science_keyword'] == 'Evolved stars: Shaping/physical structure'].index)
    short_keyword = {
        'Solar system - Trans-Neptunian Objects (TNOs)' : 'Solar System - TNOs',
        'Photon-Dominated Regions (PDR)/X-Ray Dominated Regions (XDR)': 'Photon/X-Ray Domanited Regions',
        'Luminous and Ultra-Luminous Infra-Red Galaxies (LIRG & ULIRG)': 'LIRG & ULIRG',
        'Cosmic Microwave Background (CMB)/Sunyaev-Zel\'dovich Effect (SZE)': 'CMB/Sunyaev-Zel\'dovich Effect',
        'Active Galactic Nuclei (AGN)/Quasars (QSO)': 'AGN/QSO',
        'Inter-Stellar Medium (ISM)/Molecular clouds': 'ISM & Molecular Clouds',
    }
    
    db['science_keyword'] = db['science_keyword'].replace(short_keyword)

    for missing_plot in missing_plots:
        if missing_plot == 'science_vs_bands.png':
            db['band_list'] = db['band_list'].str.split(' ')
            db['band_list'] = db['band_list'].apply(lambda x: [y.strip() for y in x])
            db = db.explode('band_list')

            db_sk_b = db.groupby(['science_keyword', 'band_list']).size().unstack(fill_value=0)

            plt.rcParams["figure.figsize"] = (28,20)
            db_sk_b.plot(kind='barh', stacked=True, color=custom_palette)
            plt.title('Science Keywords vs. ALMA Bands')
            plt.xlabel('Counts')
            plt.ylabel('Science Keywords')
            plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left',title='ALMA Bands')
            plt.savefig(os.path.join(plot_dir, 'science_vs_bands.png'))
            plt.close()

        elif missing_plot == 'science_vs_int_time.png':
            db = db[db['t_resolution'] <= 3e4]
            time_bins = np.arange(db['t_resolution'].min(), db['t_resolution'].max(), 1000)  # 1000 second bins
            db['time_bin'] = pd.cut(db['t_resolution'], bins=time_bins)

            db_sk_t = db.groupby(['science_keyword', 'time_bin']).size().unstack(fill_value=0)

            plt.rcParams["figure.figsize"] = (28,20)
            db_sk_t.plot(kind='barh', stacked=True)
            plt.title('Science Keywords vs. Integration Time')
            plt.xlabel('Counts')
            plt.ylabel('Science Keywords')
            plt.legend(title='Integration Time', loc='upper left', bbox_to_anchor=(1.01, 1))
            plt.savefig(os.path.join(plot_dir, 'science_vs_int_time.png'))
            plt.close()

        elif missing_plot == 'science_vs_source_freq.png':
            frequency_bins = np.arange(db['frequency'].min(), db['frequency'].max(), 50)  # 50 GHz bins
            db['frequency_bin'] = pd.cut(db['frequency'], bins=frequency_bins)

            db_sk_f = db.groupby(['science_keyword', 'frequency_bin']).size().unstack(fill_value=0)

            plt.rcParams["figure.figsize"] = (28,20)
            db_sk_f.plot(kind='barh', stacked=True, color=custom_palette)
            plt.title('Science Keywords vs. Source Frequency')
            plt.xlabel('Counts')
            plt.ylabel('Science Keywords')
            plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left',title='Frequency')
            plt.savefig(os.path.join(plot_dir, 'science_vs_source_freq.png')) 
            plt.close()

        elif missing_plot == 'science_vs_FoV.png':
            db['band_list'] = db['band_list'].str.split(' ')
            db['band_list'] = db['band_list'].apply(lambda x: [y.strip() for y in x])
            db = db.explode('band_list')
            db['fov'] = db['band_list'].apply(lambda x: get_fov_from_band(int(x)))
            fov_bins = np.arange(db['fov'].min(), db['fov'].max(), 10)  #  10 arcsec bins
            db['fov_bins'] = pd.cut(db['fov'], bins=fov_bins)

            db_sk_fov = db.groupby(['science_keyword', 'fov_bins']).size().unstack(fill_value=0)

            plt.rcParams["figure.figsize"] = (28,20)
            db_sk_fov.plot(kind='barh', stacked=True, color=custom_palette)
            plt.title('Science Keywords vs. FoV')
            plt.xlabel('Counts')
            plt.ylabel('Science Keywords')
            plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left',title='FoV')
            plt.savefig(os.path.join(plot_dir, 'science_vs_FoV.png'))
            plt.close()

        elif missing_plot == 'science_vs_beam_size.png':
            db['band_list'] = db['band_list'].str.split(' ')
            db['band_list'] = db['band_list'].apply(lambda x: [y.strip() for y in x])
            db = db.explode('band_list')
            db['max_baseline'] = db['antenna_arrays'].apply(lambda x: get_max_baseline_from_antenna_array(x, master_path))
            db['central_freq'] = db['band_list'].apply(lambda x: get_band_central_freq(int(x)))
            db['beam_size'] = db[['central_freq', 'max_baseline']].apply(lambda x: estimate_alma_beam_size(*x), axis=1)
            beam_size_bins = np.arange(db['beam_size'].min(), db['beam_size'].max(), 0.3)  # 0.1 arcsec bins
            db['beam_bins'] = pd.cut(db['beam_size'], bins=beam_size_bins)

            db_sk_bs = db.groupby(['science_keyword', 'beam_bins']).size().unstack(fill_value=0)

            plt.rcParams["figure.figsize"] = (28,20)
            db_sk_bs.plot(kind='barh', stacked=True, color=custom_palette)
            plt.title('Science Keywords vs. beams_size')
            plt.xlabel('Counts')
            plt.ylabel('Science Keywords')
            plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left',title='Beams Size')
            plt.savefig(os.path.join(plot_dir, 'science_vs_beam_size.png'))
            plt.close()

        elif missing_plot == 'science_vs_total_time.png':
            total_time_bins = np.arange(db['t_max'].min(), db['t_max'].max(), 500)  # 500 seconds bins
            db['Ttime_bins'] = pd.cut(db['t_max'], bins=total_time_bins)
            
            db_sk_Tt = db.groupby(['science_keyword', 'Ttime_bins']).size().unstack(fill_value=0)

            plt.rcParams["figure.figsize"] = (28,20)
            db_sk_Tt.plot(kind='barh', stacked=True, color=custom_palette)
            plt.title('Science Keywords vs. Total Time')
            plt.xlabel('Counts')
            plt.ylabel('Science Keywords')
            plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left',title='Total Time')
            plt.savefig(os.path.join(plot_dir, 'science_vs_total_time.png'))
            plt.close()
    
def query_for_metadata_by_science_type(metadata_name, main_path):
    plot_science_keywords_distributions(main_path)
    science_keywords, scientific_categories = get_science_types()
    path = os.path.join(main_path, "metadata", metadata_name)

    print(f'Please take a look at distributions in plots folder: {main_path}/plots')
    print('Available science keywords:')
    for i, keyword in enumerate(science_keywords):
        print(f'{i}: {keyword}')
    print('\nAvailable scientific categories:')
    for i, category in enumerate(scientific_categories):
        print(f'{i}: {category}')

    # Input for selection using space-separated values or empty for no filter
    science_keyword_number = input('Select the Science Keyword by number, separate by space, leave empty for all: ')
    scientific_category_number = input('Select the Scientific Category by number, separate by space, leave empty for all: ')
    band = input('Select observing bands, separate by space, leave empty for all: ')
    fov_input = input("Select FOV range as min FOV and max FOV separated by space, a single value is interpreted as the max, or leave empty for no filters: ")
    time_resolution_input = input("Select time resolution range as min max separated by space, a single value is interpreted as the max, or leave empty for no filters: ")
    total_time_input = input("Select total time range as min max separated by space, a single value is interpreted as the max, or leave empty for no filters: ")
    frequency_input = input("Select the source frequency range as min max separated by space, a single value is interpreted as the max, or leave empty for no filters: ")

    # Convert input selections to filters
    science_keyword = [science_keywords[int(i)] for i in science_keyword_number.split()] if science_keyword_number else None
    scientific_category = [scientific_categories[int(i)] for i in scientific_category_number.split()] if scientific_category_number else None
    bands = [int(x) for x in band.split()] if band else None

    # Convert input ranges to tuples or None
    fovs = [float(fov) for fov in fov_input.split()] if fov_input else None
    if isinstance(fovs, list):
        if len(fovs) > 1:
            fov_range = tuple([fovs[0], fovs[1]])
        else: 
            fov_range = tuple([0., fovs[0]])
    else:
        fov_range = None
    time_resolutions = [float(time_res) for time_res in time_resolution_input.split()] if time_resolution_input else None
    if isinstance(time_resolutions, list):
        if len(time_resolutions) > 1:
            time_resolution_range = tuple([time_resolutions[0], time_resolutions[1]])
        else:
            time_resolution_range = tuple([0., time_resolutions[0]])
    else:
        time_resolution_range = None
    total_times = [float(total_time) for total_time in total_time_input.split()] if total_time_input else None
    if isinstance(total_times, list):
        if len(total_times) > 1:
            total_time_range = tuple([total_times[0], total_times[1]])
        else:
            total_time_range = tuple([0., total_times[0]])
    else:
        total_time_range = None
    frequencies = [float(frequency) for frequency in frequency_input.split()] if frequency_input else None
    if isinstance(frequencies, list):
        if len(frequencies) > 1:
            frequency_range = tuple([frequencies[0], frequencies[1]])
        else:
            frequency_range = tuple([0., frequencies[0]])
    else:
        frequency_range = None

    # Query the database with all filters
    df = query_by_science_type(science_keyword, scientific_category, bands, fov_range, time_resolution_range, total_time_range, frequency_range)
    df = df.drop_duplicates(subset='member_ous_uid').drop(df[df['science_keyword'] == ''].index)

    # Rename columns and select relevant data
    rename_columns = {
        'target_name': 'ALMA_source_name',
        'pwv': 'PWV',
        'schedblock_name': 'SB_name',
        'velocity_resolution': 'Vel.res.',
        'spatial_resolution': 'Ang.res.',
        's_ra': 'RA',
        's_dec': 'Dec',
        's_fov': 'FOV',
        't_resolution': 'Int.Time',
        't_max': 'Total.Time',
        'cont_sensitivity_bandwidth': 'Cont_sens_mJybeam',
        'sensitivity_10kms': 'Line_sens_10kms_mJybeam',
        'obs_release_date': 'Obs.date',
        'band_list': 'Band',
        'bandwidth': 'Bandwidth',
        'frequency': 'Freq',
        'frequency_support': 'Freq.sup.'
    }
    df.rename(columns=rename_columns, inplace=True)
    database = df[['ALMA_source_name', 'Band', 'PWV', 'SB_name', 'Vel.res.', 'Ang.res.', 'RA', 'Dec', 'FOV', 'Int.Time',
                   'Total.Time', 'Cont_sens_mJybeam', 'Line_sens_10kms_mJybeam', 'Obs.date', 'Bandwidth', 'Freq',
                   'Freq.sup.', 'antenna_arrays']]
    database.loc[:, 'Obs.date'] = database['Obs.date'].apply(lambda x: x.split('T')[0])
    database.to_csv(path, index=False)

    print(f'Metadata saved to {path}\n')
    return database

def get_antennas_distances_from_reference(antenna_config):
    f = open(antenna_config)
    lines = f.readlines()
    nlines = len(lines)
    frefant = int((nlines - 1) // 2)
    f.close()
    zx, zy, zz, zztot = [], [], [], []
    for i in range(3,nlines):
        stuff = lines[i].split()
        zx.append(float(stuff[0]))
        zy.append(float(stuff[1]))
        zz.append(float(stuff[2]))
    nant = len(zx)
    nref = int(frefant)
    for i in range(0,nant):
        zxref = zx[i]-zx[nref]
        zyref = zy[i]-zy[nref]
        zzref = zz[i]-zz[nref]
        zztot.append(np.sqrt(zxref**2+zyref**2+zzref**2))
    return zztot, frefant

def generate_prms(antbl,scaleF):
    """
    This function generates the phase rms for the atmosphere
    as a function of antenna baseline length.
    It is based on the structure function of the atmosphere and 
    it gives 30 deg phase rms at 10000m = 10km.

    Input: 
    antbl = antenna baseline length in meters
    scaleF = scale factor for the phase rms
    Output:
    prms = phase rms
    """
    Lrms = 1.0/52.83 * antbl**0.8     # phase rms ~0.8 power to 10 km
    Hrms = 3.0 * antbl**0.25          # phase rms `0.25 power beyond 10 km
    if antbl < 10000.0:
        prms = scaleF*Lrms
    if antbl >= 10000.0:
        prms = scaleF*Hrms
    return prms

def simulate_atmospheric_noise(sim_output_dir, project, scale, ms, antennalist):
    zztot, frefant = get_antennas_distances_from_reference(antennalist)
    gaincal(
        vis=ms,
        caltable=os.path.join(sim_output_dir, project + "_atmosphere.gcal"),
        refant=str(frefant), #name of the reference antenna
        minsnr=0.00, #ignore solution with SNR below this
        calmode="p", #phase
        solint='inf', #solution interval,
    )
    tb = table()
    tb.open(os.path.join(sim_output_dir, project + "_atmosphere.gcal"), nomodify=False)
    yant = tb.getcol('ANTENNA1')
    ytime = tb.getcol('TIME')
    ycparam = tb.getcol('CPARAM')
    nycparam = ycparam.copy()
    nant = len(yant)
    for i in range(nant):
        antbl = zztot[yant[i]]
        # get rms phase for each antenna
        prms = generate_prms(antbl,scale)
        # determine random GAUSSIAN phase error from rms phase
        perror = random.gauss(0,prms)
        # adding a random phase error to the solution, it will be 
        # substituted by a frequency that depends from frequency
        # of observation and baseline length
        perror = perror + random.gauss(0, 0.05 * perror)
        # convert phase error to complex number
        rperror = np.cos(perror*pi/180.0)
        iperror = np.sin(perror*pi/180.0)
        nycparam[0][0][i] = 1.0*complex(rperror,iperror)  #X POL
        nycparam[1][0][i] = 1.0*complex(rperror,iperror)  #Y POL  ASSUMED SAME
    tb.putcol('CPARAM', nycparam)
    tb.flush()
    tb.close()
    applycal(
        vis = ms,
        gaintable = os.path.join(sim_output_dir, project + "_atmosphere.gcal")
    )
    #os.system("rm -rf " + os.path.join(sim_output_dir, project + "_atmosphere.gcal"))
    return 

def simulate_gain_errors(ms, amplitude: float = 0.01):
    sm = casa_simulator()
    sm.openfromms(ms)
    sm.setseed(42)
    sm.setgain(mode='fbm', amplitude=[amplitude])
    sm.corrupt()
    sm.close()
    return

def _ms2resolve_transpose(arr):
    my_asserteq(arr.ndim, 3)
    return np.ascontiguousarray(np.transpose(arr, (0, 2,1)))

def my_asserteq(*args):
    for aa in args[1:]:
        if args[0] != aa:
            raise RuntimeError(f"{args[0]} != {aa}")

def ms_to_npz(ms, dirty_cube, datacolumn='CORRECTED_DATA', output_file='test.npz'):
    tb = table()
    tb.open(ms)
    
    #get frequency info from dirty cube
    with fits.open(dirty_cube, memmap=False) as hdulist: 
            npol, nz, nx, ny = np.shape(hdulist[0].data)
            header=hdulist[0].header
    crdelt3 = header['CDELT3']
    crval3 = header['CRVAL3']
    wave = ((crdelt3 * (np.arange(0, nz, 1))) + crval3) #there will be problems, channels       are not of the same width in real data

    vis = tb.getcol(datacolumn)
    vis = np.ascontiguousarray(_ms2resolve_transpose(vis))

    wgt = tb.getcol('WEIGHT')
    wgt = np.repeat(wgt[:,None],128,axis=1)
    #this is to get vis and wgt on the same shape if ms has column weighted_spectrum this       should be different
    wgt = np.ascontiguousarray(_ms2resolve_transpose(wgt))

    uvw = np.transpose(tb.getcol('UVW'))

    np.savez_compressed(output_file,
                    freq = wave,
                    vis= vis, 
                    weight= wgt,
                    polarization=[9,12], 
                    antpos0=uvw,
                    antpos1=tb.getcol('ANTENNA1'),
                    antpos2=tb.getcol('ANTENNA2'),
                    antpos3=tb.getcol('TIME'))
