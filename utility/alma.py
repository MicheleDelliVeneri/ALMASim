from astropy.constants import c
import astropy.units as U
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
import math
import pyvo
import numpy as np 
import pandas as pd
import os 
import sys
from casatasks import exportfits, simobserve, tclean, gaincal, applycal
from casatools import table
from casatools import simulator as casa_simulator
import random
from math import pi
import matplotlib.pyplot as plt
import seaborn as sns

# Metadata related functions

def estimate_alma_beam_size(central_frequency_ghz, max_baseline_km):
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

  return beam_size_arcsec

def estimate_alma_beam_size_from_db(row):
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
  central_frequency_ghz = row['central_freq'].values * U.GHz
  max_baseline_km = row['max_baseline'].values * U.Km
   # Input validation
  if central_frequency_ghz <= 0 or max_baseline_km <= 0:
    raise ValueError("Central frequency and maximum baseline must be positive values.")

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

  return beam_size_arcsec

def get_fov_from_band(band, antenna_diameter: int = 12):
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
    max_baseline = 2 * np.max(np.sqrt(positions[:, 0]**2 + positions[:, 1]**2 + positions[:, 2]**2)) / 1000
    return max_baseline

def get_max_baseline_from_antenna_array(antenna_array, master_path):
    antenna_coordinates = pd.read_csv(os.path.join(master_path, 'antenna_config', 'antenna_coordinates.csv'))
    obs_antennas = antenna_array.split(' ')
    obs_antennas = [antenna.split(':')[0] for antenna in obs_antennas]
    obs_coordinates = antenna_coordinates[antenna_coordinates['name'].isin(obs_antennas)]
    max_baseline = 2 * np.max(np.sqrt(obs_coordinates['x'].values**2 + obs_coordinates['y'].values**2 + obs_coordinates['z'].values**2)) / 1000
    return max_baseline

def query_observations(service, member_ous_uid, target_name):
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
            """

    result = service.search(query).to_table().to_pandas()

    return result

def query_all_targets(service, targets):
    """Query observations for all predefined targets and compile the results into a single DataFrame.

    Parameters:
    service (pyvo.dal.TAPService): A TAPService instance for querying the database.
    targets (list of tuples): A list where each tuple contains (target_name, member_ous_uid).

    Returns:
    pandas.DataFrame: A DataFrame containing the results for all queried targets.
    """
    results = []

    for target_name, member_ous_uid in targets:
        result = query_observations(service, member_ous_uid, target_name)
        results.append(result)

    # Concatenate all DataFrames into a single DataFrame
    df = pd.concat(results, ignore_index=True)

    return df

def query_for_metadata_by_targets(targets, path, service_url: str = "https://almascience.eso.org/tap"):
    """Query for metadata for all predefined targets and compile the results into a single DataFrame.

    Parameters:
    service_url (str): A TAPService http address for querying the database.
    targets (list of tuples): A list where each tuple contains (target_name, member_ous_uid).
    path (str): The path to save the results to.

    Returns:
    pandas.DataFrame: A DataFrame containing the results for all queried targets.
    """
    # Create a TAPService instance (replace 'your_service_url' with the actual URL)
    service = pyvo.dal.TAPService(service_url)
    # Query all targets and compile the results
    df = query_all_targets(service, targets)
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

def get_science_types(service):
    query = f"""  
            SELECT science_keyword, scientific_category  
            FROM ivoa.obscore  
            WHERE science_observation = 'T'    
            """
    db = service.search(query).to_table().to_pandas()
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
                        keyword != 'Exo-planets' and 
                        keyword != 'Galaxy structure &evolution')]
    
    return  unique_keywords, scientific_category
    
def query_by_science_type(service, science_keyword=None, scientific_category=None, band=None):
    """Query for all science observations of given member OUS UID and target name, selecting all columns of interest.

    Parameters:
    service (pyvo.dal.TAPService): A TAPService instance for querying the database.

    Returns:
    pandas.DataFrame: A table of query results.
    """
    if science_keyword == None:
        science_keyword = ""
    if scientific_category == None:
        scientific_category = ""
    if band == None:
        band = ""
    print('Querying for science keyword/s: ', science_keyword)
    print('And scientific category/ies: ', scientific_category)
    print('And band/s: ', band)
    if type(science_keyword) == list and len(science_keyword) == 1:
        science_keyword = science_keyword[0]
        science_keyword_query = f"science_keyword like '%{science_keyword}%'"
    elif type(science_keyword) == list and len(science_keyword) > 1:
        science_keyword = "', '".join(science_keyword)
        science_keyword_query = f"science_keyword in ('{science_keyword}')"
    else:
        science_keyword_query = f"science_keyword like '%{science_keyword}%'"
    if type(scientific_category) == list and len(scientific_category) == 1:
        scientific_category = scientific_category[0]
        scientific_category_query = f"scientific_category like '%{scientific_category}%'"
    elif type(scientific_category) == list and len(scientific_category) > 1:
        scientific_category = "', '".join(scientific_category)
        scientific_category_query = f"scientific_category in ('{scientific_category}')"
    else:
        scientific_category_query = f"scientific_category like '%{scientific_category}%'"
    if type(band) == list and len(band) == 1:
        band = band[0]
        band_query = f"band_list like '%{band}%'"
    elif type(band) == list and len(band) > 1:
        band = "', '".join(band)
        band_query = f"band_list in ('{band}')"
    else:
        band_query = f"band_list like '%{band}%'"

    query = f"""
            SELECT *
            FROM ivoa.obscore
            WHERE {science_keyword_query}
            AND {scientific_category_query}
            AND is_mosaic = 'F'
            AND {band_query}
            """

    result = service.search(query).to_table().to_pandas()

    return result

def plot_science_keywords_distributions(service, master_path, output_dir):
    query = """  
            SELECT science_keyword, band_list, member_ous_uid, frequency, t_resolution, t_max, antenna_arrays
            FROM ivoa.obscore  
            WHERE science_observation = 'T'    
            """

    plot_dir = os.path.join(output_dir, 'plots')
    custom_palette = sns.color_palette("tab20")
    sns.set_palette(custom_palette)
    db = service.search(query).to_table().to_pandas()
    db = db.drop_duplicates(subset='member_ous_uid')
    db = db.drop(db[db['science_keyword'] == ''].index)
    # Splitting the science keywords at commas
    db['science_keyword'] = db['science_keyword'].str.split(',')
    db['science_keyword'] = db['science_keyword'].str.strip()
    db['band_list'] = db['band_list'].str.split(' ')
    db['band_list'] = db['band_list'].str.strip()
    db = db.dropna()
    db['max_baseline'] = db['antenna_arrays'].apply(lambda x: get_max_baseline_from_antenna_array(x, master_path))
    db['central_freq'] = db['band_list'].apply(lambda x: get_band_central_freq(int(x)))
    db['fov'] = db['band_list'].apply(lambda x: get_fov_from_band(int(x)))
    beam_sizes = db.apply(estimate_alma_beam_size_from_db, axis=1)
    
    # TESTING 
    #Checking Freq. distribution
    plt.hist(db['fov'], bins=50, alpha=0.75)
    plt.title('FOV Distribution')
    plt.xlabel('FOV arcsec')
    plt.ylabel('Count')
    plt.savefig(os.path.join(plot_dir, 'fov_dir.png'))

    #Checking time integration distribution < 30000 s 
    plt.hist(db['t_max'], bins=100, alpha=0.75, log=True)
    plt.title('Total Time Distribution')
    plt.xlabel('Total Time (s)')
    plt.ylabel('Count')
    plt.xscale('log')
    plt.savefig(os.path.join(plot_dir, 'tottime_dir.png'))

    plt.hist(db['beam_size'], bins=50, alpha=0.75)
    plt.title('Beam Distribution')
    plt.xlabel('Beam arcsec')
    plt.ylabel('Count')
    plt.savefig(os.path.join(plot_dir, 'bs_dir.png'))

    # Exploding to have one row for each combination of science keyword and band
    db = db.explode(['science_keyword', 'band_list', 'frequency', 't_resolution', 't_max', 'max_baseline', 'central_freq', 'fov', 'beam_size'])

    db = db[db['t_resolution'] <= 3e4]
    frequency_bins = np.arange(db['frequency'].min(), db['frequency'].max(), 50)  # 50 GHz bins
    db['frequency_bin'] = pd.cut(db['frequency'], bins=frequency_bins)
    time_bins = np.arange(db['t_resolution'].min(), db['t_resolution'].max(), 1000)  # 1000 second bins
    db['time_bin'] = pd.cut(db['t_resolution'], bins=time_bins)

    db_sk_b = db.groupby(['science_keyword', 'band_list']).size().unstack(fill_value=0)
    db_sk_f = db.groupby(['science_keyword', 'frequency_bin']).size().unstack(fill_value=0)
    db_sk_t = db.groupby(['science_keyword', 'time_bin']).size().unstack(fill_value=0)
    db_sk_fov = db.groupby(['science_keyword', 'fov']).size().unstack(fill_value=0)
    db_sk_bs = db.groupby(['science_keyword', 'beam_size']).size().unstack(fill_value=0)
    
    

    plt.rcParams["figure.figsize"] = (14,18)
    db_sk_b.plot(kind='barh', stacked=True, color=custom_palette)
    plt.title('Science Keywords vs. ALMA Bands')
    plt.xlabel('Counts')
    plt.ylabel('Science Keywords')
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left',title='ALMA Bands')
    plt.savefig(os.path.join(plot_dir, 'science_vs_bands.png'))

    plt.rcParams["figure.figsize"] = (14,18)
    db_sk_t.plot(kind='barh', stacked=True)
    plt.title('Science Keywords vs. Integration Time')
    plt.xlabel('Counts')
    plt.ylabel('Science Keywords')
    plt.legend(title='Integration Time', loc='upper left', bbox_to_anchor=(1.01, 1))
    plt.savefig(os.path.join(plot_dir, 'science_vs_int_time.png'))

    plt.rcParams["figure.figsize"] = (14,18)
    db_sk_f.plot(kind='barh', stacked=True, color=custom_palette)
    plt.title('Science Keywords vs. Source Frequency')
    plt.xlabel('Counts')
    plt.ylabel('Science Keywords')
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left',title='Frequency')
    plt.savefig(os.path.join(plot_dir, 'science_vs_source_freq.png')) 

def query_for_metadata_by_science_type(metadata_name, main_path, output_dir, service_url: str = "https://almascience.eso.org/tap"):
    service = pyvo.dal.TAPService(service_url)
    science_keywords, scientific_categories = get_science_types(service)
    path = os.path.join(main_path, "metadata", metadata_name)
    plot_science_keywords_distributions(service, main_path, output_dir)
    print('Please take a look at distributions in plots folder: {output_dir}/plots')
    #plt.rcParams["figure.figsize"] = (14,18)
    #counts.plot(kind='barh', stacked=True)
    #plt.title('Science Keywords vs. ALMA Bands')
    #plt.xlabel('Counts')
    #plt.ylabel('Science Keywords')
    #plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left',title='ALMA Bands')
    #plt.show()
    print('Available science keywords:')
    for i in range(len(science_keywords)):
        print(f'{i}: {science_keywords[i]}')   
    print('\nAvailable scientific categories:')
    for i in range(len(scientific_categories)):
        print(f'{i}: {scientific_categories[i]}')
    science_keyword_number = input('Select the Science Keyword by number, if you want to select multiple numbers separate them by a space, leave empty for all: ' )
    scientific_category_number = input('Select the Scientific Category by number, if you want to select multiple numbers separate them by a space, leave empty for all: ' )
    band = input('Select observing bands, if you want to select multiple bands separate them by a space, leave empty for all: ')
    if science_keyword_number == "":
        science_keyword = None
    else:
        science_keyword_number = [int(x) for x in science_keyword_number.split(' ') if x != '']
        science_keyword = [science_keywords[i] for i in science_keyword_number]

    duplicates = ['Evolved stars: Shaping/physical structure', 'Exo-planets', 'Galaxy structure &evolution']
    original = ['Evolved stars - Shaping/physical structure', 'Exoplanets', 'Galaxy structure & evolution']
    for i in range(len(original)):
        if original[i] in [science_keyword]:
            science_keywords.append(duplicates[i])
    if scientific_category_number == "":
        scientific_category = None
    else:
        scientific_category_number = [int(x) for x in scientific_category_number.split(' ') if x != '']
        scientific_category = [scientific_categories[i] for i in scientific_category_number]
    if band == "":
        bands = None
    else:
        bands = [int(x) for x in band.split(' ') if x != '']
    df = query_by_science_type(service, science_keyword, scientific_category, bands)
    df = df.drop_duplicates(subset='member_ous_uid')
    df = df.drop(df[df['science_keyword'] == ''].index)
    
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
        solint='inf', #solution interval
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
