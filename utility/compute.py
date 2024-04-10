import numpy as np
import astropy.units as U
from casatasks import exportfits, simobserve, tclean, gaincal, applycal
from casatools import table
from casatools import simulator as casa_simulator
import sys
import os
import pandas as pd
import utility.alma as ual
import utility.astro as uas
def simulator(inx, main_dir, output_dir, tng_dir, project_name, ra, dec, band, ang_res, vel_res, fov, obs_date, 
              pwv, int_time, total_time, bandwidth, freq, freq_support, antenna_array, n_pix, 
              n_channels, source_type, tng_subhaloid, ncpu, rest_frequency, redshift):
    """
    Runs a simulation for a given set of input parameters.

    Args:
        inx (int): Index of the simulation.
        main_dir (str): Path to the main directory.
        output_dir (str): Path to the output directory.
        ra (float): Right Ascension of the source in degrees.
        dec (float): Declination of the source in degrees.
        band (str): ALMA band to use for the simulation.
        ang_res (float): Angular resolution of the simulation in arcseconds.
        vel_res (float): Velocity resolution of the simulation in km/s.
        fov (float): Field of view of the simulation in arcseconds.
        obs_date (str): Observation date for the simulation.
        pwv (float): Precipitable water vapor for the simulation.
        int_time (float): Integration time for the simulation in seconds.
        total_time (float): Total time for the simulation in seconds.
        bandwidth (float): Bandwidth of the simulation in GHz.
        freq (float): Frequency of the simulation in GHz.
        freq_support (str): Frequency support for the simulation.
        antenna_array (str): Antenna array to use for the simulation.
        n_pix (int): Number of pixels for the simulation cube.
        n_channels (int): Number of channels for the simulation cube.
        source_type (str): Type of source to simulate (point, gaussian, extended, diffuse).
        tng_subhaloid (int): Subhaloid ID for the TNG simulation.
        ncpu (int): Number of CPU cores to use for the simulation.
    """
    ra = ra * U.deg
    dec = dec * U.deg
    ang_res = ang_res * U.arcsec
    vel_res = vel_res * U.km / U.s
    int_time = int_time * U.s
    total_time = total_time * U.s
    band_range = ual.get_band_range(int(band))
    band_range = band_range[1] - band_range[0]
    band_range = band_range * U.GHz
    source_freq = freq * U.GHz
    central_freq = ual.get_band_central_freq(int(band)) * U.GHz
    sim_output_dir = os.path.join(output_dir, project_name + '_{}'.format(inx))
    if not os.path.exists(sim_output_dir):
        os.makedirs(sim_output_dir)
    ual.generate_antenna_config_file_from_antenna_array(antenna_array, main_dir, sim_output_dir)
    antennalist = os.path.join(sim_output_dir, "antenna.cfg")
    antenna_name = 'antenna'
    max_baseline = ual.get_max_baseline_from_antenna_config(antennalist) * U.km
    pos_string = uas.convert_to_j2000_string(ra.value, dec.value)
    if redshift is None:
        rest_frequency = rest_frequency * U.GHz
        redshift = uas.compute_redshift(rest_frequency, source_freq)
    #rest_frequency = 115.271 * U.GHz
    else:
        rest_frequency = uas.compute_rest_frequency_from_redshift(source_freq, redshift) * U.GHz

    
    brightness = uas.sample_from_brightness_given_redshift(vel_res, rest_frequency.value, os.path.join(main_dir, 'brightnes', 'CO10.dat'), redshift)
    if source_type == 'extended':
        snapshot = uas.redshift_to_snapshot(redshift)
        outpath = os.path.join(tng_dir, 'TNG100-1', 'output', 'snapdir_0{}'.format(snapshot))
        part_num = uas.get_particles_num(tng_dir, outPath, snapshot, int(tng_subhalo_id))

