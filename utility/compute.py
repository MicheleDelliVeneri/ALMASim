import numpy as np
import astropy.units as U
from casatasks import exportfits, simobserve, tclean, gaincal, applycal
from casatools import table
from casatools import simulator as casa_simulator
import sys
import os
import random
import pandas as pd
import utility.alma as ual
import utility.astro as uas
import utility.skymodels as usm
import utility.plotting as upl
import shutil
from os.path import isfile

def remove_non_numeric(text):
  """Removes non-numeric characters from a string.

  Args:
      text: The string to process.

  Returns:
      A new string containing only numeric characters and the decimal point (.).
  """
  numbers = "0123456789."
  return "".join(char for char in text if char in numbers)

def remove_logs(folder_path):
    for filename in os.listdir(folder_path):
        # Check if the file ends with '.log'
        if filename.endswith('.log'):
            # Construct the full path to the file
            file_path = os.path.join(folder_path, filename)
            # Remove the file
            os.remove(file_path)

def simulator(inx, main_dir, output_dir, tng_dir, project_name, ra, dec, band, ang_res, vel_res, fov, obs_date, 
              pwv, int_time, total_time, bandwidth, freq, freq_support, antenna_array, n_pix, 
              n_channels, source_type, tng_api_key, ncpu, rest_frequency, redshift, save_secondary=False, 
              inject_serendipitous=False):
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
    print('Running simulation {}'.format(inx))
    ra = ra * U.deg
    dec = dec * U.deg
    ang_res = ang_res * U.arcsec
    vel_res = vel_res * U.km / U.s
    int_time = int_time * U.s
    total_time = total_time * U.s
    freq_support = freq_support.split(' U ')[0].split(',')[1]
    freq_sup = float(remove_non_numeric(freq_support)) * U.kHz
    freq_sup = freq_sup.to(U.MHz)   
    band_range = ual.get_band_range(int(band))
    band_range = band_range[1] - band_range[0]
    band_range = band_range * U.GHz
    source_freq = freq * U.GHz
    central_freq = ual.get_band_central_freq(int(band)) * U.GHz
    sim_output_dir = os.path.join(output_dir, project_name + '_{}'.format(inx))
    
    if not os.path.exists(sim_output_dir):
        os.makedirs(sim_output_dir)
    os.chdir(output_dir)
    ual.generate_antenna_config_file_from_antenna_array(antenna_array, main_dir, sim_output_dir)
    antennalist = os.path.join(sim_output_dir, "antenna.cfg")
    antenna_name = 'antenna'
    max_baseline = ual.get_max_baseline_from_antenna_config(antennalist) * U.km
    pos_string = uas.convert_to_j2000_string(ra.value, dec.value)
    if redshift is None:
        rest_frequency = rest_frequency * U.GHz
        redshift = uas.compute_redshift(rest_frequency, source_freq)
    else:
        rest_frequency = uas.compute_rest_frequency_from_redshift(source_freq, redshift) * U.GHz

    print('Redshift: {}'.format(redshift))
    print('Rest frequency: {} GHz'.format(round(rest_frequency.value, 2)))
    print('Source frequency: {} GHz'.format(round(source_freq.value, 2)))
    print('Band: ', band)
    print('Velocity resolution: {} Km/s'.format(round(vel_res.value, 2)))
    print('Angular resolution: {} arcsec'.format(round(ang_res.value, 3)))
    if source_type == 'extended':
        snapshot = uas.redshift_to_snapshot(redshift)
        print('Snapshot: {}'.format(snapshot))
        tng_subhaloid = uas.get_subhaloids_from_db(1, main_dir, snapshot)
        print('Subhaloid ID: {}'.format(tng_subhaloid))
        outpath = os.path.join(tng_dir, 'TNG100-1', 'output', 'snapdir_0{}'.format(snapshot))
        part_num = uas.get_particles_num(tng_dir, outpath, snapshot, int(tng_subhaloid), tng_api_key)
        print('Number of particles: {}'.format(part_num))
        while part_num == 0:
            print('No particles found. Checking another subhalo.')
            tng_subhaloid = uas.get_subhaloids_from_db(1, main_dir, snapshot)
            outpath = os.path.join(tng_dir, 'TNG100-1', 'output', 'snapdir_0{}'.format(snapshot))
            part_num = uas.get_particles_num(tng_dir, outpath, snapshot, int(tng_subhaloid), tng_api_key)
            print('Number of particles: {}'.format(part_num))
    else:
        snapshot = None
        tng_subhaloid = None

    brightness = uas.sample_from_brightness_given_redshift(vel_res, rest_frequency.value, os.path.join(main_dir, 'brightnes', 'CO10.dat'), redshift)
    line_name = uas.get_line_name(rest_frequency.value)
    print('{} Brightness: {}'.format(line_name, round(brightness, 4)))
    fov =  ual.get_fov_from_band(int(band))
    beam_size = ual.estimate_alma_beam_size(central_freq, max_baseline)
    cell_size = beam_size / 5
    if n_pix is None: 
        #cell_size = beam_size / 5
        n_pix = int(1.5 * fov / cell_size)
    else:
        cell_size = fov / n_pix
        # just added
        #beam_size = cell_size * 5
    if n_channels is None:
        n_channels = int(band_range / freq_sup)
    
    print('Field of view: {}'.format(fov))
    print('Beam size: {} '.format(beam_size))
    print('Cell size: {} '.format(cell_size))
    central_channel_index = n_channels // 2
    source_channel_index = int(central_channel_index * source_freq / central_freq)
    # LUCA BRIGHTNESS FUNCTION n_canali, band_range, freq_sup, band, central_freq)
    datacube = usm.DataCube(
        n_px_x=n_pix, 
        n_px_y=n_pix,
        n_channels=n_channels, 
        px_size=cell_size, 
        channel_width=freq_sup, 
        velocity_centre=central_freq, 
        ra=ra, 
        dec=dec)
    wcs = datacube.wcs
    if source_type == 'point':
        pos_x, pos_y, _ = wcs.sub(3).wcs_world2pix(ra, dec, central_freq, 0)
        pos_z = int(source_channel_index)
        fwhm_z = np.random.randint(3, 10)   
        datacube = usm.insert_pointlike(datacube, brightness, pos_x, pos_y, pos_z, fwhm_z, n_pix, n_channels)
    elif source_type == 'gaussian':
        pos_x, pos_y, _ = wcs.sub(3).wcs_world2pix(ra, dec, central_freq, 0)
        pos_z = int(source_channel_index)
        fwhm_x = np.random.randint(3, 10)
        fwhm_y = np.random.randint(3, 10)
        fwhm_z = np.random.randint(3, 10)
        angle = np.random.randint(0, 180)
        datacube = usm.insert_gaussian(datacube, brightness, pos_x, pos_y, pos_z, fwhm_x, fwhm_y, fwhm_z, angle, n_pix, n_channels)
    elif source_type == 'extended':
        datacube = usm.insert_extended(datacube, tng_dir, snapshot, int(tng_subhaloid), redshift, ra, dec, tng_api_key, ncpu)

    if inject_serendipitous == True:
        if source_type != 'gaussian':
            fwhm_x = np.random.randint(3, 10)
            fwhm_y = np.random.randint(3, 10)
            fwhm_z = np.random.randint(3, 10)
        datacube = usm.insert_serendipitous(datacube, brightness, fwhm_x, fwhm_y, fwhm_z, n_pix, n_channels)
    
    filename = os.path.join(sim_output_dir, 'skymodel_{}.fits'.format(inx))
    usm.write_datacube_to_fits(datacube, filename)
    del datacube
    upl.plot_skymodel(filename, inx, output_dir, show=False)
    #skymodel, sky_header = uas.load_fits(filename)
    #sim_brightness = np.max(skymodel)
    
    #if sim_brightness != brightness:
    #    print('Detected peak is: {} re-normalizing'.format(sim_brightness) )
    #    flattened_skymodel = np.ravel(skymodel)
    #    t_min = 0
    #    t_max = brightness
    #    skymodel_norm = (flattened_skymodel - np.min(flattened_skymodel)) / (np.max(flattened_skymodel) - np.min(flattened_skymodel)) * (t_max - t_min) + t_min
    ##    skymodel = np.reshape(skymodel_norm, np.shape(skymodel))
     #   uas.write_numpy_to_fits(skymodel, sky_header, filename)
    project_name = project_name + '_{}'.format(inx)
    os.chdir(output_dir)
    uas.write_sim_parameters(os.path.join(output_dir, 'sim_params_{}.txt'.format(inx)),
                            ra, dec, ang_res, vel_res, int_time, total_time, band, central_freq,
                            source_freq, redshift, brightness, fov, beam_size, cell_size, n_pix, 
                            n_channels, snapshot, tng_subhaloid)
    simobserve(
        project=project_name, 
        skymodel=filename,
        obsmode="int",
        setpointings=True,
        thermalnoise="tsys-atm",
        antennalist=antennalist,
        indirection=pos_string,
        incell="{}arcsec".format(cell_size.value),
        incenter='{}GHz'.format(central_freq.value),
        inwidth="{}MHz".format(freq_sup.value),
        integration="{}s".format(int_time.value),
        totaltime="{}s".format(total_time.value),
        user_pwv=pwv,
        verbose=True,
        overwrite=True,
        graphics="none",
        )
    
    scale = random.uniform(0, 1)
    ms_path = os.path.join(sim_output_dir, "{}.{}.noisy.ms".format(project_name, antenna_name))
    ual.simulate_atmospheric_noise(sim_output_dir, project_name, scale, ms_path, antennalist)
    gain_error_amp = random.gauss(0, 0.1)
    ual.simulate_gain_errors(ms_path, gain_error_amp)
    tclean(
        vis=ms_path,
        imagename=os.path.join(sim_output_dir, '{}.{}'.format(project_name, antenna_name)),
        imsize=[int(n_pix), int(n_pix)],
        cell="{}".format(beam_size),
        phasecenter=pos_string,
        specmode="cube",
        niter=0,
        fastnoise=False,
        calcpsf=True,
        pbcor=True,
        pblimit=0.2, 
        )
    exportfits(imagename=os.path.join(project_name, '{}.{}.image'.format(project_name, antenna_name)), 
       fitsimage=os.path.join(output_dir, "dirty_cube_" + str(inx) +".fits"), overwrite=True)
    exportfits(imagename=os.path.join(project_name, '{}.{}.skymodel'.format(project_name, antenna_name)), 
        fitsimage=os.path.join(output_dir, "clean_cube_" + str(inx) +".fits"), overwrite=True)
    upl.plotter(inx, output_dir, beam_size)
    if save_secondary == True:
        exportfits(imagename=os.path.join(project_name, '{}.{}.psf'.format(project_name, antenna_name)),
              fitsimage=os.path.join(output_dir, "psf_" + str(inx) +".fits"), overwrite=True)
        exportfits(imagename=os.path.join(project_name, '{}.{}.pb'.format(project_name, antenna_name)),
                fitsimage=os.path.join(output_dir, "pb_" + str(inx) +".fits"), overwrite=True)
        ual.ms_to_npz(os.path.join(project_name, "{}.{}.noisy.ms".format(project_name, antenna_name)),
              dirty_cube=os.path.join(output_dir, "dirty_cube_" + str(inx) +".fits"),
              datacolumn='CORRECTED_DATA',
              output_file=os.path.join(output_dir, "ms_" + str(inx) +".npz"))
    shutil.rmtree(sim_output_dir)
    