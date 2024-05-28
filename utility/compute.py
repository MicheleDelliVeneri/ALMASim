import numpy as np
import astropy.units as U
from astropy.constants import c
from astropy.time import Time
import sys
import os
import random
import pandas as pd
import utility.alma as ual
import utility.astro as uas
import utility.skymodels as usm
import utility.plotting as upl
import utility.interferometer as uin
import shutil
from os.path import isfile
import math
from datetime import date
import time
from time import strftime, gmtime
#from casatasks import exportfits, simobserve, tclean, gaincal, applycal
#from casatools import table
#from casatools import simulator as casa_simulator

def check_dir_exists(absolute_path):
    if not absolute_path.startswith(os.path.sep):
        absolute_path = os.path.sep + absolute_path
    output_dir = absolute_path.split(os.path.sep)[-1]
    parent_dir = os.path.join(os.path.sep, *absolute_path.split(os.path.sep)[:-1])
    if not os.path.exists(parent_dir):
        return False
    else: 
        return True

def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def closest_power_of_2(x):
    op = math.floor if bin(x)[3] != "1" else math.ceil
    return 2 ** op(math.log(x, 2))

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

def load_metadata(main_path, metadata_name):
    if '.csv' not in metadata_name:
            metadata_name = metadata_name.split('.')[0]
            metadata_name = metadata_name + '.csv'
    try:
        metadata = pd.read_csv(os.path.join(main_path, "metadata", metadata_name))
        print('Metadata contains {} samples'.format(len(metadata)))
        return metadata
    except FileNotFoundError:
        print("File not found. Please enter the metadata name again.")
        new_metadata_name = input("Enter metadata name: ")
        if '.csv' not in metadata_name:
            new_metadata_name = new_metadata_name.split('.')[0]
            new_metadata_name = new_metadata_name + '.csv'
        return load_metadata(main_path, new_metadata_name)

def simulator(inx, main_dir, output_dir, tng_dir, project_name, ra, dec, band, ang_res, vel_res, fov, obs_date, 
              pwv, int_time, total_time, bandwidth, freq, freq_support, cont_sens, antenna_array, n_pix, 
              n_channels, source_type, tng_api_key, ncpu, rest_frequency, redshift, lum_infrared, snr,
              n_lines, line_names, save_secondary=False, inject_serendipitous=False):
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
    
    start = time.time()
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
    obs_date = Time(obs_date + 'T00:00:00', format='isot', scale='utc').to_value("mjd")
    if not os.path.exists(sim_output_dir):
        os.makedirs(sim_output_dir)
    os.chdir(output_dir)
    print('Angular resolution: {}'.format(ang_res))
    print('Integration Time: {}'.format(int_time))
    print('Total Observatio Time: {}'.format(total_time))
    ual.generate_antenna_config_file_from_antenna_array(antenna_array, main_dir, sim_output_dir)
    antennalist = os.path.join(sim_output_dir, "antenna.cfg")
    antenna_name = 'antenna'
    max_baseline = ual.get_max_baseline_from_antenna_config(antennalist) * U.km
    t_ang_res = c.to(U.m/U.s) / central_freq.to(U.Hz) / (max_baseline.to(U.m))
    t_ang_res = t_ang_res * (180 / math.pi) * 3600 * U.arcsec
    print('Angular Resolution computed from max baseline: {}'.format(t_ang_res))
    pos_string = uas.convert_to_j2000_string(ra.value, dec.value)
    fov =  ual.get_fov_from_band(int(band), return_value=False)
    beam_size = ual.estimate_alma_beam_size(central_freq, max_baseline, return_value=False)
    beam_solid_angle = np.pi * (beam_size / 2) ** 2
    cont_sens = cont_sens * U.mJy / (U.arcsec ** 2)
    cont_sens_jy = (cont_sens * beam_solid_angle).to(U.Jy)
    cont_sens  = cont_sens_jy  * snr
    print("Beam Size: ", beam_size)
    print("Minimum detectable continum: ", cont_sens_jy)

    cell_size = beam_size / 5
    if n_pix is None: 
        #cell_size = beam_size / 5
        n_pix = closest_power_of_2(int(1.5 * fov / cell_size))
    else:
        n_pix = closest_power_of_2(n_pix)
        cell_size = fov / n_pix
        # just added
        #beam_size = cell_size * 5
    if n_channels is None:
        n_channels = int(band_range / freq_sup)
    else:
        band_range = n_channels * freq_sup 
        band_range = band_range.to(U.GHz)
    
    
    
    
    if redshift is None:
        if isinstance(rest_frequency, np.ndarray):
            rest_frequency = np.sort(np.array(rest_frequency))[0]
        rest_frequency = rest_frequency * U.GHz
        redshift = uas.compute_redshift(rest_frequency, source_freq)
    else:
        rest_frequency = uas.compute_rest_frequency_from_redshift(main_dir, source_freq.value, redshift) * U.GHz
    continum, line_fluxes, line_names, redshift, line_frequency, source_channel_index, n_channels_nw, bandwidth, freq_sup_nw, cont_frequencies, fwhm_z, lum_infrared  = uas.process_spectral_data(
                                                                        source_type,
                                                                        main_dir,
                                                                        redshift, 
                                                                        central_freq.value,
                                                                        band_range.value,
                                                                        source_freq.value,
                                                                        n_channels,
                                                                        lum_infrared,
                                                                        cont_sens.value,
                                                                        line_names,
                                                                        n_lines,
                                                                        )
    #print(continum.shape, line_fluxes, line_names)
    if n_channels_nw != n_channels:
        freq_sup = freq_sup_nw * U.MHz
        n_channels = n_channels_nw
        band_range  = n_channels * freq_sup
    
    central_channel_index = n_channels // 2
    print('Field of view: {} arcsec'.format(round(fov.value, 3)))
    print('Beam size: {} arcsec'.format(round(beam_size.value, 4)))
    print('Cell size: {} arcsec'.format(round(cell_size.value, 4)))
    print('Central Frequency: {}'.format(central_freq))
    print('Spectral Window: {}'.format(band_range))
    print('Freq Support: {}'.format(freq_sup))
    print('Cube Dimensions: {} x {} x {}'.format(n_pix, n_pix, n_channels))
    print('Redshift: {}'.format(redshift))
    print('Source frequency: {} GHz'.format(round(source_freq.value, 2)))
    print('Band: ', band)
    print('Velocity resolution: {} Km/s'.format(round(vel_res.value, 2)))
    print('Angular resolution: {} arcsec'.format(round(ang_res.value, 3)))
    print('Infrared Luminosity: {:.2e}'.format(lum_infrared))
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

    if type(line_names) == list or isinstance(line_names, np.ndarray):
        for line_name, line_flux in zip(line_names, line_fluxes): 
            print('Simulating Line {} Flux: {:.3e} at z {}'.format(line_name, line_flux, redshift))
    else:
        print('Simulating Line {} Flux: {} at z {}'.format(line_names[0], line_fluxes[0], redshift))
    print('Simulating Continum Flux: {:.2e}'.format(np.mean(continum)))
    print('Continuum Sensitity: {:.2e}'.format(cont_sens))
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
    fwhm_x, fwhm_y, angle = None, None, None
    if source_type == 'point':
        pos_x, pos_y, _ = wcs.sub(3).wcs_world2pix(ra, dec, central_freq, 0)
        pos_z = [int(index) for index in source_channel_index]
        datacube = usm.insert_pointlike(datacube, continum, line_fluxes, int(pos_x), int(pos_y), pos_z, fwhm_z, n_channels)
    elif source_type == 'gaussian':
        pos_x, pos_y, _ = wcs.sub(3).wcs_world2pix(ra, dec, central_freq, 0)
        pos_z = [int(index) for index in source_channel_index]
        fwhm_x = np.random.randint(3, 10) 
        fwhm_y = np.random.randint(3, 10)    
        angle = np.random.randint(0, 180)
        datacube = usm.insert_gaussian(datacube, continum, line_fluxes, int(pos_x), int(pos_y), pos_z, fwhm_x, fwhm_y, fwhm_z,
                                         angle, n_pix, n_channels)
    elif source_type == 'extended':
        
        datacube = usm.insert_extended(datacube, tng_dir, snapshot, int(tng_subhaloid), redshift, ra, dec, tng_api_key, ncpu)

    uas.write_sim_parameters(os.path.join(output_dir, 'sim_params_{}.txt'.format(inx)),
                            ra, dec, ang_res, vel_res, int_time, total_time, band, band_range, central_freq,
                            redshift, line_fluxes, line_names, line_frequency, 
                            continum, fov, beam_size, cell_size, n_pix, 
                            n_channels, snapshot, tng_subhaloid, lum_infrared, fwhm_z, source_type, fwhm_x, fwhm_y, angle)

    if inject_serendipitous == True:
        if source_type != 'gaussian':
            fwhm_x = np.random.randint(3, 10)
            fwhm_y = np.random.randint(3, 10)
        datacube = usm.insert_serendipitous(datacube, continum, cont_sens.value, line_fluxes, line_names, line_frequency, 
                                            freq_sup.value, pos_z, fwhm_x, fwhm_y, fwhm_z, n_pix, n_channels, 
                                            os.path.join(output_dir, 'sim_params_{}.txt'.format(inx)))
    
    filename = os.path.join(sim_output_dir, 'skymodel_{}.fits'.format(inx))
    print('Writing datacube to {}'.format(filename))
    usm.write_datacube_to_fits(datacube, filename, obs_date)
    print('Done')
    del datacube
    upl.plot_skymodel(filename, inx, output_dir, line_names, line_frequency, source_channel_index, cont_frequencies, show=False)
    
    project_name = project_name + '_{}'.format(inx)
    os.chdir(output_dir)
    
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
        #graphics="none",
        )
    print('Simulated observation for {}'.format(project_name))    
    
    scale = random.uniform(0, 1)
    ms_path = os.path.join(sim_output_dir, "{}.{}.noisy.ms".format(project_name, antenna_name))
    print('Simulating atmospheric noise ....')
    ual.simulate_atmospheric_noise(sim_output_dir, project_name, scale, ms_path, antennalist)
    print('Simulating gain errors ....')
    gain_error_amp = random.gauss(0, 0.1)
    ual.simulate_gain_errors(ms_path, gain_error_amp)
    print('Done')
    print('Fourier inverting ....')
    tclean(
        vis=ms_path,
        imagename=os.path.join(sim_output_dir, '{}.{}'.format(project_name, antenna_name)),
        imsize=[int(n_pix), int(n_pix)],
        cell="{}".format(cell_size),
        phasecenter=pos_string,
        specmode="cube",
        niter=0,
        fastnoise=False,
        calcpsf=True,
        pbcor=True,
        pblimit=0.2, 
        )
    print('Created dirty cube for {}'.format(project_name))
    print('Exporting...')
    exportfits(imagename=os.path.join(project_name, '{}.{}.image'.format(project_name, antenna_name)), 
       fitsimage=os.path.join(output_dir, "dirty_cube_" + str(inx) +".fits"), overwrite=True)
    exportfits(imagename=os.path.join(project_name, '{}.{}.skymodel'.format(project_name, antenna_name)), 
        fitsimage=os.path.join(output_dir, "clean_cube_" + str(inx) +".fits"), overwrite=True)
    shutil.copy(os.path.join(sim_output_dir, "{}.{}.observe.png".format(project_name, antenna_name)), 
                os.path.join(output_dir, 'plots', 'observation_' + str(inx) + '.png'))
    shutil.copy(os.path.join(sim_output_dir, "{}.{}.skymodel.png".format(project_name, antenna_name)), 
                os.path.join(output_dir, 'plots', 'skymodel_observation_' + str(inx) + '.png'))

    clean, clean_header = uas.load_fits(os.path.join(output_dir, "clean_cube_" + str(inx) +".fits"))
    dirty, dirty_header = uas.load_fits(os.path.join(output_dir, "dirty_cube_" + str(inx) +".fits"))
    sky_total_flux = np.nansum(clean)
    #if np.nanmin(dirty) < 0:
    #    dirty = dirty + np.nanmin(dirty)
    dirty_total_flux = np.nansum(dirty)
    if sky_total_flux != dirty_total_flux:
        print('Normalizing')
        dirty = dirty * (sky_total_flux / dirty_total_flux)
        dirty_total_flux = np.nansum(dirty)
        print('Total Flux detected in dirty cube after normalization: {}'.format(round(dirty_total_flux, 2)))
        uas.write_numpy_to_fits(dirty, dirty_header, os.path.join(output_dir, "dirty_cube_" + str(inx) +".fits"))
    del clean
    del dirty
    upl.plotter(inx, output_dir, beam_size, line_names, line_frequency, source_channel_index, cont_frequencies)
    if save_secondary == True:
        exportfits(imagename=os.path.join(project_name, '{}.{}.psf'.format(project_name, antenna_name)),
              fitsimage=os.path.join(output_dir, "psf_" + str(inx) +".fits"), overwrite=True)
        exportfits(imagename=os.path.join(project_name, '{}.{}.pb'.format(project_name, antenna_name)),
                fitsimage=os.path.join(output_dir, "pb_" + str(inx) +".fits"), overwrite=True)
        ual.ms_to_npz(os.path.join(project_name, "{}.{}.ms".format(project_name, antenna_name)),
                dirty_cube=os.path.join(output_dir, "clean_cube_" + str(inx) +".fits"),
                datacolumn='DATA',
                output_file=os.path.join(output_dir, "ms_" + str(inx) +".npz"))
        ual.ms_to_npz(os.path.join(project_name, "{}.{}.noisy.ms".format(project_name, antenna_name)),
              dirty_cube=os.path.join(output_dir, "dirty_cube_" + str(inx) +".fits"),
              datacolumn='CORRECTED_DATA',
              output_file=os.path.join(output_dir, "ms_dirty_" + str(inx) +".npz"))
    print('Finished')
    stop = time.time()
    print('Execution took {} seconds'.format(strftime("%H:%M:%S", gmtime(stop - start))))
    shutil.rmtree(sim_output_dir)
    
def simulator2(inx, main_dir, output_dir, tng_dir, galaxy_zoo_dir, project_name, ra, dec, band, ang_res, vel_res, fov, obs_date, 
              pwv, int_time, total_time, bandwidth, freq, freq_support, cont_sens, antenna_array, n_pix, 
              n_channels, source_type, tng_api_key, ncpu, rest_frequency, redshift, lum_infrared, snr,
              n_lines, line_names, save_secondary=False, inject_serendipitous=False):
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
    start = time.time()
    second2hour = 1 / 3600
    ra = ra * U.deg
    dec = dec * U.deg
    fov = fov * 3600 * U.arcsec
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
    #obs_date = Time(obs_date + 'T00:00:00', format='isot', scale='utc').to_value("mjd")
    if not os.path.exists(sim_output_dir):
        os.makedirs(sim_output_dir)
    os.chdir(output_dir)
    print('RA: {}'.format(ra))
    print('DEC: {}'.format(dec))
    print('Integration Time: {}'.format(int_time))
    ual.generate_antenna_config_file_from_antenna_array(antenna_array, main_dir, sim_output_dir)
    antennalist = os.path.join(sim_output_dir, "antenna.cfg")
    antenna_name = 'antenna'
    max_baseline = ual.get_max_baseline_from_antenna_config(antennalist) * U.km
    #t_ang_res = c.to(U.m/U.s) / central_freq.to(U.Hz) / (max_baseline.to(U.m))
    #t_ang_res = t_ang_res * (180 / math.pi) * 3600 * U.arcsec
    #print('Angular Resolution computed from max baseline: {}'.format(t_ang_res))
    #pos_string = uas.convert_to_j2000_string(ra.value, dec.value)
    #fov =  ual.get_fov_from_band(int(band), return_value=False)
    print('Field of view: {} arcsec'.format(round(fov.value, 3)) )
    beam_size = ual.estimate_alma_beam_size(central_freq, max_baseline, return_value=False)
    beam_solid_angle = np.pi * (beam_size / 2) ** 2
    cont_sens = cont_sens * U.mJy / (U.arcsec ** 2)
    cont_sens_jy = (cont_sens * beam_solid_angle).to(U.Jy)
    cont_sens  = cont_sens_jy  * snr
    print("Minimum detectable continum: ", cont_sens_jy)
    cell_size = beam_size / 5
    if n_pix is None: 
        #cell_size = beam_size / 5
        n_pix = closest_power_of_2(int(1.5 * fov / cell_size))
    else:
        n_pix = closest_power_of_2(n_pix)
        cell_size = fov / n_pix
        # just added
        #beam_size = cell_size * 5
    if n_channels is None:
        n_channels = int(band_range / freq_sup)
    else:
        band_range = n_channels * freq_sup 
        band_range = band_range.to(U.GHz)
    if redshift is None:
        if isinstance(rest_frequency, np.ndarray):
            rest_frequency = np.sort(np.array(rest_frequency))[0]
        rest_frequency = rest_frequency * U.GHz
        redshift = uas.compute_redshift(rest_frequency, source_freq)
    else:
        rest_frequency = uas.compute_rest_frequency_from_redshift(main_dir, source_freq.value, redshift) * U.GHz
    continum, line_fluxes, line_names, redshift, line_frequency, source_channel_index, n_channels_nw, bandwidth, freq_sup_nw, cont_frequencies, fwhm_z, lum_infrared  = uas.process_spectral_data(
                                                                        source_type,
                                                                        main_dir,
                                                                        redshift, 
                                                                        central_freq.value,
                                                                        band_range.value,
                                                                        source_freq.value,
                                                                        n_channels,
                                                                        lum_infrared,
                                                                        cont_sens.value,
                                                                        line_names,
                                                                        n_lines,
                                                                        )
    #print(continum.shape, line_fluxes, line_names)
    if n_channels_nw != n_channels:
        freq_sup = freq_sup_nw * U.MHz
        n_channels = n_channels_nw
        band_range  = n_channels * freq_sup
    central_channel_index = n_channels // 2
    print('Beam size: {} arcsec'.format(round(beam_size.value, 4)))
    print('Central Frequency: {}'.format(central_freq))
    print('Spectral Window: {}'.format(band_range))
    print('Freq Support: {}'.format(freq_sup))
    print('Cube Dimensions: {} x {} x {}'.format(n_pix, n_pix, n_channels))
    print('Redshift: {}'.format(round(redshift, 3)))
    print('Source frequency: {} GHz'.format(round(source_freq.value, 2)))
    print('Band: ', band)
    print('Velocity resolution: {} Km/s'.format(round(vel_res.value, 2)))
    print('Angular resolution: {} arcsec'.format(round(ang_res.value, 3)))
    print('Infrared Luminosity: {:.2e}'.format(lum_infrared))
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
    if type(line_names) == list or isinstance(line_names, np.ndarray):
        for line_name, line_flux in zip(line_names, line_fluxes): 
            print('Simulating Line {} Flux: {:.3e} at z {}'.format(line_name, line_flux, redshift))
    else:
        print('Simulating Line {} Flux: {} at z {}'.format(line_names[0], line_fluxes[0], redshift))
    print('Simulating Continum Flux: {:.2e}'.format(np.mean(continum)))
    print('Continuum Sensitity: {:.2e}'.format(cont_sens))
    print('Generating skymodel cube ...')
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
    fwhm_x, fwhm_y, angle = None, None, None
    if source_type == 'point':
        pos_x, pos_y, _ = wcs.sub(3).wcs_world2pix(ra, dec, central_freq, 0)
        pos_z = [int(index) for index in source_channel_index]
        datacube = usm.insert_pointlike(datacube, continum, line_fluxes, int(pos_x), int(pos_y), pos_z, fwhm_z, n_channels)
    elif source_type == 'gaussian':
        pos_x, pos_y, _ = wcs.sub(3).wcs_world2pix(ra, dec, central_freq, 0)
        pos_z = [int(index) for index in source_channel_index]
        fwhm_x = np.random.randint(3, 10) 
        fwhm_y = np.random.randint(3, 10)    
        angle = np.random.randint(0, 180)
        datacube = usm.insert_gaussian(datacube, continum, line_fluxes, int(pos_x), int(pos_y), pos_z, fwhm_x, fwhm_y, fwhm_z,
                                         angle, n_pix, n_channels)
    elif source_type == 'extended':
        datacube = usm.insert_extended(datacube, tng_dir, snapshot, int(tng_subhaloid), redshift, ra, dec, tng_api_key, ncpu)
    elif source_type == 'diffuse':
        print('To be implemented')
    elif source_type == 'galaxy-zoo':
        galaxy_path = os.path.join(galaxy_zoo_dir, 'images_gz2',  'images')
        pos_z = [int(index) for index in source_channel_index]
        datacube = usm.insert_galaxy_zoo(datacube, continum, line_fluxes, pos_z, fwhm_z, n_pix, n_channels, galaxy_path)
    
    uas.write_sim_parameters(os.path.join(output_dir, 'sim_params_{}.txt'.format(inx)),
                            ra, dec, ang_res, vel_res, int_time, total_time, band, band_range, central_freq,
                            redshift, line_fluxes, line_names, line_frequency, 
                            continum, fov, beam_size, cell_size, n_pix, 
                            n_channels, snapshot, tng_subhaloid, lum_infrared, fwhm_z, source_type, fwhm_x, fwhm_y, angle)

    if inject_serendipitous == True:
        if source_type != 'gaussian':
            fwhm_x = np.random.randint(3, 10)
            fwhm_y = np.random.randint(3, 10)
        datacube = usm.insert_serendipitous(datacube, continum, cont_sens.value, line_fluxes, line_names, line_frequency, 
                                            freq_sup.value, pos_z, fwhm_x, fwhm_y, fwhm_z, n_pix, n_channels, 
                                            os.path.join(output_dir, 'sim_params_{}.txt'.format(inx)))
    #filename = os.path.join(sim_output_dir, 'skymodel_{}.fits'.format(inx))
    #print('Writing datacube to {}'.format(filename))
    #usm.write_datacube_to_fits(datacube, filename, obs_date)
    model = datacube._array.to_value(datacube._array.unit).T
    totflux = np.sum(model) 
    print(f'Total Flux injected in model cube: {round(totflux, 3)} Jy')
    print('Done')
    del datacube
    print('Observing with ALMA')
    #upl.plot_skymodel(filename, inx, output_dir, line_names, line_frequency, source_channel_index, cont_frequencies, show=False)
    uin.Interferometer(inx, model, main_dir, output_dir, ra, dec, central_freq, band_range, fov, antenna_array, cont_sens.value * 3, 
                        int_time.value * second2hour, obs_date)
    print('Finished')
    stop = time.time()
    print('Execution took {} seconds'.format(strftime("%H:%M:%S", gmtime(stop - start))))
    shutil.rmtree(sim_output_dir)