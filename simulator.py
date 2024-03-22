from datetime import date 
import tempfile
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time
from astropy import wcs
temp_dir = tempfile.TemporaryDirectory()
import math
import os
import random
import time
from math import pi
from random import choices
from time import gmtime, strftime
from typing import Optional
import astropy.constants as C
import astropy.units as U
import h5py
import illustris_python as il
import nifty8 as ift
from astropy.constants import c
import astropy.cosmology.units as cu
from astropy.cosmology import Planck13
from astropy.time import Time
from astropy.wcs import WCS
from casatasks import exportfits, simobserve, tclean, gaincal, applycal
from casatools import table
from casatools import simulator as casa_simulator
from Hdecompose.atomic_frac import atomic_frac
from illustris_python.snapshot import getSnapOffsets, loadSubset
#from martini import DataCube, Martini
from martini.sources.sph_source import SPHSource
from martini.spectral_models import GaussianSpectrum
from martini.sph_kernels import (AdaptiveKernel, CubicSplineKernel,
                                 GaussianKernel, find_fwhm,  WendlandC2Kernel)
from natsort import natsorted
from spectral_cube import SpectralCube
from tqdm import tqdm
import psutil
import subprocess
import bisect
from os.path import isfile, expanduser
import six
from itertools import product
from scipy.signal import fftconvolve
import pyvo
from scipy.optimize import curve_fit

os.environ['MPLCONFIGDIR'] = temp_dir.name
pd.options.mode.chained_assignment = None  

def sample_from_brightneses(input_csv, n):
  """
  This function reads an input csv containing values of measured brightnesses for a given target type, 
  fits this distribution and then samples n brightness values from saild distribution.
  INPUT: 
  - input_csv = path to the csv file containing the brightnesses
  - n = number of brightness values to sample from the distribution
  OUTPUT: 
  - brightneses - list of sampled brightnesses from the distribution 
  """
  return brightnesses 
  
def closest_power_of_2(x):
    op = math.floor if bin(x)[3] != "1" else math.ceil
    return 2 ** op(math.log(x, 2))

def load_fits(inFile):
    hdu_list = fits.open(inFile)
    data = hdu_list[0].data
    header = hdu_list[0].header
    hdu_list.close()
    return data, header

def save_fits(outfile, data, header):
    hdu = fits.PrimaryHDU(data, header=header)
    hdul = fits.HDUList([hdu])
    hdul.writeto(outfile, overwrite=True)

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
                    row = [x for x in line.split("\t")][:2]
                else:
                    row = [x for x in line.split(" ")][:2]
                positions.append([float(x) for x in row])  
    positions = np.array(positions)
    max_baseline = 2 * np.max(np.sqrt(positions[:, 0]**2 + positions[:, 1]**2)) / 1000
    return max_baseline

def compute_beam_size_from_max_baseline(max_baseline, freq):
    """
    max baseline in km, freq in GHz.
    Returns the beam size in arcsec
    """
    return 76 / (max_baseline * freq)

def get_data_from_hdf(file):
    data = list()
    column_names = list()
    r = h5py.File(file, 'r')
    for key in r.keys():
        if key == 'Snapshot_99':
            group = r[key]
            for key2 in group.keys():
                column_names.append(key2)
                data.append(group[key2])
    values = np.array(data)
    r.close()
    db = pd.DataFrame(values.T, columns=column_names)     
    return db   

def get_subhaloids_from_db(n, 
    #                       limit0, 
    #                       limit1
                           ):
    file = os.path.join('metadata', 'morphologies_deeplearn.hdf5')
    db = get_data_from_hdf(file)
    catalogue = db[['SubhaloID', 'P_Late', 'P_S0', 'P_Sab']]
    catalogue.sort_values(by=['P_Late'], inplace=True, ascending=False)
    catalogue.head(10)
    ellipticals = catalogue[(catalogue['P_Late'] > 0.6) & (catalogue['P_S0'] < 0.5) & (catalogue['P_Sab'] < 0.5)]
    lenticulars = catalogue[(catalogue['P_S0'] > 0.6) & (catalogue['P_Late'] < 0.5) & (catalogue['P_Sab'] < 0.5)]
    spirals = catalogue[(catalogue['P_Sab'] > 0.6) & (catalogue['P_Late'] < 0.5) & (catalogue['P_S0'] < 0.5)]

    ellipticals['sum'] = ellipticals['P_S0'].values + ellipticals['P_Sab'].values
    lenticulars['sum'] = lenticulars['P_Late'].values + lenticulars['P_Sab'].values

    spirals['sum'] = spirals['P_Late'].values + spirals['P_S0'].values
    ellipticals.sort_values(by=['sum'], inplace=True, ascending=True)
    lenticulars.sort_values(by=['sum'], inplace=True, ascending=True)
    spirals.sort_values(by=['sum'], inplace=True, ascending=True)
    ellipticals_ids = ellipticals['SubhaloID'].values
    lenticulars_ids = lenticulars['SubhaloID'].values
    spirals_ids = spirals['SubhaloID'].values
    sample_n = n // 3
    
    #n_0 = choices(ellipticals_ids[(elliptical_ids > limit0) & ( ellipticals_ids < limit1)], k=sample_n)
    #n_1 = choices(spirals_ids[(spirals_ids > limit0) & (spirals_ids < limit1)], k=sample_n)
    #n_2 = choices(lenticulars_ids[(lenticulars_ids > limit0) & (lenticular_ids < limit1)], k=n - 2 * sample_n)
    n_0 = choices(ellipticals_ids, k=sample_n)
    n_1 = choices(spirals_ids, k=sample_n)
    n_2 = choices(lenticulars_ids, k=n - 2 * sample_n)
    ids = np.concatenate((n_0, n_1, n_2)).astype(int)
    return ids

def get_band_central_freq(band):
    if band == 3:
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

def my_asserteq(*args):
    for aa in args[1:]:
        if args[0] != aa:
            raise RuntimeError(f"{args[0]} != {aa}")

def _ms2resolve_transpose(arr):
    my_asserteq(arr.ndim, 3)
    return np.ascontiguousarray(np.transpose(arr, (0, 2,1)))

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

def get_fov_from_band(band):
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
    fov = 1.22 * wavelength / 12
    # fov in arcsec
    fov = fov * 206264.806
    return fov

def get_fov(bands):
    fovs = []
    for band in bands:
        fovs.append(get_fov_from_band(band))
    return np.array(fovs)

def get_spatial_resolution(band, config_number):
    spatial_resolution_dict = {
    1 : [3.38, 2.25, 1.83, 1.47, 0.98, 0.74, 0.52, 0.39],
    2 : [2.30, 1.53, 1.24, 1.00, 0.67, 0.50, 0.35, 0.26],
    3 : [1.42, 0.94, 0.77, 0.62, 0.41, 0.31, 0.22, 0.16],
    4 : [0.92, 0.61, 0.50, 0.40, 0.27, 0.20, 0.14, 0.11],
    5 : [0.55, 0.36, 0.30, 0.24, 0.16, 0.12, 0.084, 0.063],
    6 : [0.31, 0.20, 0.17, 0.13, 0.089, 0.067, 0.047, 0.035],
    7 : [0.21, 0.14, 0.11, 0.092, 0.061, 0.046, 0.033, 0.024],
    8 : [0.096, 0.064, 0.052, 0.042, 0.028, 0.021, 0.015, 0.011],
    9 : [0.057, 0.038, 0.031, 0.025, 0.017, 0.012, 0.088],
    10 : [0.042, 0.028, 0.023, 0.018, 0.012, 0.0091] 
    }
    if config_number == 9:
        assert band <= 9, 'band should be less than 9 for antenna configuration 9'
    elif config_number == 10:
        assert band <= 8, 'band should be less than 8 for antenna configuration 10'
    return spatial_resolution_dict[config_number][band - 3]

def get_antenna_config_from_date(obs_date):
    date_to_cycle = [
        (date(2023, 10, 1), date(2024, 9, 30), 10),
        (date(2022, 9, 30), date(2023, 10, 1), 9),
        (date(2021, 10, 1), date(2022, 10, 1), 8),
        (date(2019, 9, 30), date(2021, 10, 1), 7),
        (date(2018, 10, 1), date(2019, 9, 30), 6),
        (date(2017, 10, 1), date(2018, 10, 1), 5),
        (date(2016, 9, 30), date(2017, 10, 1), 4),
        (date(2015, 10, 1), date(2016, 10, 1), 3),
     ]
    date_to_conf = [
        (date(2015, 10, 13), date(2015, 12, 7), 8),
        (date(2015, 12, 7), date(2016, 3, 7), 1),
        (date(2016, 3, 7), date(2016, 5, 2), 2),
        (date(2016, 5, 2), date(2016, 5, 30), 3),
        (date(2016, 5, 30), date(2016, 7, 18), 4),
        (date(2016, 7, 18), date(2016, 8, 22), 5),
        (date(2016, 8, 22), date(2016, 9, 30), 6),
        (date(2016, 9, 30), date(2016, 11, 14), 4),
        (date(2016, 11, 14), date(2016, 12, 5), 3),
        (date(2016, 12, 5), date(2017, 3, 5), 2),
        (date(2017, 3, 5), date(2017, 4, 9), 1),
        (date(2017, 4, 9), date(2017, 4, 30), 3),
        (date(2017, 4, 30), date(2017, 7, 31), 5),
        (date(2017, 7, 31), date(2017, 9, 4), 7),
        (date(2017, 9, 4), date(2017, 9, 11), 8),
        (date(2017, 9, 11), date(2017, 10, 1), 9),
        (date(2017, 10, 1), date(2017, 10, 23), 10),
        (date(2017, 10, 23), date(2017, 11, 6), 9),
        (date(2017, 11, 6), date(2017, 11, 27), 8),
        (date(2017, 11, 27), date(2017, 12, 11), 7),
        (date(2017, 12, 11), date(2018, 1, 8), 6),
        (date(2018, 1, 8), date(2018, 2, 1), 5),
        (date(2018, 2, 1), date(2018, 4, 2), 4),
        (date(2018, 4, 2), date(2018, 5, 7), 3),
        (date(2018, 5, 7), date(2018, 6, 4), 2),
        (date(2018, 6, 4), date(2018, 7, 23), 1),
        (date(2018, 7, 23), date(2018, 8, 20), 2),
        (date(2018, 8, 20), date(2018, 9, 3), 3),
        (date(2018, 9, 3), date(2018, 9, 17), 4),
        (date(2018, 9, 17), date(2018, 10, 1), 5),
        (date(2018, 10, 1), date(2018, 10, 15), 6),
        (date(2018, 10, 15), date(2018, 11, 26), 5),
        (date(2018, 11, 26), date(2018, 12, 17), 4),
        (date(2018, 12, 17), date(2019, 1, 7), 3),
        (date(2019, 1, 7), date(2019, 1, 21), 2),
        (date(2019, 1, 21), date(2019, 3, 18), 1),
        (date(2019, 3, 18), date(2019, 4, 1), 2),
        (date(2019, 4, 1), date(2019, 4, 15), 3),
        (date(2019, 4, 15), date(2019, 5, 6), 4),
        (date(2019, 6, 3), date(2019, 7, 8), 9),
        (date(2019, 7, 8), date(2019, 7, 29), 8),
        (date(2019, 7, 29), date(2019, 9, 16), 7),
        (date(2019, 9, 16), date(2019, 9, 30), 6),
        (date(2019, 9, 30), date(2019, 10, 21), 4),
        (date(2019, 10, 21), date(2019, 11, 18), 3),
        (date(2019, 11, 18), date(2019, 12, 2), 2),
        (date(2019, 12, 2), date(2019, 12, 23), 1),
        (date(2019, 12, 23), date(2020, 1, 13), 2),
        (date(2020, 1, 13), date(2020, 3, 3), 3),
        (date(2020, 3, 3), date(2020, 3, 19), 4),
        (date(2020, 3, 19), date(2021, 5, 10), 5),
        (date(2021, 5, 10), date(2021, 7, 21), 6),
        (date(2021, 7, 21), date(2021, 8, 2), 7),
        (date(2021, 8, 2), date(2021, 8, 23), 8),
        (date(2021, 8, 23), date(2021, 10, 1), 9),
        (date(2021, 10, 1), date(2021, 11, 1), 8),
        (date(2021, 11, 1), date(2021, 11, 29), 7),
        (date(2021, 11, 29), date(2021, 12, 13), 6),
        (date(2021, 12, 13), date(2021, 12, 20), 5),
        (date(2021, 12, 20), date(2022, 1, 17), 4),
        (date(2022, 1, 17), date(2022, 2, 1), 3),
        (date(2022, 3, 1), date(2022, 3, 21), 1),
        (date(2022, 3, 21), date(2022, 4, 18), 2),
        (date(2022, 4, 18), date(2022, 5, 23), 3),
        (date(2022, 5, 23), date(2022, 6, 20), 4),
        (date(2022, 6, 20), date(2022, 7, 11), 5),
        (date(2022, 7, 11), date(2022, 8, 1), 6),
        (date(2022, 8, 1), date(2022, 8, 22), 5),
        (date(2022, 8, 22), date(2022, 9, 12), 4),
        (date(2022, 9,  30), date(2023, 1, 9), 3),
        (date(2023, 1, 9), date(2023, 3, 20), 4),
        (date(2023, 3, 20), date(2023, 4, 17), 5),
        (date(2023, 4, 17), date(2023, 5, 22), 6),
        (date(2023, 5, 22), date(2023, 6, 19), 7),
        (date(2023, 6, 19), date(2023, 7, 10), 8),
        (date(2023, 7, 10), date(2023, 7, 31), 9),
        (date(2023, 7, 31), date(2023, 9, 30), 10),
        (date(2023, 8, 21), date(2023, 9, 11), 8),
        (date(2023, 9, 11), date(2023, 10, 1), 9),
        (date(2023, 10, 1), date(2023, 10, 20), 8),
        (date(2023, 10, 20), date(2023, 11, 10), 7),
        (date(2023, 11, 10), date(2023, 12, 1), 6),
        (date(2023, 12, 1), date(2023, 12, 20), 5),
        (date(2023, 12, 20), date(2024, 1, 10), 4),
        (date(2024, 1, 10), date(2024, 2, 1), 3),
        (date(2024, 3, 1), date(2024, 3, 26), 1),
        (date(2024, 3, 26), date(2024, 4, 20), 2),
        (date(2024, 4, 20), date(2024, 5, 10), 3),
        (date(2024, 5, 10), date(2024, 5, 31), 4),
        (date(2024, 5, 31), date(2024, 6, 23), 5),
        (date(2024, 6, 23), date(2024, 7, 28), 6),
        (date(2024, 7, 28), date(2024, 8, 18), 5),
        (date(2024, 8, 18), date(2024, 9, 10), 4),
        (date(2024, 9, 10), date(2024, 9, 30), 3),
    ]
    obs_date = obs_date.split('-')
    obs_date = date(int(obs_date[0]), int(obs_date[1]), int(obs_date[2][:2]))
    print('Observation Date:', obs_date)
    for start, end, cycle in date_to_cycle:
        if start <= obs_date < end:
            break
    for start, end, antenna_config in date_to_conf:
        if start <= obs_date < end:
            break
    print('Computed Cycle {} Antenna Config {}'.format(cycle, antenna_config))
    return antenna_config, cycle

def get_info_from_reference(reference_path, plot_dir, i):
    date_to_cycle = [
        (date(2022, 9, 30), date(2023, 10, 1), 9),
        (date(2021, 10, 1), date(2022, 10, 1), 8),
        (date(2019, 9, 30), date(2021, 10, 1), 7),
        (date(2018, 10, 1), date(2019, 9, 30), 6),
        (date(2017, 10, 1), date(2018, 10, 1), 5),
        (date(2016, 9, 30), date(2017, 10, 1), 4),
        (date(2015, 10, 1), date(2016, 10, 1), 3),
     ]
    date_to_conf = [
        (date(2015, 10, 13), date(2015, 12, 7), 8),
        (date(2015, 12, 7), date(2016, 3, 7), 1),
        (date(2016, 3, 7), date(2016, 5, 2), 2),
        (date(2016, 5, 2), date(2016, 5, 30), 3),
        (date(2016, 5, 30), date(2016, 7, 18), 4),
        (date(2016, 7, 18), date(2016, 8, 22), 5),
        (date(2016, 8, 22), date(2016, 9, 30), 6),
        (date(2016, 9, 30), date(2016, 11, 14), 4),
        (date(2016, 11, 14), date(2016, 12, 5), 3),
        (date(2016, 12, 5), date(2017, 3, 5), 2),
        (date(2017, 3, 5), date(2017, 4, 9), 1),
        (date(2017, 4, 9), date(2017, 4, 30), 3),
        (date(2017, 4, 30), date(2017, 7, 31), 5),
        (date(2017, 7, 31), date(2017, 9, 4), 7),
        (date(2017, 9, 4), date(2017, 9, 11), 8),
        (date(2017, 9, 11), date(2017, 10, 1), 9),
        (date(2017, 10, 1), date(2017, 10, 23), 10),
        (date(2017, 10, 23), date(2017, 11, 6), 9),
        (date(2017, 11, 6), date(2017, 11, 27), 8),
        (date(2017, 11, 27), date(2017, 12, 11), 7),
        (date(2017, 12, 11), date(2018, 1, 8), 6),
        (date(2018, 1, 8), date(2018, 2, 1), 5),
        (date(2018, 2, 1), date(2018, 4, 2), 4),
        (date(2018, 4, 2), date(2018, 5, 7), 3),
        (date(2018, 5, 7), date(2018, 6, 4), 2),
        (date(2018, 6, 4), date(2018, 7, 23), 1),
        (date(2018, 7, 23), date(2018, 8, 20), 2),
        (date(2018, 8, 20), date(2018, 9, 3), 3),
        (date(2018, 9, 3), date(2018, 9, 17), 4),
        (date(2018, 9, 17), date(2018, 10, 1), 5),
        (date(2018, 10, 1), date(2018, 10, 15), 6),
        (date(2018, 10, 15), date(2018, 11, 26), 5),
        (date(2018, 11, 26), date(2018, 12, 17), 4),
        (date(2018, 12, 17), date(2019, 1, 7), 3),
        (date(2019, 1, 7), date(2019, 1, 21), 2),
        (date(2019, 1, 21), date(2019, 3, 18), 1),
        (date(2019, 3, 18), date(2019, 4, 1), 2),
        (date(2019, 4, 1), date(2019, 4, 15), 3),
        (date(2019, 4, 15), date(2019, 5, 6), 4),
        (date(2019, 6, 3), date(2019, 7, 8), 9),
        (date(2019, 7, 8), date(2019, 7, 29), 8),
        (date(2019, 7, 29), date(2019, 9, 16), 7),
        (date(2019, 9, 16), date(2019, 9, 30), 6),
        (date(2019, 9, 30), date(2019, 10, 21), 4),
        (date(2019, 10, 21), date(2019, 11, 18), 3),
        (date(2019, 11, 18), date(2019, 12, 2), 2),
        (date(2019, 12, 2), date(2019, 12, 23), 1),
        (date(2019, 12, 23), date(2020, 1, 13), 2),
        (date(2020, 1, 13), date(2020, 3, 3), 3),
        (date(2020, 3, 3), date(2020, 3, 19), 4),
        (date(2020, 3, 19), date(2021, 5, 10), 5),
        (date(2021, 5, 10), date(2021, 7, 21), 6),
        (date(2021, 7, 21), date(2021, 8, 2), 7),
        (date(2021, 8, 2), date(2021, 8, 23), 8),
        (date(2021, 8, 23), date(2021, 10, 1), 9),
        (date(2021, 10, 1), date(2021, 11, 1), 8),
        (date(2021, 11, 1), date(2021, 11, 29), 7),
        (date(2021, 11, 29), date(2021, 12, 13), 6),
        (date(2021, 12, 13), date(2021, 12, 20), 5),
        (date(2021, 12, 20), date(2022, 1, 17), 4),
        (date(2022, 1, 17), date(2022, 2, 1), 3),
        (date(2022, 3, 1), date(2022, 3, 21), 1),
        (date(2022, 3, 21), date(2022, 4, 18), 2),
        (date(2022, 4, 18), date(2022, 5, 23), 3),
        (date(2022, 5, 23), date(2022, 6, 20), 4),
        (date(2022, 6, 20), date(2022, 7, 11), 5),
        (date(2022, 7, 11), date(2022, 8, 1), 6),
        (date(2022, 8, 1), date(2022, 8, 22), 5),
        (date(2022, 8, 22), date(2022, 9, 12), 4),
        (date(2022, 9,  30), date(2023, 1, 9), 3),
        (date(2023, 1, 9), date(2023, 3, 20), 4),
        (date(2023, 3, 20), date(2023, 4, 17), 5),
        (date(2023, 4, 17), date(2023, 5, 22), 6),
        (date(2023, 5, 22), date(2023, 6, 19), 7),
        (date(2023, 6, 19), date(2023, 7, 10), 8),
        (date(2023, 7, 10), date(2023, 7, 31), 9),
        (date(2023, 7, 31), date(2023, 9, 30), 10),
        (date(2023, 8, 21), date(2023, 9, 11), 8),
        (date(2023, 9, 11), date(2023, 10, 1), 9),
    ]
    img, header = load_fits(reference_path)

    shape = [s for s in img.shape if s != 1]
    unique, counts = np.unique(shape, return_counts=True)
    if len(unique) > 1:
        unique = unique[np.argsort(counts)]
        n_channels, n_pix = unique[0], unique[1]
    else:
        n_channels, n_pix = 1, unique[0]
    ra = header['CRVAL1']
    dec = header['CRVAL2']
    inbright = np.max(img)
    obs_date = header["DATE-OBS"].split('-')
    obs_date = date(int(obs_date[0]), int(obs_date[1]), int(obs_date[2][:2]))
    restfreq = header['RESTFRQ']
    for start, end, cycle in date_to_cycle:
        if start <= obs_date < end:
            break
    for start, end, antenna_config in date_to_conf:
        if start <= obs_date < end:
            break
    plot_reference(img, i, plot_dir)
    return [ra, dec, n_pix, n_channels, inbright, restfreq, cycle, antenna_config]

def get_pos(x_radius, y_radius, z_radius):
    x = np.random.randint(-x_radius , x_radius)
    y = np.random.randint(-y_radius, y_radius)
    z = np.random.randint(-z_radius, z_radius)
    return (x, y, z)

def distance_2d(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def distance_1d(p1, p2):
    return math.sqrt((p1-p2)**2)

def distance_3d(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)

def get_iou_1d(bb1, bb2):
    assert(bb1['z1'] < bb1['z2'])
    assert(bb2['z1'] < bb2['z2'])
    z_left = max(bb1['z1'], bb2['z1'])
    z_right = min(bb1['z2'], bb2['z2'])
    if z_right < z_left:
        return 0.0
    intersection = z_right - z_left
    bb1_area = bb1['z2'] - bb1['z1']
    bb2_area = bb2['z2'] - bb2['z1']
    union = bb1_area + bb2_area - intersection
    return intersection / union

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def sample_positions(pos_x, pos_y, pos_z, fwhm_x, fwhm_y, fwhm_z,  
                     n_components, fwhm_xs, fwhm_ys, fwhm_zs,
                     xy_radius, z_radius, sep_xy, sep_z):
    sample = []
    i = 0
    n = 0
    while (len(sample) < n_components) and (n < 1000):
        new_p = get_pos(xy_radius, xy_radius, z_radius)
        new_p = int(new_p[0] + pos_x), int(new_p[1] + pos_y), int(new_p[2] + pos_z)
        if len(sample) == 0:
            spatial_dist = distance_2d((new_p[0],new_p[1]), (pos_x, pos_y))
            freq_dist = distance_1d(new_p[2], pos_z)
            if  spatial_dist < sep_xy or freq_dist < sep_z:
                n += 1
                continue
            else:
                spatial_iou = get_iou(
                        {'x1': new_p[0] - fwhm_xs[i], 
                         'x2': new_p[0] + fwhm_xs[i], 
                         'y1': new_p[1] - fwhm_ys[i], 
                         'y2': new_p[1] + fwhm_ys[i]},
                        {'x1': pos_x - fwhm_x, 
                         'x2': pos_x + fwhm_x, 
                         'y1': pos_y - fwhm_y, 
                         'y2': pos_y + fwhm_y})
                freq_iou = get_iou_1d(
                        {'z1': new_p[2] - fwhm_zs[i], 'z2': new_p[2] + fwhm_zs[i]}, 
                        {'z1': pos_z - fwhm_z, 'z2': pos_z + fwhm_z})
                if spatial_iou > 0.1 or freq_iou > 0.1:
                    n += 1
                    continue
                else:
                    sample.append(new_p)
                    i += 1
                    n = 0
                    print('Found {}st component'.format(len(sample)))
        else:
            spatial_distances = [distance_2d((new_p[0], new_p[1]), (p[0], p[1])) for p in sample]
            freq_distances = [distance_1d(new_p[2], p[2]) for p in sample]
            checks = [spatial_dist < sep_xy or freq_dist < sep_z for spatial_dist, freq_dist in zip(spatial_distances, freq_distances)]
            if any(checks) is True:
                n += 1
                continue
            else:
                spatial_iou = [get_iou(
                        {'x1': new_p[0] - fwhm_xs[i], 
                         'x2': new_p[0] + fwhm_xs[i], 
                         'y1': new_p[1] - fwhm_ys[i], 
                         'y2': new_p[1] + fwhm_ys[i]},
                        {'x1': p[0] - fwhm_xs[j], 
                         'x2': p[0] + fwhm_xs[j], 
                         'y1': p[1] - fwhm_ys[j], 
                         'y2': p[1] + fwhm_ys[j]}) for j, p in enumerate(sample)]
                freq_iou = [get_iou_1d(
                        {'z1': new_p[2] - fwhm_zs[i], 'z2': new_p[2] + fwhm_zs[i]}, 
                        {'z1': p[2] - fwhm_zs[j], 'z2': p[2] + fwhm_zs[j]}) for j, p in enumerate(sample)]
                checks = [spatial_iou > 0.1 or freq_iou > 0.1 for spatial_iou, freq_iou in zip(spatial_iou, freq_iou)]
                if any(checks) is True:
                    n += 1
                    continue
                else:
                    i += 1
                    n = 0
                    sample.append(new_p)
                    print('Found {}st component'.format(len(sample)))
          
    return sample

def threedgaussian(amplitude, spind, chan, center_x, center_y, width_x, width_y, angle, idxs):
    angle = pi/180. * angle
    rcen_x = center_x * np.cos(angle) - center_y * np.sin(angle)
    rcen_y = center_x * np.sin(angle) + center_y * np.cos(angle)
    xp = idxs[0] * np.cos(angle) - idxs[1] * np.sin(angle)
    yp = idxs[0] * np.sin(angle) + idxs[1] * np.cos(angle)
    v1 = 230e9 - (64 * 10e6)
    v2 = v1+10e6*chan
    g = (10**(np.log10(amplitude) + (spind) * np.log10(v1/v2))) * \
        np.exp(-(((rcen_x-xp)/width_x)**2+((rcen_y-yp)/width_y)**2)/2.)
    return g

def gaussian(x, amp, cen, fwhm):
    """
    Generates a 1D Gaussian given the following input parameters:
    x: position
    amp: amplitude
    fwhm: fwhm
    level: level
    """
    return amp*np.exp(-(x-cen)**2/(2*(fwhm/2.35482)**2))

def diffuse_signal(n_px):
    ift.random.push_sseq(random.randint(1, 1000))
    space = ift.RGSpace((2*n_px, 2*n_px))
    args = {
        'offset_mean': 24,
        'offset_std': (1, 0.1),
        'fluctuations': (5., 1.),
        'loglogavgslope': (-3.5, 0.5),
        'flexibility': (1.2, 0.4),
        'asperity': (0.2, 0.2)
    }

    cf = ift.SimpleCorrelatedField(space, **args)
    exp_cf = ift.exp(cf)
    random_pos = ift.from_random(exp_cf.domain)
    sample = np.log(exp_cf(random_pos))
    data = sample.val[0:n_px, 0:n_px]
    normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    return normalized_data

def insert_pointlike(datacube, amplitude, pos_x, pos_y, pos_z, fwhm_z, n_px, n_chan):
    z_idxs = np.arange(0, n_chan)
    g = gaussian(z_idxs, 1, pos_z, fwhm_z)
    ts = np.zeros((n_px, n_px))
    ts[int(pos_x), int(pos_y)] = amplitude
    for z in range(datacube._array.shape[2]):
        slice_ = ts + g[z] * ts
        datacube._array[:, :, z] += slice_ * U.Jy * U.pix**-2
    return datacube

def insert_gaussian(datacube, amplitude, pos_x, pos_y, pos_z, fwhm_x, fwhm_y, fwhm_z, pa, n_px, n_chan):
    z_idxs = np.arange(0, n_chan)
    idxs = np.indices([n_px, n_px])
    g = gaussian(z_idxs, 1, pos_z, fwhm_z)
    for z in range(datacube._array.shape[2]):
        ts = threedgaussian(amplitude, 0, z, pos_x, pos_y, fwhm_x, fwhm_y, pa, idxs)
        slice_ = ts + g[z] * ts
        datacube._array[:, :, z] += slice_ * U.Jy * U.pix**-2
    return datacube

def insert_diffuse(datacube, pos_z, fwhm_z, fov, n_chan, n_px_x, n_px_y):
    z_idxs = np.arange(0, n_chan)
    idxs = np.indices([n_px_x, n_px_y])
    g = gaussian(z_idxs, 1, pos_z, fwhm_z)
    ts = diffuse_signal( n_px_x)
    ts = ts/np.nanmax(ts)
    ts = np.nan_to_num(ts)
    for z in range(datacube._array.shape[2]):
        slice_ = ts + g[z] * ts
        datacube._array[:, :, z] += slice_ * U.Jy * U.pix**-2
    return datacube

def write_datacube_to_fits(
    datacube,
    filename,
    channels="frequency",
    overwrite=True,
    ):
        """
        Output the DataCube to a FITS-format file.

        Parameters
        ----------
        filename : string
            Name of the file to write. '.fits' will be appended if not already
            present.

        channels : {'frequency', 'velocity'}, optional
            Type of units used along the spectral axis in output file.
            (Default: 'frequency'.)

        overwrite: bool, optional
            Whether to allow overwriting existing files. (Default: True.)
        """

        datacube.drop_pad()
        if channels == "frequency":
            datacube.freq_channels()
        elif channels == "velocity":
            datacube.velocity_channels()
        else:
            raise ValueError(
                "Unknown 'channels' value "
                "(use 'frequency' or 'velocity'."
            )

        filename = filename if filename[-5:] == ".fits" else filename + ".fits"

        wcs_header = datacube.wcs.to_header()
        wcs_header.rename_keyword("WCSAXES", "NAXIS")
        header = fits.Header()
        if len(datacube._array.shape) == 3: 
            header.append(("SIMPLE", "T"))
            header.append(("BITPIX", 16))
            header.append(("NAXIS", wcs_header["NAXIS"]))
            header.append(("NAXIS1", datacube.n_px_x))
            header.append(("NAXIS2", datacube.n_px_y))
            header.append(("NAXIS3", datacube.n_channels))
            header.append(("EXTEND", "T"))
            header.append(("CDELT1", wcs_header["CDELT1"]))
            header.append(("CRPIX1", wcs_header["CRPIX1"]))
            header.append(("CRVAL1", wcs_header["CRVAL1"]))
            header.append(("CTYPE1", wcs_header["CTYPE1"]))
            header.append(("CUNIT1", wcs_header["CUNIT1"]))
            header.append(("CDELT2", wcs_header["CDELT2"]))
            header.append(("CRPIX2", wcs_header["CRPIX2"]))
            header.append(("CRVAL2", wcs_header["CRVAL2"]))
            header.append(("CTYPE2", wcs_header["CTYPE2"]))
            header.append(("CUNIT2", wcs_header["CUNIT2"]))
            header.append(("CDELT3", wcs_header["CDELT3"]))
            header.append(("CRPIX3", wcs_header["CRPIX3"]))
            header.append(("CRVAL3", wcs_header["CRVAL3"]))
            header.append(("CTYPE3", wcs_header["CTYPE3"]))
            header.append(("CUNIT3", wcs_header["CUNIT3"]))
        else:
            header.append(("SIMPLE", "T"))
            header.append(("BITPIX", 16))
            header.append(("NAXIS", wcs_header["NAXIS"]))
            header.append(("NAXIS1", datacube.n_px_x))
            header.append(("NAXIS2", datacube.n_px_y))
            header.append(("NAXIS3", datacube.n_channels))
            header.append(("NAXIS4", 1))
            header.append(("EXTEND", "T"))
            header.append(("CDELT1", wcs_header["CDELT1"]))
            header.append(("CRPIX1", wcs_header["CRPIX1"]))
            header.append(("CRVAL1", wcs_header["CRVAL1"]))
            header.append(("CTYPE1", wcs_header["CTYPE1"]))
            header.append(("CUNIT1", wcs_header["CUNIT1"]))
            header.append(("CDELT2", wcs_header["CDELT2"]))
            header.append(("CRPIX2", wcs_header["CRPIX2"]))
            header.append(("CRVAL2", wcs_header["CRVAL2"]))
            header.append(("CTYPE2", wcs_header["CTYPE2"]))
            header.append(("CUNIT2", wcs_header["CUNIT2"]))
            header.append(("CDELT3", wcs_header["CDELT3"]))
            header.append(("CRPIX3", wcs_header["CRPIX3"]))
            header.append(("CRVAL3", wcs_header["CRVAL3"]))
            header.append(("CTYPE3", wcs_header["CTYPE3"]))
            header.append(("CUNIT3", wcs_header["CUNIT3"]))
            header.append(("CDELT4", wcs_header["CDELT4"]))
            header.append(("CRPIX4", wcs_header["CRPIX4"]))
            header.append(("CRVAL4", wcs_header["CRVAL4"]))
            header.append(("CTYPE4", wcs_header["CTYPE4"]))
            header.append(("CUNIT4", "PAR"))
        header.append(("EPOCH", 2000))
        # header.append(('BLANK', -32768)) #only for integer data
        header.append(("BSCALE", 1.0))
        header.append(("BZERO", 0.0))
        datacube_array_units = datacube._array.unit
        header.append(
            ("DATAMAX", np.max(datacube._array.to_value(datacube_array_units)))
        )
        header.append(
            ("DATAMIN", np.min(datacube._array.to_value(datacube_array_units)))
        )
        
        # long names break fits format, don't let the user set this
        header.append(("OBJECT", "MOCK"))
        header.append(("BUNIT", datacube_array_units.to_string("fits")))
        header.append(("MJD-OBS", Time.now().to_value("mjd")))
        header.append(("BTYPE", "Intensity"))
        header.append(("SPECSYS", wcs_header["SPECSYS"]))

        # flip axes to write
        hdu = fits.PrimaryHDU(
            header=header, data=datacube._array.to_value(datacube_array_units).T
        )
        hdu.writeto(filename, overwrite=overwrite)

        if channels == "frequency":
            datacube.velocity_channels()
        return

def write_diffuse_datacube_to_fits(
    datacube,
    filename, 
    channels='frequency',
    overwrite=True
    ):
    """
        Output the DataCube to a FITS-format file.

        Parameters
        ----------
        filename : string
            Name of the file to write. '.fits' will be appended if not already
            present.

        channels : {'frequency', 'velocity'}, optional
            Type of units used along the spectral axis in output file.
            (Default: 'frequency'.)

        overwrite: bool, optional
            Whether to allow overwriting existing files. (Default: True.)
    """
    datacube.drop_pad()
    if channels == "frequency":
        datacube.freq_channels()
    elif channels == "velocity":
        datacube.velocity_channels()
    else:
        raise ValueError(
            "Unknown 'channels' value "
            "(use 'frequency' or 'velocity'."
        )
    
    filename = filename if filename[-5:] == ".fits" else filename + ".fits"
    wcs_header = datacube.wcs.to_header()
    wcs_header.rename_keyword("WCSAXES", "NAXIS")
    header = fits.Header()
    if len(datacube._array.shape) == 3: 
        header.append(("SIMPLE", "T"))
        header.append(("BITPIX", 16))
        header.append(("NAXIS", wcs_header["NAXIS"]))
        header.append(("NAXIS1", datacube.n_px_x))
        header.append(("NAXIS2", datacube.n_px_y))
        header.append(("NAXIS3", datacube.n_channels))
        header.append(("EXTEND", "T"))
        header.append(("CDELT1", wcs_header["CDELT1"]))
        header.append(("CRPIX1", wcs_header["CRPIX1"]))
        header.append(("CRVAL1", wcs_header["CRVAL1"]))
        header.append(("CTYPE1", wcs_header["CTYPE1"]))
        header.append(("CUNIT1", wcs_header["CUNIT1"]))
        header.append(("CDELT2", wcs_header["CDELT2"]))
        header.append(("CRPIX2", wcs_header["CRPIX2"]))
        header.append(("CRVAL2", wcs_header["CRVAL2"]))
        header.append(("CTYPE2", wcs_header["CTYPE2"]))
        header.append(("CUNIT2", wcs_header["CUNIT2"]))
        header.append(("CDELT3", 6.105420846558E+04))
        header.append(("CRPIX3", 1))
        header.append(("CRVAL3", 1.098740492600E+11))
        header.append(("CTYPE3", 'FREQ'))
        header.append(("CUNIT3", 'Hz'))
        header.append(("EPOCH", 2000))
        header.append(("VELREF", 257))
        header.append(("OBJECT", "MOCK"))
        header.append(("MJD-OBS", Time.now().to_value("mjd")))
        header.append(("BTYPE", "Intensity"))
        header.append(("SPECSYS",'LSRK'))
        header.append(("RESTFREQ", 1.099060000000E+11 ))
    else:
        header.append(("SIMPLE", "T"))
        header.append(("BITPIX", 16))
        header.append(("NAXIS", wcs_header["NAXIS"]))
        header.append(("NAXIS1", datacube.n_px_x))
        header.append(("NAXIS2", datacube.n_px_y))
        header.append(("NAXIS3", datacube.n_channels))
        header.append(("NAXIS4", 1))
        header.append(("EXTEND", "T"))
        header.append(("CDELT1", wcs_header["CDELT1"]))
        header.append(("CRPIX1", wcs_header["CRPIX1"]))
        header.append(("CRVAL1", wcs_header["CRVAL1"]))
        header.append(("CTYPE1", wcs_header["CTYPE1"]))
        header.append(("CUNIT1", wcs_header["CUNIT1"]))
        header.append(("CDELT2", wcs_header["CDELT2"]))
        header.append(("CRPIX2", wcs_header["CRPIX2"]))
        header.append(("CRVAL2", wcs_header["CRVAL2"]))
        header.append(("CTYPE2", wcs_header["CTYPE2"]))
        header.append(("CUNIT2", wcs_header["CUNIT2"]))
        header.append(("CDELT3", 6.105420846558E+04))
        header.append(("CRPIX3", 1))
        header.append(("CRVAL3", 1.098740492600E+11))
        header.append(("CTYPE3", 'FREQ'))
        header.append(("CUNIT3", 'Hz'))
        header.append(("CDELT4", wcs_header["CDELT4"]))
        header.append(("CRPIX4", wcs_header["CRPIX4"]))
        header.append(("CRVAL4", wcs_header["CRVAL4"]))
        header.append(("CTYPE4", wcs_header["CTYPE4"]))
        header.append(("CUNIT4", "PAR"))
        header.append(("EPOCH", 2000))
        header.append(("VELREF", 257))
        header.append(("BSCALE", 1.0))
        header.append(("BZERO", 0.0))
        header.append(("OBJECT", "MOCK"))
        header.append(("MJD-OBS", Time.now().to_value("mjd")))
        header.append(("BTYPE", "Intensity"))
        header.append(("SPECSYS",'LSRK'))
        header.append(("RESTFREQ", 1.099060000000E+11 ))
    datacube_array_units = datacube._array.unit
    hdu = fits.PrimaryHDU(
            header=header, data=datacube._array.to_value(datacube_array_units).T
        )
    hdu.writeto(filename, overwrite=overwrite)

def write_numpy_to_fits(array, header, path):
    hdu = fits.PrimaryHDU(
            header=header, data=array
        )
    hdu.writeto(path, overwrite=True)

def generate_gaussian_skymodel(id, data_dir, n_sources, n_px, n_channels, bandwidth, 
                               fwhm_x, fwhm_y, fwhm_z, pixel_size, fov,
                               spatial_resolution, central_frequency, 
                               frequency_resolution, ra, dec, pa, min_sep_spatial, 
                               min_sep_frequency, rest_frequency, serendipitous, plot_dir):
    """
    Generates a gaussian skymodel with n_sources sources
    Input:
    id (int): id of the skymodel
    data_dir (str): directory where the skymodel will be saved
    n_sources (int): number of sources
    n_px (int): number of pixels in the image
    n_channels (int): number of channels in the image
    bandwidth (float): bandwidth in MHz
    distance (float): distance of the source from the observer in Mpc
    fwhm_x (float): fwhm in x direction in arcsec
    fwhm_y (float): fwhm in y direction in arcsec
    fwhm_z (float): fwhm in z direction in MHz
    pixel_size (float): pixel size in arcsec
    fov (float): field of view in arcsec
    spatial_resolution (float): spatial resolution in arcsec
    central_frequency (float): central frequency in GHz
    frequency_resolution (float): frequency resolution in MHz
    pa (float): position angle in degrees\
    min_sep_spatial (float): minimum separation between sources in arcsec
    min_sep_frequency (float): minimum separation between sources in MHz
    """
    fwhm_x = int(fwhm_x.value / pixel_size.value)
    fwhm_y = int(fwhm_y.value / pixel_size.value)
    fwhm_z = int(fwhm_z.value / frequency_resolution.value)
    print(fwhm_x, fwhm_y, fwhm_z)
    if rest_frequency == 1420.4:
        hI_rest_frequency = rest_frequency * U.MHz
    else:
        hI_rest_frequency = rest_frequency * 10 ** -6 * U.MHz
    radio_hI_equivalence = U.doppler_radio(hI_rest_frequency)
    central_velocity = central_frequency.to(U.km / U.s, equivalencies=radio_hI_equivalence)
    velocity_resolution = frequency_resolution.to(U.km / U.s, equivalencies=radio_hI_equivalence)
    print('Number of Pixels', n_px)
    datacube = DataCube(
        n_px_x = n_px,
        n_px_y = n_px,
        n_channels = n_channels, 
        px_size = spatial_resolution,
        channel_width = velocity_resolution,
        velocity_centre=central_velocity, 
        ra = ra,
        dec = dec,
        )
    wcs = datacube.wcs
    pos_x, pos_y, _ = wcs.sub(3).wcs_world2pix(ra, dec, central_velocity, 0)
    pos_z = n_channels // 2
    print('Generating central source at position ({}, {}, {})'.format(int(pos_x), int(pos_y), int(pos_z)))
    datacube = insert_gaussian(datacube, 1, pos_x, pos_y, pos_z, fwhm_x, fwhm_y, fwhm_z, pa, n_px, n_channels)
    #xy_radius = fov / pixel_size * 0.3
    #z_radius = 0.3 * bandwidth * U.MHz / frequency_resolution
    xy_radius = n_px / 4
    z_radius = n_channels / 2
    if serendipitous is True:
        print('Generating central source and {} serendipitous companions in a radius of {} pixels in the x and y directions and {} pixels in the z direction\n'.format(n_sources, int(xy_radius), int(z_radius)))
    min_sep_xy = min_sep_spatial / pixel_size
    min_sep_z = min_sep_frequency / frequency_resolution
    if serendipitous is True:
        if fwhm_x == 1:
            fwhm_x += 1
        if fwhm_y == 1:
            fwhm_y += 1
        if fwhm_z <= 2:
            fwhm_z = 3
        fwhm_xs = np.random.randint(1, fwhm_x, n_sources)
        fwhm_ys = np.random.randint(1, fwhm_y, n_sources)
        fwhm_zs = np.random.randint(2, fwhm_z, n_sources)
        amplitudes = np.random.rand(n_sources)
        sample_coords = sample_positions(pos_x, pos_y, pos_z, 
                                     fwhm_x, fwhm_y, fwhm_z,
                                     n_sources, fwhm_xs, fwhm_ys, fwhm_zs,
                                     xy_radius, z_radius, min_sep_xy.value, min_sep_z.value)
    
        pas = np.random.randint(0, 360, n_sources)
        for c_id, choords in tqdm(enumerate(sample_coords), total=len(sample_coords),):
            print('{}:\nLocation: {}\nSize X: {} Y: {} Z: {}'.format(c_id, choords, fwhm_xs[c_id], fwhm_ys[c_id], fwhm_zs[c_id]))
            datacube = insert_gaussian(datacube, amplitudes[c_id], choords[0], choords[1], choords[2], fwhm_xs[c_id], fwhm_ys[c_id], fwhm_zs[c_id], pas[c_id], n_px, n_channels)
    filename = os.path.join(data_dir, 'skymodel_{}.fits'.format(id))
    write_datacube_to_fits(datacube, filename)
    plot_skymodel(filename, id, plot_dir)
    print('Skymodel saved to {}'.format(filename))
    del datacube
    return filename

def generate_pointlike_skymodel(id, data_dir, rest_frequency, 
                                frequency_resolution, fwhm_z, central_frequency, 
                                n_px, n_channels, ra, dec, 
                                spatial_resolution, serendipitous, plot_dir):
    fwhm_z = int(fwhm_z.value / frequency_resolution.value)
    print('Line FWHM in channels:', fwhm_z)
    if fwhm_z < 3:
        fwhm_z = 3
    #if rest_frequency == 1420.4:
    #    hI_rest_frequency = rest_frequency * U.MHz
    #else:
    #    hI_rest_frequency = rest_frequency * 10 ** -6 * U.MHz
    hI_rest_frequency = rest_frequency * U.MHz
    radio_hI_equivalence = U.doppler_radio(hI_rest_frequency)
    central_velocity = central_frequency.to(U.km / U.s, equivalencies=radio_hI_equivalence)
    velocity_resolution = frequency_resolution.to(U.km / U.s, equivalencies=radio_hI_equivalence)
    datacube = DataCube(
        n_px_x = n_px,
        n_px_y = n_px,
        n_channels = n_channels, 
        px_size = spatial_resolution,
        channel_width = velocity_resolution,
        velocity_centre=central_velocity, 
        ra = ra,
        dec = dec,
        )
    wcs = datacube.wcs
    pos_x, pos_y, _ = wcs.sub(3).wcs_world2pix(ra, dec, central_velocity, 0)
    pos_z = n_channels // 2
    print('Generating point-like source at position ({}, {}, {})'.format(int(pos_x), int(pos_y), int(pos_z)))
    datacube = insert_pointlike(datacube, 1, pos_x, pos_y, pos_z, fwhm_z, n_px, n_channels)
    if serendipitous is True:
        print('Generating central serendipitous companions')
        xy_radius = n_px / 4
        z_radius = n_channels / 2
        min_sep_xy = 3
        min_sep_z = 10
        n_sources = random.randint(1, 5)
        amplitudes = np.random.rand(n_sources)
        for i in range(n_sources):
            while True:
                x = np.random.randint(int(0.25 * n_px), int(0.75 * n_px))
                y = np.random.randint(int(0.25 * n_px), int(0.75 * n_px))
                z = np.random.randint(int(0.25 * n_channels), int(0.75 * n_channels))
                if abs(x - pos_x) >= min_sep_xy and abs(y - pos_y) >= min_sep_xy and abs(z - pos_z) >= min_sep_z:
                    break
            s_fwhm_z = np.random.randint(1, fwhm_z)
            datacube = insert_pointlike(datacube, amplitudes[i], x, y, z, s_fwhm_z, n_px, n_channels)
    filename = os.path.join(data_dir, 'skymodel_{}.fits'.format(id))
    write_datacube_to_fits(datacube, filename)
    plot_skymodel(filename, id, plot_dir)
    print('Skymodel saved to {}'.format(filename))
    del datacube
    return filename
     
def generate_diffuse_skymodel(id, data_dir, n_px, n_channels,
                              fwhm_z, fov, spatial_resolution, central_frequency, 
                              frequency_resolution, ra, dec, rest_frequency, plot_dir):
    """
    Generates a gaussian skymodel with n_sources sources
    Input:
    id (int): id of the skymodel
    data_dir (str): directory where the skymodel will be saved
    n_sources (int): number of sources
    n_px (int): number of pixels in the image
    n_channels (int): number of channels in the image
    bandwidth (float): bandwidth in MHz
    distance (float): distance of the source from the observer in Mpc
    fwhm_x (float): fwhm in x direction in arcsec
    fwhm_y (float): fwhm in y direction in arcsec
    fwhm_z (float): fwhm in z direction in MHz
    pixel_size (float): pixel size in arcsec
    fov (float): field of view in arcsec
    spatial_resolution (float): spatial resolution in arcsec
    central_frequency (float): central frequency in GHz
    frequency_resolution (float): frequency resolution in MHz
    pa (float): position angle in degrees\
    min_sep_spatial (float): minimum separation between sources in arcsec
    min_sep_frequency (float): minimum separation between sources in MHz
    plot (bool): if True, plots the skymodel
    plot_dir (str): directory where the plots will be saved
    """
    fwhm_z = int(fwhm_z.value / frequency_resolution.value)
    if rest_frequency == 1420.4:
        hI_rest_frequency = rest_frequency * U.MHz
    else:
        hI_rest_frequency = rest_frequency * 10 ** -6 * U.MHz
    radio_hI_equivalence = U.doppler_radio(hI_rest_frequency)
    central_velocity = central_frequency.to(U.km / U.s, equivalencies=radio_hI_equivalence)
    velocity_resolution = frequency_resolution.to(U.km / U.s, equivalencies=radio_hI_equivalence)
    datacube = DataCube(
    n_px_x = n_px,
    n_px_y = n_px,
    n_channels = n_channels, 
    px_size = spatial_resolution,
    channel_width = velocity_resolution,
    velocity_centre=central_velocity, 
    ra = ra,
    dec = dec,
    )
    pos_z = n_channels // 2
    datacube = insert_diffuse(datacube, pos_z, fwhm_z, fov, n_channels, n_px, n_px)
    filename = os.path.join(data_dir, 'skymodel_{}.fits'.format(id))
    write_diffuse_datacube_to_fits(datacube, filename)
    print('Skymodel saved to {}'.format(filename))
    plot_skymodel(filename, id, plot_dir)
    del datacube
    return filename

def generate_lensing_skymodel(id, data_dir):
    return

def find_distance(TNGSnap, subhaloID, n_px, n_channels, 
                            frequency_resolution, spatial_resolution,
                            ra, dec, x_rot, y_rot, TNGBasePath, distance, ncpu):
    source = myTNGSource(TNGSnap, subhaloID,
                       distance= distance * U.Mpc,
                       rotation = {'L_coords': (x_rot, y_rot)},
                       basePath = TNGBasePath,
                       ra = 0. * U.deg,
                       dec = 0. * U.deg,)
    if ra == 0.0 * U.deg and dec == 0.0 * U.deg:
        ra = source.ra
        dec = source.dec
    
    datacube = DataCube(
        n_px_x = n_px,
        n_px_y = n_px,
        n_channels = n_channels, 
        px_size = 10 * U.arcsec,
        channel_width=frequency_resolution,
        velocity_centre=source.vsys, 
        ra = ra,
        dec = dec,
    )
    spectral_model = GaussianSpectrum(
        sigma="thermal"
    )
    sph_kernel =  WendlandC2Kernel()
    M = Martini(
        source=source,
        datacube=datacube,
        sph_kernel=sph_kernel,
        spectral_model=spectral_model,
        quiet=False,
        find_distance=True)
    particle_percentage = M._compute_particles_num()
    print('Initial particle percentage: {}%'.format(particle_percentage))
    while particle_percentage < 80:
        print('Particle percentage is too low, increasing')
        distance += 10
        source = myTNGSource(TNGSnap, subhaloID,
                       distance= distance * U.Mpc,
                       rotation = {'L_coords': (x_rot, y_rot)},
                       basePath = TNGBasePath,
                       ra = 0. * U.deg,
                       dec = 0. * U.deg,)
        if ra == 0.0 * U.deg and dec == 0.0 * U.deg:
            ra = source.ra
            dec = source.dec
    
        datacube = DataCube(
            n_px_x = n_px,
            n_px_y = n_px,
            n_channels = n_channels, 
            px_size = spatial_resolution * U.arcsec,
            channel_width=frequency_resolution,
            velocity_centre=source.vsys, 
            ra = ra,
            dec = dec,
        )
        spectral_model = GaussianSpectrum(
            sigma="thermal"
        )
        sph_kernel =  WendlandC2Kernel()
        M = Martini(
            source=source,
            datacube=datacube,
            sph_kernel=sph_kernel,
            spectral_model=spectral_model,
            quiet=False)
        particle_percentage = M._compute_particles_num()
        print('Particle percentage: {}%'.format(particle_percentage))
    return distance

def insert_extended_skymodel(TNGSnap, subhaloID, n_px, n_channels, 
                            frequency_resolution, spatial_resolution,
                            ra, dec, x_rot, y_rot, TNGBasePath, distance, ncpu):
    source = myTNGSource(TNGSnap, subhaloID,
                       distance= distance * U.Mpc,
                       rotation = {'L_coords': (x_rot, y_rot)},
                       basePath = TNGBasePath,
                       ra = 0. * U.deg,
                       dec = 0. * U.deg,)
    if ra == 0.0 * U.deg and dec == 0.0 * U.deg:
        ra = source.ra
        dec = source.dec
    
    datacube = DataCube(
        n_px_x = n_px,
        n_px_y = n_px,
        n_channels = n_channels, 
        px_size = 10 * U.arcsec,
        channel_width=frequency_resolution,
        velocity_centre=source.vsys, 
        ra = source.ra,
        dec = source.dec,
    )
    spectral_model = GaussianSpectrum(
        sigma="thermal"
    )
    sph_kernel =  WendlandC2Kernel()
    M = Martini(
        source=source,
        datacube=datacube,
        sph_kernel=sph_kernel,
        spectral_model=spectral_model,
        quiet=False, 
        find_distance=False)
    M.insert_source_in_cube(skip_validation=True, progressbar=True, ncpu=ncpu)
    return M, source, datacube

def generate_extended_skymodel(id, data_dir, n_px, n_channels, pixel_size,
                               central_frequency, frequency_resolution, spatial_resolution,
                               TNGBasePath, TNGSnap, subhaloID, ra, dec, 
                               rest_frequency, plot_dir, ncpu):
    #distance = np.random.randint(1, 5) * U.Mpc
    x_rot = np.random.randint(0, 360) * U.deg
    y_rot = np.random.randint(0, 360) * U.deg
    simulation_str = TNGBasePath.split('/')[-1]
    TNGBasePath = os.path.join(TNGBasePath, 'TNG100-1', 'output')

    #distance, channel_width = get_distance(n_px, n_channels, x_rot, y_rot, 
    #                                       TNGSnap, subhaloID, TNGBasePath, factor=4)
    data_header = loadHeader(TNGBasePath, TNGSnap)
    z = data_header["Redshift"] * cu.redshift
    
    #distance = z.to(U.Mpc, cu.redshift_distance(Planck13, kind="comoving"))
    #distance= 10
    #distance = find_distance(TNGSnap, subhaloID, n_px, n_channels, 
    #                        frequency_resolution, spatial_resolution,
    #                        ra, dec, x_rot, y_rot, TNGBasePath, distance, ncpu)
    #print('Found optimal distance: {} Mpc'.format(distance))
    distance = 50
    print('Generating extended source from subhalo {} - {} at {} with rotation angles {} and {} in the X and Y planes'.format(simulation_str, subhaloID, distance, x_rot, y_rot))
    print('Source generated, injecting into datacube')
    if rest_frequency == 1420.4:
        hI_rest_frequency = rest_frequency * U.MHz
    else:
        hI_rest_frequency = rest_frequency * 10 ** -6 * U.MHz
    radio_hI_equivalence = U.doppler_radio(hI_rest_frequency)
    central_velocity = central_frequency.to(U.km / U.s, equivalencies=radio_hI_equivalence)
    velocity_resolution = frequency_resolution.to(U.km / U.s, equivalencies=radio_hI_equivalence)
    M, source, datacube = insert_extended_skymodel(TNGSnap, subhaloID, n_px, n_channels, 
                                frequency_resolution, spatial_resolution,
                                ra, dec, x_rot, y_rot, TNGBasePath, distance, ncpu)
    initial_mass_ratio = M.inserted_mass / M.source.input_mass * 100
    print('Mass ratio: {}%'.format(initial_mass_ratio))
    mass_ratio = initial_mass_ratio
    while mass_ratio < 80:
        if mass_ratio < 10:
            distance = distance * 8
        elif mass_ratio < 20:
            distance = distance * 5
        elif mass_ratio < 30:
            distance = distance * 2
        else:       
            distance = distance * 1.5
        print('Injecting source at distance {}'.format(distance))
        M, source, datacube = insert_extended_skymodel(TNGSnap, subhaloID, n_px, n_channels, 
                                frequency_resolution, spatial_resolution,
                                ra, dec, x_rot, y_rot, TNGBasePath, distance, ncpu)
        mass_ratio = M.inserted_mass / M.source.input_mass * 100
        print('Mass ratio: {}%'.format(mass_ratio))
    print('Datacube generated, inserting source')
    
    print('Source inserted, saving skymodel')
    M.write_hdf5(os.path.join(data_dir, 'skymodel_{}.hdf5'.format(str(id))), channels='velocity')
    f = h5py.File(os.path.join(data_dir, 'skymodel_{}.hdf5'.format(str(id))),'r')
    vch = f['channel_mids'][()] / 1E3 - source.distance.to(U.Mpc).value*70  # m/s to km/s
    f.close()
    os.remove(os.path.join(data_dir, 'skymodel_{}.hdf5'.format(str(id))))
    filename = os.path.join(data_dir, 'skymodel_{}.fits'.format(id))
    #M.write_fits(filename, channels='velocity')
    write_datacube_to_fits(M.datacube, filename)
    print('Skymodel saved to {}'.format(filename))
    plot_skymodel(filename, id, plot_dir)
    del datacube
    del M
    return filename

# -------------------------- Modified functions from illustris-tng -------------------------- #

def partTypeNum(partType):
    """ Mapping between common names and numeric particle types. """
    if str(partType).isdigit():
        return int(partType)
        
    if str(partType).lower() in ['gas','cells']:
        return 0
    if str(partType).lower() in ['dm','darkmatter']:
        return 1
    if str(partType).lower() in ['dmlowres']:
        return 2 # only zoom simulations, not present in full periodic boxes
    if str(partType).lower() in ['tracer','tracers','tracermc','trmc']:
        return 3
    if str(partType).lower() in ['star','stars','stellar']:
        return 4 # only those with GFM_StellarFormationTime>0
    if str(partType).lower() in ['wind']:
        return 4 # only those with GFM_StellarFormationTime<0
    if str(partType).lower() in ['bh','bhs','blackhole','blackholes']:
        return 5
    
    raise Exception("Unknown particle type name.")

def gcPath(basePath, snapNum, chunkNum=0):
    """ Return absolute path to a group catalog HDF5 file (modify as needed). """
    gcPath = basePath + '/groups_%03d/' % snapNum
    filePath1 = gcPath + 'groups_%03d.%d.hdf5' % (snapNum, chunkNum)
    filePath2 = gcPath + 'fof_subhalo_tab_%03d.%d.hdf5' % (snapNum, chunkNum)

    if isfile(expanduser(filePath1)):
        return filePath1
    return filePath2

def offsetPath(basePath, snapNum):
    """ Return absolute path to a separate offset file (modify as needed). """
    offsetPath = basePath + '/../postprocessing/offsets/offsets_%03d.hdf5' % snapNum

    return offsetPath

def loadObjects(basePath, snapNum, gName, nName, fields):
    """ Load either halo or subhalo information from the group catalog. """
    result = {}

    # make sure fields is not a single element
    if isinstance(fields, six.string_types):
        fields = [fields]

    # load header from first chunk
    with h5py.File(gcPath(basePath, snapNum), 'r') as f:

        header = dict(f['Header'].attrs.items())

        if 'N'+nName+'_Total' not in header and nName == 'subgroups':
            nName = 'subhalos' # alternate convention

        result['count'] = f['Header'].attrs['N' + nName + '_Total']

        if not result['count']:
            print('warning: zero groups, empty return (snap=' + str(snapNum) + ').')
            return result

        # if fields not specified, load everything
        if not fields:
            fields = list(f[gName].keys())

        for field in fields:
            # verify existence
            if field not in f[gName].keys():
                raise Exception("Group catalog does not have requested field [" + field + "]!")

            # replace local length with global
            shape = list(f[gName][field].shape)
            shape[0] = result['count']

            # allocate within return dict
            result[field] = np.zeros(shape, dtype=f[gName][field].dtype)

    # loop over chunks
    wOffset = 0

    for i in range(header['NumFiles']):
        f = h5py.File(gcPath(basePath, snapNum, i), 'r')

        if not f['Header'].attrs['N'+nName+'_ThisFile']:
            continue  # empty file chunk

        # loop over each requested field
        for field in fields:
            if field not in f[gName].keys():
                raise Exception("Group catalog does not have requested field [" + field + "]!")

            # shape and type
            shape = f[gName][field].shape

            # read data local to the current file
            if len(shape) == 1:
                result[field][wOffset:wOffset+shape[0]] = f[gName][field][0:shape[0]]
            else:
                result[field][wOffset:wOffset+shape[0], :] = f[gName][field][0:shape[0], :]

        wOffset += shape[0]
        f.close()

    # only a single field? then return the array instead of a single item dict
    if len(fields) == 1:
        return result[fields[0]]

    return result

def loadSubhalos(basePath, snapNum, fields=None):
    """ Load all subhalo information from the entire group catalog for one snapshot
       (optionally restrict to a subset given by fields). """

    return loadObjects(basePath, snapNum, "Subhalo", "subgroups", fields)

def loadHalos(basePath, snapNum, fields=None):
    """ Load all halo information from the entire group catalog for one snapshot
       (optionally restrict to a subset given by fields). """

    return loadObjects(basePath, snapNum, "Group", "groups", fields)

def loadHeader(basePath, snapNum):
    """ Load the group catalog header. """
    with h5py.File(gcPath(basePath, snapNum), 'r') as f:
        header = dict(f['Header'].attrs.items())

    return header

def load(basePath, snapNum):
    """ Load complete group catalog all at once. """
    r = {}
    r['subhalos'] = loadSubhalos(basePath, snapNum)
    r['halos']    = loadHalos(basePath, snapNum)
    r['header']   = loadHeader(basePath, snapNum)
    return r

def loadSingle(basePath, snapNum, haloID=-1, subhaloID=-1):
    """ Return complete group catalog information for one halo or subhalo. """
    if (haloID < 0 and subhaloID < 0) or (haloID >= 0 and subhaloID >= 0):
        raise Exception("Must specify either haloID or subhaloID (and not both).")

    gName = "Subhalo" if subhaloID >= 0 else "Group"
    searchID = subhaloID if subhaloID >= 0 else haloID

    # old or new format
    if 'fof_subhalo' in gcPath(basePath, snapNum):
        # use separate 'offsets_nnn.hdf5' files
        with h5py.File(offsetPath(basePath, snapNum), 'r') as f:
            offsets = f['FileOffsets/'+gName][()]
    else:
        # use header of group catalog
        with h5py.File(gcPath(basePath, snapNum), 'r') as f:
            offsets = f['Header'].attrs['FileOffsets_'+gName]

    offsets = searchID - offsets
    fileNum = np.max(np.where(offsets >= 0))
    groupOffset = offsets[fileNum]

    # load halo/subhalo fields into a dict
    result = {}

    with h5py.File(gcPath(basePath, snapNum, fileNum), 'r') as f:
        for haloProp in f[gName].keys():
            result[haloProp] = f[gName][haloProp][groupOffset]

    return result

def snapPath(basePath, snapNum, chunkNum=0):
    """ Return absolute path to a snapshot HDF5 file (modify as needed). """
    snapPath = basePath + '/snapdir_' + str(snapNum).zfill(3) + '/'
    filePath1 = snapPath + 'snap_' + str(snapNum).zfill(3) + '.' + str(chunkNum) + '.hdf5'
    filePath2 = filePath1.replace('/snap_', '/snapshot_')

    if isfile(filePath1):
        return filePath1
    return filePath2

def getNumPart(header):
    """ Calculate number of particles of all types given a snapshot header. """
    if 'NumPart_Total_HighWord' not in header:
        return header['NumPart_Total'] # new uint64 convention

    nTypes = 6

    nPart = np.zeros(nTypes, dtype=np.int64)
    for j in range(nTypes):
        nPart[j] = header['NumPart_Total'][j] | (header['NumPart_Total_HighWord'][j] << 32)

    return nPart

def loadSubset(basePath, snapNum, partType, fields=None, subset=None, mdi=None, sq=True, float32=False, outPath=None):
    """ Load a subset of fields for all particles/cells of a given partType.
        If offset and length specified, load only that subset of the partType.
        If mdi is specified, must be a list of integers of the same length as fields,
        giving for each field the multi-dimensional index (on the second dimension) to load.
          For example, fields=['Coordinates', 'Masses'] and mdi=[1, None] returns a 1D array
          of y-Coordinates only, together with Masses.
        If sq is True, return a numpy array instead of a dict if len(fields)==1.
        If float32 is True, load any float64 datatype arrays directly as float32 (save memory). """
    result = {}

    ptNum = partTypeNum(partType)
    gName = "PartType" + str(ptNum)

    # make sure fields is not a single element
    if isinstance(fields, six.string_types):
        fields = [fields]

    # load header from first chunk
    with h5py.File(snapPath(basePath, snapNum), 'r') as f:

        header = dict(f['Header'].attrs.items())
        nPart = getNumPart(header)

        # decide global read size, starting file chunk, and starting file chunk offset
        if subset:
            offsetsThisType = subset['offsetType'][ptNum] - subset['snapOffsets'][ptNum, :]

            fileNum = np.max(np.where(offsetsThisType >= 0))
            fileOff = offsetsThisType[fileNum]
            numToRead = subset['lenType'][ptNum]
        else:
            fileNum = 0
            fileOff = 0
            numToRead = nPart[ptNum]

        result['count'] = numToRead

        if not numToRead:
            # print('warning: no particles of requested type, empty return.')
            return result

        # find a chunk with this particle type
        i = 1
        while gName not in f:
            if os.path.isfile(snapPath(basePath, snapNum, i)):
                print('Found')
                f = h5py.File(snapPath(basePath, snapNum, i), 'r')
            else:
                print('Not Found')
                api_key = '8f578b92e700fae3266931f4d785f82c'
                url = f'http://www.tng-project.org/api/TNG100-1/files/snapshot-{str(snapNum)}'
                subdir = os.path.join('output', 'snapdir_0{}'.format(str(i)))
                cmd = f'wget -q --progress=bar  --content-disposition --header="API-Key:{api_key}" {url}.{i}.hdf5'
                print(f'Downloading {message} {i} ...')
                if outPath is not None:
                    os.chdir(outPath)
                subprocess.check_call(cmd, shell=True)
                print('Done.')
                f = h5py.File(snapPath(basePath, snapNum, i), 'r')
            i += 1

        # if fields not specified, load everything
        if not fields:
            fields = list(f[gName].keys())

        for i, field in enumerate(fields):
            # verify existence
            if field not in f[gName].keys():
                raise Exception("Particle type ["+str(ptNum)+"] does not have field ["+field+"]")

            # replace local length with global
            shape = list(f[gName][field].shape)
            shape[0] = numToRead

            # multi-dimensional index slice load
            if mdi is not None and mdi[i] is not None:
                if len(shape) != 2:
                    raise Exception("Read error: mdi requested on non-2D field ["+field+"]")
                shape = [shape[0]]

            # allocate within return dict
            dtype = f[gName][field].dtype
            if dtype == np.float64 and float32: dtype = np.float32
            result[field] = np.zeros(shape, dtype=dtype)

    # loop over chunks
    wOffset = 0
    origNumToRead = numToRead

    while numToRead:
        if not os.path.isfile(snapPath(basePath, snapNum, fileNum)):
            print(f'Particles are found in Snapshot {fileNum} which is not present on disk')
            # move directory to the correct directory data !!!
            api_key = '8f578b92e700fae3266931f4d785f82c'
            url = f'http://www.tng-project.org/api/TNG100-1/files/snapshot-{str(snapNum)}'
            subdir = os.path.join('output', 'snapdir_0{}'.format(str(fileNum)))
            savePath = os.path.join(basePath, 'snapdir_0{}'.format(str(snapNum)))
            cmd = f'wget -P {savePath} -q --progress=bar  --content-disposition --header="API-Key:{api_key}" {url}.{fileNum}.hdf5'
            if outPath is not None:
                os.chdir(outPath)
            print(f'Downloading Snapshot {fileNum} in {savePath}...')
            subprocess.check_call(cmd, shell=True)
            print('Done.')
        print('Checking File {}...'.format(fileNum))
        f = h5py.File(snapPath(basePath, snapNum, fileNum), 'r')

        # no particles of requested type in this file chunk?
        if gName not in f:
            f.close()
            fileNum += 1
            fileOff  = 0
            continue

        # set local read length for this file chunk, truncate to be within the local size
        numTypeLocal = f['Header'].attrs['NumPart_ThisFile'][ptNum]

        numToReadLocal = numToRead

        if fileOff + numToReadLocal > numTypeLocal:
            numToReadLocal = numTypeLocal - fileOff

        #print('['+str(fileNum).rjust(3)+'] off='+str(fileOff)+' read ['+str(numToReadLocal)+\
        #      '] of ['+str(numTypdeLocal)+'] remaining = '+str(numToRead-numToReadLocal))

        # loop over each requested field for this particle type
        for i, field in enumerate(fields):
            # read data local to the current file
            if mdi is None or mdi[i] is None:
                result[field][wOffset:wOffset+numToReadLocal] = f[gName][field][fileOff:fileOff+numToReadLocal]
            else:
                result[field][wOffset:wOffset+numToReadLocal] = f[gName][field][fileOff:fileOff+numToReadLocal, mdi[i]]

        wOffset   += numToReadLocal
        numToRead -= numToReadLocal
        fileNum   += 1
        fileOff    = 0  # start at beginning of all file chunks other than the first
        print('Loading File {}...'.format(fileNum))
        f.close()

    # verify we read the correct number
    if origNumToRead != wOffset:
        raise Exception("Read ["+str(wOffset)+"] particles, but was expecting ["+str(origNumToRead)+"]")

    # only a single field? then return the array instead of a single item dict
    if sq and len(fields) == 1:
        return result[fields[0]]

    return result

def getSnapOffsets(basePath, snapNum, id, type):
    """ Compute offsets within snapshot for a particular group/subgroup. """
    r = {}
    print(f'Checking offset in Snapshot {snapNum} for grouphalo {id}')
    # old or new format
    if 'fof_subhalo' in gcPath(basePath, snapNum):
        # use separate 'offsets_nnn.hdf5' files
        with h5py.File(offsetPath(basePath, snapNum), 'r') as f:
            groupFileOffsets = f['FileOffsets/'+type][()]
            r['snapOffsets'] = np.transpose(f['FileOffsets/SnapByType'][()])  # consistency
    else:
        # load groupcat chunk offsets from header of first file
        with h5py.File(gcPath(basePath, snapNum), 'r') as f:
            groupFileOffsets = f['Header'].attrs['FileOffsets_'+type]
            r['snapOffsets'] = f['Header'].attrs['FileOffsets_Snap']

    # calculate target groups file chunk which contains this id
    groupFileOffsets = int(id) - groupFileOffsets
    fileNum = np.max(np.where(groupFileOffsets >= 0))
    groupOffset = groupFileOffsets[fileNum]

    # load the length (by type) of this group/subgroup from the group catalog
    with h5py.File(gcPath(basePath, snapNum, fileNum), 'r') as f:
        r['lenType'] = f[type][type+'LenType'][groupOffset, :]

    # old or new format: load the offset (by type) of  this group/subgroup within the snapshot
    if 'fof_subhalo' in gcPath(basePath, snapNum):
        with h5py.File(offsetPath(basePath, snapNum), 'r') as f:
            r['offsetType'] = f[type+'/SnapByType'][id, :]

            # add TNG-Cluster specific offsets if present
            if 'OriginalZooms' in f:
                for key in f['OriginalZooms']:
                    r[key] = f['OriginalZooms'][key][()] 
    else:
        with h5py.File(gcPath(basePath, snapNum, fileNum), 'r') as f:
            r['offsetType'] = f['Offsets'][type+'_SnapByType'][groupOffset, :]

    return r

def loadSubhalo(basePath, snapNum, id, partType, fields=None):
    """ Load all particles/cells of one type for a specific subhalo
        (optionally restricted to a subset fields). """
    # load subhalo length, compute offset, call loadSubset
    subset = getSnapOffsets(basePath, snapNum, id, "Subhalo")
    return loadSubset(basePath, snapNum, partType, fields, subset=subset)

def loadHalo(basePath, snapNum, id, partType, fields=None):
    """ Load all particles/cells of one type for a specific halo
        (optionally restricted to a subset fields). """
    # load halo length, compute offset, call loadSubset
    subset = getSnapOffsets(basePath, snapNum, id, "Group")
    return loadSubset(basePath, snapNum, partType, fields, subset=subset)

def loadOriginalZoom(basePath, snapNum, id, partType, fields=None):
    """ Load all particles/cells of one type corresponding to an
        original (entire) zoom simulation. TNG-Cluster specific.
        (optionally restricted to a subset fields). """
    # load fuzz length, compute offset, call loadSubset                                                                     
    subset = getSnapOffsets(basePath, snapNum, id, "Group")

    # identify original halo ID and corresponding index
    halo = loadSingle(basePath, snapNum, haloID=id)
    assert 'GroupOrigHaloID' in halo, 'Error: loadOriginalZoom() only for the TNG-Cluster simulation.'
    orig_index = np.where(subset['HaloIDs'] == halo['GroupOrigHaloID'])[0][0]

    # (1) load all FoF particles/cells
    subset['lenType'] = subset['GroupsTotalLengthByType'][orig_index, :]
    subset['offsetType'] = subset['GroupsSnapOffsetByType'][orig_index, :]

    data1 = loadSubset(basePath, snapNum, partType, fields, subset=subset)

    # (2) load all non-FoF particles/cells
    subset['lenType'] = subset['OuterFuzzTotalLengthByType'][orig_index, :]
    subset['offsetType'] = subset['OuterFuzzSnapOffsetByType'][orig_index, :]

    data2 = loadSubset(basePath, snapNum, partType, fields, subset=subset)

    # combine and return
    if isinstance(data1, np.ndarray):
        return np.concatenate((data1,data2), axis=0)
    
    data = {'count':data1['count']+data2['count']}
    for key in data1.keys():
        if key == 'count': continue
        data[key] = np.concatenate((data1[key],data2[key]), axis=0)
    return data

def get_particles_num(basePath, outputPath, snapNum, subhaloID):
    basePath = os.path.join(basePath, "TNG100-1", "output", )
    print('Looking for Subhalo %d in snapshot %d' % (subhaloID, snapNum))
    partType = 'gas'
    subset = getSnapOffsets(basePath, snapNum, subhaloID, "Subhalo")
    subhalo = loadSubset(basePath, snapNum, partType, subset=subset)
    os.chdir(basePath)
    gas = il.snapshot.loadSubhalo(basePath, snapNum, subhaloID, partType)
    if 'Coordinates' in gas.keys():
        gas_num = len(gas['Coordinates'])
    else:
        gas_num = 0
    return gas_num

# -------------------------- Modified functions from martini -------------------------- #

class myTNGSource(SPHSource):
    def __init__(
        self,
        snapNum,
        subID,
        basePath=None,
        distance=3.0 * U.Mpc,
        vpeculiar=0 * U.km / U.s,
        rotation={"rotmat": np.eye(3)},
        ra=0.0 * U.deg,
        dec=0.0 * U.deg,
    ):
        X_H = 0.76
        full_fields_g = (
            "Masses",
            "Velocities",
            "InternalEnergy",
            "ElectronAbundance",
            "Density",
            "CenterOfMass",
            "GFM_Metals",
        )
        mdi_full = [None, None, None, None, None, None, 0]
        mini_fields_g = (
            "Masses",
            "Velocities",
            "InternalEnergy",
            "ElectronAbundance",
            "Density",
            "Coordinates",
        )
        data_header = loadHeader(basePath, snapNum)
        data_sub = loadSingle(basePath, snapNum, subhaloID=subID)
        haloID = data_sub["SubhaloGrNr"]
        subset_g = getSnapOffsets(basePath, snapNum, haloID, "Group")
        try:
            data_g = loadSubset(
                    basePath,
                    snapNum,
                    "gas",
                    fields=full_fields_g,
                    subset=subset_g,
                    mdi=mdi_full,
                )

            minisnap = False
        except Exception as exc:
            print(exc.args)
            if ("Particle type" in exc.args[0]) and ("does not have field" in exc.args[0]):
                data_g.update(
                        loadSubset(
                            basePath,
                            snapNum,
                            "gas",
                            fields=("CenterOfMass",),
                            subset=subset_g,
                            sq=False,
                        )
                    )
                minisnap = True
                X_H_g = X_H
            else:
                raise
        X_H_g = (
                X_H if minisnap else data_g["GFM_Metals"])  # only loaded column 0: Hydrogen
        a = data_header["Time"]
        z = data_header["Redshift"]
        h = data_header["HubbleParam"]
        xe_g = data_g["ElectronAbundance"]
        rho_g = data_g["Density"] * 1e10 / h * U.Msun * np.power(a / h * U.kpc, -3)
        u_g = data_g["InternalEnergy"]  # unit conversion handled in T_g
        mu_g = 4 * C.m_p.to(U.g).value / (1 + 3 * X_H_g + 4 * X_H_g * xe_g)
        gamma = 5.0 / 3.0  # see http://www.tng-project.org/data/docs/faq/#gen4
        T_g = (gamma - 1) * u_g / C.k_B.to(U.erg / U.K).value * 1e10 * mu_g * U.K
        m_g = data_g["Masses"] * 1e10 / h * U.Msun
        # cast to float64 to avoid underflow error
        nH_g = U.Quantity(rho_g * X_H_g / mu_g, dtype=np.float64) / C.m_p
        # In TNG_corrections I set f_neutral = 1 for particles with density
        # > .1cm^-3. Might be possible to do a bit better here, but HI & H2
        # tables for TNG will be available soon anyway.
        fatomic_g = atomic_frac(
            z, nH_g, T_g, rho_g, X_H_g, onlyA1=True, TNG_corrections=True
            )
        mHI_g = m_g * X_H_g * fatomic_g
        try:
            xyz_g = data_g["CenterOfMass"] * a / h * U.kpc
        except KeyError:
            xyz_g = data_g["Coordinates"] * a / h * U.kpc
        vxyz_g = data_g["Velocities"] * np.sqrt(a) * U.km / U.s
        V_cell = (
            data_g["Masses"] / data_g["Density"] * np.power(a / h * U.kpc, 3)
            )  # Voronoi cell volume
        r_cell = np.power(3.0 * V_cell / 4.0 / np.pi, 1.0 / 3.0).to(U.kpc)
        # hsm_g has in mind a cubic spline that =0 at r=h, I think
        hsm_g = 2.5 * r_cell * find_fwhm(CubicSplineKernel().kernel)
        xyz_centre = data_sub["SubhaloPos"] * a / h * U.kpc
        xyz_g -= xyz_centre
        vxyz_centre = data_sub["SubhaloVel"] * np.sqrt(a) * U.km / U.s
        vxyz_g -= vxyz_centre
        super().__init__(
            distance=distance,
            vpeculiar=vpeculiar,
            rotation=rotation,
            ra=ra,
            dec=dec,
            h=h,
            T_g=T_g,
            mHI_g=mHI_g,
            xyz_g=xyz_g,
            vxyz_g=vxyz_g,
            hsm_g=hsm_g,
        )
        return

class DataCube(object):
    """
    Handles creation and management of the data cube itself.

    Basic usage simply involves initializing with the parameters listed below.
    More advanced usage might arise if designing custom classes for other sub-
    modules, especially beams. To initialize a DataCube from a saved state, see
    DataCube.load_state.

    Parameters
    ----------
    n_px_x : int, optional
        Pixel count along the x (RA) axis. Even integers strongly preferred.
        (Default: 256.)

    n_px_y : int, optional
        Pixel count along the y (Dec) axis. Even integers strongly preferred.
        (Default: 256.)

    n_channels : int, optional
        Number of channels along the spectral axis. (Default: 64.)

    px_size : Quantity, with dimensions of angle, optional
        Angular scale of one pixel. (Default: 15 arcsec.)

    channel_width : Quantity, with dimensions of velocity or frequency, optional
        Step size along the spectral axis. Can be provided as a velocity or a
        frequency. (Default: 4 km/s.)

    velocity_centre : Quantity, with dimensions of velocity or frequency, optional
        Velocity (or frequency) of the centre along the spectral axis.
        (Default: 0 km/s.)

    ra : Quantity, with dimensions of angle, optional
        Right ascension of the cube centroid. (Default: 0 deg.)

    dec : Quantity, with dimensions of angle, optional
        Declination of the cube centroid. (Default: 0 deg.)

    stokes_axis : bool, optional
        Whether the datacube should be initialized with a Stokes' axis. (Default: False.)

    See Also
    --------
    load_state
    """

    def __init__(
        self,
        n_px_x=256,
        n_px_y=256,
        n_channels=64,
        px_size=15.0 * U.arcsec,
        channel_width=4.0 * U.km * U.s**-1,
        velocity_centre=0.0 * U.km * U.s**-1,
        ra=0.0 * U.deg,
        dec=0.0 * U.deg,
        stokes_axis=False,
    ):
        self.HIfreq = 1.420405751e9 * U.Hz
        self.stokes_axis = stokes_axis
        datacube_unit = U.Jy * U.pix**-2
        self._array = np.zeros((n_px_x, n_px_y, n_channels)) * datacube_unit
        if self.stokes_axis:
            self._array = self._array[..., np.newaxis]
        self.n_px_x, self.n_px_y, self.n_channels = n_px_x, n_px_y, n_channels
        self.px_size = px_size
        self.arcsec2_to_pix = (
            U.Jy * U.pix**-2,
            U.Jy * U.arcsec**-2,
            lambda x: x / self.px_size**2,
            lambda x: x * self.px_size**2,
        )
        self.velocity_centre = velocity_centre.to(
            U.m / U.s, equivalencies=U.doppler_radio(self.HIfreq)
        )
        self.channel_width = np.abs(
            (
                velocity_centre.to(
                    channel_width.unit, equivalencies=U.doppler_radio(self.HIfreq)
                )
                + 0.5 * channel_width
            ).to(U.m / U.s, equivalencies=U.doppler_radio(self.HIfreq))
            - (
                velocity_centre.to(
                    channel_width.unit, equivalencies=U.doppler_radio(self.HIfreq)
                )
                - 0.5 * channel_width
            ).to(U.m / U.s, equivalencies=U.doppler_radio(self.HIfreq))
        )
        self.ra = ra
        self.dec = dec
        self.padx = 0
        self.pady = 0
        self._freq_channel_mode = False
        self._init_wcs()
        self._channel_mids()
        self._channel_edges()

        return

    def _init_wcs(self):
        self.wcs = wcs.WCS(naxis=3)
        self.wcs.wcs.crpix = [
            self.n_px_x / 2.0 + 0.5,
            self.n_px_y / 2.0 + 0.5,
            self.n_channels / 2.0 + 0.5,
        ]
        self.units = [U.deg, U.deg, U.m / U.s]
        self.wcs.wcs.cunit = [unit.to_string("fits") for unit in self.units]
        self.wcs.wcs.cdelt = [
            -self.px_size.to_value(self.units[0]),
            self.px_size.to_value(self.units[1]),
            self.channel_width.to_value(
                self.units[2], equivalencies=U.doppler_radio(self.HIfreq)
            ),
        ]
        self.wcs.wcs.crval = [
            self.ra.to_value(self.units[0]),
            self.dec.to_value(self.units[1]),
            self.velocity_centre.to_value(
                self.units[2], equivalencies=U.doppler_radio(self.HIfreq)
            ),
        ]
        self.wcs.wcs.ctype = ["RA---TAN", "DEC--TAN", "VRAD"]
        self.wcs.wcs.specsys = "GALACTOC"
        if self.stokes_axis:
            self.wcs = wcs.utils.add_stokes_axis_to_wcs(self.wcs, self.wcs.wcs.naxis)
        return

    def _channel_mids(self):
        """
        Calculate the centres of the channels from the coordinate system.
        """
        pixels = (
            np.zeros(self.n_channels),
            np.zeros(self.n_channels),
            np.arange(self.n_channels) - 0.5,
        )
        if self.stokes_axis:
            pixels = pixels + (np.zeros(self.n_channels),)
        self.channel_mids = (
            self.wcs.wcs_pix2world(
                *pixels,
                0,
            )[2]
            * self.units[2]
        )
        return

    def _channel_edges(self):
        """
        Calculate the edges of the channels from the coordinate system.
        """
        pixels = (
            np.zeros(self.n_channels + 1),
            np.zeros(self.n_channels + 1),
            np.arange(self.n_channels + 1) - 1,
        )
        if self.stokes_axis:
            pixels = pixels + (np.zeros(self.n_channels + 1),)
        self.channel_edges = (
            self.wcs.wcs_pix2world(
                *pixels,
                0,
            )[2]
            * self.units[2]
        )
        return

    def spatial_slices(self):
        """
        Return an iterator over the spatial 'slices' of the cube.

        Returns
        -------
        out : iterator
            Iterator over the spatial 'slices' of the cube.
        """
        s = np.s_[..., 0] if self.stokes_axis else np.s_[...]
        return iter(self._array[s].transpose((2, 0, 1)))

    def spectra(self):
        """
        Return an iterator over the spectra (one in each spatial pixel).

        Returns
        -------
        out : iterator
            Iterator over the spectra (one in each spatial pixel).
        """
        s = np.s_[..., 0] if self.stokes_axis else np.s_[...]
        return iter(self._array[s].reshape(self.n_px_x * self.n_px_y, self.n_channels))

    def freq_channels(self):
        """
        Convert spectral axis to frequency units.
        """
        if self._freq_channel_mode:
            return

        self.wcs.wcs.cdelt[2] = -np.abs(
            (
                (self.wcs.wcs.crval[2] + 0.5 * self.wcs.wcs.cdelt[2]) * self.units[2]
            ).to_value(U.Hz, equivalencies=U.doppler_radio(self.HIfreq))
            - (
                (self.wcs.wcs.crval[2] - 0.5 * self.wcs.wcs.cdelt[2]) * self.units[2]
            ).to_value(U.Hz, equivalencies=U.doppler_radio(self.HIfreq))
        )
        self.wcs.wcs.crval[2] = (self.wcs.wcs.crval[2] * self.units[2]).to_value(
            U.Hz, equivalencies=U.doppler_radio(self.HIfreq)
        )
        self.wcs.wcs.ctype[2] = "FREQ"
        self.units[2] = U.Hz
        self.wcs.wcs.cunit[2] = self.units[2].to_string("fits")
        self._freq_channel_mode = True
        self._channel_mids()
        self._channel_edges()
        return

    def velocity_channels(self):
        """
        Convert spectral axis to velocity units.
        """
        if not self._freq_channel_mode:
            return

        self.wcs.wcs.cdelt[2] = np.abs(
            (
                (self.wcs.wcs.crval[2] - 0.5 * self.wcs.wcs.cdelt[2]) * self.units[2]
            ).to_value(U.m / U.s, equivalencies=U.doppler_radio(self.HIfreq))
            - (
                (self.wcs.wcs.crval[2] + 0.5 * self.wcs.wcs.cdelt[2]) * self.units[2]
            ).to_value(U.m / U.s, equivalencies=U.doppler_radio(self.HIfreq))
        )
        self.wcs.wcs.crval[2] = (self.wcs.wcs.crval[2] * self.units[2]).to_value(
            U.m / U.s, equivalencies=U.doppler_radio(self.HIfreq)
        )
        self.wcs.wcs.ctype[2] = "VRAD"
        self.units[2] = U.m * U.s**-1
        self.wcs.wcs.cunit[2] = self.units[2].to_string("fits")
        self._freq_channel_mode = False
        self._channel_mids()
        self._channel_edges()
        return

    def add_pad(self, pad):
        """
        Resize the cube to add a padding region in the spatial direction.

        Accurate convolution with a beam requires a cube padded according to
        the size of the beam kernel (its representation sampled on a grid with
        the same spacing). The beam class is required to handle defining the
        size of pad required.

        Parameters
        ----------
        pad : 2-tuple (or other sequence)
            Number of pixels to add in the x (RA) and y (Dec) directions.

        See Also
        ----------
        drop_pad
        """

        if self.padx > 0 or self.pady > 0:
            raise RuntimeError("Tried to add padding to already padded datacube array.")
        tmp = self._array
        shape = (self.n_px_x + pad[0] * 2, self.n_px_y + pad[1] * 2, self.n_channels)
        if self.stokes_axis:
            shape = shape + (1,)
        self._array = np.zeros(shape)
        self._array = self._array * tmp.unit
        xregion = np.s_[pad[0] : -pad[0]] if pad[0] > 0 else np.s_[:]
        yregion = np.s_[pad[1] : -pad[1]] if pad[1] > 0 else np.s_[:]
        self._array[xregion, yregion, ...] = tmp
        extend_crpix = [pad[0], pad[1], 0]
        if self.stokes_axis:
            extend_crpix.append(0)
        self.wcs.wcs.crpix += np.array(extend_crpix)
        self.padx, self.pady = pad
        return

    def drop_pad(self):
        """
        Remove the padding added using add_pad.

        After convolution, the pad region contains meaningless information and
        can be discarded.

        See Also
        --------
        add_pad
        """

        if (self.padx == 0) and (self.pady == 0):
            return
        self._array = self._array[self.padx : -self.padx, self.pady : -self.pady, ...]
        retract_crpix = [self.padx, self.pady, 0]
        if self.stokes_axis:
            retract_crpix.append(0)
        self.wcs.wcs.crpix -= np.array(retract_crpix)
        self.padx, self.pady = 0, 0
        return

    def copy(self):
        """
        Produce a copy of the DataCube.

        May be especially useful to create multiple datacubes with differing
        intermediate steps.

        Returns
        -------
        out : DataCube
            Copy of the DataCube object.
        """
        in_freq_channel_mode = self._freq_channel_mode
        if in_freq_channel_mode:
            self.velocity_channels()
        copy = DataCube(
            self.n_px_x,
            self.n_px_y,
            self.n_channels,
            self.px_size,
            self.channel_width,
            self.velocity_centre,
            self.ra,
            self.dec,
        )
        copy.padx, copy.pady = self.padx, self.pady
        copy.wcs = self.wcs
        copy._freq_channel_mode = self._freq_channel_mode
        copy.channel_edges = self.channel_edges
        copy.channel_mids = self.channel_mids
        copy._array = self._array.copy()
        return copy

    def save_state(self, filename, overwrite=False):
        """
        Write a file from which the current DataCube state can be
        re-initialized (see DataCube.load_state). Note that h5py must be
        installed for use. NOT for outputting mock observations, for this
        see Martini.write_fits and Martini.write_hdf5.

        Parameters
        ----------
        filename : str
            File to write.

        overwrite : bool
            Whether to allow overwriting existing files (default: False).

        See Also
        --------
        load_state
        """
        import h5py

        mode = "w" if overwrite else "w-"
        with h5py.File(filename, mode=mode) as f:
            array_unit = self._array.unit
            f["_array"] = self._array.to_value(array_unit)
            f["_array"].attrs["datacube_unit"] = str(array_unit)
            f["_array"].attrs["n_px_x"] = self.n_px_x
            f["_array"].attrs["n_px_y"] = self.n_px_y
            f["_array"].attrs["n_channels"] = self.n_channels
            px_size_unit = self.px_size.unit
            f["_array"].attrs["px_size"] = self.px_size.to_value(px_size_unit)
            f["_array"].attrs["px_size_unit"] = str(px_size_unit)
            channel_width_unit = self.channel_width.unit
            f["_array"].attrs["channel_width"] = self.channel_width.to_value(
                channel_width_unit
            )
            f["_array"].attrs["channel_width_unit"] = str(channel_width_unit)
            velocity_centre_unit = self.velocity_centre.unit
            f["_array"].attrs["velocity_centre"] = self.velocity_centre.to_value(
                velocity_centre_unit
            )
            f["_array"].attrs["velocity_centre_unit"] = str(velocity_centre_unit)
            ra_unit = self.ra.unit
            f["_array"].attrs["ra"] = self.ra.to_value(ra_unit)
            f["_array"].attrs["ra_unit"] = str(ra_unit)
            dec_unit = self.dec.unit
            f["_array"].attrs["dec"] = self.dec.to_value(dec_unit)
            f["_array"].attrs["dec_unit"] = str(self.dec.unit)
            f["_array"].attrs["padx"] = self.padx
            f["_array"].attrs["pady"] = self.pady
            f["_array"].attrs["_freq_channel_mode"] = int(self._freq_channel_mode)
            f["_array"].attrs["stokes_axis"] = self.stokes_axis
        return

    @classmethod
    def load_state(cls, filename):
        """
        Initialize a DataCube from a state saved using DataCube.save_state.
        Note that h5py must be installed for use. Note that ONLY the DataCube
        state is restored, other modules and their configurations are not
        affected.

        Parameters
        ----------
        filename : str
            File to open.

        Returns
        -------
        out : martini.DataCube
            A suitably initialized DataCube object.

        See Also
        --------
        save_state
        """
        import h5py

        with h5py.File(filename, mode="r") as f:
            n_px_x = f["_array"].attrs["n_px_x"]
            n_px_y = f["_array"].attrs["n_px_y"]
            n_channels = f["_array"].attrs["n_channels"]
            px_size = f["_array"].attrs["px_size"] * U.Unit(
                f["_array"].attrs["px_size_unit"]
            )
            channel_width = f["_array"].attrs["channel_width"] * U.Unit(
                f["_array"].attrs["channel_width_unit"]
            )
            velocity_centre = f["_array"].attrs["velocity_centre"] * U.Unit(
                f["_array"].attrs["velocity_centre_unit"]
            )
            ra = f["_array"].attrs["ra"] * U.Unit(f["_array"].attrs["ra_unit"])
            dec = f["_array"].attrs["dec"] * U.Unit(f["_array"].attrs["dec_unit"])
            stokes_axis = bool(f["_array"].attrs["stokes_axis"])
            D = cls(
                n_px_x=n_px_x,
                n_px_y=n_px_y,
                n_channels=n_channels,
                px_size=px_size,
                channel_width=channel_width,
                velocity_centre=velocity_centre,
                ra=ra,
                dec=dec,
                stokes_axis=stokes_axis,
            )
            D._init_wcs()
            D.add_pad((f["_array"].attrs["padx"], f["_array"].attrs["pady"]))
            if bool(f["_array"].attrs["_freq_channel_mode"]):
                D.freq_channels()
            D._array = f["_array"] * U.Unit(f["_array"].attrs["datacube_unit"])
        return D

    def __repr__(self):
        """
        Print the contents of the data cube array itself.

        Returns
        -------
        out : str
            Text representation of the DataCube._array contents.
        """
        return self._array.__repr__()

class Martini:
    """
    Creates synthetic HI data cubes from simulation data.

    Usual use of martini involves first creating instances of classes from each
    of the required and optional sub-modules, then creating a Martini with
    these instances as arguments. The object can then be used to create
    synthetic observations, usually by calling `insert_source_in_cube`,
    (optionally) `add_noise`, (optionally) `convolve_beam` and `write_fits` in
    order.

    Parameters
    ----------
    source : an instance of a class derived from martini.source._BaseSource
        A description of the HI emitting object, including position, geometry
        and an interface to the simulation data (SPH particle masses,
        positions, etc.). Sources leveraging the simobj package for reading
        simulation data (github.com/kyleaoman/simobj) and a few test sources
        (e.g. single particle) are provided, creation of customized sources,
        for instance to leverage other interfaces to simulation data, is
        straightforward. See sub-module documentation.

    datacube : martini.DataCube instance
        A description of the datacube to create, including pixels, channels,
        sky position. See sub-module documentation.

    beam : an instance of a class derived from beams._BaseBeam, optional
        A description of the beam for the simulated telescope. Given a
        description, either mathematical or as an image, the creation of a
        custom beam is straightforward. See sub-module documentation.

    noise : an instance of a class derived from noise._BaseNoise, optional
        A description of the simulated noise. A simple Gaussian noise model is
        provided; implementation of other noise models is straightforward. See
        sub-module documentation.

    sph_kernel : an instance of a class derived from sph_kernels._BaseSPHKernel
        A description of the SPH smoothing kernel. Check simulation
        documentation for the kernel used in a particular simulation, and
        SPH kernel submodule documentation for guidance.

    spectral_model : an instance of a class derived from \
    spectral_models._BaseSpectrum
        A description of the HI line produced by a particle of given
        properties. A Dirac-delta spectrum, and both fixed-width and
        temperature-dependent Gaussian line models are provided; implementing
        other models is straightforward. See sub-module documentation.

    quiet : bool
        If True, suppress output to stdout. (Default: False)

    See Also
    --------
    martini.sources
    martini.DataCube
    martini.beams
    martini.noise
    martini.sph_kernels
    martini.spectral_models

    Examples
    --------
    More detailed examples can be found in the examples directory in the github
    distribution of the package.

    The following example illustrates basic use of martini, using a (very!)
    crude model of a gas disk. This example can be run by doing
    'from martini import demo; demo()'::

        # ------make a toy galaxy----------
        N = 500
        phi = np.random.rand(N) * 2 * np.pi
        r = []
        for L in np.random.rand(N):

            def f(r):
                return L - 0.5 * (2 - np.exp(-r) * (np.power(r, 2) + 2 * r + 2))

            r.append(fsolve(f, 1.0)[0])
        r = np.array(r)
        # exponential disk
        r *= 3 / np.sort(r)[N // 2]
        z = -np.log(np.random.rand(N))
        # exponential scale height
        z *= 0.5 / np.sort(z)[N // 2] * np.sign(np.random.rand(N) - 0.5)
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        xyz_g = np.vstack((x, y, z)) * U.kpc
        # linear rotation curve
        vphi = 100 * r / 6.0
        vx = -vphi * np.sin(phi)
        vy = vphi * np.cos(phi)
        # small pure random z velocities
        vz = (np.random.rand(N) * 2.0 - 1.0) * 5
        vxyz_g = np.vstack((vx, vy, vz)) * U.km * U.s**-1
        T_g = np.ones(N) * 8e3 * U.K
        mHI_g = np.ones(N) / N * 5.0e9 * U.Msun
        # ~mean interparticle spacing smoothing
        hsm_g = np.ones(N) * 4 / np.sqrt(N) * U.kpc
        # ---------------------------------

        source = SPHSource(
            distance=3.0 * U.Mpc,
            rotation={"L_coords": (60.0 * U.deg, 0.0 * U.deg)},
            ra=0.0 * U.deg,
            dec=0.0 * U.deg,
            h=0.7,
            T_g=T_g,
            mHI_g=mHI_g,
            xyz_g=xyz_g,
            vxyz_g=vxyz_g,
            hsm_g=hsm_g,
        )

        datacube = DataCube(
            n_px_x=128,
            n_px_y=128,
            n_channels=32,
            px_size=10.0 * U.arcsec,
            channel_width=10.0 * U.km * U.s**-1,
            velocity_centre=source.vsys,
        )

        beam = GaussianBeam(
            bmaj=30.0 * U.arcsec, bmin=30.0 * U.arcsec, bpa=0.0 * U.deg, truncate=4.0
        )

        noise = GaussianNoise(rms=3.0e-5 * U.Jy * U.beam**-1)

        spectral_model = GaussianSpectrum(sigma=7 * U.km * U.s**-1)

        sph_kernel = CubicSplineKernel()

        M = Martini(
            source=source,
            datacube=datacube,
            beam=beam,
            noise=noise,
            spectral_model=spectral_model,
            sph_kernel=sph_kernel,
        )

        M.insert_source_in_cube()
        M.add_noise()
        M.convolve_beam()
        M.write_beam_fits(beamfile, channels="velocity")
        M.write_fits(cubefile, channels="velocity")
        print(f"Wrote demo fits output to {cubefile}, and beam image to {beamfile}.")
        try:
            M.write_hdf5(hdf5file, channels="velocity")
        except ModuleNotFoundError:
            print("h5py package not present, skipping hdf5 output demo.")
        else:
            print(f"Wrote demo hdf5 output to {hdf5file}.")
    """

    def __init__(
        self,
        source=None,
        datacube=None,
        beam=None,
        noise=None,
        sph_kernel=None,
        spectral_model=None,
        quiet=False,
        find_distance=False,
    ):
        self.quiet = quiet
        self.find_distance = find_distance
        if source is not None:
            self.source = source
        else:
            raise ValueError("A source instance is required.")
        if datacube is not None:
            self.datacube = datacube
        else:
            raise ValueError("A datacube instance is required.")
        self.beam = beam
        self.noise = noise
        if sph_kernel is not None:
            self.sph_kernel = sph_kernel
        else:
            raise ValueError("An SPH kernel instance is required.")
        if spectral_model is not None:
            self.spectral_model = spectral_model
        else:
            raise ValueError("A spectral model instance is required.")

        if self.beam is not None:
            self.beam.init_kernel(self.datacube)
            self.datacube.add_pad(self.beam.needs_pad())

        self.source._init_skycoords()
        self.source._init_pixcoords(self.datacube)  # after datacube is padded

        self.sph_kernel._init_sm_lengths(source=self.source, datacube=self.datacube)
        self.sph_kernel._init_sm_ranges()
        if self.find_distance == False:
            self._prune_particles()  # prunes both source, and kernel if applicable
            self.spectral_model.init_spectra(self.source, self.datacube)
            self.inserted_mass = 0

        return

    def convolve_beam(self):
        """
        Convolve the beam and DataCube.
        """

        if self.beam is None:
            warn("Skipping beam convolution, no beam object provided to " "Martini.")
            return

        unit = self.datacube._array.unit
        for spatial_slice in self.datacube.spatial_slices():
            # use a view [...] to force in-place modification
            spatial_slice[...] = (
                fftconvolve(spatial_slice, self.beam.kernel, mode="same") * unit
            )
        self.datacube.drop_pad()
        self.datacube._array = self.datacube._array.to(
            U.Jy * U.beam**-1,
            equivalencies=U.beam_angular_area(self.beam.area),
        )
        if not self.quiet:
            print(
                "Beam convolved.",
                "  Data cube RMS after beam convolution:"
                f" {np.std(self.datacube._array):.2e}",
                f"  Maximum pixel: {self.datacube._array.max():.2e}",
                "  Median non-zero pixel:"
                f" {np.median(self.datacube._array[self.datacube._array > 0]):.2e}",
                sep="\n",
            )
        return

    def add_noise(self):
        """
        Insert noise into the DataCube.
        """

        if self.noise is None:
            warn("Skipping noise, no noise object provided to Martini.")
            return

        # this unit conversion means noise can be added before or after source insertion:
        noise_cube = (
            self.noise.generate(self.datacube, self.beam)
            .to(
                U.Jy * U.arcsec**-2,
                equivalencies=U.beam_angular_area(self.beam.area),
            )
            .to(self.datacube._array.unit, equivalencies=[self.datacube.arcsec2_to_pix])
        )
        self.datacube._array = self.datacube._array + noise_cube
        if not self.quiet:
            print(
                "Noise added.",
                f"  Noise cube RMS: {np.std(noise_cube):.2e} (before beam convolution).",
                "  Data cube RMS after noise addition (before beam convolution): "
                f"{np.std(self.datacube._array):.2e}",
                sep="\n",
            )
        return

    def _prune_particles(self):
        """
        Determines which particles cannot contribute to the DataCube and
        removes them to speed up calculation. Assumes the kernel is 0 at
        distances greater than the kernel size (which may differ from the
        SPH smoothing length).
        """

        if not self.quiet:
            print(
                f"Source module contained {self.source.npart} particles with total HI"
                f" mass of {self.source.mHI_g.sum():.2e}."
            )
        spectrum_half_width = (
            self.spectral_model.half_width(self.source) / self.datacube.channel_width
        )
        reject_conditions = (
            (
                self.source.pixcoords[:2] + self.sph_kernel.sm_ranges[np.newaxis]
                < 0 * U.pix
            ).any(axis=0),
            self.source.pixcoords[0] - self.sph_kernel.sm_ranges
            > (self.datacube.n_px_x + self.datacube.padx * 2) * U.pix,
            self.source.pixcoords[1] - self.sph_kernel.sm_ranges
            > (self.datacube.n_px_y + self.datacube.pady * 2) * U.pix,
            self.source.pixcoords[2] + 4 * spectrum_half_width * U.pix < 0 * U.pix,
            self.source.pixcoords[2] - 4 * spectrum_half_width * U.pix
            > self.datacube.n_channels * U.pix,
        )
        reject_mask = np.zeros(self.source.pixcoords[0].shape)
        for condition in reject_conditions:
            reject_mask = np.logical_or(reject_mask, condition)
        self.source.apply_mask(np.logical_not(reject_mask))
        # most kernels ignore this line, but required by AdaptiveKernel
        self.sph_kernel._apply_mask(np.logical_not(reject_mask))
        if not self.quiet:
            print(
                f"Pruned particles that will not contribute to data cube, "
                f"{self.source.npart} particles remaining with total HI mass of "
                f"{self.source.mHI_g.sum():.2e}."
            )
        return
    
    def _compute_particles_num(self):
        new_source = self.source
        new_sph_kernel = self.sph_kernel
        initial_npart = self.source.npart
        spectrum_half_width = (
            self.spectral_model.half_width(new_source) / self.datacube.channel_width
        )
        reject_conditions = (
            (
                new_source.pixcoords[:2] + new_sph_kernel.sm_ranges[np.newaxis]
                < 0 * U.pix
            ).any(axis=0),
            new_source.pixcoords[0] - new_sph_kernel.sm_ranges
            > (self.datacube.n_px_x + self.datacube.padx * 2) * U.pix,
            new_source.pixcoords[1] - new_sph_kernel.sm_ranges
            > (self.datacube.n_px_y + self.datacube.pady * 2) * U.pix,
            new_source.pixcoords[2] + 4 * spectrum_half_width * U.pix < 0 * U.pix,
            new_source.pixcoords[2] - 4 * spectrum_half_width * U.pix
            > self.datacube.n_channels * U.pix,
        )
        reject_mask = np.zeros(new_source.pixcoords[0].shape)
        for condition in reject_conditions:
            reject_mask = np.logical_or(reject_mask, condition)
        new_source.apply_mask(np.logical_not(reject_mask))
        # most kernels ignore this line, but required by AdaptiveKernel
        new_sph_kernel._apply_mask(np.logical_not(reject_mask))
        final_npart = new_source.npart
        del new_source
        del new_sph_kernel
        return final_npart / initial_npart * 100

    def _evaluate_pixel_spectrum(self, ranks_and_ij_pxs, progressbar=True):
        """
        Add up contributions of particles to the spectrum in a pixel.

        This is the core loop of MARTINI. It is embarrassingly parallel. To support
        parallel excecution we accept storing up to a copy of the entire (future) datacube
        in one-pixel pieces. This avoids the need for concurrent access to the datacube
        by parallel processes, which would in the simplest case duplicate a copy of the
        datacube array per parallel process! In realistic use cases the memory overhead
        from a the equivalent of a second datacube array should be minimal - memory-
        limited applications should be limited by the memory consumed by particle data,
        which is not duplicated in parallel execution.

        The arguments that differ between parallel ranks must be bundled into one for
        compatibility with `multiprocess`.

        Parameters
        ----------
        rank_and_ij_pxs : tuple
            A 2-tuple containing an integer (cpu "rank" in the case of parallel execution)
            and a list of 2-tuples specifying the indices (i, j) of pixels in the grid.

        Returns
        -------
        out : list
            A list containing 2-tuples. Each 2-tuple contains and "insertion slice" that
            is an index into the datacube._array instance held by this martini instance
            where the pixel spectrum is to be placed, and a 1D array containing the
            spectrum, whose length must match the length of the spectral axis of the
            datacube.
        """
        result = list()
        rank, ij_pxs = ranks_and_ij_pxs
        if progressbar:
            ij_pxs = tqdm(ij_pxs, position=rank)
        for ij_px in ij_pxs:
            ij = np.array(ij_px)[..., np.newaxis] * U.pix
            mask = (
                np.abs(ij - self.source.pixcoords[:2]) <= self.sph_kernel.sm_ranges
            ).all(axis=0)
            weights = self.sph_kernel._px_weight(
                self.source.pixcoords[:2, mask] - ij, mask=mask
            )
            insertion_slice = (
                np.s_[ij_px[0], ij_px[1], :, 0]
                if self.datacube.stokes_axis
                else np.s_[ij_px[0], ij_px[1], :]
            )
            result.append(
                (
                    insertion_slice,
                    (self.spectral_model.spectra[mask] * weights[..., np.newaxis]).sum(
                        axis=-2
                    ),
                )
            )
        return result

    def _insert_pixel(self, insertion_slice, insertion_data):
        """
        Insert the spectrum for a single pixel into the datacube array.

        Parameters
        ----------
        insertion_slice : integer, tuple or slice
            Index into the datacube's _array specifying the insertion location.
        insertion data : array-like
            1D array containing the spectrum at the location specified by insertion_slice.
        """
        self.datacube._array[insertion_slice] = insertion_data
        return

    def insert_source_in_cube(self, skip_validation=False, progressbar=None, ncpu=1):
        """
        Populates the DataCube with flux from the particles in the source.

        Parameters
        ----------
        skip_validation : bool, optional
            SPH kernel interpolation onto the DataCube is approximated for
            increased speed. For some combinations of pixel size, distance
            and SPH smoothing length, the approximation may break down. The
            kernel class will check whether this will occur and raise a
            RuntimeError if so. This validation can be skipped (at the cost
            of accuracy!) by setting this parameter True. (Default: False.)

        progressbar : bool, optional
            A progress bar is shown by default. Progress bars work, with perhaps
            some visual glitches, in parallel. If martini was initialised with
            `quiet` set to `True`, progress bars are switched off unless explicitly
            turned on. (Default: None.)

        ncpu : int
            Number of processes to use in main source insertion loop. Using more than
            one cpu requires the `multiprocess` module (n.b. not the same as
            `multiprocessing`). (Default: 1)

        """

        assert self.spectral_model.spectra is not None

        if progressbar is None:
            progressbar = not self.quiet

        self.sph_kernel._confirm_validation(noraise=skip_validation, quiet=self.quiet)

        ij_pxs = list(
            product(
                np.arange(self.datacube._array.shape[0]),
                np.arange(self.datacube._array.shape[1]),
            )
        )

        if ncpu == 1:
            for insertion_slice, insertion_data in self._evaluate_pixel_spectrum(
                (0, ij_pxs), progressbar=progressbar
            ):
                self._insert_pixel(insertion_slice, insertion_data)
        else:
            # not multiprocessing, need serialization from dill not pickle
            from multiprocess import Pool

            with Pool(processes=ncpu) as pool:
                for result in pool.imap_unordered(
                    lambda x: self._evaluate_pixel_spectrum(x, progressbar=progressbar),
                    [(icpu, ij_pxs[icpu::ncpu]) for icpu in range(ncpu)],
                ):
                    for insertion_slice, insertion_data in result:
                        self._insert_pixel(insertion_slice, insertion_data)

        self.datacube._array = self.datacube._array.to(
            U.Jy / U.arcsec**2, equivalencies=[self.datacube.arcsec2_to_pix]
        )
        pad_mask = (
            np.s_[
                self.datacube.padx : -self.datacube.padx,
                self.datacube.pady : -self.datacube.pady,
                ...,
            ]
            if self.datacube.padx > 0 and self.datacube.pady > 0
            else np.s_[...]
        )
        inserted_flux = (
            self.datacube._array[pad_mask].sum() * self.datacube.px_size**2
        )
        inserted_mass = (
            2.36e5
            * U.Msun
            * self.source.distance.to_value(U.Mpc) ** 2
            * inserted_flux.to_value(U.Jy)
            * self.datacube.channel_width.to_value(U.km / U.s)
        )
        self.inserted_mass = inserted_mass
        if not self.quiet:
            print(
                "Source inserted.",
                f"  Flux in cube: {inserted_flux:.2e}",
                f"  Mass in cube (assuming distance {self.source.distance:.2f}):"
                f" {inserted_mass:.2e}",
                f"    [{inserted_mass / self.source.input_mass * 100:.0f}%"
                f" of initial source mass]",
                f"  Maximum pixel: {self.datacube._array.max():.2e}",
                "  Median non-zero pixel:"
                f" {np.median(self.datacube._array[self.datacube._array > 0]):.2e}",
                sep="\n",
            )
        return

    def write_fits(
        self,
        filename,
        channels="frequency",
        overwrite=True,
    ):
        """
        Output the DataCube to a FITS-format file.

        Parameters
        ----------
        filename : string
            Name of the file to write. '.fits' will be appended if not already
            present.

        channels : {'frequency', 'velocity'}, optional
            Type of units used along the spectral axis in output file.
            (Default: 'frequency'.)

        overwrite: bool, optional
            Whether to allow overwriting existing files. (Default: True.)
        """

        self.datacube.drop_pad()
        if channels == "frequency":
            self.datacube.freq_channels()
        elif channels == "velocity":
            self.datacube.velocity_channels()
        else:
            raise ValueError(
                "Martini.write_fits: Unknown 'channels' value "
                "(use 'frequency' or 'velocity')."
            )

        filename = filename if filename[-5:] == ".fits" else filename + ".fits"

        wcs_header = self.datacube.wcs.to_header()
        wcs_header.rename_keyword("WCSAXES", "NAXIS")

        header = fits.Header()
        header.append(("SIMPLE", "T"))
        header.append(("BITPIX", 16))
        header.append(("NAXIS", wcs_header["NAXIS"]))
        header.append(("NAXIS1", self.datacube.n_px_x))
        header.append(("NAXIS2", self.datacube.n_px_y))
        header.append(("NAXIS3", self.datacube.n_channels))
        if self.datacube.stokes_axis:
            header.append(("NAXIS4", 1))
        header.append(("EXTEND", "T"))
        header.append(("CDELT1", wcs_header["CDELT1"]))
        header.append(("CRPIX1", wcs_header["CRPIX1"]))
        header.append(("CRVAL1", wcs_header["CRVAL1"]))
        header.append(("CTYPE1", wcs_header["CTYPE1"]))
        header.append(("CUNIT1", wcs_header["CUNIT1"]))
        header.append(("CDELT2", wcs_header["CDELT2"]))
        header.append(("CRPIX2", wcs_header["CRPIX2"]))
        header.append(("CRVAL2", wcs_header["CRVAL2"]))
        header.append(("CTYPE2", wcs_header["CTYPE2"]))
        header.append(("CUNIT2", wcs_header["CUNIT2"]))
        header.append(("CDELT3", wcs_header["CDELT3"]))
        header.append(("CRPIX3", wcs_header["CRPIX3"]))
        header.append(("CRVAL3", wcs_header["CRVAL3"]))
        header.append(("CTYPE3", wcs_header["CTYPE3"]))
        header.append(("CUNIT3", wcs_header["CUNIT3"]))
        if self.datacube.stokes_axis:
            header.append(("CDELT4", wcs_header["CDELT4"]))
            header.append(("CRPIX4", wcs_header["CRPIX4"]))
            header.append(("CRVAL4", wcs_header["CRVAL4"]))
            header.append(("CTYPE4", wcs_header["CTYPE4"]))
            header.append(("CUNIT4", "PAR"))
        header.append(("EPOCH", 2000))
        header.append(("INSTRUME", "MARTINI", martini_version))
        # header.append(('BLANK', -32768)) #only for integer data
        header.append(("BSCALE", 1.0))
        header.append(("BZERO", 0.0))
        datacube_array_units = self.datacube._array.unit
        header.append(
            ("DATAMAX", np.max(self.datacube._array.to_value(datacube_array_units)))
        )
        header.append(
            ("DATAMIN", np.min(self.datacube._array.to_value(datacube_array_units)))
        )
        header.append(("ORIGIN", "astropy v" + astropy_version))
        # long names break fits format, don't let the user set this
        header.append(("OBJECT", "MOCK"))
        if self.beam is not None:
            header.append(("BPA", self.beam.bpa.to_value(U.deg)))
        header.append(("OBSERVER", "K. Oman"))
        # header.append(('NITERS', ???))
        # header.append(('RMS', ???))
        # header.append(('LWIDTH', ???))
        # header.append(('LSTEP', ???))
        header.append(("BUNIT", datacube_array_units.to_string("fits")))
        # header.append(('PCDEC', ???))
        # header.append(('LSTART', ???))
        header.append(("DATE-OBS", Time.now().to_value("fits")))
        # header.append(('LTYPE', ???))
        # header.append(('PCRA', ???))
        # header.append(('CELLSCAL', ???))
        if self.beam is not None:
            header.append(("BMAJ", self.beam.bmaj.to_value(U.deg)))
            header.append(("BMIN", self.beam.bmin.to_value(U.deg)))
        header.append(("BTYPE", "Intensity"))
        header.append(("SPECSYS", wcs_header["SPECSYS"]))

        # flip axes to write
        hdu = fits.PrimaryHDU(
            header=header, data=self.datacube._array.to_value(datacube_array_units).T
        )
        hdu.writeto(filename, overwrite=overwrite)

        if channels == "frequency":
            self.datacube.velocity_channels()
        return

    def write_beam_fits(self, filename, channels="frequency", overwrite=True):
        """
        Output the beam to a FITS-format file.

        The beam is written to file, with pixel sizes, coordinate system, etc.
        similar to those used for the DataCube.

        Parameters
        ----------
        filename : string
            Name of the file to write. '.fits' will be appended if not already
            present.

        channels : {'frequency', 'velocity'}, optional
            Type of units used along the spectral axis in output file.
            (Default: 'frequency'.)

        overwrite: bool, optional
            Whether to allow overwriting existing files. (Default: True.)

        Raises
        ------
        ValueError
            If Martini was initialized without a beam.
        """

        if self.beam is None:
            raise ValueError(
                "Martini.write_beam_fits: Called with beam set " "to 'None'."
            )
        assert self.beam.kernel is not None
        if channels == "frequency":
            self.datacube.freq_channels()
        elif channels == "velocity":
            self.datacube.velocity_channels()
        else:
            raise ValueError(
                "Martini.write_beam_fits: Unknown 'channels' "
                "value (use 'frequency' or 'velocity'."
            )

        filename = filename if filename[-5:] == ".fits" else filename + ".fits"

        wcs_header = self.datacube.wcs.to_header()

        beam_kernel_units = self.beam.kernel.unit
        header = fits.Header()
        header.append(("SIMPLE", "T"))
        header.append(("BITPIX", 16))
        # header.append(('NAXIS', self.beam.kernel.ndim))
        header.append(("NAXIS", 3))
        header.append(("NAXIS1", self.beam.kernel.shape[0]))
        header.append(("NAXIS2", self.beam.kernel.shape[1]))
        header.append(("NAXIS3", 1))
        header.append(("EXTEND", "T"))
        header.append(("BSCALE", 1.0))
        header.append(("BZERO", 0.0))
        # this is 1/arcsec^2, is this right?
        header.append(("BUNIT", beam_kernel_units.to_string("fits")))
        header.append(("CRPIX1", self.beam.kernel.shape[0] // 2 + 1))
        header.append(("CDELT1", wcs_header["CDELT1"]))
        header.append(("CRVAL1", wcs_header["CRVAL1"]))
        header.append(("CTYPE1", wcs_header["CTYPE1"]))
        header.append(("CUNIT1", wcs_header["CUNIT1"]))
        header.append(("CRPIX2", self.beam.kernel.shape[1] // 2 + 1))
        header.append(("CDELT2", wcs_header["CDELT2"]))
        header.append(("CRVAL2", wcs_header["CRVAL2"]))
        header.append(("CTYPE2", wcs_header["CTYPE2"]))
        header.append(("CUNIT2", wcs_header["CUNIT2"]))
        header.append(("CRPIX3", 1))
        header.append(("CDELT3", wcs_header["CDELT3"]))
        header.append(("CRVAL3", wcs_header["CRVAL3"]))
        header.append(("CTYPE3", wcs_header["CTYPE3"]))
        header.append(("CUNIT3", wcs_header["CUNIT3"]))
        header.append(("SPECSYS", wcs_header["SPECSYS"]))
        header.append(("BMAJ", self.beam.bmaj.to_value(U.deg)))
        header.append(("BMIN", self.beam.bmin.to_value(U.deg)))
        header.append(("BPA", self.beam.bpa.to_value(U.deg)))
        header.append(("BTYPE", "beam    "))
        header.append(("EPOCH", 2000))
        header.append(("OBSERVER", "K. Oman"))
        # long names break fits format
        header.append(("OBJECT", "MOCKBEAM"))
        header.append(("INSTRUME", "MARTINI", martini_version))
        header.append(("DATAMAX", np.max(self.beam.kernel.to_value(beam_kernel_units))))
        header.append(("DATAMIN", np.min(self.beam.kernel.to_value(beam_kernel_units))))
        header.append(("ORIGIN", "astropy v" + astropy_version))

        # flip axes to write
        hdu = fits.PrimaryHDU(
            header=header,
            data=self.beam.kernel.to_value(beam_kernel_units)[..., np.newaxis].T,
        )
        hdu.writeto(filename, overwrite=True)

        if channels == "frequency":
            self.datacube.velocity_channels()
        return

    def write_hdf5(
        self,
        filename,
        channels="frequency",
        overwrite=True,
        memmap=False,
        compact=False,
    ):
        """
        Output the DataCube and Beam to a HDF5-format file. Requires the h5py
        package.

        Parameters
        ----------
        filename : string
            Name of the file to write. '.hdf5' will be appended if not already
            present.

        channels : {'frequency', 'velocity'}, optional
            Type of units used along the spectral axis in output file.
            (Default: 'frequency'.)

        overwrite: bool, optional
            Whether to allow overwriting existing files. (Default: True.)

        memmap: bool, optional
            If True, create a file-like object in memory and return it instead
            of writing file to disk. (Default: False.)

        compact: bool, optional
            If True, omit pixel coordinate arrays to save disk space. In this
            case pixel coordinates can still be reconstructed from FITS-style
            keywords stored in the FluxCube attributes. (Default: False.)
        """

        import h5py

        self.datacube.drop_pad()
        if channels == "frequency":
            self.datacube.freq_channels()
        elif channels == "velocity":
            pass
        else:
            raise ValueError(
                "Martini.write_fits: Unknown 'channels' value "
                "(use 'frequency' or 'velocity')."
            )

        filename = filename if filename[-5:] == ".hdf5" else filename + ".hdf5"

        wcs_header = self.datacube.wcs.to_header()

        mode = "w" if overwrite else "x"
        driver = "core" if memmap else None
        h5_kwargs = {"backing_store": False} if memmap else dict()
        f = h5py.File(filename, mode, driver=driver, **h5_kwargs)
        datacube_array_units = self.datacube._array.unit
        s = np.s_[..., 0] if self.datacube.stokes_axis else np.s_[...]
        f["FluxCube"] = self.datacube._array.to_value(datacube_array_units)[s]
        c = f["FluxCube"]
        origin = 0  # index from 0 like numpy, not from 1
        if not compact:
            xgrid, ygrid, vgrid = np.meshgrid(
                np.arange(self.datacube._array.shape[0]),
                np.arange(self.datacube._array.shape[1]),
                np.arange(self.datacube._array.shape[2]),
            )
            cgrid = (
                np.vstack(
                    (
                        xgrid.flatten(),
                        ygrid.flatten(),
                        vgrid.flatten(),
                        np.zeros(vgrid.shape).flatten(),
                    )
                ).T
                if self.datacube.stokes_axis
                else np.vstack(
                    (
                        xgrid.flatten(),
                        ygrid.flatten(),
                        vgrid.flatten(),
                    )
                ).T
            )
            wgrid = self.datacube.wcs.all_pix2world(cgrid, origin)
            ragrid = wgrid[:, 0].reshape(self.datacube._array.shape)[s]
            decgrid = wgrid[:, 1].reshape(self.datacube._array.shape)[s]
            chgrid = wgrid[:, 2].reshape(self.datacube._array.shape)[s]
            f["RA"] = ragrid
            f["RA"].attrs["Unit"] = wcs_header["CUNIT1"]
            f["Dec"] = decgrid
            f["Dec"].attrs["Unit"] = wcs_header["CUNIT2"]
            f["channel_mids"] = chgrid
            f["channel_mids"].attrs["Unit"] = wcs_header["CUNIT3"]
        c.attrs["AxisOrder"] = "(RA,Dec,Channels)"
        c.attrs["FluxCubeUnit"] = str(self.datacube._array.unit)
        c.attrs["deltaRA_in_RAUnit"] = wcs_header["CDELT1"]
        c.attrs["RA0_in_px"] = wcs_header["CRPIX1"] - 1
        c.attrs["RA0_in_RAUnit"] = wcs_header["CRVAL1"]
        c.attrs["RAUnit"] = wcs_header["CUNIT1"]
        c.attrs["RAProjType"] = wcs_header["CTYPE1"]
        c.attrs["deltaDec_in_DecUnit"] = wcs_header["CDELT2"]
        c.attrs["Dec0_in_px"] = wcs_header["CRPIX2"] - 1
        c.attrs["Dec0_in_DecUnit"] = wcs_header["CRVAL2"]
        c.attrs["DecUnit"] = wcs_header["CUNIT2"]
        c.attrs["DecProjType"] = wcs_header["CTYPE2"]
        c.attrs["deltaV_in_VUnit"] = wcs_header["CDELT3"]
        c.attrs["V0_in_px"] = wcs_header["CRPIX3"] - 1
        c.attrs["V0_in_VUnit"] = wcs_header["CRVAL3"]
        c.attrs["VUnit"] = wcs_header["CUNIT3"]
        c.attrs["VProjType"] = wcs_header["CTYPE3"]
        if self.beam is not None:
            c.attrs["BeamPA"] = self.beam.bpa.to_value(U.deg)
            c.attrs["BeamMajor_in_deg"] = self.beam.bmaj.to_value(U.deg)
            c.attrs["BeamMinor_in_deg"] = self.beam.bmin.to_value(U.deg)
        c.attrs["DateCreated"] = str(Time.now())
        #c.attrs["MartiniVersion"] = martini_version
        #c.attrs["AstropyVersion"] = astropy_version
        if self.beam is not None:
            if self.beam.kernel is None:
                raise ValueError(
                    "Martini.write_hdf5: Called with beam present but beam kernel"
                    " uninitialized."
                )
            beam_kernel_units = self.beam.kernel.unit
            f["Beam"] = self.beam.kernel.to_value(beam_kernel_units)[..., np.newaxis]
            b = f["Beam"]
            b.attrs["BeamUnit"] = self.beam.kernel.unit.to_string("fits")
            b.attrs["deltaRA_in_RAUnit"] = wcs_header["CDELT1"]
            b.attrs["RA0_in_px"] = self.beam.kernel.shape[0] // 2
            b.attrs["RA0_in_RAUnit"] = wcs_header["CRVAL1"]
            b.attrs["RAUnit"] = wcs_header["CUNIT1"]
            b.attrs["RAProjType"] = wcs_header["CTYPE1"]
            b.attrs["deltaDec_in_DecUnit"] = wcs_header["CDELT2"]
            b.attrs["Dec0_in_px"] = self.beam.kernel.shape[1] // 2
            b.attrs["Dec0_in_DecUnit"] = wcs_header["CRVAL2"]
            b.attrs["DecUnit"] = wcs_header["CUNIT2"]
            b.attrs["DecProjType"] = wcs_header["CTYPE2"]
            b.attrs["deltaV_in_VUnit"] = wcs_header["CDELT3"]
            b.attrs["V0_in_px"] = 0
            b.attrs["V0_in_VUnit"] = wcs_header["CRVAL3"]
            b.attrs["VUnit"] = wcs_header["CUNIT3"]
            b.attrs["VProjType"] = wcs_header["CTYPE3"]
            b.attrs["BeamPA"] = self.beam.bpa.to_value(U.deg)
            b.attrs["BeamMajor_in_deg"] = self.beam.bmaj.to_value(U.deg)
            b.attrs["BeamMinor_in_deg"] = self.beam.bmin.to_value(U.deg)
            b.attrs["DateCreated"] = str(Time.now())
            #b.attrs["MartiniVersion"] = martini_version
            #b.attrs["AstropyVersion"] = astropy_version

        if channels == "frequency":
            self.datacube.velocity_channels()
        if memmap:
            return f
        else:
            f.close()
            return

    def reset(self):
        """
        Re-initializes the DataCube with zero-values.
        """
        init_kwargs = dict(
            n_px_x=self.datacube.n_px_x,
            n_px_y=self.datacube.n_px_y,
            n_channels=self.datacube.n_channels,
            px_size=self.datacube.px_size,
            channel_width=self.datacube.channel_width,
            velocity_centre=self.datacube.velocity_centre,
            ra=self.datacube.ra,
            dec=self.datacube.dec,
            stokes_axis=self.datacube.stokes_axis,
        )
        self.datacube = DataCube(**init_kwargs)
        if self.beam is not None:
            self.datacube.add_pad(self.beam.needs_pad())
        return

# ------------------------ #
def plot_moments(FluxCube, vch, path):
    np.seterr(all='ignore')
    fig = plt.figure(figsize=(16, 5))
    sp1 = fig.add_subplot(1,3,1)
    sp2 = fig.add_subplot(1,3,2)
    sp3 = fig.add_subplot(1,3,3)
    rms = np.std(FluxCube[:16, :16])  # noise in a corner patch where there is little signal
    clip = np.where(FluxCube > 5 * rms, 1, 0)
    mom0 = np.sum(FluxCube, axis=-1)
    mask = np.where(mom0 > .02, 1, np.nan)
    mask = np.ones_like(mom0)
    mom1 = np.sum(FluxCube * clip * vch, axis=-1) / mom0
    mom2 = np.sqrt(np.sum(FluxCube * clip * np.power(vch - mom1[..., np.newaxis], 2), axis=-1)) / mom0
    im1 = sp1.imshow(mom0.T, cmap='Greys', aspect=1.0, origin='lower')
    plt.colorbar(im1, ax=sp1, label='mom0 [Jy/beam]')
    im2 = sp2.imshow((mom1*mask).T, cmap='RdBu', aspect=1.0, origin='lower')
    plt.colorbar(im2, ax=sp2, label='mom1 [km/s]')
    im3 = sp3.imshow((mom2*mask).T, cmap='magma', aspect=1.0, origin='lower', vmin=0, vmax=300)
    plt.colorbar(im3, ax=sp3, label='mom2 [km/s]')
    for sp in sp1, sp2, sp3:
        sp.set_xlabel('x [px = arcsec/10]')
        sp.set_ylabel('y [px = arcsec/10]')
    plt.subplots_adjust(wspace=.3)
    plt.savefig(path)

def _gen_particle_coords(source, datacube):
    # pixels indexed from 0 (not like in FITS!) for better use with numpy
    origin = 0
    skycoords = source.sky_coordinates
    return (
        np.vstack(
            datacube.wcs.sub(3).wcs_world2pix(
                skycoords.ra.to(datacube.units[0]),
                skycoords.dec.to(datacube.units[1]),
                skycoords.radial_velocity.to(datacube.units[2]),
                origin,
            )
        )
        * U.pix
    )

def get_distance(n_px, n_channels, 
                 x_rot, y_rot, TNGSnapshotID, TNGSubhaloID, 
                 TNGBasePath, factor):
    
    distance = 1 * U.Mpc
    i = 0
    source = myTNGSource(snapNum=TNGSnapshotID, 
                        subID=TNGSubhaloID,
                        distance=distance,
                        rotation = {'L_coords': (x_rot, y_rot)},
                        basePath = TNGBasePath,
                        ra = 0. * U.deg,
                        dec = 0. * U.deg,)
    datacube = DataCube(
        n_px_x = n_px,
        n_px_y = n_px,
        n_channels = n_channels, 
        px_size = 10.0 * U.arcsec,
        channel_width=10.0 * U.km * U.s**-1,
        velocity_centre=source.vsys, 
        ra = source.ra,
        dec = source.dec,
    )
    coordinates = _gen_particle_coords(source, datacube)
    min_x, max_x = np.min(coordinates[0,:]), np.max(coordinates[0,:])
    min_y, max_y = np.min(coordinates[1,:]), np.max(coordinates[1,:])
    min_z, max_z = np.min(coordinates[2,:]), np.max(coordinates[2,:])
    dist_x = max_x - min_x
    dist_y = max_y - min_y
    dist_z = max_z - min_z
    while (dist_x > factor * n_px* U.pix) or (dist_y > factor * n_px* U.pix):
        i += 1
        if i < 10:
            distance += 10 * U.Mpc
        elif i > 10 and i < 20:
            distance += 100 * U.Mpc
        source = myTNGSource(
                        snapNum=TNGSnapshotID, 
                        subID=TNGSubhaloID,
                        distance=distance,
                        rotation = {'L_coords': (x_rot, y_rot)},
                        basePath = TNGBasePath,
                        ra = 0. * U.deg,
                        dec = 0. * U.deg,)
        datacube = DataCube(
            n_px_x = n_px,
            n_px_y = n_px,
            n_channels = n_channels, 
            px_size = 10.0 * U.arcsec,
            channel_width=10.0 * U.km * U.s**-1,
            velocity_centre=source.vsys, 
            ra = source.ra,
            dec = source.dec,
        )
        coordinates = _gen_particle_coords(source, datacube)
        min_x, max_x = np.min(coordinates[0,:]), np.max(coordinates[0,:])
        min_y, max_y = np.min(coordinates[1,:]), np.max(coordinates[1,:])
        dist_x = max_x - min_x
        dist_y = max_y - min_y
    source = myTNGSource(
                        snapNum=TNGSnapshotID, 
                        subID=TNGSubhaloID,
                        distance=distance,
                        rotation = {'L_coords': (x_rot, y_rot)},
                        basePath = TNGBasePath,
                        ra = 0. * U.deg,
                        dec = 0. * U.deg,)
    channel_width=10.0 * U.km * U.s**-1
    while dist_z > factor * n_channels * U.pix:
        channel_width += 1.0 * U.km * U.s**-1
        datacube = DataCube(
            n_px_x = n_px,
            n_px_y = n_px,
            n_channels = n_channels, 
            px_size = 10.0 * U.arcsec,
            channel_width=channel_width,
            velocity_centre=source.vsys, 
            ra = source.ra,
            dec = source.dec,
        )
        coordinates = _gen_particle_coords(source, datacube)
        min_z, max_z = np.min(coordinates[2,:]), np.max(coordinates[2,:])
        dist_z = max_z - min_z
        
    return distance, channel_width

def get_mem_gb():
    mem_bytes = psutil.virtual_memory ().total # total physical memory in bytes
    mem_gib = mem_bytes / (1024.**3) # convert to gigabytes
    return str(mem_gib)

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

def simulate_atmospheric_noise(project, scale, ms, antennalist):
    zztot, frefant = get_antennas_distances_from_reference(antennalist)
    gaincal(
        vis=ms,
        caltable=project + "_atmosphere.gcal",
        refant=str(frefant), #name of the reference antenna
        minsnr=0.01, #ignore solution with SNR below this
        calmode="p", #phase
        solint='inf', #solution interval
    )
    tb = table()
    tb.open(project + "_atmosphere.gcal", nomodify=False)
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
        nycparam[0][0][i] = 1.0*np.complex(rperror,iperror)  #X POL
        nycparam[1][0][i] = 1.0*np.complex(rperror,iperror)  #Y POL  ASSUMED SAME
    tb.putcol('CPARAM', nycparam)
    tb.flush()
    tb.close()
    applycal(
        vis = ms,
        gaintable = project + "_atmosphere.gcal"
    )
    os.system("rm -rf " + project + "_atmosphere.gcal")
    return 

def simulate_gain_errors(ms, amplitude: float = 0.01):
    sm = casa_simulator()
    sm.openfromms(ms)
    sm.setseed(42)
    sm.setgain(mode='fbm', amplitude=[amplitude])
    sm.corrupt()
    sm.close()
    return

def simulate_antenna_position_errors(antenna_list, amplitude):
    
    
    return

def check_parameters(i, data_dir, main_path, project_name, output_dir, plot_dir, band, antenna_name
                     , inbright, bandwidth, inwidth, integration, totaltime, ra, dec, pwv, rest_frequency, snr,
                     get_skymodel, source_type, TNGBasePath, TNGSnapshotID, TNGSubhaloID, plot, save_ms, save_psf,
                     save_pb, crop, serendipitous, n_pxs, n_channels):
    if isinstance(i, list) or isinstance(i, np.ndarray):
        i = i[0]
    if isinstance(data_dir, list) or isinstance(data_dir, np.ndarray):
        data_dir = data_dir[0]
    if isinstance(main_path, list) or isinstance(main_path, np.ndarray):
        main_path = main_path[0]
    if isinstance(project_name, list) or isinstance(project_name, np.ndarray):
        project_name = project_name[0]
    if isinstance(output_dir, list) or isinstance(output_dir, np.ndarray):
        output_dir = output_dir[0]
    if isinstance(plot_dir, list) or isinstance(plot_dir, np.ndarray):
        plot_dir = plot_dir[0]
    if isinstance(band, list) or isinstance(band, np.ndarray):
        band = band[0]
    if isinstance(antenna_name, list) or isinstance(antenna_name, np.ndarray):
        antenna_name = antenna_name[0]
    if isinstance(inbright, list) or isinstance(inbright, np.ndarray):
        inbright = inbright[0]
    if isinstance(bandwidth, list) or isinstance(bandwidth, np.ndarray):
        bandwidth = bandwidth[0]
    if isinstance(inwidth, list) or isinstance(inwidth, np.ndarray):
        inwidth = inwidth[0]
    if isinstance(integration, list) or isinstance(integration, np.ndarray):
        integration = integration[0]
    if isinstance(totaltime, list) or isinstance(totaltime, np.ndarray):
        totaltime = totaltime[0]
    if isinstance(ra, list) or isinstance(ra, np.ndarray):
        ra = ra[0]
    if isinstance(dec, list) or isinstance(dec, np.ndarray):
        dec = dec[0]
    if isinstance(pwv, list) or isinstance(pwv, np.ndarray):
        pwv = pwv[0]
    if isinstance(rest_frequency, list) or isinstance(rest_frequency, np.ndarray):
        rest_frequency = rest_frequency[0]
    if isinstance(snr, list) or isinstance(snr, np.ndarray):
        snr = snr[0]
    if isinstance(get_skymodel, list) or isinstance(get_skymodel, np.ndarray):
        get_skymodel = get_skymodel[0]
    if isinstance(source_type, list) or isinstance(source_type, np.ndarray):
        source_type = source_type[0]
    if isinstance(TNGBasePath, list) or isinstance(TNGBasePath, np.ndarray):
        TNGBasePath = TNGBasePath[0]
    if isinstance(TNGSnapshotID, list) or isinstance(TNGSnapshotID, np.ndarray):
        TNGSnapshotID = TNGSnapshotID[0]
    if isinstance(TNGSubhaloID, list) or isinstance(TNGSubhaloID, np.ndarray):
        TNGSubhaloID = TNGSubhaloID[0]
    if isinstance(plot, list) or isinstance(plot, np.ndarray):
        plot = plot[0]
    if isinstance(save_ms, list) or isinstance(save_ms, np.ndarray):
        save_ms = save_ms[0]
    if isinstance(crop, list) or isinstance(crop, np.ndarray):
        crop = crop[0]
    if isinstance(n_pxs, list) or isinstance(n_pxs, np.ndarray):
        n_pxs = n_pxs[0]
    if isinstance(n_channels, list) or isinstance(n_channels, np.ndarray):
        n_channels = n_channels[0]
    if isinstance(save_psf, list) or isinstance(save_psf, np.ndarray):
        save_psf = save_psf[0]
    if isinstance(save_pb, list) or isinstance(save_pb, np.ndarray):
        save_pb = save_pb[0]
    if isinstance(serendipitous, list) or isinstance(serendipitous, np.ndarray):
        serendipitous = serendipitous[0]
    return i, data_dir, main_path, project_name, output_dir, plot_dir, band, antenna_name, inbright, bandwidth, inwidth, integration, totaltime, ra, dec, pwv, rest_frequency, snr, get_skymodel, source_type, TNGBasePath, TNGSnapshotID, TNGSubhaloID, plot, save_ms, save_psf, save_pb, crop, serendipitous, n_pxs, n_channels

def write_simulation_parameters(path, band, bandwidth, central_freq, cell_size, fov, 
                                spatial_resolution, cycle, antenna_name, inbright, beam_size, 
                                integration, totaltime, TNGBasePath, TNGSnapshotID, TNGSubhaloID, 
                                n_px, n_channels):
  """
  Writes simulation parameters to a text file at the specified path.

  Args:
    path: The path to save the text file.
    band: The band name (e.g., 'L' band).
    bandwidth: Bandwidth in MHz.
    central_freq: Central frequency in GHz.
    cell_size: Pixel size in arcseconds.
    fov: Field of view in arcseconds.
    spatial_resolution: Spatial resolution in arcseconds.
    cycle: Cycle number.
    antenna_name: Antenna configuration name.
    inbright: Brightness in Jy/px.
    beam_size: Beam size in arcseconds.
    integration: Integration time in seconds.
    totaltime: Total time in seconds.
    TNGBasePath: Path to TNG base directory.
    TNGSnapshotID: TNG snapshot ID.
    TNGSubhaloID: TNG subhalo ID.
    n_px: Number of pixels in each dimension of the cube.
    n_channels: Number of channels.
  """

  with open(path, 'w') as f:
    f.write('Simulation Parameters:\n')
    f.write('Band: {}\n'.format(band))
    f.write('Bandwidth: {} MHz\n'.format(bandwidth))
    f.write('Central Frequency: {} GHz\n'.format(central_freq))
    f.write('Pixel size: {} arcsec\n'.format(cell_size))
    f.write('Fov: {} arcsec\n'.format(fov))
    f.write('Spatial_resolution: {} arcsec\n'.format(spatial_resolution))
    f.write('Cycle: {}\n'.format(cycle))
    f.write('Antenna Configuration: {}\n'.format(antenna_name))
    f.write('Inbright: {} Jy/px\n'.format(inbright))
    f.write('Beam Size: {} arcsec\n'.format(beam_size))
    f.write('Integration Time: {} s\n'.format(integration))
    f.write('Total Time: {} s\n'.format(totaltime))
    f.write('TNG Base Path: {}\n'.format(TNGBasePath))
    f.write('TNG Snapshot ID: {}\n'.format(TNGSnapshotID))
    f.write('TNG Subhalo ID: {}\n'.format(TNGSubhaloID))
    f.write('Cube Size: {} x {} x {} pixels\n'.format(n_px, n_px, n_channels))
    f.close()

def simulator(i: int, data_dir: str, main_path: str, project_name: str, 
              output_dir: str, plot_dir: str, band: int, antenna_name: str, inbright: float, 
              bandwidth: int, inwidth: float, integration: int, totaltime: int, ra: float, dec: float,
              pwv: float, rest_frequency: float, snr: float, get_skymodel: bool, 
              source_type: str, TNGBasePath: str, TNGSnapshotID: int, TNGSubhaloID: int,
              plot: bool, save_ms: bool, save_psf: bool, save_pb: bool, crop: bool, 
              serendipitous: bool, run_tclean: bool, niter: int,
              n_pxs: Optional[int] = None, 
              n_channels: Optional[int] = None,
              n_workers: Optional[int] = 1,
              ncpu: Optional[int] = 1):
    """
    Input:
    i: index of the file to be simulated
    data_dir: directory where the data is stored
    main_path: path to the ALMASim Folder on your machine
    project_name: name of the project
    output_dir: directory where the output is stored
    plot_dir: directory where the plots are stored
    band (int): band to be simulated (3 - 10)
    antenna_name (str): name of the antenna configuration (alma.cycle9.3.1 - alma.cycle9.3.10)
    inbright (float): brightness of the source in Jy/px (0.000001 - 0.1)
    bandwidth (float): bandwidth in MHz (1000)
    inwidth (float): channel width in MHz (10 - 100)
    integration (float): integration time in seconds (0.1 - 10)
    totaltime (float): total time in seconds (1000 - 7000)
    pwv (float): precipitable water vapor in mm (0.1 - 1)
    snr (float): signal to noise ratio (5 - 30)
    get_skymodel (bool): if True, skymodels are loaded from the data_dir, else they are generated
    source_type (str): type of source to generate: "point", "diffuse" or "extended"
    TNGBasePath (str): path to the IllustrisTNG folder on your machine,
    TNGSnapshotID (int): snapshot of the IllustrisTNG simulation,
    TNGsubhaloID (int): subhaloID of the source in the IllustrisTNG simulation,
    TNG_api_key (str): your API key to access the IllustrisTNG simulation,
    plot (bool): if True, simulations plots are stored
    save_ms (bool): if True, measurement sets are stored
    crop (bool): if True, cubes are cropped to the size of the beam times 1.5 or to n_pxs
    serendipitous (bool): if True, serendipitous sources are added to the cube
    run_tclean (bool): if True, tclean is run on the measurement set
    niter (int): number of iterations for tclean
    n_pxs (int): Optional number of pixels in the x and y direction, if present crop is set to True
    n_channels (int): Optional number of channels in the z direction
    ncpu (int): number of cpu to use in parallel
    Output:
    None
    """
    start = time.time()
    flatten = False
    i, data_dir, main_path, project_name, output_dir, plot_dir, band, antenna_name, inbright, \
        bandwidth, inwidth, integration, totaltime, ra, dec, pwv, rest_frequency, snr, get_skymodel, \
        source_type, TNGBasePath, TNGSnapshotID, TNGSubhaloID, plot, save_ms, save_psf, save_pb, crop, \
        serendipitous, n_pxs, n_channels = check_parameters(i, data_dir, main_path, project_name, 
                                                            output_dir, plot_dir, band, antenna_name, 
                                                            inbright, bandwidth, inwidth, integration, 
                                                            totaltime, ra, dec, pwv, rest_frequency, snr, 
                                                            get_skymodel, source_type, TNGBasePath, 
                                                            TNGSnapshotID, TNGSubhaloID, plot, save_ms, 
                                                            save_psf, save_pb, crop, serendipitous, n_pxs, 
                                                            n_channels)
    os.chdir(output_dir)
    project = project_name + '_{}'.format(i)
    if not os.path.exists(project):
        os.mkdir(project)
    cycle = os.path.split(antenna_name)[0]
    antenna_name = os.path.split(antenna_name)[1]
    config_number = int(antenna_name.split('.')[-1])
    spatial_resolution = get_spatial_resolution(band, config_number)
    central_freq= get_band_central_freq(band)
    antennalist = os.path.join(main_path, "antenna_config", cycle, antenna_name + '.cfg')
    max_baseline = get_max_baseline_from_antenna_config(antennalist)
    beam_size = compute_beam_size_from_max_baseline(max_baseline, central_freq)
    cell_size = beam_size / 5
    # FoV only depends from the band (central frequency)
    fov = get_fov([band])[0]
   
    if n_channels is None or n_channels == 1:
        flatten = True
        n_channels = int(bandwidth / inwidth)
        #inbright = inbright / n_channels
    if n_pxs is None:
        n_px = int(fov / cell_size)
    elif n_pxs is not None and crop is False:
        n_px = int(n_pxs)
    else:
        n_px = int(fov / cell_size)
    # number of pixels must be even
    print('Simulation Parameters given Band and Spatial Resolution for simulation {}'.format(i))
    print('Band ', band)
    print('Bandwidth ', bandwidth, ' MHz')
    print('Central Frequency ', central_freq, ' GHz')
    print('Pixel size ', cell_size, ' arcsec')
    print('Fov ', fov, ' arcsec')
    print('Spatial_resolution ', spatial_resolution, ' arcsec')
    print('Cycle ', cycle)
    print('Antenna Configuration ', antenna_name)
    print('Inbright ', inbright, ' Jy/px')
    print('Beam Size: ', beam_size, ' arcsec')
    print('Integration Time ', integration, ' s')
    print('Total Time ', totaltime, ' s')
    print('TNG Base Path ', TNGBasePath)
    print('TNG Snapshot ID ', TNGSnapshotID)
    print('TNG Subhalo ID ', TNGSubhaloID)
    print('Cube Size: {} x {} x {} pixels'.format(n_px, n_px, n_channels))
    write_simulation_parameters(os.path.join(output_dir, 'simulation_parameters_{}.txt'.format(i)), band, bandwidth, central_freq, cell_size, fov, spatial_resolution, cycle, antenna_name, inbright, beam_size, integration, totaltime, TNGBasePath, TNGSnapshotID, TNGSubhaloID, n_px, n_channels)
    if n_pxs is not None:
        print('Cube will be cropped to {} x {} x {} pixels'.format(n_pxs, n_pxs, n_channels))
    print('\n# ------------------------ #\n')
    skymodel_time = time.time()
    if get_skymodel is True:
        print('Reading Skymodel from {}'.format(data_dir))
        print('\n')
        files = natsorted(os.listdir(data_dir))
        files = [os.path.join(data_dir, file) for file in files if '.fits' in file]
        filename = files[i]
    else:
        print('Generating {} Skymodel'.format(source_type))
        if source_type == "extended":
            print('Generating Extended Emission Skymodel from TNG')
            print('\n')
            filename = generate_extended_skymodel(i, output_dir, n_px, n_channels, 
                                                  spatial_resolution * U.arcsec,
                                                  central_freq * U.GHz,
                                                  inwidth * U.MHz, 
                                                  spatial_resolution,
                                                  TNGBasePath, 
                                                  TNGSnapshotID, TNGSubhaloID, 
                                                  ra * U.deg, dec * U.deg, 
                                                  rest_frequency, plot_dir, ncpu)
        elif source_type == "gaussian":
            print('Generating Gaussian Skymodel')
            if serendipitous == True:
                n_sources = np.random.randint(1, 5)
            else:
                n_sources = 0
            fwhm_x = 1.5 * cell_size * np.random.rand() + cell_size
            fwhm_y = 1.5 * cell_size  * np.random.rand() + cell_size
            fwhm_z = 0.1 * bandwidth * np.random.rand() + inwidth
            pa = np.random.randint(0, 360)
            print('Number of Sources ', n_sources)
            print('FWHM_x ', fwhm_x, ' arcsec')
            print('FWHM_y ', fwhm_y, ' arcsec')
            print('FWHM_z ', fwhm_z, ' MHz')
            print('PA ', pa, ' deg')

            min_sep_spatial = 1.5 * cell_size
            min_sep_frequency = 1.5 * inwidth
            filename = generate_gaussian_skymodel(i, output_dir, n_sources,
                                                  n_px, n_channels, bandwidth, 
                                                  fwhm_x * U.arcsec, fwhm_y * U.arcsec, 
                                                  fwhm_z * U.MHz,
                                                  cell_size * U.arcsec,  
                                                  fov * U.arcsec, 
                                                  spatial_resolution * U.arcsec, 
                                                  central_freq * U.GHz,
                                                  inwidth * U.GHz,
                                                  ra * U.deg, dec * U.deg,
                                                  pa, 
                                                  min_sep_spatial, 
                                                  min_sep_frequency,
                                                  rest_frequency, 
                                                  serendipitous,
                                                  plot_dir)
        elif source_type == "diffuse":
            print('Generating Diffuse Emission Skymodel')
            fwhm_z = (0.1 * bandwidth - 0.01 * bandwidth)*np.random.rand() + 0.01 * bandwidth
            print('FWHM_z ', fwhm_z, ' MHz')
            filename = generate_diffuse_skymodel(i, output_dir, n_px, n_channels,
                                                 fwhm_z * U.MHz, fov * U.arcsec, 
                                                 spatial_resolution * U.arcsec, 
                                                 central_freq * U.GHz, inwidth * U.GHz, ra * U.deg, 
                                                 dec * U.deg, rest_frequency, plot_dir)
        elif source_type == "point" or source_type == 'QSO':
            print('Generating Point Source Skymodel')
            fwhm_z = 0.1 * bandwidth * np.random.rand() + inwidth
            print('FWHM_z ', fwhm_z, ' MHz')
            # there is no inbright in here, that is why all inbrights are set to 2, fix this by adding the 
            # inbright and setting the maximum flux to inbright
            filename = generate_pointlike_skymodel(i, output_dir, rest_frequency, 
                                                   inwidth * U.MHz, fwhm_z * U.MHz,
                                                   central_freq * U.GHz, n_px, 
                                                   n_channels, ra * U.deg, dec * U.deg,
                                                   spatial_resolution * U.arcsec, 
                                                   serendipitous, plot_dir)
        elif source_type == "lens":
            print('Generating Lensing Skymodel')
            filename = generate_lensing_skymodel()
    final_skymodel_time = time.time()
    print('# ------------------------ #')
    print('Simulating ALMA Observation of the Skymodel')
    print('Brightness', inbright, ' Jy/px')
    skymodel, sky_header = load_fits(filename)
    true_brightness = np.max(skymodel)
    min_brightness = np.min(skymodel)
    print('True Brightness', true_brightness, ' Jy/px')
    print('Minimum Brightness', min_brightness, ' Jy/px')
    if inbright != true_brightness:
        print('Adjusting Brightness')
        flattened_skymodel = np.ravel(skymodel)
        t_min = 0
        t_max = inbright
        skymodel_norm = (flattened_skymodel - np.min(flattened_skymodel)) / (np.max(flattened_skymodel) - np.min(flattened_skymodel)) * (t_max - t_min) + t_min
        skymodel = np.reshape(skymodel_norm, np.shape(skymodel))
        print('New Brightness', np.max(skymodel), ' Jy/px')
        write_numpy_to_fits(skymodel, sky_header, filename)
    sim_time = time.time()
    simobserve(
        project=project, 
        skymodel=filename,
        inbright="{}Jy/pix".format(inbright),
        incell="{}arcsec".format(cell_size),
        indirection="J2000 19h30m00 -40d00m00",
        incenter='{}GHz'.format(central_freq),
        inwidth="{}MHz".format(inwidth),
        setpointings=True,
        integration="{}s".format(integration),
        obsmode="int",
        antennalist=antennalist,
        totaltime="{}s".format(totaltime),
        thermalnoise="tsys-atm",
        user_pwv=pwv,
        graphics="none",
        verbose=False,
        overwrite=True)
    
    # Adding atmosphere noise
    #scale = random.uniform(0.3, 1)
    scale = 0.5
    print('Adding Atmospheric Noise using a scale factor of {} for thropospheric phase'.format(scale))
    # scale is a multiplicative factor for the thropospheric phase 
    # which is a delay in the propagation of radio waves in the atmosphere
    # caused by the refractive index of the throphosphere
    simulate_atmospheric_noise(
        os.path.join(output_dir, project), 
        scale, 
        os.path.join(output_dir, project, "{}.{}.noisy.ms".format(project, antenna_name)), 
        antennalist)
    gain_error_amplitude = random.gauss(0.001, 0.1)
    simulate_gain_errors(
        os.path.join(output_dir, project, "{}.{}.noisy.ms".format(project, antenna_name)),
        gain_error_amplitude
    )

    tclean(
        vis=os.path.join(project, "{}.{}.noisy.ms".format(project, antenna_name)),
        imagename=os.path.join(project, '{}.{}'.format(project, antenna_name)),
        imsize=[int(n_px), int(n_px)],
        cell="{}arcsec".format(cell_size),
        specmode="cube",
        niter=0,
        fastnoise=False,
        calcpsf=True,
        pbcor=True,
        pblimit=0.2,
        )

    print('Saving Dirty and Clean Cubes')
    exportfits(imagename=os.path.join(project, '{}.{}.image'.format(project, antenna_name)), 
           fitsimage=os.path.join(output_dir, "dirty_cube_" + str(i) +".fits"), overwrite=True)
    exportfits(imagename=os.path.join(project, '{}.{}.skymodel'.format(project, antenna_name)), 
           fitsimage=os.path.join(output_dir, "clean_cube_" + str(i) +".fits"), overwrite=True)
    if save_psf is True:
        exportfits(imagename=os.path.join(project, '{}.{}.psf'.format(project, antenna_name)),
              fitsimage=os.path.join(output_dir, "psf_" + str(i) +".fits"), overwrite=True)
    if save_pb is True:
        exportfits(imagename=os.path.join(project, '{}.{}.pb'.format(project, antenna_name)),
                fitsimage=os.path.join(output_dir, "pb_" + str(i) +".fits"), overwrite=True)
    final_sim_time = time.time()
    
    if save_ms is True:
        print('# ------------------------ #')
        print('Saving Measurement Set')
        save_time = time.time()
        ms_to_npz(os.path.join(project, "{}.{}.noisy.ms".format(project, antenna_name)),
              dirty_cube=os.path.join(output_dir, "dirty_cube_" + str(i) +".fits"),
              datacolumn='CORRECTED_DATA',
              output_file=os.path.join(output_dir, "{}.{}.noisy_".format(project, antenna_name) + str(i) +".npz"))
        final_Save_time = time.time()

    # Cutting out the central region of the dirty and clean cubes
    clean, clean_header = load_fits(os.path.join(output_dir, "clean_cube_" + str(i) +".fits"))
    dirty, dirty_header = load_fits(os.path.join(output_dir, "dirty_cube_" + str(i) +".fits"))
    sky_total_flux = np.nansum(skymodel)
    dirty_total_flux = np.nansum(dirty)
    if sky_total_flux != dirty_total_flux:
        print('Dirty Cube total flux is different from the Sky Model total flux')
        print('Dirty Cube total flux: ', dirty_total_flux, ' Jy/px')
        print('Sky Model total flux: ', sky_total_flux, ' Jy/px')
        print('Normalizing Sky Model Cube to the Observed total flux')
        clean = clean * dirty_total_flux / sky_total_flux 
        print('Sky Model total flux after normalization: ', np.sum(clean), ' Jy/px')
        write_numpy_to_fits(clean, clean_header, os.path.join(output_dir, "clean_cube_" + str(i) +".fits"))
    if crop == True:
        left = int((clean.shape[-1] - n_pxs) / 2)
        clean_cube = clean[:, :,  left:left+int(n_pxs), left:left+int(n_pxs)]
        dirty_cube = dirty[:, :, left:left+int(n_pxs), left:left+int(n_pxs)]
        if flatten == True:
            clean_cube = np.expand_dims(np.sum(clean_cube, axis=1), axis=1)
            dirty_cube = np.expand_dims(np.sum(dirty_cube, axis=1), axis=1)
        
        write_numpy_to_fits(clean_cube, clean_header, os.path.join(output_dir, "clean_cube_" + str(i) +".fits"))
        write_numpy_to_fits(dirty_cube, dirty_header, os.path.join(output_dir, "dirty_cube_" + str(i) +".fits"))
    if run_tclean is True:
        print('Running tClean')
        tclean_time = time.time()
        tclean(
        vis=os.path.join(project, "{}.{}.noisy.ms".format(project, antenna_name)),
        imagename=os.path.join(project, '{}.{}'.format(project, antenna_name)),
        imsize=[int(n_px), int(n_px)],
        cell="{}arcsec".format(cell_size),
        specmode="cube",
        niter=niter,
        fastnoise=False,
        calcpsf=True,
        pbcor=True,
        pblimit=0.2,
        )
        final_tclean_time = time.time()
        exportfits(imagename=os.path.join(project, '{}.{}.image'.format(project, antenna_name)), 
           fitsimage=os.path.join(output_dir, "tclean_cube_" + str(i) +".fits"), overwrite=True)
        print('tClean performed {} iterations in {} seconds'.format(niter, strftime("%H:%M:%S", gmtime(final_tclean_time - tclean_time))))
    
    print('Deleting junk files')
    #shutil.rmtree(project)
    os.remove(os.path.join(output_dir, "skymodel_" + str(i) +".fits"))
    if plot is True:
        print('Saving Plots')
        plotter(i, output_dir, plot_dir, run_tclean, band, cycle, inbright, beam_size, cell_size, antenna_name, fwhm_z)
    stop = time.time()
    print('Skymodel Generated in {} seconds'.format(strftime("%H:%M:%S", gmtime(final_skymodel_time - skymodel_time))))
    print('Simulation Took {} seconds'.format(strftime("%H:%M:%S", gmtime(final_sim_time - sim_time))))
    if save_ms is True:
        print('Saving Took {} seconds'.format(strftime("%H:%M:%S", gmtime(final_Save_time - save_time))))
    print('Execution took {} seconds'.format(strftime("%H:%M:%S", gmtime(stop - start))))
    return None

def plotter(i, output_dir, plot_dir, run_tclean, band, cycle, inbright, beam_size, pixel_size, antenna_config, fwhm_z):
    clean, _ = load_fits(os.path.join(output_dir, 'clean_cube_{}.fits'.format(i)))
    dirty, _ = load_fits(os.path.join(output_dir, 'dirty_cube_{}.fits'.format(i)))
    beam_solid_angle = np.pi * (beam_size / 2) ** 2
    pixel_solid_angle = pixel_size ** 2
    pix_to_beam = beam_solid_angle / pixel_solid_angle

    if run_tclean is True:
        tclean, _ = load_fits(os.path.join(output_dir, 'tclean_cube_{}.fits'.format(i)))
    if len(clean.shape) > 3:
        clean = clean[0]
        dirty = dirty[0]
        if run_tclean is True:
            tclean = tclean[0]
    if clean.shape[0] > 1:
        clean_spectrum = np.sum(clean[:, :, :], axis=(1, 2))
        dirty_spectrum = np.where(dirty < 0, 0, dirty)
        dirty_spectrum = np.nansum(dirty_spectrum[:, :, :], axis=(1, 2))
        if run_tclean is True:
            tclean_spectrum = np.where(tclean < 0, 0, tclean)
            tclean_spectrum = np.nansum(tclean_spectrum[:, :, :], axis=(1, 2))
        clean_image = np.sum(clean[:, :, :], axis=0)[np.newaxis, :, :]
        dirty_image = np.nansum(dirty[:, :, :], axis=0)[np.newaxis, :, :]
        focused_image = np.nansum(dirty[int(dirty.shape[0] - fwhm_z):int(dirty.shape[0] + fwhm_z), :, :], axis=0)[np.newaxis, :, :]
        if run_tclean is True:
            tclean_image = np.nansum(tclean[:, :, :], axis=0)[np.newaxis, :, :]
    else:
        clean_image = clean.copy()
        dirty_image = dirty.copy()
        if run_tclean is True:
            tclean_image = tclean.copy()
    if run_tclean is True:
        fig, ax = plt.subplots(1, 4, figsize=(24, 5))
    else:
        fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    ax[0].imshow(clean_image[0] * pix_to_beam, origin='lower')
    ax[1].imshow(dirty_image[0] * pix_to_beam, origin='lower')
    if run_tclean is True:
        ax[3].imshow(tclean_image[0] * pix_to_beam, origin='lower')
    plt.colorbar(ax[0].imshow(clean_image[0] * pix_to_beam, origin='lower'), ax=ax[0], label='Jy/beam')
    plt.colorbar(ax[1].imshow(dirty_image[0] * pix_to_beam, origin='lower'), ax=ax[1], label='Jy/beam')
    if run_tclean is True:
        plt.colorbar(ax[3].imshow(tclean_image[0] * pix_to_beam, origin='lower'), ax=ax[2], label='Jy/beam')
    x_size, y_size = clean_image[0].shape
    ax[2].imshow(focused_image[0] * pix_to_beam, origin='lower')
    plt.colorbar(ax[2].imshow(focused_image[0] * pix_to_beam, origin='lower'), ax=ax[2], label='Jy/beam')
    xticks = np.arange(0, x_size, step=x_size // 5)
    yticks = np.arange(0, y_size, step=x_size // 5)
    ax[0].set_title('Sky Model Image')
    ax[1].set_title('ALMA Observed Image')
    ax[0].set_xlabel('RA (arcsec)')
    ax[0].set_ylabel('DEC (arcsec)')
    ax[0].set_xticks(xticks)
    ax[0].set_xticklabels(np.round(xticks * pixel_size, 2))
    ax[0].set_yticks(yticks)
    ax[0].set_yticklabels(np.round(yticks * pixel_size, 2))
    ax[1].set_xlabel('RA (arcsec)')
    ax[1].set_ylabel('DEC (arcsec)')
    ax[1].set_xticks(xticks)
    ax[1].set_xticklabels(np.round(xticks * pixel_size, 2))
    ax[1].set_yticks(yticks)
    ax[1].set_yticklabels(np.round(yticks * pixel_size, 2))
    ax[0].add_patch(Ellipse((10, 10), beam_size/ pixel_size, beam_size / pixel_size,
                            color='white', fill=False, linewidth=2))
    ax[0].text(10 + beam_size / pixel_size + 2, 10, f'Clean Beam with size: {round(beam_size, 2)} arcsec',
            verticalalignment='center', horizontalalignment='left', color='white', fontsize=8)
    ax[2].set_xlabel('RA (arcsec)')
    ax[2].set_ylabel('DEC (arcsec)')
    ax[2].set_xticks(xticks)
    ax[2].set_xticklabels(np.round(xticks * pixel_size, 2))
    ax[2].set_yticks(yticks)
    ax[2].set_yticklabels(np.round(yticks * pixel_size, 2))
    ax[2].add_patch(Ellipse((10, 10), beam_size/ pixel_size, beam_size / pixel_size,  color='white', fill=False, linewidth=2))
    ax[2].set_title('Focused Primary Source')
    if run_tclean is True:
        ax[3].set_xlabel('x [pixels]')
        ax[3].set_ylabel('y [pixels]')
        ax[3].set_title('tClean Image')
        ax[3].text(0.95, 0.95, 'Band: {}\n\nCycle: {}\n\n Antenna Config: {}\n\nBright: {} Jy/beam\n\nPixel Size {} arcsec'.format(band, cycle[-1], antenna_config[-1],
                   round(inbright * pix_to_beam, 2), round(pixel_size, 2)), verticalalignment='top', horizontalalignment='right',
                   transform=ax[2].transAxes, color='white', fontsize=8)
    else:
        ax[1].text(0.95, 0.95, 'Band: {}\n\nCycle: {}\n\n Antenna Config: {}\n\nBright: {} Jy/beam\n\nPixel Size {} arcsec'.format(band, cycle[-1], antenna_config[-1],
                   round(inbright * pix_to_beam, 2), round(pixel_size, 2)), verticalalignment='top', horizontalalignment='right',
                   transform=ax[1].transAxes, color='white', fontsize=8)
    plt.savefig(os.path.join(plot_dir, 'sim_{}.png'.format(i)))
    plt.close()
    if clean.shape[0] > 1:
        if run_tclean is True:
            fig, ax = plt.subplots(1, 3, figsize=(18, 5))
        else:
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].plot(clean_spectrum * pix_to_beam)
        ax[1].plot(dirty_spectrum * pix_to_beam)
        if run_tclean is True:
            ax[2].plot(tclean_spectrum * pix_to_beam)
        ax[0].set_title('Clean Sky Model Spectrum')
        ax[1].set_title('ALMA Simulated Spectrum')
        ax[0].set_xlabel('Channel')
        ax[0].set_ylabel('Jy/beam')
        ax[1].set_xlabel('Channel')
        ax[1].set_ylabel('Jy/beam')
        if run_tclean is True:
            ax[2].set_title('tClean Spectrum')
            ax[2].set_xlabel('Channel')
            ax[2].set_ylabel('Jy/beam')
        plt.savefig(os.path.join(plot_dir, 'sim_spectrum_{}.png'.format(i)))
        plt.close()

def plot_reference(reference, i, plot_dir):
    if len(reference.shape) > 3:
        reference = reference[0]
    if reference.shape[0] > 1:
        reference_spectrum = np.sum(reference[:, :, :], axis=(1, 2))
        reference_image = np.sum(reference[:, :, :], axis=0)[np.newaxis, :, :]
    else:
        reference_image = reference.copy()
    plt.figure(figsize=(5, 5))
    plt.imshow(reference_image[0], origin='lower')
    plt.colorbar()
    plt.title('Reference Image')
    plt.savefig(os.path.join(plot_dir, 'reference_{}.png'.format(i)))
    plt.close()
    if reference.shape[0] > 1:
        plt.figure(figsize=(5, 5))
        plt.plot(reference_spectrum)
        plt.title('Reference Spectrum')
        plt.savefig(os.path.join(plot_dir, 'reference_spectrum_{}.png'.format(i)))
        plt.close()

def plot_skymodel(path, i, plot_dir):
    skymodel, _ = load_fits(path)
    if len(skymodel.shape) > 3:
        skymodel = skymodel[0]
    skymodel_spectrum = np.sum(skymodel[:, :, :], axis=(1, 2))
    skymodel_image = np.sum(skymodel[:, :, :], axis=0)[np.newaxis, :, :]
    plt.figure(figsize=(5, 5))
    plt.imshow(skymodel_image[0], origin='lower')
    plt.colorbar()
    plt.title('skymodel Image')
    plt.savefig(os.path.join(plot_dir, 'skymodel_{}.png'.format(i)))
    plt.close()
    if skymodel.shape[0] > 1:
        plt.figure(figsize=(5, 5))
        plt.plot(skymodel_spectrum)
        plt.title('skymodel Spectrum')
        plt.savefig(os.path.join(plot_dir, 'skymodel_spectrum_{}.png'.format(i)))
        plt.close()

def download_TNG_data(path, api_key: str='8f578b92e700fae3266931f4d785f82c', TNGSnapshotID: int=99, TNGSubhaloID: list=[0]):
    """
    Downloads TNG100-1 simulation data for a given snapshot from the TNG project website using the specified API key.
    
    Args:
    - path (str): The path to the directory where the data will be downloaded.
    - api_key (str): The API key to use for downloading the data. Defaults to a public key.
    - TNGSnapshotID (int): The snapshot ID of the simulation data to download. Defaults to 99.
    - TNGSubhaloID (list): The subhalo IDs of the simulation data to download. Defaults to [0].
    Returns:
    - None
    """

    # Define the URLs for the simulation data
    urls = [
        (f'http://www.tng-project.org/api/TNG100-1/files/snapshot-{str(TNGSnapshotID)}', os.path.join('output', 'snapdir_0{}'.format(str(TNGSnapshotID))), 'Snapshot'),
        (f'http://www.tng-project.org/api/TNG100-1/files/simulation.hdf5', '', 'Simulation File'),
        (f'https://www.tng-project.org/api/TNG100-1/files/offsets.{str(TNGSnapshotID)}.hdf5', os.path.join('postprocessing', 'offsets'), 'Offsets File'),
    ]
    
    # Create the main directory for the downloaded data
    if not os.path.exists(path):
        os.mkdir(path)
    main_path = os.path.join(path, 'TNG100-1')
    if not os.path.exists(main_path):
        os.mkdir(main_path)
    print(f'Downloading TNG100-1 data to {main_path}...')
    # Download the simulation data
    for i, (url, subdir, message) in enumerate(urls):
        output_path = os.path.join(main_path,  subdir)
        os.makedirs(output_path, exist_ok=True)
        os.chdir(output_path)
        if i == 0:
            for id in TNGSubhaloID:
                cmd = f'wget -q --progress=bar  --content-disposition --header="API-Key:{api_key}" {url}.{id}.hdf5'
                if not os.path.isfile(os.path.join(output_path, f'snap_0{TNGSnapshotID}.{id}.hdf5')):
                    print(f'Downloading {message} {id} ...')
                    subprocess.check_call(cmd, shell=True)
                    print('Done.')
        elif i == 1:
            print(f'Downloading {message} ...')
            if not os.path.isfile(os.path.join(output_path, 'simulation.hdf5')):
                cmd = f'wget -q  --progress=bar   --content-disposition --header="API-Key:{api_key}" {url}'
                if not os.path.isfile(url.split('/')[-1]):
                    subprocess.check_call(cmd, shell=True)
                    print('Done.')
        else:
            print(f'Downloading {message} ...')
            if not os.path.isfile(os.path.join(output_path, f'offsets_{TNGSnapshotID}.hdf5')):
                cmd = f'wget -q  --progress=bar   --content-disposition --header="API-Key:{api_key}" {url}'
                if not os.path.isfile(url.split('/')[-1]):
                    subprocess.check_call(cmd, shell=True)
                    print('Done.')
    print('All downloads complete.')
    
    return

def check_TNGBasePath(TNGBasePath: str, TNGSnapshotID: int, TNGSubhaloID: list):
    """
    Check if TNGBasePath exists and contains the following subfolders: output and postprocessing.
    """
    if TNGBasePath is not None:
        if os.path.exists(TNGBasePath):
            subfolders = [os.path.join('TNG100-1', 'output'), 
                          os.path.join('TNG100-1', 'postprocessing')]
            for subfolder in subfolders:
                if not os.path.exists(os.path.join(TNGBasePath, subfolder)):
                    print(f"Error: {subfolder} subfolder not found in TNGBasePath.")
                    return False
                elif len(os.listdir(os.path.join(TNGBasePath, subfolder))) == 0:
                    print(f"Error: {subfolder} subfolder is empty.")
                    return False
                
            # Check if the files relative to the specified snapshot and subhalo IDs exist

            # Check if the snapdir exists
            snapdir = os.path.join(TNGBasePath, 'TNG100-1', 'output', f'snapdir_0{TNGSnapshotID}')
            if not os.path.exists(snapdir):
                print(f"Error: snapdir_0{TNGSnapshotID} not found in TNGBasePath.")
                return False
            elif len(os.listdir(snapdir)) == 0:
                print(f"Error: snapdir_0{TNGSnapshotID} is empty.")
                return False
            # Check if all the files relative to the specified subhalo IDs exist
            elif len(os.listdir(snapdir)) > 0:
                filelist = [os.path.join(snapdir, 'snap_0{}.{}.hdf5'.format(TNGSnapshotID, id)) for id in TNGSubhaloID]
                for file_ in filelist:
                    if not os.path.exists(file_):
                        print(f"Error: {file_} not found in TNGBasePath.")
                        return False
                    elif os.path.getsize(file_) == 0:
                        print(f"Error: {file_} is empty.")
                        return False
            # check if groupdir exists 
            groupdir = os.path.join(TNGBasePath, 'TNG100-1', 'output', f'groups_0{TNGSnapshotID}')
            if not os.path.exists(groupdir):
                print(f"Error: groups_0{TNGSnapshotID} not found in TNGBasePath.")
                return False
            elif len(os.listdir(groupdir)) == 0:
                print(f"Error: groups_0{TNGSnapshotID} is empty.")
                return False
            # Check if all the files relative to the specified subhalo IDs exist
            elif len(os.listdir(groupdir)) > 0:
                filelist = [os.path.join(groupdir, 'fof_subhalo_tab_0{}.{}.hdf5'.format(TNGSnapshotID, id)) for id in TNGSubhaloID]
                for file_ in filelist:
                    if not os.path.exists(file_):
                        print(f"Error: {file_} not found in TNGBasePath.")
                        return False
                    elif os.path.getsize(file_) == 0:
                        print(f"Error: {file_} is empty.")
                        return False
            return True
        else:
            print("Error: TNGBasePath does not exist.")
            return False
    else:
        print("Error: TNGBasePath not specified.")
        return None

def check_TNGData(path, api_key: str='8f578b92e700fae3266931f4d785f82c', TNGSnapshotID: int=99):
    if not os.path.exists(path):
        os.mkdir(path)
    main_path = os.path.join(path, 'TNG100-1')
    if not os.path.exists(main_path):
        os.mkdir(main_path)
    subfolders = [os.path.join(main_path, 'output'), 
                          os.path.join(main_path, 'postprocessing')]
    return

def get_subhalorange(basePath, snapNum, subhaloIDs):
    partType = 'gas'
    filenums, limits = [], []
    offsetPath = os.path.join(basePath, "TNG100-1", "postprocessing/offsets/offsets_%03d.hdf5" % snapNum)
    basePath = os.path.join(basePath, "TNG100-1", "output", )
    with h5py.File(offsetPath, "r") as f:
        offsets = f["FileOffsets/" + 'Subhalo'][()]
        unique, counts = np.unique(offsets, return_counts=True)
        for subhaloID in subhaloIDs:
            index = bisect.bisect(unique, subhaloID)
            limits.append([offsets[index - 1], offsets[index + counts[index]]])
            filenums.append(np.arange(index, index + counts[index]))

    #with h5py.File(il.snapshot.snapPath(basePath, snapNum), 'r') as f:
    #   for subhaloID in subhaloIDs:
    #        subset = il.snapshot.getSnapOffsets(basePath, snapNum, subhaloID, "Subhalo")
    #        header = dict(f['Header'].attrs.items())
    #        nPart = il.snapshot.getNumPart(header)
    ##        ptNum = il.snapshot.partTypeNum(partType)
    #        gName = "PartType" + str(ptNum)
    #        offsetsThisType = subset['offsetType'][ptNum] - subset['snapOffsets'][ptNum, :]
    #        fileNum = np.max(np.where(offsetsThisType >= 0))
    #        fileOff = offsetsThisType[fileNum]
    #        numToRead = subset['lenType'][ptNum]
    #        print(fileNum)
    #        print(np.where(unique == fileNum))
    #        filenums.append(np.arange(fileNum, fileNum + counts[][0]))
            
    filenums = np.array(filenums).flatten().tolist()
    return filenums, limits

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

def query_for_metadata(targets, path, service_url: str = "https://almascience.eso.org/tap"):
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
    'band_list': 'Band'
    }
    # Rename the columns in the DataFrame
    df.rename(columns=rename_columns, inplace=True)
    database = df[['ALMA_source_name', 'Band', 'PWV', 'SB_name', 'Vel.res', 'Ang.res', 'RA', 'Dec', 'FOV', 'Int.Time', 'Total.Time', 'Cont_sens_mJybeam', 'Line_sens_10kms_mJybeam', 'Obs.date']]
    database['Obs.date'] = database['Obs.date'].apply(lambda x: x.split('T')[0])
    database.to_csv(path, index=False)
    return database

def luminosity_to_jy(velocity, data, rest_frequency: float = 115.27):
        """
        This function takes as input a pandas db containing luminosities in K km s-1 pc2, redshifts, and luminosity distances in Mpc, 
        and returns the brightness values in Jy.
        
        Parameters:
        velocity (float): The velocity dispersion assumed for the line (Km s-1).
        data (pandas.DataFrame): A pandas DataFrame containing the data.
        rest_frequency (float): The rest frequency of the line in GHz. Defaults to 115.27 GHz for CO(1-0).

        Output:
        sigma: numpy.ndarray: An array of brightness values in Jy.

        """
        alpha = 3.255 * 10**7
        sigma = (data['Luminosity(K km s-1 pc2)'] * ( (1 + data['#redshift']) * rest_frequency **2)) / (alpha * velocity * (data['luminosity distance(Mpc)']**2))
        redshift = data['#redshift'].values
        return sigma

def exponential_func(x, a, b):
        """
        Exponential function used to fit the data.
        """
        return a * np.exp(-b * x)

def sample_from_brightness(n, velocity, rest_frequency, data_path):
    """
    Generates n samples of brightness values based on an exponential fit to the data.
    
    Parameters:
    n (int): Number of samples to generate.
    velocity (float): The velocity dispersion assumed for the line.
    data_path (str): Path to the CSV file containing the data.
    
    Returns:
    pd.DataFrame: A DataFrame containing the sampled brightness values and corresponding redshifts.
    """
    # Read the data from the CSV file
    data = pd.read_csv(data_path, sep='\t')
    # Calculate the brightness values (sigma) using the provided velocity
    sigma = luminosity_to_jy(velocity, data, rest_frequency)
    # Extract the redshift values from the data
    redshift = data['#redshift'].values
    # Generate evenly spaced redshifts for sampling
    np.random.seed(42)
    sampled_redshifts = np.linspace(min(redshift), max(redshift), n)
    # Fit an exponential curve to the data
    popt, pcov = curve_fit(exponential_func, redshift, sigma, )
    # Use the fitted parameters to calculate the sampled brightness values
    sampled_sigma = exponential_func(sampled_redshifts, *popt) + np.min(sigma)
    print(np.mean(sigma), np.min(sigma), np.max(sigma))
    plt.scatter(redshift, sigma, label='Data')
    plt.scatter(sampled_redshifts, sampled_sigma, label='Polinomial Fit')
    plt.xlabel('Redshift')
    plt.ylabel('Brightness (Jy)')
    plt.legend()
    plt.show()
    # Return the sampled brightness values and the corresponding redshifts
    return pd.DataFrame(zip(sampled_redshifts, sampled_sigma), columns=['Redshift', 'Brightness(Jy)'])
