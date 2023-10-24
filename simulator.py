from datetime import date 
import tempfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
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
from martini import DataCube, Martini
from martini.sources.sph_source import SPHSource
from martini.spectral_models import GaussianSpectrum
from martini.sph_kernels import (AdaptiveKernel, CubicSplineKernel,
                                 GaussianKernel, find_fwhm)
from natsort import natsorted
from spectral_cube import SpectralCube
from tqdm import tqdm
import psutil
import subprocess

os.environ['MPLCONFIGDIR'] = temp_dir.name
pd.options.mode.chained_assignment = None  

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

def get_subhaloids_from_db(n, limit):
    file = 'morphologies_deeplearn.hdf5'
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
    
    n_0 = choices(ellipticals_ids[ellipticals_ids < limit], k=sample_n)
    n_1 = choices(spirals_ids[spirals_ids < limit], k=sample_n)
    n_2 = choices(lenticulars_ids[lenticulars_ids < limit], k=n - 2 * sample_n)
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

def get_fov(bands):
    light_speed = c.to(U.m / U.s).value
    fovs = []
    for band in bands:
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
        fovs.append(fov)
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
    sample = exp_cf(random_pos)
    data = sample.val[0:n_px, 0:n_px]
    return data

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
    xy_radius = fov / pixel_size * 0.3
    z_radius = 0.3 * bandwidth * U.MHz / frequency_resolution
    if serendipitous is True:
        print('Generating central source and {} serendipitous companions in a radius of {} pixels in the x and y directions and {} pixels in the z direction\n'.format(n_sources, int(xy_radius), int(z_radius)))
    min_sep_xy = min_sep_spatial / pixel_size
    min_sep_z = min_sep_frequency / frequency_resolution
    if serendipitous is True:
        fwhm_xs = np.random.randint(2, 10, n_sources)
        fwhm_ys = np.random.randint(2, 10, n_sources)
        fwhm_zs = np.random.randint(2, 30, n_sources)
        amplitudes = np.random.rand(n_sources)
        sample_coords = sample_positions(pos_x, pos_y, pos_z, 
                                     fwhm_x, fwhm_y, fwhm_z,
                                     n_sources, fwhm_xs, fwhm_ys, fwhm_zs,
                                     xy_radius.value, z_radius.value, min_sep_xy.value, min_sep_z.value)
    
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
                                spatial_resolution, plot_dir):
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
    wcs = datacube.wcs
    pos_x, pos_y, _ = wcs.sub(3).wcs_world2pix(ra, dec, central_velocity, 0)
    pos_z = n_channels // 2
    print('Generating point-like source at position ({}, {}, {})'.format(int(pos_x), int(pos_y), int(pos_z)))
    datacube = insert_pointlike(datacube, 1, pos_x, pos_y, pos_z, fwhm_z, n_px, n_channels)
    filename = os.path.join(data_dir, 'skymodel_{}.fits'.format(id))
    write_datacube_to_fits(datacube, filename)
    plot_skymodel(filename, id, plot_dir)
    print('Skymodel saved to {}'.format(filename))
    del datacube
    return filename
     
def generate_diffuse_skymodel(id, data_dir, n_px, n_channels,
                              fwhm_z, fov,
                              spatial_resolution, central_frequency, 
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

def partTypeNum(partType):
    """
    Mapping between common names and numeric particle types.

    Reproduced from the illustris_python toolkit.
    """
    if str(partType).isdigit():
        return int(partType)

    if str(partType).lower() in ["gas", "cells"]:
        return 0
    if str(partType).lower() in ["dm", "darkmatter"]:
        return 1
    if str(partType).lower() in ["tracer", "tracers", "tracermc", "trmc"]:
        return 3
    if str(partType).lower() in ["star", "stars", "stellar"]:
        return 4  # only those with GFM_StellarFormationTime>0
    if str(partType).lower() in ["wind"]:
        return 4  # only those with GFM_StellarFormationTime<0
    if str(partType).lower() in ["bh", "bhs", "blackhole", "blackholes"]:
        return 5

    raise Exception("Unknown particle type name.")

def getNumPart(header):
    """
    Calculate number of particles of all types given a snapshot header.

    Reproduced from the illustris_python toolkit.
    """
    nTypes = 6

    nPart = np.zeros(nTypes, dtype=np.int64)
    for j in range(nTypes):
        nPart[j] = header["NumPart_Total"][j] | (
            header["NumPart_Total_HighWord"][j] << 32
        )

    return nPart

def gcPath(basePath, snapNum, chunkNum=0):
    """
    Return absolute path to a group catalog HDF5 file (modify as needed).

    Reproduced from the illustris_python toolkit.
    """
    gcPath = basePath + "/groups_%03d/" % snapNum
    filePath1 = gcPath + "groups_%03d.%d.hdf5" % (snapNum, chunkNum)
    filePath2 = gcPath + "fof_subhalo_tab_%03d.%d.hdf5" % (snapNum, chunkNum)

    if os.path.isfile(filePath1):
        return filePath1
    return filePath2

def offsetPath(basePath, snapNum):
    """
    Return absolute path to a separate offset file (modify as needed).

    Reproduced from the illustris_python toolkit.
    """
    offsetPath = basePath + "/../postprocessing/offsets/offsets_%03d.hdf5" % snapNum

    return offsetPath

def snapPath(basePath, snapNum, chunkNum=0):
    """Return absolute path to a snapshot HDF5 file (modify as needed)."""
    snapPath = basePath + "/snapdir_" + str(snapNum).zfill(3) + "/"
    filePath = snapPath + "snap_" + str(snapNum).zfill(3)
    filePath += "." + str(chunkNum) + ".hdf5"
    return filePath

def loadSingle(basePath, snapNum, haloID=-1, subhaloID=-1):
    """
    Return complete group catalog information for one halo or subhalo.

    Reproduced from the illustris_python toolkit.
    """
    import h5py

    if (haloID < 0 and subhaloID < 0) or (haloID >= 0 and subhaloID >= 0):
        raise Exception("Must specify either haloID or subhaloID (and not both).")

    gName = "Subhalo" if subhaloID >= 0 else "Group"
    searchID = subhaloID if subhaloID >= 0 else haloID

    # old or new format
    if "fof_subhalo" in gcPath(basePath, snapNum):
        # use separate 'offsets_nnn.hdf5' files
        with h5py.File(offsetPath(basePath, snapNum), "r") as f:
            offsets = f["FileOffsets/" + gName][()]
    else:
        # use header of group catalog
        with h5py.File(gcPath(basePath, snapNum), "r") as f:
            offsets = f["Header"].attrs["FileOffsets_" + gName]

    offsets = searchID - offsets
    fileNum = np.max(np.where(offsets >= 0))
    groupOffset = offsets[fileNum]

    # load halo/subhalo fields into a dict
    result = {}

    with h5py.File(gcPath(basePath, snapNum, fileNum), "r") as f:
        for haloProp in f[gName].keys():
            result[haloProp] = f[gName][haloProp][groupOffset]

    return result

def loadSubset(
    basePath,
    snapNum,
    partType,
    fields=None,
    subset=None,
    mdi=None,
    sq=True,
    float32=False,
    ):
    """
    Load a subset of fields for all particles/cells of a given partType.
    If offset and length specified, load only that subset of the partType.
    If mdi is specified, must be a list of integers of the same length as fields,
    giving for each field the multi-dimensional index (on the second dimension) to load.
      For example, fields=['Coordinates', 'Masses'] and mdi=[1, None] returns a 1D array
      of y-Coordinates only, together with Masses.
    If sq is True, return a numpy array instead of a dict if len(fields)==1.
    If float32 is True, load any float64 datatype arrays directly as float32
    (save memory).

    Reproduced from the illustris_python toolkit.
    """
    import h5py
    import six

    result = {}

    ptNum = partTypeNum(partType)
    gName = "PartType" + str(ptNum)

    # make sure fields is not a single element
    if isinstance(fields, six.string_types):
        fields = [fields]

    # load header from first chunk
    with h5py.File(snapPath(basePath, snapNum), "r") as f:
        header = dict(f["Header"].attrs.items())
        nPart = getNumPart(header)

        # decide global read size, starting file chunk, and starting file chunk offset
        if subset:
            offsetsThisType = (
                subset["offsetType"][ptNum] - subset["snapOffsets"][ptNum, :]
            )

            fileNum = np.max(np.where(offsetsThisType >= 0))
            fileOff = offsetsThisType[fileNum]
            numToRead = subset["lenType"][ptNum]
        else:
            fileNum = 0
            fileOff = 0
            numToRead = nPart[ptNum]

        result["count"] = numToRead

        if not numToRead:
            # print('warning: no particles of requested type, empty return.')
            return result

        # find a chunk with this particle type
        i = 1
        while gName not in f:
            f = h5py.File(snapPath(basePath, snapNum, i), "r")
            i += 1

        # if fields not specified, load everything
        if not fields:
            fields = list(f[gName].keys())

        for i, field in enumerate(fields):
            # verify existence
            if field not in f[gName].keys():
                raise Exception(
                    "Particle type ["
                    + str(ptNum)
                    + "] does not have field ["
                    + field
                    + "]"
                )

            # replace local length with global
            shape = list(f[gName][field].shape)
            shape[0] = numToRead

            # multi-dimensional index slice load
            if mdi is not None and mdi[i] is not None:
                if len(shape) != 2:
                    raise Exception(
                        "Read error: mdi requested on non-2D field [" + field + "]"
                    )
                shape = [shape[0]]

            # allocate within return dict
            dtype = f[gName][field].dtype
            if dtype == np.float64 and float32:
                dtype = np.float32
            result[field] = np.zeros(shape, dtype=dtype)

    # loop over chunks
    wOffset = 0
    origNumToRead = numToRead

    while numToRead:
        f = h5py.File(snapPath(basePath, snapNum, fileNum), "r")

        # no particles of requested type in this file chunk?
        if gName not in f:
            f.close()
            fileNum += 1
            fileOff = 0
            continue

        # set local read length for this file chunk, truncate to be within the local size
        numTypeLocal = f["Header"].attrs["NumPart_ThisFile"][ptNum]

        numToReadLocal = numToRead

        if fileOff + numToReadLocal > numTypeLocal:
            numToReadLocal = numTypeLocal - fileOff

        # loop over each requested field for this particle type
        for i, field in enumerate(fields):
            # read data local to the current file
            if mdi is None or mdi[i] is None:
                result[field][wOffset : wOffset + numToReadLocal] = f[gName][field][
                    fileOff : fileOff + numToReadLocal
                ]
            else:
                result[field][wOffset : wOffset + numToReadLocal] = f[gName][field][
                    fileOff : fileOff + numToReadLocal, mdi[i]
                ]

        wOffset += numToReadLocal
        numToRead -= numToReadLocal
        fileNum += 1
        fileOff = 0  # start at beginning of all file chunks other than the first

        f.close()

    # verify we read the correct number
    if origNumToRead != wOffset:
        raise Exception(
            "Read ["
            + str(wOffset)
            + "] particles, but was expecting ["
            + str(origNumToRead)
            + "]"
        )

    # only a single field? then return the array instead of a single item dict
    if sq and len(fields) == 1:
        return result[fields[0]]

    return result

def loadHeader(basePath, snapNum):
    """
    Load the group catalog header.

    Reproduced from the illustris_python toolkit.
    """
    import h5py

    with h5py.File(gcPath(basePath, snapNum), "r") as f:
        header = dict(f["Header"].attrs.items())

    return header
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

def generate_extended_skymodel(id, data_dir, n_px, n_channels, pixel_size,
                               central_frequency, frequency_resolution, 
                               TNGBasePath, TNGSnap, subhaloID, ra, dec, 
                               rest_frequency, plot_dir):
    #distance = np.random.randint(1, 5) * U.Mpc
    x_rot = np.random.randint(0, 360) * U.deg
    y_rot = np.random.randint(0, 360) * U.deg
    simulation_str = TNGBasePath.split('/')[-1]
    TNGBasePath = os.path.join(TNGBasePath, 'TNG100-1', 'output')

    #distance, channel_width = get_distance(n_px, n_channels, x_rot, y_rot, 
    #                                       TNGSnap, subhaloID, TNGBasePath, factor=4)
    data_header = loadHeader(TNGBasePath, TNGSnap)
    z = data_header["Redshift"] * cu.redshift
    
    distance = z.to(U.Mpc, cu.redshift_distance(Planck13, kind="comoving"))

    print('Generating extended source from subhalo {} - {} at {} with rotation angles {} and {} in the X and Y planes'.format(simulation_str, subhaloID, distance, x_rot, y_rot))
    
    source = myTNGSource(TNGSnap, subhaloID,
                       distance=distance,
                       rotation = {'L_coords': (x_rot, y_rot)},
                       basePath = TNGBasePath,
                       ra = 0. * U.deg,
                       dec = 0. * U.deg,)
    if rest_frequency == 1420.4:
        hI_rest_frequency = rest_frequency * U.MHz
    else:
        hI_rest_frequency = rest_frequency * 10 ** -6 * U.MHz
    radio_hI_equivalence = U.doppler_radio(hI_rest_frequency)
    central_velocity = central_frequency.to(U.km / U.s, equivalencies=radio_hI_equivalence)
    velocity_resolution = frequency_resolution.to(U.km / U.s, equivalencies=radio_hI_equivalence)
    
    if ra == 0.0 * U.deg and dec == 0.0 * U.deg:
        ra = source.ra
        dec = source.dec
    
    datacube = DataCube(
        n_px_x = n_px,
        n_px_y = n_px,
        n_channels = n_channels, 
        px_size = pixel_size,
        channel_width=frequency_resolution,
        velocity_centre=source.vsys, 
        ra = ra,
        dec = dec,
    )
    spectral_model = GaussianSpectrum(
        sigma="thermal"
    )
    sph_kernel = AdaptiveKernel(
    (
        CubicSplineKernel(),
        GaussianKernel(truncate=6),
    )

    )

    M = Martini(
        source=source,
        datacube=datacube,
        sph_kernel=sph_kernel,
        spectral_model=spectral_model,
        quiet=True)
    
    M.insert_source_in_cube(skip_validation=True, progressbar=True)
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

def simulator(i: int, data_dir: str, main_path: str, project_name: str, 
              output_dir: str, plot_dir: str, band: int, antenna_name: str, inbright: float, 
              bandwidth: int, inwidth: float, integration: int, totaltime: int, ra: float, dec: float,
              pwv: float, rest_frequency: float, snr: float, get_skymodel: bool, 
              source_type: str, TNGBasePath: str, TNGSnapshotID: int, 
              TNGSubhaloID: int,
              plot: bool, save_ms: bool, save_psf: bool, save_pb: bool, crop: bool, serendipitous: bool, 
              n_pxs: Optional[int] = None, 
              n_channels: Optional[int] = None ):
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
    n_pxs (int): Optional number of pixels in the x and y direction, if present crop is set to True
    n_channels (int): Optional number of channels in the z direction
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
        inbright = inbright / n_channels
    if n_pxs is None:
        n_px = int(fov / cell_size)
    elif n_pxs is not None and crop is False:
        n_px = int(n_pxs)
    else:
        n_px = int(fov / cell_size)
    # number of pixels must be even
    print('Simulation Parameters given Band and Spatial Resolution')
    print('Band ', band)
    print('Bandwidth ', bandwidth, ' MHz')
    print('Central Frequency ', central_freq, ' GHz')
    print('Pixel size ', cell_size, ' arcsec')
    print('Fov ', fov, ' arcsec')
    print('Spatial_resolution ', spatial_resolution, ' arcsec')
    print('Cycle ', cycle)
    print('Antenna Configuration ', antenna_name)
    print('Beam Size: ', beam_size, ' arcsec')
    print('TNG Base Path ', TNGBasePath)
    print('TNG Snapshot ID ', TNGSnapshotID)
    print('TNG Subhalo ID ', TNGSubhaloID)
    print('Cube Size: {} x {} x {} pixels'.format(n_px, n_px, n_channels))
    if n_pxs is not None:
        print('Cube will be cropped to {} x {} x {} pixels'.format(n_pxs, n_pxs, n_channels))
    print('# ------------------------ #')
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
                                                  inwidth * U.MHz, TNGBasePath, 
                                                  TNGSnapshotID, TNGSubhaloID, 
                                                  ra * U.deg, dec * U.deg, 
                                                  rest_frequency, plot_dir)
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
            filename = generate_diffuse_skymodel(i, output_dir, n_px, n_channels, bandwidth, 
                                                 fwhm_z * U.MHz, cell_size * U.arcsec, 
                                                 fov * U.arcsec, spatial_resolution * U.arcsec, 
                                                 central_freq * U.GHz, inwidth * U.GHz, ra * U.deg, 
                                                 dec * U.deg, rest_frequency, plot_dir)
        elif source_type == "point":
            print('Generating Point Source Skymodel')
            fwhm_z = 0.1 * bandwidth * np.random.rand() + inwidth
            print('FWHM_z ', fwhm_z, ' MHz')
            filename = generate_pointlike_skymodel(i, output_dir, rest_frequency, 
                                                   inwidth * U.MHz, fwhm_z * U.MHz,
                                                   central_freq * U.GHz, n_px, 
                                                   n_channels, ra * U.deg, dec * U.deg,
                                                   spatial_resolution * U.arcsec, plot_dir)
        elif source_type == "lens":
            print('Generating Lensing Skymodel')
            filename = generate_lensing_skymodel()
    final_skymodel_time = time.time()
    print('# ------------------------ #')
    print('Simulating ALMA Observation of the Skymodel')
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
    if crop == True:
        left = int((clean.shape[-1] - n_pxs) / 2)
        clean_cube = clean[:, :,  left:left+int(n_pxs), left:left+int(n_pxs)]
        dirty_cube = dirty[:, :, left:left+int(n_pxs), left:left+int(n_pxs)]
        if flatten == True:
            clean_cube = np.expand_dims(np.sum(clean_cube, axis=1), axis=1)
            dirty_cube = np.expand_dims(np.sum(dirty_cube, axis=1), axis=1)
        
        write_numpy_to_fits(clean_cube, clean_header, os.path.join(output_dir, "clean_cube_" + str(i) +".fits"))
        write_numpy_to_fits(dirty_cube, dirty_header, os.path.join(output_dir, "dirty_cube_" + str(i) +".fits"))

    print('Deleting junk files')
    #shutil.rmtree(project)
    os.remove(os.path.join(output_dir, "skymodel_" + str(i) +".fits"))
    if plot is True:
        print('Saving Plots')
        plotter(i, output_dir, plot_dir)
    stop = time.time()
    print('Skymodel Generated in {} seconds'.format(strftime("%H:%M:%S", gmtime(final_skymodel_time - skymodel_time))))
    print('Simulation Took {} seconds'.format(strftime("%H:%M:%S", gmtime(final_sim_time - sim_time))))
    if save_ms is True:
        print('Saving Took {} seconds'.format(strftime("%H:%M:%S", gmtime(final_Save_time - save_time))))
    print('Execution took {} seconds'.format(strftime("%H:%M:%S", gmtime(stop - start))))
    return

def plotter(i, output_dir, plot_dir):
    clean, _ = load_fits(os.path.join(output_dir, 'clean_cube_{}.fits'.format(i)))
    dirty, _ = load_fits(os.path.join(output_dir, 'dirty_cube_{}.fits'.format(i)))
    if len(clean.shape) > 3:
        clean = clean[0]
        dirty = dirty[0]
    if clean.shape[0] > 1:
        clean_spectrum = np.sum(clean[:, :, :], axis=(1, 2))
        dirty_spectrum = np.where(dirty < 0, 0, dirty)
        dirty_spectrum = np.nansum(dirty_spectrum[:, :, :], axis=(1, 2))
        clean_image = np.sum(clean[:, :, :], axis=0)[np.newaxis, :, :]
        dirty_image = np.nansum(dirty[:, :, :], axis=0)[np.newaxis, :, :]
    else:
        clean_image = clean.copy()
        dirty_image = dirty.copy()
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(clean_image[0], origin='lower')
    ax[1].imshow(dirty_image[0], origin='lower')
    plt.colorbar(ax[0].imshow(clean_image[0], origin='lower'), ax=ax[0])
    plt.colorbar(ax[1].imshow(dirty_image[0], origin='lower'), ax=ax[1])
    ax[0].set_title('Clean Sky Model Image')
    ax[1].set_title('ALMA Simulated Image')
    plt.savefig(os.path.join(plot_dir, 'sim_{}.png'.format(i)))
    plt.close()
    if clean.shape[0] > 1:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].plot(clean_spectrum)
        ax[1].plot(dirty_spectrum)
        ax[0].set_title('Clean Sky Model Spectrum')
        ax[1].set_title('ALMA Simulated Spectrum')
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
        (f'http://www.tng-project.org/api/TNG100-1/files/groupcat-{str(TNGSnapshotID)}', os.path.join('output', 'groups_0{}'.format(str(TNGSnapshotID))), 'Group Catalog'),
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
        if i < 2:
            for id in TNGSubhaloID:
                cmd = f'wget -q --progress=bar  --content-disposition --header="API-Key:{api_key}" {url}.{id}.hdf5'
                if 'snapdir' in output_path:
                    if not os.path.isfile(os.path.join(output_path, f'snap_0{TNGSnapshotID}.{id}.hdf5')):
                        print(f'Downloading {message} {id} ...')
                        subprocess.check_call(cmd, shell=True)
                        print('Done.')
                else:
                    if not os.path.isfile(os.path.join(output_path, f'fof_subhalo_tab_0{TNGSnapshotID}.{id}.hdf5')):
                        print(f'Downloading {message} {id} ...')
                        subprocess.check_call(cmd, shell=True)
                        print('Done.')
        else:
            print(f'Downloading {message} ...')
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

def get_subhalorange(basePath, snapNum, subhaloIDs):
    basePath = os.path.join(basePath, "TNG100-1", "postprocessing/offsets/offsets_%03d.hdf5" % snapNum)
    gName = "Subhalo" if max(subhaloIDs) >= 0 else "Group"
    with h5py.File(basePath, "r") as f:
        offsets = f["FileOffsets/" + gName][()]
    limit = offsets[max(subhaloIDs) + 1]
    n_subhalo = np.max(np.where(offsets == limit))
    return limit, n_subhalo
