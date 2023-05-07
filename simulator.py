import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
import tempfile
temp_dir = tempfile.TemporaryDirectory()
from casatasks import simobserve, tclean, exportfits
import os
import shutil
from casatools import table
from astropy.constants import c
from astropy.time import Time
import astropy.units as U
from martini.sources import TNGSource
from martini import DataCube, Martini
from martini.beams import GaussianBeam
from martini.noise import GaussianNoise
from martini.spectral_models import GaussianSpectrum
from martini.sph_kernels import AdaptiveKernel, GaussianKernel, CubicSplineKernel, DiracDeltaKernel
from natsort import natsorted
import math
from math import pi
from tqdm import tqdm
import time
from time import strftime, gmtime
import dask
from typing import Optional
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from spectral_cube import SpectralCube
import h5py
from random import choices
os.environ['MPLCONFIGDIR'] = temp_dir.name
pd.options.mode.chained_assignment = None  

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

def get_subhaloids_from_db(n):
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
    sample_n = 100 // 3

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

def get_fov(bands):
    light_speed = c.to(U.m / U.s).value
    fovs = []
    for band in bands:
        if band == 1:
            central_freq = 43 * U.GHz  
        elif band == 3:
            central_freq = 100 * U.GHz
        elif band == 4:
            central_freq = 143 * U.GHz
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
            central_freq = 850 * U.GHz
           
        central_freq = central_freq.to(U.Hz).value
        central_freq_s = 1 / central_freq
        wavelength = light_speed * central_freq_s
        fov = 1.13 * wavelength / 12
        fov = fov * 180 / np.pi * 3600
        fovs.append(fov)
    return np.array(fovs)

spatial_resolution_dict = {
    'alma.cycle9.3.1' : [3.38, 2.25, 1.83, 1.47, 0.98, 0.74, 0.52, 0.39],
    'alma.cycle9.3.2' : [2.30, 1.53, 1.24, 1.00, 0.67, 0.50, 0.35, 0.26],
    'alma.cycle9.3.3' : [1.42, 0.94, 0.77, 0.62, 0.41, 0.31, 0.22, 0.16],
    'alma.cycle9.3.4' : [0.92, 0.61, 0.50, 0.40, 0.27, 0.20, 0.14, 0.11],
    'alma.cycle9.3.5' : [0.55, 0.36, 0.30, 0.24, 0.16, 0.12, 0.084, 0.063],
    'alma.cycle9.3.6' : [0.31, 0.20, 0.17, 0.13, 0.089, 0.067, 0.047, 0.035],
    'alma.cycle9.3.7' : [0.21, 0.14, 0.11, 0.092, 0.061, 0.046, 0.033, 0.024],
    'alma.cycle9.3.8' : [0.096, 0.064, 0.052, 0.042, 0.028, 0.021, 0.015, 0.011],
    'alma.cycle9.3.9' : [0.057, 0.038, 0.031, 0.025, 0.017, 0.012, 0.088],
    'alma.cycle9.3.10' : [0.042, 0.028, 0.023, 0.018, 0.012, 0.0091]
   
}

def get_spatial_resolution(band, antenna_name):
    if antenna_name == 'alma.cycle.9.3.9':
        assert band <= 9, 'band should be less than 9 for antenna configuration 9'
    elif antenna_name == 'alma.cycle.9.3.10':
        assert band <= 8, 'band should be less than 8 for antenna configuration 10'
    return spatial_resolution_dict[antenna_name][band - 3]

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

def insert_gaussian(id, c_id, datacube, amplitude, pos_x, pos_y, pos_z, fwhm_x, fwhm_y, fwhm_z, pa, n_px, n_chan, plot, plot_dir):
    z_idxs = np.arange(0, n_chan)
    idxs = np.indices([n_px, n_px])
    g = gaussian(z_idxs, 1, pos_z, fwhm_z)
    for z in range(datacube._array.shape[2]):
        ts = threedgaussian(amplitude, 0, z, pos_x, pos_y, fwhm_x, fwhm_y, pa, idxs)
        slice_ = ts + g[z] * ts
        datacube._array[:, :, z, 0] += slice_ * U.Jy * U.pix**-2
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

def generate_gaussian_skymodel(id, data_dir, n_sources, n_px, n_channels, bandwidth, 
                               fwhm_x, fwhm_y, fwhm_z, pixel_size, fov,
                               spatial_resolution, central_frequency, 
                               frequency_resolution, pa, min_sep_spatial, min_sep_frequency, plot, plot_dir):
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
    fwhm_x = int(fwhm_x.value / pixel_size.value)
    fwhm_y = int(fwhm_y.value / pixel_size.value)
    fwhm_z = int(fwhm_z.value / frequency_resolution.value)
    ra = 0 * U.deg
    dec = 0 * U.deg
    hI_rest_frequency = 1420.4 * U.MHz
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
    c_id = 0
    datacube = insert_gaussian(id, c_id, datacube, 1, pos_x, pos_y, pos_z, fwhm_x, fwhm_y, fwhm_z, pa, n_px, n_channels, plot, plot_dir)
    xy_radius = fov / pixel_size * 0.3
    z_radius = 0.3 * bandwidth * U.MHz / frequency_resolution
    print('Generating central source and {} serendipitous companions in a radius of {} pixels in the x and y directions and {} pixels in the z direction\n'.format(n_sources, xy_radius, z_radius))
    min_sep_xy = min_sep_spatial / pixel_size
    min_sep_z = min_sep_frequency / frequency_resolution
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
        datacube = insert_gaussian(id, c_id + 1, datacube, amplitudes[c_id], choords[0], choords[1], choords[2], fwhm_xs[c_id], fwhm_ys[c_id], fwhm_zs[c_id], pas[c_id], n_px, n_channels, plot, plot_dir)
    filename = os.path.join(data_dir, 'skymodel_{}.fits'.format(id))
    write_datacube_to_fits(datacube, filename)
    print('Skymodel saved to {}'.format(filename))
    del datacube
    return filename

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
                 x_rot, y_rot, simulation_str, TNGSnapshotID, TNGSubhaloID, 
                 api_key, data_dir):
    distance = 1 * U.Mpc
    source = TNGSource(simulation_str, TNGSnapshotID, TNGSubhaloID,
                       distance=distance,
                       rotation = {'L_coords': (x_rot, y_rot)},
                       cutout_dir = data_dir,
                       api_key = api_key,
                       ra = 0. * U.deg,
                       dec = 0. * U.deg,
                       )
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
    while (min_x < - 0.5 * n_px * U.pix) or (max_x > 0.5 *n_px * U.pix) or (min_y < 0.5 * n_px * U.pix) or (max_y > 0.5 * n_px * U.pix):
        distance += 10 * U.Mpc
        source = TNGSource(simulation_str, TNGSnapshotID, TNGSubhaloID,
                       distance=distance,
                       rotation = {'L_coords': (x_rot, y_rot)},
                       cutout_dir = os. getcwd(),
                       api_key = api_key,
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
    source = TNGSource(simulation_str, TNGSnapshotID, TNGSubhaloID,
                       distance=distance,
                       rotation = {'L_coords': (x_rot, y_rot)},
                       cutout_dir = os. getcwd(),
                       api_key = api_key,
                       ra = 0. * U.deg,
                       dec = 0. * U.deg,)
    channel_width=10.0 * U.km * U.s**-1
    while (min_z < 0 * U.pix) or (max_z > n_channels * U.pix):
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
        
    return distance, channel_width

def generate_extended_skymodel(id, data_dir, n_px, n_channels, 
                               spatial_resolution, central_frequency, frequency_resolution, 
                               TNGBasePath, TNGSnap, subhaloID, api_key, plot, plot_dir):
    #distance = np.random.randint(1, 5) * U.Mpc
    x_rot = np.random.randint(0, 360) * U.deg
    y_rot = np.random.randint(0, 360) * U.deg
    simulation_str = TNGBasePath.split('/')[-1]

    distance, channel_width = get_distance(n_px, n_channels, x_rot, y_rot,
                         simulation_str, TNGSnap, subhaloID, api_key, data_dir)


    print('Generating extended source from subhalo {} - {} at {} with rotation angles {} and {} in the X and Y planes'.format(simulation_str, subhaloID, distance, x_rot, y_rot))
    
    source = TNGSource(simulation_str, TNGSnap, subhaloID,
                       distance=distance,
                       rotation = {'L_coords': (x_rot, y_rot)},
                       cutout_dir = TNGBasePath,
                       api_key = api_key,
                       ra = 0. * U.deg,
                       dec = 0. * U.deg,)
    hI_rest_frequency = 115.27120*U.GHz 
    radio_hI_equivalence = U.doppler_radio(hI_rest_frequency)
    central_velocity = central_frequency.to(U.km / U.s, equivalencies=radio_hI_equivalence)
    velocity_resolution = frequency_resolution.to(U.km / U.s, equivalencies=radio_hI_equivalence)
   
    
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
    spectral_model = GaussianSpectrum(
        sigma="thermal"
    )
    sph_kernel = AdaptiveKernel(
    (
        CubicSplineKernel(),
        GaussianKernel(truncate=6),
    ),
    vebose=False,

    )

    M = Martini(
        source=source,
        datacube=datacube,
        sph_kernel=sph_kernel,
        spectral_model=spectral_model)
    
    M.insert_source_in_cube(skip_validation=True)
    M.write_hdf5(os.path.join(data_dir, 'skymodel_{}.hdf5'.format(str(id))), channels='velocity')
    f = h5py.File(os.path.join(data_dir, 'skymodel_{}.hdf5'.format(str(id))),'r')
    vch = f['channel_mids'][()] / 1E3 - source.distance.to(U.Mpc).value*70  # m/s to km/s
    f.close()
    os.remove(os.path.join(data_dir, 'skymodel_{}.hdf5'.format(str(id))))
    filename = os.path.join(data_dir, 'skymodel_{}.fits'.format(id))
    #M.write_fits(filename, channels='velocity')
    write_datacube_to_fits(M.datacube, filename)
    print('Skymodel saved to {}'.format(filename))
    if plot is True:
        SkyCube = M.datacube._array.value
        plot_moments(SkyCube[:, :, :, 0], vch, os.path.join(plot_dir, 'skymodel_{}.png'.format(str(id))))
    del datacube
    del M
    return filename

def simulator(i: int, data_dir: str, main_path: str, project_name: str, 
              output_dir: str, plot_dir: str, band: int, antenna_name: str, inbright: float, 
              bandwidth: int, inwidth: float, integration: int, totaltime: int, 
              pwv: float, snr: float, get_skymodel: bool, 
              extended: bool, TNGBasePath: str, TNGSnaphotID: int, 
              TNGSubhaloID: int, TNG_api_key: str,
              plot: bool, save_ms: bool, crop: bool, n_pxs: Optional[int] = None, 
              n_channels: Optional[int] = None):
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
    extended (bool): if True, extended sources are simulated, else point sources
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
    os.chdir(output_dir)
    project = project_name + '_{}'.format(i)
    if not os.path.exists(project):
        os.mkdir(project)
    spatial_resolution = get_spatial_resolution(band, antenna_name)
    central_freq= get_band_central_freq(band)
    fov = get_fov([band])[0]
    pixel_size = spatial_resolution / 7
    n_px = int(1.5 * fov / pixel_size)
    if n_channels is None:
        n_channels = int(bandwidth / inwidth)
    # number of pixels must be even
    if n_px % 2 != 0:
        n_px += 1
    xy_radius = n_px // 2
    length = int(math.sqrt(2) * xy_radius)
    if length % 2 != 0:
        length += 1
    

    print('Simulation Parameters given Band and Spatial Resolution')
    print('Band ', band)
    print('Bandwidth ', bandwidth, ' MHz')
    print('Central Frequency ', central_freq, ' GHz')
    print('pixel_size ', pixel_size, ' arcsec')
    print('fov ', fov, ' arcsec')
    print('spatial_resolution ', spatial_resolution, ' arcsec')
    print('Antenna Configuration ', antenna_name)
    print('Cube Size: {} x {} x {} pixels'.format(n_px, n_px, n_channels))
    print('Final size will be {} x {} x {} pixels'.format(length, length, n_channels))
    print('# ------------------------ #')
    skymodel_time = time.time()
    if get_skymodel is True:
        print('Reading Skymodel from {}'.format(data_dir))
        print('\n')
        files = natsorted(os.listdir(data_dir))
        files = [os.path.join(data_dir, file) for file in files if '.fits' in file]
        filename = files[i]
    else:
        if extended is True:
            print('Generating Extended Emission Skymodel from TNG')
            print('\n')
            filename = generate_extended_skymodel(i, output_dir, n_px, n_channels, 
                                                  spatial_resolution * U.arcsec, central_freq * U.GHz,
                                                  inwidth * U.MHz, TNGBasePath, TNGSnaphotID, TNGSubhaloID, 
                                                  TNG_api_key, plot, plot_dir) 
        else:
            print('Generating Gaussian Skymodel')
            n_sources = np.random.randint(1, 5)
            fwhm_x = (0.1 * fov - 0.01 * fov)*np.random.rand() + 0.01 * fov
            fwhm_y = (0.1 * fov - 0.01 * fov)*np.random.rand() + 0.01 * fov
            fwhm_z = (0.1 * bandwidth - 0.01 * bandwidth)*np.random.rand() + 0.01 * bandwidth
            pa = np.random.randint(0, 360)
            print('Number of Sources ', n_sources)
            print('FWHM_x ', fwhm_x, ' arcsec')
            print('FWHM_y ', fwhm_y, ' arcsec')
            print('FWHM_z ', fwhm_z, ' MHz')
            print('PA ', pa, ' deg')

            min_sep_spatial = 1.5 * pixel_size
            min_sep_frequency = 1.5 * inwidth
            filename = generate_gaussian_skymodel(i, output_dir, n_sources,
                                                  n_px, n_channels, bandwidth, 
                                                  fwhm_x * U.arcsec, fwhm_y * U.arcsec, 
                                                  fwhm_z * U.MHz,
                                                  pixel_size * U.arcsec,  
                                                  fov * U.arcsec, 
                                                  spatial_resolution * U.arcsec, 
                                                  central_freq * U.GHz,
                                                  inwidth * U.MHz, 
                                                  pa, 
                                                  min_sep_spatial, 
                                                  min_sep_frequency, 
                                                  plot, 
                                                  plot_dir)
    
    final_skymodel_time = time.time()
    noise_time = time.time()
    antennalist = os.path.join(main_path, "antenna_config", antenna_name + '.cfg')
    print('# ------------------------ #')
    print('Generating Noise')
    simobserve(
        project=project, 
        skymodel=filename,
        inbright="{}Jy/pix".format(inbright),
        incell="{}arcsec".format(pixel_size),
        indirection="J2000 19h30m00 -40d00m00",
        incenter='{}GHz'.format(central_freq),
        inwidth="{}MHz".format(inwidth),
        setpointings=True,
        integration="{}s".format(integration),
        mapsize=["{}arcsec".format(fov)],
        maptype="square",
        obsmode="int",
        antennalist=antennalist,
        totaltime="{}s".format(totaltime),
        thermalnoise="tsys-atm",
        user_pwv=pwv,
        seed=11111,
        graphics="none",
        verbose=False,
        overwrite=True)
    
    tclean(
        vis=os.path.join(project, "{}.{}.noisy.ms".format(project, antenna_name)),
        imagename=os.path.join(project, '{}.{}'.format(project, antenna_name)),
        imsize=[int(n_px), int(n_px)],
        cell="{}arcsec".format(pixel_size),
        phasecenter="J2000 19h30m00 -40d00m00",
        specmode="cube",
        niter=0,
        fastnoise=False,
        calcpsf=True,
        pbcor=True,
        pblimit=0.2, 
        )
    exportfits(imagename=os.path.join(project, '{}.{}.image'.format(project, antenna_name)), 
           fitsimage=os.path.join(output_dir, "noise_cube_" + str(i) +".fits"), overwrite=True)
    noise, header = load_fits(os.path.join(output_dir, "noise_cube_" + str(i) +".fits"))
    noise_rmse = np.sqrt(np.nanmean(noise**2))
    print('Measured Noise RMSE is {} Jy/pix'.format(noise_rmse))
    final_noise_time = time.time()
    inbright = snr * noise_rmse
    print('Setting new inbright to: {} Jy/pix'.format(round(inbright, 4)))
    print('# ------------------------ #')
    print('Simulating ALMA Observation of the Skymodel')
    sim_time = time.time()
    simobserve(
        project=project, 
        skymodel=filename,
        inbright="{}Jy/pix".format(inbright),
        incell="{}arcsec".format(pixel_size),
        indirection="J2000 19h30m00 -40d00m00",
        incenter='{}GHz'.format(central_freq),
        inwidth="{}MHz".format(inwidth),
        setpointings=True,
        integration="{}s".format(integration),
        mapsize=["{}arcsec".format(fov)],
        maptype="square",
        obsmode="int",
        antennalist=antennalist,
        totaltime="{}s".format(totaltime),
        thermalnoise="tsys-atm",
        user_pwv=pwv,
        seed=11111,
        graphics="none",
        verbose=False,
        overwrite=True)
    
    tclean(
        vis=os.path.join(project, "{}.{}.noisy.ms".format(project, antenna_name)),
        imagename=os.path.join(project, '{}.{}'.format(project, antenna_name)),
        imsize=[int(n_px), int(n_px)],
        cell="{}arcsec".format(pixel_size),
        phasecenter="J2000 19h30m00 -40d00m00",
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
    if n_pxs is not None:
        crop = True
    if crop is True:
        if dirty.shape[1] > 1:
            clean_cube = SpectralCube(clean[0], wcs=WCS(clean_header).dropaxis(3))
            dirty_cube = SpectralCube(dirty[0], wcs=WCS(dirty_header).dropaxis(3))
            if n_pxs is not None:
                length = n_pxs
            left = int((clean_cube.shape[1] - length) / 2)
            clean_cube = clean_cube[:, left:left+length, left:left+length]
            dirty_cube = dirty_cube[:, left:left+length, left:left+length]
            clean_cube.write(os.path.join(output_dir, "clean_cube_" + str(i) +".fits"), overwrite=True)
            dirty_cube.write(os.path.join(output_dir, "dirty_cube_" + str(i) +".fits"), overwrite=True)
        else:
            left = int((clean_cube.shape[1] - length) / 2)
            clean_cube = clean_cube[0, left:left+length, left:left+length]
            dirty_cube = dirty_cube[0, left:left+length, left:left+length]
            save_fits(os.path.join(output_dir, "clean_cube_" + str(i) +".fits"), clean_cube, WCS(clean_header).dropaxis(2).to_header())
            save_fits(os.path.join(output_dir, "dirty_cube_" + str(i) +".fits"), dirty_cube, WCS(dirty_header).dropaxis(2).to_header())

    print('Deleting junk files')
    shutil.rmtree(project)
    os.remove(os.path.join(output_dir, "noise_cube_" + str(i) +".fits"))
    os.remove(os.path.join(output_dir, "skymodel_" + str(i) +".fits"))
    if plot is True:
        print('Saving Plots')
        plotter(i, output_dir, plot_dir)
    stop = time.time()
    print('Skymodel Generated in {} seconds'.format(strftime("%H:%M:%S", gmtime(final_skymodel_time - skymodel_time))))
    print('Noise Computation Took {} seconds'.format(strftime("%H:%M:%S", gmtime(final_noise_time - noise_time))))
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
        dirty_spectrum = np.nansum(dirty[:, :, :], axis=(1, 2))
        clean_image = np.sum(clean[:, :, :], axis=0)
        dirty_image = np.nansum(dirty[:, :, :], axis=0)
    else:
        clean_image = clean.copy()
        dirty_image = dirty.copy()
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(clean_image, origin='lower')
    ax[1].imshow(dirty_image, origin='lower')
    plt.colorbar(ax[0].imshow(clean_image, origin='lower'), ax=ax[0])
    plt.colorbar(ax[1].imshow(dirty_image, origin='lower'), ax=ax[1])
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
    


i = 3
data_dir = '/media/storage'
main_path = '/home/deepfocus/ALMASim'
plot_dir = 'extended_plots'
output_dir = 'extended_sims_test_0'
project_name = 'sim'
band = 6
antenna_name = 'alma.cycle9.3.3'
inbright = 0.01
bandwidth = 1280
inwidth = 10
integration = 10
totaltime = 4500
pwv = 0.3
snr = 30
get_skymodel = False
extended = True
TNGBasePath = '/media/storage/TNG100-1'
TNGSnapshotID = 99
TNGSubhaloID = 487363
api_key = "8f578b92e700fae3266931f4d785f82c"
plot = True
save_ms = False
crop = False
n_pxs = None
n_channels = None

if __name__ == '__main__':
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    output_dir = os.path.join(data_dir, output_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    plot_dir = os.path.join(output_dir, plot_dir)
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)
    simulator(i, data_dir, main_path, project_name, 
              output_dir, plot_dir, band, antenna_name, inbright,
              bandwidth, inwidth, integration, totaltime, 
              pwv, snr, get_skymodel, extended, TNGBasePath, 
              TNGSnapshotID, TNGSubhaloID, api_key,
              plot, save_ms, crop, n_pxs, n_channels)