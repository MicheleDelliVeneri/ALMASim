from ast import arguments
from email.policy import default
from cv2 import _OUTPUT_ARRAY_DEPTH_MASK_16F
import numpy as np
import pandas as pd
import argparse
from astropy.io import fits 
import os
from photutils.aperture import CircularAnnulus, CircularAperture
from tqdm import tqdm
from radio_beam import Beam
import cv2
import astropy.units as u


def measure_snr(img, box):
    y0, x0, y1, x1 = box
    xc, yc = 180, 180
    r0, r1 = 1.6 * (x1 - x0), 2.6 * (x1 - x0)
    r = 0.5 * (x1 - x0)

    noise_aperture = CircularAnnulus((xc, yc), r0 / 2, r1 / 2 )
    mask = noise_aperture.to_mask(method='center')

    source_aperture = CircularAperture((xc, yc), r)
    aperture_mask = source_aperture.to_mask()

    noise_p = mask.multiply(img)
    noise_p = noise_p[mask.data > 0]
    source_p = aperture_mask.multiply(img)
    source_p = source_p[aperture_mask.data > 0.]
    std = np.std(noise_p)
    mean = np.mean(source_p)
    snr = mean / std
    #print('Source Mean: ', mean)
    #print('Noise RMS: ', std)
    return snr

def generate_noise(rms):
    ch, rows, cols = 128, 360, 360
    mean = 0
    gauss = np.random.normal(mean, rms, (ch, rows, cols))
    return gauss

def add_noise(cube, rms):
    bmaj = 0.886 * u.arcsec
    bmin = 0.7691 * u.arcsec
    noise = generate_noise(rms)
    pix_scale = 0.1 * u.arcsec
    gk = np.array(Beam(bmaj, bmin).as_kernel(pix_scale))
    noisy_cube = np.zeros(cube.shape)
    for z in range(cube.shape[0]):
        noisy_cube[z] = cube[z] + cv2.filter2D(noise[z], -1, gk)
    return noisy_cube

def measure_params(input_dataframe, output_dir, n, rms_noise):
    cont, flux, peak, snr = [], [], [], []
    for i in tqdm(range(n)):
        source_params = input_dataframe.loc[input_dataframe.ID == i]
        boxes = np.array(source_params[["y0", "x0", "y1", "x1"]].values)
        dirty_cube = fits.getdata(os.path.join(output_dir, "dirty_cube_{}.fits".format(str(i))))[0]
        if rms_noise != 0.:
            dirty_cube = add_noise(dirty_cube, rms_noise)
            hdu = fits.PrimaryHDU(data=dirty_cube.astype(np.float32))
            hdu.writeto(os.path.join(output_dir, "dirty_cube_{}.fits".format(str(i))), overwrite=True)
        dirty_img = np.sum(dirty_cube, axis=0)
        for j, box in enumerate(boxes):
            source_param = source_params.iloc[j, :]
            z, fwhm_z = int(source_param['z']), int(source_param['fwhm_z'])
            y0, x0, y1, x1 = box
            source_pixels = dirty_cube[z - fwhm_z: z + fwhm_z, y0: y1, x0: x1]
            cont_pixels = np.concatenate((dirty_cube[: z - fwhm_z, y0:y1, x0:x1], dirty_cube[z + fwhm_z: , y0:y1, x0:x1]), axis=0)
            cont.append(np.mean(cont_pixels))
            flux.append(np.sum(source_pixels))
            peak.append(np.max(source_pixels))
            snr.append(measure_snr(dirty_img, box))
    cont = np.array(cont).astype(np.float32)
    flux = np.array(flux).astype(np.float32)
    peak = np.array(peak).astype(np.float32)
    snr = np.array(snr).astype(np.float32)
    input_dataframe['continuum'] = cont
    input_dataframe['flux'] = flux
    input_dataframe['peak'] = peak
    input_dataframe['snr'] = snr
    name = os.path.join(output_dir, 'params.csv')
    input_dataframe.to_csv(name, index=False)



        

parser = argparse.ArgumentParser()
parser.add_argument('input_dir', type=str,
                    help='the directory where the simulated model cubes and the params.csv file are stored')
parser.add_argument('output_dir', type=str,
                    help='the directory where the simulated cubes are stored')
parser.add_argument('catalogue_name', type=str, help='the name of the .csv file')
parser.add_argument('noise_rms', type=float, default='RMS of the noise to add to the cube over the standard noise')

args = parser.parse_args()
input_dir = args.input_dir
output_dir = args.output_dir
catalogue_name = args.catalogue_name
rms_noise = args.noise_rms

input_dataframe = pd.read_csv(os.path.join(input_dir, catalogue_name))
n = len(list(os.listdir(input_dir))) - 1
measure_params(input_dataframe, output_dir, n, rms_noise)
