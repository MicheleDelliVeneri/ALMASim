import sys
import os
import pandas as pd
current_path = os.getcwd()
parent_dir = os.path.join(current_path, "..")
print("Current working directory:", current_path)
print("Path to the parent directory:",parent_dir)
sys.path.append(parent_dir)
import numpy as np
import astropy.units as U
from casatasks import exportfits, simobserve, tclean, gaincal, applycal
from casatools import table
from casatools import simulator as casa_simulator
import random
import shutil
from astropy.constants import c
import math
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from scipy.optimize import curve_fit
import  utility.skymodels as sku
import utility.alma as au
import utility.astro as uas
import utility.plotting as uplt
import utility.skymodels as usk

continum = [0.00032362, 0.00032942, 0.00033533, 0.00034136, 0.00034753, 0.00035381,
        0.00036021, 0.00036672, 0.00037327, 0.00037988, 0.00038661, 0.00039345,
        0.00040031, 0.00040552, 0.00041079, 0.00041612, 0.00042153, 0.000426,
        0.00042939, 0.00043281, 0.00043626, 0.00043974, 0.00044337, 0.00044704,
        0.00045075, 0.00045449, 0.00046024, 0.00047049, 0.00048097, 0.00049168,
        0.00050263, 0.00051723, 0.00053354, 0.00055037, 0.00056773]

flux = [1.28982205e-06, 1.52e-6]
n_channels = len(continum)
cental_index = n_channels // 2
source_index = [8, 15] 
source_flux = [flux[i] + continum[source_index[i]] for i in range(len(source_index))]
source_fwhm = [2, 3]
fwhm_x, fwhm_y = 3, 3
pos_x, pos_y = 64, 64
n_px = 128

datacube = usk.DataCube(
        n_px_x=n_px, 
        n_px_y=n_px,
        n_channels=n_channels)

datacube = usk.insert_pointlike(datacube, continum, source_flux, pos_x, pos_y, source_index, source_fwhm, n_channels)
print(np.sum(datacube._array), np.sum(continum) + np.sum(source_flux))

datacube = usk.DataCube(
        n_px_x=n_px, 
        n_px_y=n_px,
        n_channels=n_channels)

datacube = usk.insert_gaussian(datacube, continum, source_flux, pos_x, pos_y, source_index, fwhm_x, fwhm_y, 
        source_fwhm, 0, n_px, n_channels)
print(np.sum(datacube._array), np.sum(continum) + np.sum(source_flux))
