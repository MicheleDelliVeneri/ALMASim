import numpy as np
from math import pi
import random
from astropy.io import fits
import pandas as pd
import os
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing
import time
import glob
import argparse
from skimage.measure import regionprops, label
import math


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


def generate_component(master_cube, boxes, amp, line_amp, pos_x, pos_y, fwhm_x, fwhm_y, pos_z, fwhm_z, pa, spind):
    z_idxs = np.arange(0, 128)
    idxs = np.indices([360, 360])
    g = gaussian(z_idxs, line_amp, pos_z, fwhm_z)
    cube = np.zeros((128, 360, 360))
    for z in range(cube.shape[0]):
        ts = threedgaussian(amp, spind, z, pos_x, pos_y,
                            fwhm_x, fwhm_y, pa, idxs)
        cube[z] += ts + g[z] * ts
    img = np.sum(cube, axis=0)
    tseg = (img - np.min(img)) / (np.max(img) - np.min(img))
    std = np.std(tseg)
    tseg[tseg >= 3 * std] = 1
    tseg = tseg.astype(int)

    props = regionprops(label(tseg, connectivity=2))
    y0, x0, y1, x1 = props[0].bbox
    boxes.append([y0, x0, y1, x1])
    master_cube += cube
    return master_cube


def genpt(posxy):
    return (random.choice(posxy), random.choice(posxy))


def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def make_cube(i, data_dir, amps, xyposs, fwhms, angles, line_centres, line_fwhms, spectral_indexes):
    n_components = random.randint(2, 5)
    params = []
    columns = ['ID', 'amp', 'line_amp', 'pa', 'spind',
               'fwhm_x', 'fwhm_y', 'fwhm_z', 'z']
    pa = np.random.choice(angles)
    amp = np.random.choice(amps)
    line_amp = np.random.choice(amps)
    limit = amp + line_amp * amp
    pos_x, pos_y = 180, 180
    pos_z = np.random.choice(line_centres)
    fwhm_z = np.random.choice(line_fwhms)
    fwhm_x = np.random.choice(fwhms)
    fwhm_y = np.random.choice(fwhms)
    spind = np.random.choice(spectral_indexes)
    master_cube = np.zeros((128, 360, 360))
    boxes = []
    master_cube = generate_component(
        master_cube, boxes, amp, line_amp, pos_x, pos_y, fwhm_x, fwhm_y, pos_z, fwhm_z, pa, spind)
    params.append([int(i), round(amp, 2), round(line_amp, 2), round(pa, 2), round(spind, 2),
                   round(fwhm_x, 2), round(fwhm_y, 2), round(fwhm_z, 2), round(pos_z, 2)])
    mindist = 20
    sample = []
    while len(sample) < n_components:
        newp = genpt(xyposs)
        for p in sample:
            if distance(newp, p) < mindist:
                break
        else:
            sample.append(newp)

    for j in range(n_components):
        pos_x = sample[j][0]
        pos_y = sample[j][1]
        fwhm_x = np.random.choice(fwhms)
        fwhm_y = np.random.choice(fwhms)
        pa = np.random.choice(angles)
        spind = np.random.choice(spectral_indexes)
        pos_z = np.random.choice(line_centres)
        fwhm_z = np.random.choice(line_fwhms)
        temp = 99
        while temp > limit:
            amp = np.random.choice(amps)
            line_amp = np.random.choice(amps)
            temp = amp + line_amp * amp
        master_cube = generate_component(
            master_cube, boxes, amp, line_amp, pos_x, pos_y, fwhm_x, fwhm_y, pos_z, fwhm_z, pa, spind)
        params.append([int(i), round(amp, 2), round(line_amp, 2), round(pa, 2), round(spind, 2),
                       round(fwhm_x, 2), round(fwhm_y, 2), round(fwhm_z, 2), round(pos_z, 2)])
    hdu = fits.PrimaryHDU(data=master_cube.astype(np.float32))
    hdu.writeto(data_dir + '/gauss_cube_{}.fits'.format(str(i)), overwrite=True)
    boxes = np.array(boxes)
    params = np.array(params)
    df = pd.DataFrame(params, columns=columns)
    xs = boxes[:, 1] + 0.5 * (boxes[:, 3] - boxes[:, 1])
    ys = boxes[:, 0] + 0.5 * (boxes[:, 2] - boxes[:, 0])

    df['x'] = xs
    df['y'] = ys
    df['x0'] = boxes[:, 1]
    df['y0'] = boxes[:, 0]
    df['x1'] = boxes[:, 3]
    df['y1'] = boxes[:, 2]
    df.to_csv(os.path.join(data_dir, 'params_' + str(i) + '.csv'), index=False)

def get_band_central_freq(band):
    if band == 3:
        return '100GHz'
    elif band == 6:
        return '250GHz'
    elif band == 9:
        return '650GHz'


xyposs = np.arange(100, 250).astype(float)
fwhms = np.linspace(2., 8., num=100).astype(float)
angles = np.linspace(0, 90, num=100).astype(float)
line_centres = np.arange(10, 110).astype(float)
line_fwhms = np.linspace(3, 10, num=100).astype(float)
spectral_indexes = np.linspace(-2, 2, num=100).astype(float)
amps = np.arange(1, 5, 0.1)



parser = argparse.ArgumentParser()
parser.add_argument("data_dir", type=str,
                    help='The directory in wich the simulated model cubes are stored;')
parser.add_argument("output_dir", type=str,
                    help='The directory in which the alma simulations will be stored;')
parser.add_argument("csv_name", type=str,
                    help='The name of the .csv file in which to store the simulated source parameters;')
parser.add_argument('n', type=int, help='The number of cubes to generate;')
parser.add_argument('antenna_config', type=str, default='antenna_config/alma.cycle9.3.1.cfg',
        help="The antenna configuration file, if set to None random antenna configurations are sampled from the list of available configurations")
parser.add_argument('spatial_resolution', type=float, default=0.1, 
        help='Spatial resolution in arcseconds, if set to None random resolutions are sampled from real observations')
parser.add_argument('integration_time', type=int, default=2400,
        help='Total observation time, if set to None random observation times are sampled from real observations')
parser.add_argument('coordinates', type=str, default="J2000 03h59m59.96s -34d59m59.50s",
        help='Coordinates of the target in the sky as a J2000 string, if set to None a random direction is sampled from real observations')
parser.add_argument('band', type=int, default=6,
        help='ALMA Observing band which determines the central frequency of observation')
parser.add_argument('bandwidth', type=int, default=1000, 
                    help='observation bandwidht in MHz')
parser.add_argument('frequency_resolution', type=float, default=10,
                    help='frequency resolution in MHz')



args = parser.parse_args()

data_dir = args.data_dir
output_dir = args.output_dir
csv_name = args.csv_name
n = args.n
antenna_config = args.antenna_config
spatial_resolution = args.spatial_resolution
integration_time = args.integration_time
coordinates = args.coordinates
band = args.band
bandwidth = args.bandwidth
frequency_resolution = args.frequency_resolution
\
antenna_configs = os.listdir('antenna_config')

# Selecting the Antenna Configuration
get_antennas = False               
if antenna_config is None:
    get_antennas = True

get_spatial_resolution = False
if spatial_resolution is None:
    get_spatial_resolution = True

get_integration_time = False
if integration_time is None:
    get_integration_time = True

# Select Central Frequency From Band
if band is not None:
    central_freq = get_band_central_freq(band)
else:
    central_freq = get_band_central_freq(np.random.choice([3, 6, 9]))


if not os.path.exists(data_dir):
    os.mkdir(data_dir)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

n_cores = multiprocessing.cpu_count() // 4

if __name__ == '__main__':
    start = time.time()
    print('Generating Model Cubes ...')
    Parallel(n_cores)(delayed(make_cube)(i, data_dir,
                                         amps, xyposs, fwhms, angles, line_centres,
                                         line_fwhms, spectral_indexes) for i in tqdm(range(n)))
    print('Cubes Generated, aggregating parameters.csv and loading observational parameters')
    obs_db = pd.read_csv('obs_configuration.csv')
    files = os.path.join(data_dir, 'params_*.csv')
    files = glob.glob(files)
    df = pd.concat(map(pd.read_csv, files), ignore_index=True)
    os.system("rm -r {}/*.csv".format(data_dir))
    df = df.sort_values(by="ID")
    df.to_csv(os.path.join(data_dir, csv_name), index=False)
    print('Creating textfile for simulations')
    df = open('sims_param.csv', 'w')
    for i in range(n):
        sampled_obs_parameters = obs_db.sample(n=1, axis=0)
        if get_antennas:
            ac = os.path.join(os.getcwd(), np.random.choice(antenna_configs))
        else:
            ac = antenna_config
        if get_spatial_resolution:
            sp = str(sampled_obs_parameters['spatial_resolution'].values)+'arcsec'
        else:
            sp = str(spatial_resolution)+'arcsec'
        if get_integration_time:
            it = str(sampled_obs_parameters['integration_time'].values)+'s'
        else:
            it = str(integration_time)+'s'
        
        

        
        
        df.write(str(i) + ',' + data_dir + ',' + output_dir + ',' + ac)
        df.write('\n')
    df.close()
    print(f'Execution took {time.time() - start} seconds')
