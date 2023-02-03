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
from martini.sources import TNGSource
from martini import DataCube, Martini
from martini.beams import GaussianBeam
from martini.noise import GaussianNoise
from martini.spectral_models import GaussianSpectrum
from martini.sph_kernels import AdaptiveKernel, GaussianKernel, CubicSplineKernel, DiracDeltaKernel
import astropy.units as U
import matplotlib.pyplot as plt
import h5py

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
    hdu.writeto(data_dir + '/skymodel_cube_{}.fits'.format(str(i)), overwrite=True)
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


def make_extended_cube(i, subhaloID, plot_dir, output_dir, TNGBasePath, TNGSnap,
                           spatial_resolutions, velocity_resolutions, ras, decs,
                           n_levels, distances, x_rots, y_rots, n_px=256, n_chan=1024, save_plots=False):
    # Generate a cube from the parameters
    # params is a dictionary with the following keys
    # spatial_resolution [arcsec]
    # spectral_resolution [km/s]
    # coordinates [J2000]
    # integration time [s]

    # The following parameters are randomly sampled 
    # The antenna configuration from the anenna configuration folder
    # the distance to the source [Mpc]
    # the inclination of the source [deg]

    distance = distances[i]
    n_level = n_levels[i]
    x_rot = x_rots[i]
    y_rot = y_rots[i]
    ra = ras[i]
    dec = decs[i]
    spatial_resolution = spatial_resolutions[i]
    velocity_resolution = velocity_resolutions[i]
    tngid = subhaloID[i]
    tngsnap = TNGSnap[i]
    source = TNGSource(TNGBasePath, tngsnap, tngid,
                       distance=distance * U.Mpc,
                       rotation = {'L_coords': (x_rot * U.deg, y_rot * U.deg)},
                       ra = ra * U.deg,
                       dec = dec * U.deg,
                       )
    datacube = DataCube(
        n_px_x = n_px,
        n_px_y = n_px,
        n_channels = n_chan, 
        px_size = spatial_resolution * U.arcsec,
        channel_width = velocity_resolution * U.km / U.s,
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
        GaussianKernel(truncate=6)
    )
    )

    M = Martini(
        source=source,
        datacube=datacube,
        sph_kernel=sph_kernel,
        spectral_model=spectral_model)

    print('Creating noiseless sky cube....')
    M.insert_source_in_cube(printfreq=10)
    M.write_hdf5('skymodel_{}.hdf5'.format(str(i)), channels='velocity')
    f = h5py.File('skymodel_{}.hdf5'.format(str(i)),'r')
    SkyCube = f['FluxCube'][()]
    vch = f['channel_mids'][()] / 1E3 - source.distance.to(U.Mpc).value*70  # m/s to km/s
    f.close()
    os.remove('skymodel_{}.hdf5'.format(str(i)))
    line = np.sum(SkyCube, axis=(0, 1))
    max_line = np.max(line)
    if n_level is not None:
        noise_level = n_level * max_line
    else:
        n_level = np.random.uniform(1/20, 0.3)
        noise_level = n_level * max_line
    noise = GaussianNoise(
        rms=noise_level * U.Jy * U.arcsec**-2)

    M = Martini(
        source=source,
        datacube=datacube,
        noise=noise,
        sph_kernel=sph_kernel,
        spectral_model=spectral_model)

    M.insert_source_in_cube(printfreq=10)
    M.add_noise()
    M.write_fits(os.path.join(output_dir, 'skymodel_cube_{}.fits'.format(str(i))), channels='velocity')
    SkyCube = M.datacube._array.value.T
    if save_plots:
        plot_moments(SkyCube, vch, os.path.join(plot_dir, 'skymodel_{}.png'.format(str(i))))



class SmartFormatter(argparse.HelpFormatter):

    def _split_lines(self, text, width):
        if text.startswith('R|'):
            return text[2:].splitlines()  
        # this is the RawTextHelpFormatter._split_lines
        return argparse.HelpFormatter._split_lines(self, text, width)



parser = argparse.ArgumentParser(description='Simulate ALMA data cubes from TNG data cubes and Gaussian Simulations',
                                 formatter_class=SmartFormatter)
parser.add_argument("--data_dir", type=str, default='models',
                    help='The directory in wich the simulated model cubes are stored;')
parser.add_argument("--output_dir", type=str, default='sims', 
                    help='The directory in which the alma simulations will be stored;')
parser.add_argument("--plot_dir", type=str, help='The plot directory', default='plots')
parser.add_argument("--csv_name", type=str, default='params.csv', 
                    help='The name of the .csv file in which to store the simulated source parameters;')
parser.add_argument("--mode", type=str, default='gauss',  choices=['gauss', 'extended'],
                    help='The type of model to simulate, either gauss or extended;')
parser.add_argument('--n_simulations', type=int, help='The number of cubes to generate;', default=10, )

parser.add_argument('--antenna_config', type=str, default='antenna_config/alma.cycle9.3.1.cfg', 
        help="The antenna configuration file, if --sample params is set to True, random antenna configurations are sampled from the list of available configurations")
parser.add_argument('--spatial_resolution', type=float, default=0.1, 
        help='Spatial resolution in arcseconds, if --sample params is set to True random resolutions are sampled from real observations')
parser.add_argument('--integration_time', type=int, default=2400, 
        help='Total observation time, iif --sample params is set to True, random observation times are sampled from real observations')
parser.add_argument('--coordinates', type=str, default="J2000 03h59m59.96s -34d59m59.50s", 
        help='Coordinates of the target in the sky as a J2000 string, if --sample params is set to True a random direction is sampled from real observations')
parser.add_argument('--band', type=int, default=6,
        help='ALMA Observing band which determines the central frequency of observation')
parser.add_argument('--bandwidth', type=int, default=1000, 
                    help='observation bandwidht in MHz')
parser.add_argument('--frequency_resolution', type=float, default=10,
                    help='frequency resolution in MHz')
parser.add_argument('--velocity_resolution', type=float, default=10,
                    help='velocity resolution in km/s')
parser.add_argument('--TNGBasePath', type=str, default='TNG100-1/output/',)
parser.add_argument("--TNGSnap", type=int, help='The TNG snapshot', default=99,)
parser.add_argument('--TNGSubhaloID', type=int, help='The TNG subhalo ID', default=385350, )
parser.add_argument("--n_px", type=int, help='The number of pixels', default=256, )
parser.add_argument("--n_chan", type=int, help='The number of channels', default=128, )
parser.add_argument('--ra', type=float, help='The right ascension of the source in degrees', default=0.0,)
parser.add_argument('--dec', type=float, help='The declination of the source in degrees', default=0.0, )
parser.add_argument('--distance', type=float, help='The distance of the source in Mpc', default=0.0, )
parser.add_argument('--noise_level', type=float, help='The noise level as percentage of peak flux', default=0.3, )
parser.add_argument('--sample_params', type=str, help='Whether to sample the parameters from real observations', default=False)
parser.add_argument('--save_plots', type=str, help='Whether to save plots of the simulated cubes', default="False")
parser.add_argument('--sample_selection', type=str, 
                    help="R|"
                          "Flags to select, in case of sampling, which paramters to sample from "
                          "real observations.\n"
                          "The flags are:\n"
                          " a = sample antenna configuration\n"
                          " r = sample spatial resolution\n"
                          " t = sample integration time\n"
                          " c = sample coordinates\n"
                          " b = sample band\n"
                          " B = sample bandwidth\n"
                          " f = sample frequency resolution\n"
                          " v = sample velocity resolution\n"
                          " s = sample Snapshot\n"
                          " i = sample SubhaloID\n"
                          " C = sample Ra and Dec\n"
                          " D = sample Distance\n"
                          " N = sample Noise Level\n"
                          " e = sample all parameters\n\n"   
                          "Example: --sample_selection 'art' will sample the antenna configuration,"
                          "the spatial resolution and the integration time and take the remaining"
                          "parameters from the stdin or from the default values. ",
                          default="e"
                          )
args = parser.parse_args()
data_dir = str(args.data_dir)
output_dir = str(args.output_dir)
csv_name = str(args.csv_name)
n = int(args.n_simulations)
antenna_config = str(args.antenna_config)
spatial_resolution = float(args.spatial_resolution)
integration_time = float(args.integration_time)
coordinates = str(args.coordinates)
alma_band = int(args.band)
bandwidth = int(args.bandwidth)
frequency_resolution = float(args.frequency_resolution)
velocuty_resolution = float(args.velocity_resolution)
mode = args.mode
plot_dir = args.plot_dir
TNGSnap = args.TNGSnap
TNGSubhaloID = args.TNGSubhaloID
n_px = int(args.n_px)
n_chan = int(args.n_chan)
velocity_resolution = args.velocity_resolution
Ra = args.ra
Dec = args.dec
distances = args.distance
noise_level = args.noise_level
save_plots = args.save_plots
sample_params = args.sample_params
tngpath = args.TNGBasePath
sample_selection = args.sample_selection
if save_plots == 'False':
    save_plots = False
else:
    save_plots = True

antenna_configs = os.listdir('antenna_config')

# Selecting the Antenna Configuration
get_antennas = False    
get_spatial_resolution = False
get_integration_time = False
get_velocity_resolution = False
get_bandwidth = False
get_frequency_resolution = False
get_TNGSnap = False
get_TNGSubhalo = False
get_ra_dec = False
get_noise_level = False
get_distance = False
get_band = False
if sample_params == "True":
    if 'e' in sample_selection:
        get_antennas = True
        get_spatial_resolution = True
        get_integration_time = True
        get_bandwidth = True
        get_velocity_resolution = True
        get_frequency_resolution = True
        get_TNGSnap = True
        get_TNGSubhalo = True
        get_ra_dec = True
        get_noise_level = True
        get_distance = True
        get_band = True
    if 'a' in sample_selection:
        get_antennas = True
    if 'r' in sample_selection:
        get_spatial_resolution = True
    if 't' in sample_selection:
        get_integration_time = True
    if 'v' in sample_selection:
        get_velocity_resolution = True
    if 'f' in sample_selection:
        get_frequency_resolution = True
    if 'b' in sample_selection:
        get_band = True
    if 's' in sample_selection:
        get_TNGSnap = True
    if 'i' in sample_selection:
        get_TNGSubhalo = True
    if 'C' in sample_selection:
        get_ra_dec = True
    if 'N' in sample_selection:
        get_noise_level = True
    if 'D' in sample_selection:
        get_distance = True
    if 'B' in sample_selection:
        get_bandwidth = True



# Select Central Frequency From Band
if get_band is False:
    central_freq = get_band_central_freq(alma_band)
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
    print("loading observational parameters ...\n")
    print("-------------------------------------\n")
    obs_db = pd.read_csv('obs_configurations.csv')
    if sample_params == "True":
        print('Sampling Observational Parameters ...')
        print('Using the following additional parameters:')
        print('Models Directory: ', data_dir)
        print('Simulations Directory: ', output_dir)
        print('Plots Directory: ', plot_dir)
        print('Simulations Parameters File Name: ', csv_name)
        print('Simulation Mode: ', mode)
        print('Number of Simulations: ', n)
        print('Plot Flag: ', save_plots)
        print('-------------------------------------\n')

    print('These are the default values, but some of these may be sampled depending on your choice of the --sample_selection flags.')
    print('Using the following parameters:')
    print('Models Directory: ', data_dir)
    print('Simulations Directory: ', output_dir)
    print('Plots Directory: ', plot_dir)
    print('Simulations Parameters File Name: ', csv_name)
    print('Simulation Mode: ', mode)
    print('Number of Simulations: ', n)
    print('Antenna Configuration: ', antenna_config)
    print('Spatial Resolution: ', spatial_resolution)
    print('Integration Time: ', integration_time)
    print('Coordinates: ', coordinates)
    print('ALMA Band: ', alma_band)
    print('Bandwidth: ', bandwidth)
    print('Frequency Resolution: ', frequency_resolution)
    print('Velocity Resolution: ', velocity_resolution)
    print('TNG Snapshot: ', TNGSnap)
    print('TNG Subhalo ID: ', TNGSubhaloID)
    print('Number of Pixels: ', n_px)
    print('Number of Channels: ', n_chan)
    print('RA: ', Ra)
    print('Dec: ', Dec)
    print('Distance: ', distances)
    print('Noise Level: ', noise_level)
    print('Plot Flag: ', save_plots)
    print('Sample Parameters Flag: ', sample_params)
    print('Sample Selection Flag: ', sample_selection)
    print('-------------------------------------\n')

    print('Sampling Flags:')
    print('Get Antennas: ', get_antennas)
    print('Get Spatial Resolution: ', get_spatial_resolution)
    print('Get Integration Time: ', get_integration_time)
    print('Get Velocity Resolution: ', get_velocity_resolution)
    print('Get Bandwidth: ', get_bandwidth)
    print('Get Frequency Resolution: ', get_frequency_resolution)
    print('Get TNG Snap: ', get_TNGSnap)
    print('Get TNG Subhalo: ', get_TNGSubhalo)
    print('Get Ra Dec: ', get_ra_dec)
    print('Get Noise Level: ', get_noise_level)
    print('Get Distance: ', get_distance)
    print('Get Band: ', get_band)
    print('-------------------------------------\n')
  
    if mode == 'gauss':
        print('Generating Gaussian Model Cubes ...')
        xyposs = np.arange(100, 250).astype(float)
        fwhms = np.linspace(2., 8., num=100).astype(float)
        angles = np.linspace(0, 90, num=100).astype(float)
        line_centres = np.arange(10, 110).astype(float)
        line_fwhms = np.linspace(3, 10, num=100).astype(float)
        spectral_indexes = np.linspace(-2, 2, num=100).astype(float)
        amps = np.arange(1, 5, 0.1)
        Parallel(n_cores)(delayed(make_cube)(i, data_dir,
                                         amps, xyposs, fwhms, angles, line_centres,
                                         line_fwhms, spectral_indexes) for i in tqdm(range(n)))
        coords = np.array([coordinates for i in range(n)])
        sps = np.array([spatial_resolution for i in range(n)])
        frs = np.array([frequency_resolution for i in range(n)])
        ints = np.array([integration_time for i in range(n)])
    elif mode == 'extended':
        print('Generating Extended Model Cubes using TNG Simulations ...')
        params = obs_db.sample(n=n, axis=0)
        if get_spatial_resolution:
            sps = params['spatial_resolution [arcsec]'].values
        else:
            sps = np.array([spatial_resolution for i in range(n)])
        if get_velocity_resolution:
            vrs = params['velocity_resolution [Km/s]'].values
        else:
            vrs = np.array([velocity_resolution for i in range(n)])
        if get_bandwidth:
            bws = params['bandwidth [MHz]'].values
        else:
            bws = np.array([bandwidth for i in range(n)])
        if get_frequency_resolution:
            frs = params['frequency_resolution [MHz]'].values
        else:
            frs = np.array([frequency_resolution for i in range(n)])
        if get_integration_time:
            ints = params['integration_time [s]'].values
        else:
            ints = np.array([integration_time for i in range(n)])
        if coordinates is None:
            coords = params['J2000 coordinates'].values
        else:
            coords = np.array([coordinates for i in range(n)])
        if get_TNGSnap:
            snapIDs  = np.random.choice([99, 98, 97], n)
        else:
            snapID = TNGSnap
            snapIDs = np.array([snapID for i in range(n)])
        if get_TNGSubhalo:
            subhaloIDs = np.random.choice([385350, 385351, 385352, 385353], n)
        else:
            subhaloIDs = np.array([TNGSubhaloID for i in range(n)])
        if get_ra_dec:
            ras = params['ra [deg]'].values
            decs = params['dec [deg]'].values
        else:
            ras = np.array([Ra for i in range(n)])
            decs = np.array([Dec for i in range(n)])
        if get_distance:
            distances = np.random.choice(np.arange(3, 30, 1), n)
        else:
            distances = np.array([distances for i in range(n)])
        if get_noise_level:
            n_levels = np.random.choice(np.arange(1/20, 0.3, 0.01), n)
        else:
            n_levels = np.array([noise_level for i in range(n)])
        
        x_rots = np.random.choice(np.arange(0, 360, 1), n)
        y_rots = np.random.choice(np.arange(0, 360, 1), n)
        n_channels = list(np.array(bws / frs).astype(int))
        n_channel = max(set(n_channels), key=n_channels.count)
        print('n_channel: ', n_channel)
        Parallel(n_cores)(delayed(make_extended_cube)(i, subhaloIDs, plot_dir, data_dir, tngpath, snapIDs, 
                                                      sps, vrs, ras, decs, n_levels, distances, x_rots, y_rots,
                                                          n_px, n_channel, save_plots) for i in tqdm(range(n)))
        

    
    if mode == 'gauss':
        print('Cubes Generated, aggregating all params_*.csv into a single parameters.csv file')
        files = os.path.join(data_dir, 'params_*.csv')
        files = glob.glob(files)
        df = pd.concat(map(pd.read_csv, files), ignore_index=True)
        os.system("rm -r {}/*.csv".format(data_dir))
        df = df.sort_values(by="ID")
        df.to_csv(os.path.join(data_dir, csv_name), index=False)
    
    print('Creating textfile for simulations')
    df = open('sims_param.csv', 'w')
    for i in range(n):
        if get_antennas is True:
            ac = os.path.join(os.getcwd(), np.random.choice(antenna_configs))
        else:
            ac = antenna_config

        c = coords[i]
        sp = str(sps[i]) + 'arcsec'
        fr = str(frs[i]) + 'MHz'
        map_size = str(n_px * sps[i]) + 'arcsec'
        it = str(ints[i]) + 's'
        vr = str(vrs[i]) + 'km/s'
        ra = str(ras[i]) + 'deg'
        dec = str(decs[i]) + 'deg'
        dl = str(distances[i]) + 'Mpc'
        nl = str(n_levels[i]) 
        x_rot = str(x_rots[i]) + 'deg'
        y_rot = str(y_rots[i]) + 'deg'
        n_c = str(n_channel) + 'ch'
        sID = str(subhaloIDs[i])
        snapID = str(snapIDs[i])


        #print(data_dir, output_dir, ac, c, sp, central_freq, fr, it, map_size, n_px)
        df.write(str(i) + ',' + data_dir + ',' + output_dir + ',' + ac + ',' + c + ',' + sp + ',' +
                      central_freq + ',' + fr + ',' + it + ','  + map_size + ',' + str(n_px) +
                      ',' + vr + ',' + ra + ',' + dec + ',' + dl + ',' + nl + ',' + x_rot + ',' +
                      y_rot + ',' + n_c, ',' + sID + ',' + snapID)
        df.write('\n')
    df.close()

    df = open('sim_info.csv', 'w')
    df.write('The sims_param.csv file contains the parameters of the simulations. The columns are as follows:\n')
    df.write('1. ID: The ID of the simulation\n')
    df.write('2. data_dir: The directory where the data is stored\n')
    df.write('3. output_dir: The directory where the output sky models and dirty cubes pairs will be stored\n')
    df.write('4. antenna_config: The antenna configuration used for the simulation\n')
    df.write('5. J2000 coordinates: The coordinates of the simulated source\n')
    df.write('6. pixel_size: The pixel size in arcsecond of the simulated cube\n')
    df.write('7. central_freq: The central frequency of the simulated cube\n')
    df.write('8. channel_width: The channel width of the simulated cube in MHz\n')
    df.write('9. integration_time: The integration time of the simulated cube in seconds\n')
    df.write('10. map_size: The spatial size of the simulated cube in arcseconds\n')
    df.write('11. n_px: The number of pixels in the spatial dimensions of the cube\n')
    df.write('12. vlsr: The systemic velocity of the simulated source in km/s\n')
    df.write('13. ra: The right ascension of the simulated source in degrees\n')
    df.write('14. dec: The declination of the simulated source in degrees\n')
    df.write('15. distance: The distance of the simulated source in Mpc\n')
    df.write('16. noise_level: The noise level of the simulated cube as a fraction of the peak brightness\n')
    df.write('17. x_rot: The rotation of the simulated cube in the x direction in degrees\n')
    df.write('18. y_rot: The rotation of the simulated cube in the y direction in degrees\n')
    df.write('19. n_channel: The number of channels in the simulated cube\n')
    df.write('20. subhaloID: The ID of the subhalo from which the simulated source was taken\n')
    df.write('21. snapID: The ID of the snapshot from which the subhalo was taken\n')
    df.write('\n')
    df.write('In order to produce the dirty cubes, run the create_simulations.sh script!\n')
    df.close()

    print(f'Execution took {time.time() - start} seconds')
    print('To see the parameters of the simulations, run: cat sims_param.csv')

