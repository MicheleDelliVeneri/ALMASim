import os 
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from martini.sources import TNGSource
from martini import DataCube, Martini
from martini.beams import GaussianBeam
from martini.noise import GaussianNoise
from martini.spectral_models import GaussianSpectrum
from martini.sph_kernels import AdaptiveKernel, GaussianKernel, CubicSplineKernel, DiracDeltaKernel
import astropy.units as U
import matplotlib.pyplot as plt
import h5py
import numpy as np
import tempfile

temp_dir = tempfile.TemporaryDirectory()
os.environ['MPLCONFIGDIR'] = temp_dir.name


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


def generate_skymodel_cube(i, params, subhaloID, plot_dir, output_dir, TNGBasePath, TNGSnap,
                    n_px=256, n_chan=1024, n_level=0.3):
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

    distance = np.random.randint(3, 30) * U.Mpc
    x_rot = np.random.randint(0, 360) * U.deg
    y_rot = np.random.randint(0, 360) * U.deg

    source = TNGSource(TNGBasePath, TNGSnap, subhaloID,
                       distance=distance,
                       rotation = {'L_coords': (x_rot, y_rot)},
                       ra = 0. * U.deg,
                       dec = 0. * U.deg,
                       )
    datacube = DataCube(
        n_px_x = n_px,
        n_px_y = n_px,
        n_channels = n_chan, 
        px_size = params['spatial_resolution'] * U.arcsec,
        channel_width = params['spectral_resolution'] * U.km / U.s,
        velocity_centre=source.vsys, 
        ra = source.ra,
        dec = source.dec,
    )
    spectral_model = GaussianSpectrum(
        signa="thermal"
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
        spectral_model=spectral_model,
        sph_kernel = sph_kernel)

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
        spectral_model=spectral_model,
        sph_kernel = sph_kernel)

    M.insert_source_in_cube(printfreq=10)
    M.add_noise()
    M.write_fits(os.path.join(output_dir, 'skymodel_{}.fits'.format(str(i))), channels='velocity')
    SkyCube = M.datacube._array.value.T
    plot_moments(SkyCube, vch, os.path.join(plot_dir, 'skymodel_{}.png'.format(str(i))))





obs_params = pd.read_csv('obs_configurations.csv')

parser = argparse.ArgumentParser()
parser.add_argument("i", type=str, 
        help='the index of the simulation to be run;')
parser.add_argument("subHaloID", type=int,
                    help='The subhalo ID of the simulation to be run;')
parser.add_argument("output_dir", type=str, help='The output directory')
parser.add_argument("plot_dir", type=str, help='The plot directory')
parser.add_argument("TNGBasePath", type=str, help='The TNG base path', default='/home/rt2122/Data/TNG100-1')
parser.add_argument("TNGSnap", type=int, help='The TNG snapshot', default=99)
parser.add_argument("n_px", type=int, help='The number of pixels', default=256)
parser.add_argument("n_chan", type=int, help='The number of channels', default=128)
parser.add_argument("n_level", type=float, help='The noise level', default=0.3)


args = parser.parse_args()
subhaloId = args.subHaloID
output_dir = args.output_dir
plot_dir = args.plot_dir
i = args.i
TNGBasePath = args.TNGBasePath
TNGSnap = args.TNGSnap
n_px = args.n_px
n_chan = args.n_chan
n_level = args.n_level

id_ = np.random.randint(0, len(obs_params))   
params = obs_params.iloc[id_, :]

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

generate_skymodel_cube(i, params, subhaloId, plot_dir, output_dir, TNGBasePath, TNGSnap,
                        n_px=n_px, n_chan=n_chan, n_level=n_level)