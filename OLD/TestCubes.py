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

def plot_moments(FluxCube, path):
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


myBasePath = 'TNG100-1/output/'
mySnap = 99
myId = 385350 

source = TNGSource(
    myBasePath,
    mySnap,
    myId,
    distance=30 * U.Mpc,
    rotation={'L_coords': (60 * U.deg, 0. * U.deg)},
    ra=0. * U.deg,
    dec=0. * U.deg
)

datacube = DataCube(
    n_px_x=128,
    n_px_y=128,
    n_channels=64,
    px_size=10. * U.arcsec,
    channel_width=40. * U.km * U.s ** -1,
    velocity_centre=source.vsys,
    ra=source.ra,
    dec=source.dec
)

beam = GaussianBeam(
    bmaj=30. * U.arcsec,
    bmin=30. * U.arcsec,
    bpa=0. * U.deg,
    truncate=3.
)

noise = GaussianNoise(
    rms=2.E-6 * U.Jy * U.arcsec ** -2
)

spectral_model = GaussianSpectrum(
    sigma='thermal'
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
    beam=beam,
    noise=noise,
    spectral_model=spectral_model,
    sph_kernel=sph_kernel
)

print('Creating noiseless sky cube....')
M.insert_source_in_cube(printfreq=10)
M.write_hdf5('skymodel_test.hdf5', channels='velocity')

print('Adding noise and convolving with beam....')
M.add_noise()
M.convolve_beam()
M.write_hdf5('dirty_test.hdf5', channels='velocity')

f = h5py.File('skymodel_test.hdf5','r')
SkyCube = f['FluxCube'][()]
vch = f['channel_mids'][()] / 1E3 - source.distance.to(U.Mpc).value*70  # m/s to km/s
f.close()

f = h5py.File('dirty_test.hdf5','r')
DirtyCube = f['FluxCube'][()]
vch = f['channel_mids'][()] / 1E3 - source.distance.to(U.Mpc).value*70  # m/s to km/s
f.close()

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
im0 = ax[0].imshow(SkyCube[:, :, 32], origin='lower', cmap='viridis')
ax[0].set_xlabel('x [px = arcsec/10]')
ax[0].set_ylabel('y [px = arcsec/10]')
plt.colorbar(im0, ax=ax[0], label='Flux [Jy/beam]')
im1 = ax[1].imshow(DirtyCube[:, :, 32], origin='lower', cmap='viridis')
ax[1].set_xlabel('x [px = arcsec/10]')
ax[1].set_ylabel('y [px = arcsec/10]')
plt.colorbar(im1, ax=ax[1], label='Flux [Jy/beam]')
plt.savefig('TestCubes.png', dpi=300)


plot_moments(SkyCube, 'TestMoments_Sky.png')
plot_moments(DirtyCube, 'TestMoments_Dirty.png')