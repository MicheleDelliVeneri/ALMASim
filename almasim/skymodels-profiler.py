import numpy as np
import math
import astropy.units as U
from martini.sources import TNGSource
from martini import DataCube, Martini
from martini.spectral_models import GaussianSpectrum
from martini.sph_kernels import WendlandC2Kernel
import astropy.cosmology.units as cu
from astropy.cosmology import WMAP9
import os
from astropy.io import fits
from itertools import product
from tqdm import tqdm
import matplotlib.image as plimg
from scipy.ndimage import zoom
import nifty8 as ift
import random
from astropy.utils import NumpyRNGContext
from skimage import io
from dask import delayed, compute
from dask.distributed import Client, LocalCluster, progress
from martini import DataCube, Martini
import time


# ----------------------- UTILITY --------------------------------


def interpolate_array(arr, n_px):
    """Interpolates a 2D array to have n_px pixels while preserving aspect ratio."""
    x_zoom_factor = n_px / arr.shape[0]
    y_zoom_factor = n_px / arr.shape[1]
    return zoom(arr, [x_zoom_factor, y_zoom_factor])


def track_progress(update_progress, futures):
    if update_progress is not None:
        total_tasks = len(futures)
        completed_tasks = 0
        # Track progress
        while completed_tasks < total_tasks:
            completed_tasks = sum(f.done() for f in futures)
            progress_value = int((completed_tasks / total_tasks) * 100)
            update_progress.emit(progress_value)  # Emit progress signal
            time.sleep(1)  # Check progress every second


def gaussian(x, amp, cen, fwhm):
    """
    Generates a 1D Gaussian given the following input parameters:
    x: position
    amp: amplitude
    fwhm: fwhm
    """
    # def integrand(x, amp, cen, fwhm):
    #    return np.exp(-(x-cen)**2/(2*(fwhm/2.35482)**2))
    # integral, _ = quad(integrand, -np.inf, np.inf, args=(1, cen, fwhm))
    gaussian = np.exp(-((x - cen) ** 2) / (2 * (fwhm / 2.35482) ** 2))
    if np.sum(gaussian) != 0:
        norm = amp / np.sum(gaussian)
    else:
        norm = amp
    result = norm * gaussian
    # norm = 1 / integral
    return result


# ----------------------- POINTLIKE SIMULATIONS --------------------------------


def insert_pointlike(
    update_progress,
    datacube,
    continum,
    line_fluxes,
    pos_x,
    pos_y,
    pos_z,
    fwhm_z,
    n_chan,
):
    """
    Inserts a point source into the datacube at the specified position and amplitude.
    datacube: datacube object
    amplitude: amplitude of the point source
    pos_x: x position
    pos_y: y position
    pos_z: z position
    fwhm_z: fwhm in z
    n_px: number of pixels in the cube
    n_chan: number of channels in the cube
    """
    z_idxs = np.arange(0, n_chan)
    gs = np.zeros(n_chan)
    for i in range(len(line_fluxes)):
        gs += gaussian(z_idxs, line_fluxes[i], pos_z[i], fwhm_z[i])
        if update_progress is not None:
            update_progress.emit(i / len(line_fluxes) * 100)
    datacube._array[
        pos_x,
        pos_y,
    ] = (
        (continum + gs) * U.Jy * U.pix**-2
    )
    return datacube


# ----------------------- GAUSSIAN SIMULATIONS --------------------------------


@delayed
def gaussian2d(amp, x, y, n_px, cen_x, cen_y, fwhm_x, fwhm_y, angle):
    """
    Generates a 2D Gaussian given the following input parameters:
    x, y: positions
    amp: amplitude
    cen_x, cen_y: centers
    fwhm_x, fwhm_y: FWHMs (full width at half maximum) along x and y axes
    angle: angle of rotation (in degrees)
    """
    angle_rad = math.radians(angle)
    # Rotate coordinates
    xp = (x - cen_x) * np.cos(angle_rad) - (y - cen_y) * np.sin(angle_rad) + cen_x
    yp = (x - cen_x) * np.sin(angle_rad) + (y - cen_y) * np.cos(angle_rad) + cen_y

    gaussian = np.exp(
        -(
            (xp - cen_x) ** 2 / (2 * (fwhm_x / 2.35482) ** 2)
            + (yp - cen_y) ** 2 / (2 * (fwhm_y / 2.35482) ** 2)
        )
    )
    norm = amp / np.sum(gaussian)

    result = norm * gaussian

    return result


def insert_gaussian(
    update_progress,
    datacube,
    continum,
    line_fluxes,
    pos_x,
    pos_y,
    pos_z,
    fwhm_x,
    fwhm_y,
    fwhm_z,
    angle,
    n_px,
    n_chan,
):
    z_idxs = np.arange(0, n_chan)
    x, y = np.meshgrid(np.arange(n_px), np.arange(n_px))
    gs = np.zeros(n_chan)
    skymodel = []
    for i in range(len(line_fluxes)):
        gs += gaussian(z_idxs, line_fluxes[i], pos_z[i], fwhm_z[i])
    for z in range(0, n_chan):
        delayed_result = gaussian2d(
            gs[z] + continum[z], x, y, n_px, pos_x, pos_y, fwhm_x, fwhm_y, angle
        )
        skymodel.append(delayed_result)
    delayed_skymodel = delayed(np.stack)(skymodel, axis=0)
    futures = client.compute([delayed_skymodel])
    track_progress(update_progress, futures)
    skymodel = client.gather(futures)
    datacube._array = skymodel * U.Jy * U.pix**-2
    del skymodel
    del z_idxs, gs, delayed_result, delayed_skymodel
    return datacube


# ----------------------- GALAXY ZOO SIMULATIONS --------------------------------


@delayed
def galaxy_image(avgimg, amp):
    return avgimg * amp


def insert_galaxy_zoo(
    update_progress,
    datacube,
    continum,
    line_fluxes,
    pos_z,
    fwhm_z,
    n_px,
    n_chan,
    data_path,
):
    files = np.array(os.listdir(data_path))
    imfile = os.path.join(data_path, np.random.choice(files))
    img = plimg.imread(imfile).astype(np.float32)
    dims = np.shape(img)
    d3 = min(2, dims[2])
    avimg = np.average(img[:, :, :d3], axis=2)
    avimg -= np.min(avimg)
    avimg *= 1 / np.max(avimg)
    avimg = interpolate_array(avimg, n_px)
    avimg /= np.sum(avimg)
    z_idxs = np.arange(0, n_chan)
    gs = np.zeros(n_chan)
    skymodel = []
    for i in range(len(line_fluxes)):
        gs += gaussian(z_idxs, line_fluxes[i], pos_z[i], fwhm_z[i])
    for z in range(0, n_chan):
        delayed_result = galaxy_image(avimg, gs[z] + continum[z])
        skymodel.append(delayed_result)
    delayed_skymodel = delayed(np.stack)(skymodel, axis=0)
    futures = client.compute([delayed_skymodel])
    track_progress(update_progress, futures)
    skymodel = client.gather(futures)
    datacube._array = skymodel * U.Jy * U.pix**-2
    return datacube


# ----------------------- DIFFUSE SIMULATIONS --------------------------------


def diffuse_signal(n_px):
    ift.random.push_sseq(random.randint(1, 1000))
    space = ift.RGSpace((2 * n_px, 2 * n_px))
    args = {
        "offset_mean": 24,
        "offset_std": (1, 0.1),
        "fluctuations": (5.0, 1.0),
        "loglogavgslope": (-3.5, 0.5),
        "flexibility": (1.2, 0.4),
        "asperity": (0.2, 0.2),
    }

    cf = ift.SimpleCorrelatedField(space, **args)
    exp_cf = ift.exp(cf)
    random_pos = ift.from_random(exp_cf.domain)
    sample = np.log(exp_cf(random_pos))
    data = sample.val[0:n_px, 0:n_px]
    normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    return normalized_data


@delayed
def diffuse_image(diffuse_signal, amp):
    return diffuse_signal * amp


def insert_diffuse(
    update_progress, datacube, continum, line_fluxes, pos_z, fwhm_z, n_px, n_chan
):
    ts = diffuse_signal(n_px)
    ts = np.nan_to_num(ts)
    ts - np.min(ts)
    ts *= 1 / np.max(ts)
    z_idxs = np.arange(0, n_chan)
    gs = np.zeros(n_chan)
    skymodel = []
    for i in range(len(line_fluxes)):
        gs += gaussian(z_idxs, line_fluxes[i], pos_z[i], fwhm_z[i])
    for z in range(0, n_chan):
        delayed_result = diffuse_image(ts, gs[z] + continum[z])
        skymodel.append(delayed_result)
    delayed_skymodel = delayed(np.stack)(skymodel, axis=0)
    futures = client.compute([delayed_skymodel])
    track_progress(update_progress, futures)
    skymodel = client.gather(futures)
    datacube._array = skymodel * U.Jy * U.pix**-2
    return datacube


# ----------------------- MOLECOLAR SIMULATIONS --------------------------------


def make_extended(
    imsize,
    powerlaw=2.0,
    theta=0.0,
    ellip=1.0,
    return_fft=False,
    full_fft=True,
    randomseed=32768324,
):
    """

    Generate a 2D power-law image with a specified index and random phases.

    Adapted from https://github.com/keflavich/image_registration. Added ability
    to make the power spectra elliptical. Also changed the random sampling so
    the random phases are Hermitian (and the inverse FFT gives a real-valued
    image).

    Parameters
    ----------
    imsize : int
        Array size.
    powerlaw : float, optional
        Powerlaw index.
    theta : float, optional
        Position angle of major axis in radians. Has no effect when ellip==1.
    ellip : float, optional
        Ratio of the minor to major axis. Must be > 0 and <= 1. Defaults to
        the circular case (ellip=1).
    return_fft : bool, optional
        Return the FFT instead of the image. The full FFT is
        returned, including the redundant negative phase phases for the RFFT.
    full_fft : bool, optional
        When `return_fft=True`, the full FFT, with negative frequencies, will
        be returned. If `full_fft=False`, the RFFT is returned.
    randomseed: int, optional
        Seed for random number generator.

    Returns
    -------
    newmap : np.ndarray
        Two-dimensional array with the given power-law properties.
    full_powermap : np.ndarray
        The 2D array in Fourier space. The zero-frequency is shifted to
        the centre.
    """
    imsize = int(imsize)

    if ellip > 1 or ellip <= 0:
        raise ValueError("ellip must be > 0 and <= 1.")

    yy, xx = np.meshgrid(np.fft.fftfreq(imsize), np.fft.rfftfreq(imsize), indexing="ij")

    if ellip < 1:
        # Apply a rotation and scale the x-axis (ellip).
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        xprime = ellip * (xx * costheta - yy * sintheta)
        yprime = xx * sintheta + yy * costheta

        rr2 = xprime**2 + yprime**2

        rr = rr2**0.5
    else:
        # Circular whenever ellip == 1
        rr = (xx**2 + yy**2) ** 0.5

    # flag out the bad point to avoid warnings
    rr[rr == 0] = np.nan

    with NumpyRNGContext(randomseed):

        Np1 = (imsize - 1) // 2 if imsize % 2 != 0 else imsize // 2

        angles = np.random.uniform(0, 2 * np.pi, size=(imsize, Np1 + 1))

    phases = np.cos(angles) + 1j * np.sin(angles)

    # Rescale phases to an amplitude of unity
    phases /= np.sqrt(np.sum(phases**2) / float(phases.size))

    output = (rr ** (-powerlaw / 2.0)).astype("complex") * phases

    output[np.isnan(output)] = 0.0

    # Impose symmetry
    # From https://dsp.stackexchange.com/questions/26312/numpys-real-fft-rfft-losing-power
    if imsize % 2 == 0:
        output[1:Np1, 0] = np.conj(output[imsize:Np1:-1, 0])
        output[1:Np1, -1] = np.conj(output[imsize:Np1:-1, -1])
        output[Np1, 0] = output[Np1, 0].real + 1j * 0.0
        output[Np1, -1] = output[Np1, -1].real + 1j * 0.0

    else:
        output[1 : Np1 + 1, 0] = np.conj(output[imsize:Np1:-1, 0])
        output[1 : Np1 + 1, -1] = np.conj(output[imsize:Np1:-1, -1])

    # Zero freq components must have no imaginary part to be own conjugate
    output[0, -1] = output[0, -1].real + 1j * 0.0
    output[0, 0] = output[0, 0].real + 1j * 0.0

    if return_fft:

        if not full_fft:
            return output

        # Create the full power map, with the symmetric conjugate component
        if imsize % 2 == 0:
            power_map_symm = np.conj(output[:, -2:0:-1])
        else:
            power_map_symm = np.conj(output[:, -1:0:-1])

        power_map_symm[1::, :] = power_map_symm[:0:-1, :]

        full_powermap = np.concatenate((output, power_map_symm), axis=1)

        if not full_powermap.shape[1] == imsize:
            raise ValueError(
                "The full output should have a square shape."
                " Instead has {}".format(full_powermap.shape)
            )

        return np.fft.fftshift(full_powermap)

    newmap = np.fft.irfft2(output)

    return newmap


def molecolar_cloud(n_px):
    powerlaw = random.random() * 3.0 + 1.5
    ellip = random.random() * 0.5 + 0.5
    theta = random.random() * 2 * 3.1415927
    im = make_extended(
        n_px,
        powerlaw=powerlaw,
        theta=theta,
        ellip=ellip,
        randomseed=random.randrange(10000),
    )
    return im


@delayed
def molecolar_image(molecolar_cld, amp):
    return molecolar_cld * amp


def insert_molecolar_cloud(
    update_progress, datacube, continum, line_fluxes, pos_z, fwhm_z, n_pix, n_chan
):
    im = molecular_cloud(n_pix)
    im - np.min(im)
    im *= 1 / np.max(im)
    z_idxs = np.arange(0, n_chan)
    gs = np.zeros(n_chan)
    for i in range(len(line_fluxes)):
        gs += gaussian(z_idxs, line_fluxes[i], pos_z[i], fwhm_z[i])
    for z in range(0, n_chan):
        delayed_result = molecolar_image(im, gs[z] + continum[z])
        skymodel.append(delayed_result)
    delayed_skymodel = delayed(np.stack)(skymodel, axis=0)
    futures = client.compute([delayed_skymodel])
    track_progress(update_progress, futures)
    skymodel = client.gather(futures)
    datacube._array = skymodel * U.Jy * U.pix**-2
    return datacube


# ----------------------- HUBBLE SIMULATIONS --------------------------------


@delayed
def hubble_image(hubble, amp):
    return hubble * amp


def insert_hubble(
    update_progress,
    datacube,
    continum,
    line_fluxes,
    pos_z,
    fwhm_z,
    n_pix,
    n_chan,
    data_path,
):
    imfile = os.path.join(data_path, np.random.choice(files))
    img = io.imread(imfile).astype(np.float32)
    dims = np.shape(img)
    d3 = min(2, dims[2])
    avimg = np.average(img[:, :, :d3], axis=2)
    avimg -= np.min(avimg)
    avimg *= 1 / np.max(avimg)
    avimg = interpolate_array(avimg, n_px)
    avimg /= np.sum(avimg)
    z_idxs = np.arange(0, n_chan)
    gs = np.zeros(n_chan)
    for i in range(len(line_fluxes)):
        gs += gaussian(z_idxs, line_fluxes[i], pos_z[i], fwhm_z[i])
    for z in range(0, n_chan):
        delayed_result = hubble_image(avimg, gs[z] + continum[z])
        skymodel.append(delayed_result)
    delayed_skymodel = delayed(np.stack)(skymodel, axis=0)
    futures = client.compute([delayed_skymodel])
    track_progress(update_progress, futures)
    skymodel = client.gather(futures)
    datacube._array = skymodel * U.Jy * U.pix**-2
    return datacube


# ----------------------- TNG SIMULATIONS --------------------------------


if __name__ == "__main__":
    cluster = LocalCluster()
    client = Client(cluster, timeout=60, heartbeat_interval=10)
    update_progress = None
    n_pix = 256
    n_channels = 960
    cell_size = 0.2 * U.arcsec
    central_freq = 142 * U.GHz
    ra = 120 * U.deg
    dec = 30 * U.deg
    delta_freq = 100 * U.MHz
    datacube = DataCube(
        n_px_x=n_pix,
        n_px_y=n_pix,
        n_channels=n_channels,
        px_size=cell_size,
        channel_width=delta_freq,
        spectral_centre=central_freq,
        ra=ra,
        dec=dec,
    )
    t1 = time.time()
    pos_x = [128]
    pos_y = [128]
    pos_z = [480]
    fwhm_z = [10]
    fwhm_x = 3
    fwhm_y = 3
    angle = 0
    continum = np.ones(n_channels)
    line_fluxes = [2]
    datacube = insert_gaussian(
        update_progress,
        datacube,
        continum,
        line_fluxes,
        pos_x,
        pos_y,
        pos_z,
        fwhm_x,
        fwhm_y,
        fwhm_z,
        angle,
        n_pix,
        n_channels,
    )
    t2 = time.time()
    print(f"Time taken: {t2-t1}")
    datacube = DataCube(
        n_px_x=n_pix,
        n_px_y=n_pix,
        n_channels=n_channels,
        px_size=cell_size,
        channel_width=delta_freq,
        spectral_centre=central_freq,
        ra=ra,
        dec=dec,
    )
    datacube = insert_galaxy_zoo(
        update_progress,
        datacube,
        continum,
        line_fluxes,
        pos_z,
        fwhm_z,
        n_pix,
        n_channels,
        "/usr/Michele/GalaxyZoo/images_gz2/images",
    )
