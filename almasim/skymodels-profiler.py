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
    client,
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
    client,
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
    client,
    update_progress, 
    datacube, continum, 
    line_fluxes, pos_z, fwhm_z, n_px, n_chan
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
    client,
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
    client,
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

@delayed
def insert_pixel(self, datacube_array, insertion_slice, insertion_data):
    """
    Insert the spectrum for a single pixel into the datacube array.
    """
    datacube_array[insertion_slice] = insertion_data
    return

@delayed
def evaluate_pixel_spectrum(ranks_and_ij_pxs, datacube_array, pixcoords, 
    kernel_sm_ranges, kernel_px_weights, datacube_strokes_axis, spectral_model_spectra):
    """
    Add up contributions of particles to the spectrum in a pixel.
    """
    result = list()
    rank, ij_pxs = ranks_and_ij_pxs
    for i, ij_px in enumerate(ij_pxs):
        ij = np.array(ij_px)[..., np.newaxis] * U.pix
        mask = (
            np.abs(ij - pixcoords[:2]) <= kernel_sm_ranges
        ).all(axis=0)
        weights = kernel_px_weights(
            pixcoords[:2, mask] - ij, mask=mask
        )
        insertion_slice = (
            np.s_[ij_px[0], ij_px[1], :, 0]
            if datacube_stokes_axis
            else np.s_[ij_px[0], ij_px[1], :]
        )
        result.append(
            (
                insertion_slice,
                (self.spectral_model_spectra[mask] * weights[..., np.newaxis]).sum(
                    axis=-2
                ),
            )
        )
    return result

class MartiniMod(Martini):
    
    def _insert_source_in_cube(
        self,
        client,
        update_progress=None,
        terminal=None,
    ):
        assert self.spectral_model.spectra is not None

        self.sph_kernel._confirm_validation(noraise=True, quiet=True)

        # Scatter the datacube array across the workers
        scattered_array = client.scatter(self._datacube._array, broadcast=True)

        ij_pxs = list(
            product(
                np.arange(self._datacube._array.shape[0]),
                np.arange(self._datacube._array.shape[1]),
            )
        )

        # Parallel execution with Dask (let Dask decide how to distribute the tasks)
        delayed_results = []
        # Split the pixel grid among workers
        for icpu in range(len(ij_pxs)):
            # Directly call the delayed method (no need for explicit dask.delayed)
            delayed_result = evaluate_pixel_spectrum(
                (icpu, [ij_pxs[icpu]]), scattered_array,
                self.source.pixels_coords, 
                self.sph_kernel._sm_ranges,
                self.sph_kernel._px_weight,
                self._datacube.stokes_axis,
                self.spectral_model.spectra
            )
            delayed_results.append(delayed_result)

        # Compute all the delayed tasks in parallel
        futures = dask.compute(*delayed_results)
        track_progress(update_progress, futures)
        # Process the results and insert into the scattered datacube array
        for result in futures:
            for insertion_slice, insertion_data in result:
                insert_pixel(scattered_array, insertion_slice, insertion_data)

        # Gather the results back to the local datacube array
        self._datacube._array = client.gather(scattered_array)

        # Final operations on the datacube
        self._datacube._array = self._datacube._array.to(
            U.Jy / U.arcsec**2, equivalencies=[self._datacube.arcsec2_to_pix]
        )
        pad_mask = (
            np.s_[
                self._datacube.padx : -self._datacube.padx,
                self._datacube.pady : -self._datacube.pady,
                ...,
            ]
            if self._datacube.padx > 0 and self._datacube.pady > 0
            else np.s_[...]
        )
        inserted_flux_density = np.sum(
            self._datacube._array[pad_mask] * self._datacube.px_size**2
        ).to(U.Jy)
        inserted_mass = (
            2.36e5
            * U.Msun
            * self.source.distance.to_value(U.Mpc) ** 2
            * np.sum(
                (self._datacube._array[pad_mask] * self._datacube.px_size**2)
                .sum((0, 1))
                .squeeze()
                .to_value(U.Jy)
                * np.abs(np.diff(self._datacube.velocity_channel_edges)).to_value(
                    U.km / U.s
                )
            )
        )
        self.inserted_mass = inserted_mass
        if terminal is not None:
            terminal.add_log(
                "Source inserted.\n"
                f"  Flux density in cube: {inserted_flux_density:.2e}\n"
                f"  Mass in cube (assuming distance {self.source.distance:.2f} and a"
                f" spatially resolved source):"
                f" {inserted_mass:.2e}"
                f"    [{inserted_mass / self.source.input_mass * 100:.0f}%"
                f" of initial source mass]\n"
                f"  Maximum pixel: {self._datacube._array.max():.2e}\n"
                "  Median non-zero pixel:"
                f" {np.median(self._datacube._array[self._datacube._array > 0]):.2e}"
            )
        return

def insert_tng(
    client,
    update_progress,
    terminal,
    n_px,
    n_channels,
    freq_sup,
    snapshot,
    subhalo_id,
    distance,
    x_rot,
    y_rot,
    tngpath,
    ra,
    dec,
    api_key,
):
    source = TNGSource(
        simulation="TNG100-1",
        snapNum=snapshot,
        subID=subhalo_id,
        cutout_dir=tngpath,
        distance=distance * U.Mpc,
        rotation={"L_coords": (x_rot, y_rot)},
        ra=ra,
        dec=dec,
        api_key=api_key,
    )

    datacube = DataCube(
        n_px_x=n_px,
        n_px_y=n_px,
        n_channels=n_channels,
        px_size=10 * U.arcsec,
        channel_width=freq_sup,
        spectral_centre=source.vsys,
        ra=source.ra,
        dec=source.dec,
    )
    spectral_model = GaussianSpectrum(sigma="thermal")
    sph_kernel = WendlandC2Kernel()
    M = MartiniMod(
        source=source,
        datacube=datacube,
        sph_kernel=sph_kernel,
        spectral_model=spectral_model,
        quiet=False,
    )
    M._insert_source_in_cube(
        client, update_progress, 
        terminal,
    )
    return M

def insert_extended(
    client, 
    update_progress, 
    datacube,
     tngpath,
    snapshot,
    subhalo_id,
    redshift,
    ra,
    dec,
    api_key,
    ):
    x_rot = np.random.randint(0, 360) * U.deg
    y_rot = np.random.randint(0, 360) * U.deg
    tngpath = os.path.join(tngpath, "TNG100-1", "output")
    redshift = redshift * cu.redshift
    distance = redshift.to(U.Mpc, cu.redshift_distance(WMAP9, kind="comoving"))
    if terminal is not None:
        terminal.add_log(
            "Computed a distance of {} for redshift {}".format(distance, redshift)
        )
    distance = 50
    M = insert_tng(
        client,
        update_progress,
        terminal,
        datacube.n_px_x,
        datacube.n_channels,
        datacube.channel_width,
        snapshot,
        subhalo_id,
        distance,
        x_rot,
        y_rot,
        tngpath,
        ra,
        dec,
        api_key
    )
    initial_mass_ratio = M.inserted_mass / M.source.input_mass * 100
    if terminal is not None:
        terminal.add_log("Mass ratio: {}%".format(initial_mass_ratio))
    mass_ratio = initial_mass_ratio
    while mass_ratio < 50:
        if mass_ratio < 10:
            distance = distance * 8
        elif mass_ratio < 20:
            distance = distance * 5
        elif mass_ratio < 30:
            distance = distance * 2
        else:
            distance = distance * 1.5
        if terminal is not None:
            terminal.add_log("Injecting source at distance {}".format(distance))
        M = insert_tng(
            client,
            update_progress,
            terminal,
            datacube.n_px_x,
            datacube.n_channels,
            datacube.channel_width,
            snapshot,
            subhalo_id,
            distance,
            x_rot,
            y_rot,
            tngpath,
            ra,
            dec,
            api_key,
        )
        mass_ratio = M.inserted_mass / M.source.input_mass * 100
        if terminal is not None:
            terminal.add_log("Mass ratio: {}%".format(mass_ratio))
    if terminal is not None:
        terminal.add_log("Datacube generated, inserting source")
    return M.datacube
