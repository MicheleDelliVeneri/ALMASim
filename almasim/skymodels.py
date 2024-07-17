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


# ------------------ Martini Modification Class -------------------- #


class MartiniMod(Martini):

    def _evaluate_pixel_spectrum(
        self, ranks_and_ij_pxs, update_progress, progressbar=True
    ):
        """
        Add up contributions of particles to the spectrum in a pixel.
        This is the core loop of MARTINI. It is embarrassingly parallel. To support
        parallel excecution we accept storing up to a copy of the entire (future) datacube
        in one-pixel pieces. This avoids the need for concurrent access to the datacube
        by parallel processes, which would in the simplest case duplicate a copy of the
        datacube array per parallel process! In realistic use cases the memory overhead
        from a the equivalent of a second datacube array should be minimal - memory-
        limited applications should be limited by the memory consumed by particle data,
        which is not duplicated in parallel execution.
        The arguments that differ between parallel ranks must be bundled into one for
        compatibility with `multiprocess`.
        Parameters
        ----------
        rank_and_ij_pxs : tuple
            A 2-tuple containing an integer (cpu "rank" in the case of parallel execution)
            and a list of 2-tuples specifying the indices (i, j) of pixels in the grid.
        Returns
        -------
        out : list
            A list containing 2-tuples. Each 2-tuple contains and "insertion slice" that
            is an index into the datacube._array instance held by this martini instance
            where the pixel spectrum is to be placed, and a 1D array containing the
            spectrum, whose length must match the length of the spectral axis of the
            datacube.
        """
        result = list()
        rank, ij_pxs = ranks_and_ij_pxs
        if progressbar:
            ij_pxs = tqdm(ij_pxs, position=rank)
        for i, ij_px in enumerate(ij_pxs):
            ij = np.array(ij_px)[..., np.newaxis] * U.pix
            mask = (
                np.abs(ij - self.source.pixcoords[:2]) <= self.sph_kernel.sm_ranges
            ).all(axis=0)
            weights = self.sph_kernel._px_weight(
                self.source.pixcoords[:2, mask] - ij, mask=mask
            )
            insertion_slice = (
                np.s_[ij_px[0], ij_px[1], :, 0]
                if self.datacube.stokes_axis
                else np.s_[ij_px[0], ij_px[1], :]
            )
            result.append(
                (
                    insertion_slice,
                    (self.spectral_model.spectra[mask] * weights[..., np.newaxis]).sum(
                        axis=-2
                    ),
                )
            )
            if update_progress is not None:
                update_progress.emit(i / len(ij_pxs) * 100)
        return result

    def _insert_source_in_cube(
        self,
        update_progress=None,
        terminal=None,
        skip_validation=False,
        progressbar=None,
        ncpu=1,
        quiet=None,
    ):
        """
        Populates the :class:`~martini.datacube.DataCube` with flux from the
        particles in the source.

        Parameters
        ----------
        skip_validation : bool, optional
            SPH kernel interpolation onto the DataCube is approximated for
            increased speed. For some combinations of pixel size, distance
            and SPH smoothing length, the approximation may break down. The
            kernel class will check whether this will occur and raise a
            RuntimeError if so. This validation can be skipped (at the cost
            of accuracy!) by setting this parameter True. (Default: ``False``)

        progressbar : bool, optional
            A progress bar is shown by default. Progress bars work, with perhaps
            some visual glitches, in parallel. If martini was initialised with
            `quiet` set to `True`, progress bars are switched off unless explicitly
            turned on. (Default: ``None``)

        ncpu : int
            Number of processes to use in main source insertion loop. Using more than
            one cpu requires the `multiprocess` module (n.b. not the same as
            `multiprocessing`). (Default: ``1``)

        quiet : bool, optional
            If ``True``, suppress output to stdout. If specified, takes precedence over
            quiet parameter of class. (Default: ``None``)
        """

        assert self.spectral_model.spectra is not None

        if progressbar is None:
            progressbar = not self.quiet

        self.sph_kernel._confirm_validation(noraise=skip_validation, quiet=self.quiet)

        ij_pxs = list(
            product(
                np.arange(self._datacube._array.shape[0]),
                np.arange(self._datacube._array.shape[1]),
            )
        )

        for insertion_slice, insertion_data in self._evaluate_pixel_spectrum(
            (0, ij_pxs), update_progress, progressbar=progressbar
        ):
            self._insert_pixel(insertion_slice, insertion_data)

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
        if (quiet is None and not self.quiet) or (quiet is not None and not quiet):
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


# ------------------ SkyModels ------------------------------------- #


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


def gaussian2d(x, y, amp, cen_x, cen_y, fwhm_x, fwhm_y, angle):
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
    """
    Inserts a 3D Gaussian into the datacube at the specified position and amplitude.
    datacube: datacube object
    amplitude: amplitude of the source
    pos_x: x position
    pos_y: y position
    pos_z: z position
    fwhm_x: fwhm in x
    fwhm_y: fwhm in y
    fwhm_z: fwhm in z
    angle: angle of rotation
    n_px: number of pixels in the cube
    n_chan: number of channels in the cube
    """
    X, Y = np.meshgrid(np.arange(n_px), np.arange(n_px))
    z_idxs = np.arange(0, n_chan)
    gs = np.zeros(n_chan)
    for i in range(len(line_fluxes)):
        gs += gaussian(z_idxs, line_fluxes[i], pos_z[i], fwhm_z[i])
    for z in range(0, n_chan):
        cont = gaussian2d(X, Y, continum[z], pos_x, pos_y, fwhm_x, fwhm_y, angle)
        line = gaussian2d(X, Y, gs[z], pos_x, pos_y, fwhm_x, fwhm_y, angle)
        slice_ = cont + line
        datacube._array[:, :, z] += slice_ * U.Jy * U.pix**-2
        if update_progress is not None:
            update_progress.emit(z / n_chan * 100)
    return datacube


def interpolate_array(arr, n_px):
    """Interpolates a 2D array to have n_px pixels while preserving aspect ratio."""
    zoom_factor = n_px / arr.shape[0]
    return zoom(arr, zoom_factor)


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
    cube = np.zeros((n_px, n_px, n_chan))
    for i in range(len(line_fluxes)):
        gs += gaussian(z_idxs, line_fluxes[i], pos_z[i], fwhm_z[i])
    for z in range(0, n_chan):
        cube[:, :, z] += avimg * (continum[z] + gs[z])
        if update_progress is not None:
            update_progress.emit(z / n_chan * 100)
    datacube._array[:, :, :] = cube * U.Jy / U.pix**2
    return datacube


def insert_tng(
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
    ncpu,
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
        update_progress, terminal, skip_validation=True, progressbar=True, ncpu=ncpu
    )
    return M


def insert_extended(
    update_progress,
    terminal,
    datacube,
    tngpath,
    snapshot,
    subhalo_id,
    redshift,
    ra,
    dec,
    api_key,
    ncpu,
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
        ncpu,
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
            ncpu,
        )
        mass_ratio = M.inserted_mass / M.source.input_mass * 100
        if terminal is not None:
            terminal.add_log("Mass ratio: {}%".format(mass_ratio))
    if terminal is not None:
        terminal.add_log("Datacube generated, inserting source")
    return M.datacube


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


def insert_diffuse(
    update_progress, datacube, continum, line_fluxes, pos_z, fwhm_z, n_px, n_chan
):
    z_idxs = np.arange(0, n_chan)
    ts = diffuse_signal(n_px)
    ts = np.nan_to_num(ts)
    ts - np.min(ts)
    ts *= 1 / np.max(ts)
    cube = np.zeros((n_px, n_px, n_chan))
    gs = np.zeros(n_chan)
    for i in range(len(line_fluxes)):
        gs += gaussian(z_idxs, line_fluxes[i], pos_z[i], fwhm_z[i])
    for z in range(0, n_chan):
        cube[:, :, z] += ts * (continum[z] + gs[z])
        if update_progress is not None:
            update_progress.emit(z / n_chan * 100)
    datacube._array[:, :, :] += cube * U.Jy * U.pix**-2
    return datacube


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


def molecular_cloud(n_px):
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


def insert_molecular_cloud(
    update_progress, datacube, continum, line_fluxes, pos_z, fwhm_z, n_pix, n_chan
):
    z_idxs = np.arange(0, n_chan)
    im = molecular_cloud(n_pix)
    im - np.min(im)
    im *= 1 / np.max(im)
    gs = np.zeros(n_chan)
    for i in range(len(line_fluxes)):
        gs += gaussian(z_idxs, line_fluxes[i], pos_z[i], fwhm_z[i])
    cube = np.zeros((n_pix, n_pix, n_chan))
    for i in range(len(line_fluxes)):
        gs += gaussian(z_idxs, line_fluxes[i], pos_z[i], fwhm_z[i])
    for z in range(0, n_chan):
        cube[:, :, z] += im * (continum[z] + gs[z])
        if update_progress is not None:
            update_progress.emit(z / n_chan * 100)
    datacube._array[:, :, :] += cube * U.Jy * U.pix**-2
    return datacube


def distance_1d(p1, p2):
    return math.sqrt((p1 - p2) ** 2)


def distance_2d(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def distance_3d(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)


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
    assert bb1["x1"] < bb1["x2"]
    assert bb1["y1"] < bb1["y2"]
    assert bb2["x1"] < bb2["x2"]
    assert bb2["y1"] < bb2["y2"]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1["x1"], bb2["x1"])
    y_top = max(bb1["y1"], bb2["y1"])
    x_right = min(bb1["x2"], bb2["x2"])
    y_bottom = min(bb1["y2"], bb2["y2"])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1["x2"] - bb1["x1"]) * (bb1["y2"] - bb1["y1"])
    bb2_area = (bb2["x2"] - bb2["x1"]) * (bb2["y2"] - bb2["y1"])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def get_iou_1d(bb1, bb2):
    assert bb1["z1"] < bb1["z2"]
    assert bb2["z1"] < bb2["z2"]
    z_left = max(bb1["z1"], bb2["z1"])
    z_right = min(bb1["z2"], bb2["z2"])
    if z_right < z_left:
        return 0.0
    intersection = z_right - z_left
    bb1_area = bb1["z2"] - bb1["z1"]
    bb2_area = bb2["z2"] - bb2["z1"]
    union = bb1_area + bb2_area - intersection
    return intersection / union


def get_pos(x_radius, y_radius, z_radius):
    x = np.random.randint(-x_radius, x_radius)
    y = np.random.randint(-y_radius, y_radius)
    z = np.random.randint(-z_radius, z_radius)
    return (x, y, z)


def sample_positions(
    terminal,
    pos_x,
    pos_y,
    pos_z,
    fwhm_x,
    fwhm_y,
    fwhm_z,
    n_components,
    fwhm_xs,
    fwhm_ys,
    fwhm_zs,
    xy_radius,
    z_radius,
    sep_xy,
    sep_z,
):
    sample = []
    i = 0
    n = 0
    while (len(sample) < n_components) and (n < 1000):
        new_p = get_pos(xy_radius, xy_radius, z_radius)
        new_p = int(new_p[0] + pos_x), int(new_p[1] + pos_y), int(new_p[2] + pos_z)
        if len(sample) == 0:
            spatial_dist = distance_2d((new_p[0], new_p[1]), (pos_x, pos_y))
            freq_dist = distance_1d(new_p[2], pos_z)
            if spatial_dist < sep_xy or freq_dist < sep_z:
                n += 1
                continue
            else:
                spatial_iou = get_iou(
                    {
                        "x1": new_p[0] - fwhm_xs[i],
                        "x2": new_p[0] + fwhm_xs[i],
                        "y1": new_p[1] - fwhm_ys[i],
                        "y2": new_p[1] + fwhm_ys[i],
                    },
                    {
                        "x1": pos_x - fwhm_x,
                        "x2": pos_x + fwhm_x,
                        "y1": pos_y - fwhm_y,
                        "y2": pos_y + fwhm_y,
                    },
                )
                freq_iou = get_iou_1d(
                    {"z1": new_p[2] - fwhm_zs[i], "z2": new_p[2] + fwhm_zs[i]},
                    {"z1": pos_z - fwhm_z, "z2": pos_z + fwhm_z},
                )
                if spatial_iou > 0.1 or freq_iou > 0.1:
                    n += 1
                    continue
                else:
                    sample.append(new_p)
                    i += 1
                    n = 0
                    if terminal is not None:
                        terminal.add_log("Found {}st component".format(len(sample)))
        else:
            spatial_distances = [
                distance_2d((new_p[0], new_p[1]), (p[0], p[1])) for p in sample
            ]
            freq_distances = [distance_1d(new_p[2], p[2]) for p in sample]
            checks = [
                spatial_dist < sep_xy or freq_dist < sep_z
                for spatial_dist, freq_dist in zip(spatial_distances, freq_distances)
            ]
            if any(checks) is True:
                n += 1
                continue
            else:
                spatial_iou = [
                    get_iou(
                        {
                            "x1": new_p[0] - fwhm_xs[i],
                            "x2": new_p[0] + fwhm_xs[i],
                            "y1": new_p[1] - fwhm_ys[i],
                            "y2": new_p[1] + fwhm_ys[i],
                        },
                        {
                            "x1": p[0] - fwhm_xs[j],
                            "x2": p[0] + fwhm_xs[j],
                            "y1": p[1] - fwhm_ys[j],
                            "y2": p[1] + fwhm_ys[j],
                        },
                    )
                    for j, p in enumerate(sample)
                ]
                freq_iou = [
                    get_iou_1d(
                        {"z1": new_p[2] - fwhm_zs[i], "z2": new_p[2] + fwhm_zs[i]},
                        {"z1": p[2] - fwhm_zs[j], "z2": p[2] + fwhm_zs[j]},
                    )
                    for j, p in enumerate(sample)
                ]
                checks = [
                    spatial_iou > 0.1 or freq_iou > 0.1
                    for spatial_iou, freq_iou in zip(spatial_iou, freq_iou)
                ]
                if any(checks) is True:
                    n += 1
                    continue
                else:
                    i += 1
                    n = 0
                    sample.append(new_p)
                    if terminal is not None:
                        terminal.add_log("Found {}st component".format(len(sample)))

    return sample


def insert_serendipitous(
    terminal,
    update_progress,
    datacube,
    continum,
    cont_sens,
    line_fluxes,
    line_names,
    line_frequencies,
    freq_sup,
    pos_zs,
    fwhm_x,
    fwhm_y,
    fwhm_zs,
    n_px,
    n_chan,
    sim_params_path,
):
    wcs = datacube.wcs
    xy_radius = n_px / 4
    z_radius = n_chan / 2
    n_sources = np.random.randint(1, 5)
    # Generate fwhm for x and y
    fwhm_xs = np.random.randint(1, fwhm_x, n_sources)
    fwhm_ys = np.random.randint(1, fwhm_y, n_sources)
    # generate a random number of lines for each serendipitous source
    if len(line_fluxes) == 1:
        n_lines = np.array([1] * n_sources)
    else:
        n_lines = np.random.randint(1, 3, n_sources)
    # generate the width of the first line based on the first line of the central source
    s_fwhm_zs = np.random.randint(2, fwhm_zs[0], n_sources)
    # get posx and poy of the centtral source
    pos_x, pos_y, _ = datacube.wcs.sub(3).wcs_world2pix(
        datacube.ra, datacube.dec, datacube.spectral_centre, 0
    )
    # get a mininum separation based on spatial dimensions
    sep_x, sep_z = np.random.randint(0, xy_radius), np.random.randint(0, z_radius)
    # get the position of the first line of the central source
    pos_z = pos_zs[0]
    # get maximum continum value
    cont_peak = np.max(continum)
    # get serendipitous continum maximum
    serendipitous_norms = np.random.uniform(cont_sens, cont_peak, n_sources)
    # normalize continum to each serendipitous continum maximum
    serendipitous_conts = np.array(
        [
            continum * serendipitous_norm / cont_peak
            for serendipitous_norm in serendipitous_norms
        ]
    )
    # sample coordinates of the first line
    sample_coords = sample_positions(
        terminal,
        pos_x,
        pos_y,
        pos_z,
        fwhm_x,
        fwhm_y,
        fwhm_zs[0],
        n_sources,
        fwhm_xs,
        fwhm_ys,
        s_fwhm_zs,
        xy_radius,
        z_radius,
        sep_x,
        sep_z,
    )
    # get the rotation angles
    pas = np.random.randint(0, 360, n_sources)
    with open(sim_params_path, "w") as f:
        f.write("\n Injected {} serendipitous sources\n".format(n_sources))
        f.close()
    for c_id, choords in enumerate(sample_coords):
        with open(sim_params_path, "w") as f:
            n_line = n_lines[c_id]
            if terminal is not None:
                terminal.add_log(
                    "Simulating serendipitous source {} with {} lines".format(
                        c_id + 1, n_line
                    )
                )
            s_line_fluxes = np.random.uniform(cont_sens, np.max(line_fluxes), n_line)
            s_line_names = line_names[:n_line]
            if terminal is not None:
                for s_name, s_flux in zip(s_line_names, s_line_fluxes):
                    terminal.add_log("Line {} Flux: {}".format(s_name, s_flux))
            pos_x, pos_y, pos_z = choords
            delta = pos_z - pos_zs[0]
            pos_z = np.array([pos + delta for pos in pos_zs])[:n_line]
            s_ra, s_dec, _ = wcs.sub(3).wcs_pix2world(pos_x, pos_y, 0, 0)
            s_freq = np.array(
                [line_freq + delta * freq_sup for line_freq in line_frequencies]
            )[:n_line]
            fwhmsz = [s_fwhm_zs[0]]
            for _ in range(n_line - 1):
                fwhmsz.append(np.random.randint(2, np.random.choice(fwhm_zs, 1))[0])
            s_continum = serendipitous_conts[c_id]
            f.write("RA: {}\n".format(s_ra))
            f.write("DEC: {}\n".format(s_dec))
            f.write("FWHM_x (pixels): {}\n".format(fwhm_xs[c_id]))
            f.write("FWHM_y (pixels): {}\n".format(fwhm_ys[c_id]))
            f.write("Projection Angle: {}\n".format(pas[c_id]))
            for i in range(len(s_freq)):
                f.write(
                    f"Line: {s_line_names[i]} - Frequency: {s_freq[i]} GHz "
                    f"- Flux: {line_fluxes[i]} Jy - Width (Channels): {fwhmsz[i]}\n"
                )
            datacube = insert_gaussian(
                update_progress,
                datacube,
                s_continum,
                s_line_fluxes,
                pos_x,
                pos_y,
                pos_z,
                fwhm_xs[c_id],
                fwhm_ys[c_id],
                fwhmsz,
                pas[c_id],
                n_px,
                n_chan,
            )
            f.close()
    return datacube


def get_datacube_header(datacube, obs_date):
    datacube.drop_pad()
    datacube.freq_channels()
    wcs_header = datacube.wcs.to_header()
    wcs_header.rename_keyword("WCSAXES", "NAXIS")
    header = fits.Header()
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
    header.append(("OBJECT", "MOCK"))
    header.append(("BUNIT", datacube._array.unit.to_string("fits")))
    header.append(("MJD-OBS", obs_date))
    header.append(("BTYPE", "Intensity"))
    header.append(("SPECSYS", wcs_header["SPECSYS"]))
    return header
