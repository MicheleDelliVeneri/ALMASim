"""Molecular cloud sky model implementation."""

import random
from typing import Any

import astropy.units as U
import numpy as np
from astropy.utils import NumpyRNGContext
from dask import delayed

from .base import SkyModel
from .utils import gaussian, track_progress


def make_extended(
    imsize: int,
    powerlaw: float = 2.0,
    theta: float = 0.0,
    ellip: float = 1.0,
    return_fft: bool = False,
    full_fft: bool = True,
    randomseed: int = 32768324,
) -> np.ndarray:
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
    np.ndarray
        Two-dimensional array with the given power-law properties.
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
                "The full output should have a square shape. Instead has {}".format(
                    full_powermap.shape
                )
            )

        return np.fft.fftshift(full_powermap)

    newmap = np.fft.irfft2(output)
    return newmap


def molecular_cloud(n_px: int) -> np.ndarray:
    """Generate a molecular cloud image."""
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
def molecular_image(molecular_cld: np.ndarray, amp: float) -> np.ndarray:
    """Scale molecular cloud image by amplitude."""
    return molecular_cld * amp


class MolecularCloudSkyModel(SkyModel):
    """Molecular cloud sky model."""

    def insert(self) -> Any:
        """Insert molecular cloud source into datacube."""
        im = molecular_cloud(self.n_px)
        im -= np.min(im)
        im *= 1 / np.max(im)
        z_idxs = np.arange(0, self.n_chan)
        gs = np.zeros(self.n_chan)
        skymodel = []
        for i in range(len(self.line_fluxes)):
            gs += gaussian(z_idxs, self.line_fluxes[i], self.pos_z[i], self.fwhm_z[i])
        for z in range(0, self.n_chan):
            delayed_result = molecular_image(im, gs[z] + self.continuum[z])
            skymodel.append(delayed_result)
        delayed_skymodel = delayed(np.stack)(skymodel, axis=0)
        futures = self.client.compute([delayed_skymodel])
        track_progress(self.update_progress, futures)
        skymodel = self.client.gather(futures)
        self.datacube._array = skymodel * U.Jy * U.pix**-2
        return self.datacube
