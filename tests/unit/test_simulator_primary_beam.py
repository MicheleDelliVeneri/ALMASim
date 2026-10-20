"""The simulator's primary beam is the instrument's Airy model (MADIV-28 T4.5).

The beam used to be a Gaussian evaluated on a distance grid scaled by the
pixel size a second time, so its width followed the model's pixel size: a
25" field at 256 pixels kept 88 % of the flux at the image edge.
"""

from __future__ import annotations

import numpy as np
import pytest

from almasim.services.instrument.constants import ARCSEC_PER_RAD, SPEED_OF_LIGHT_M_S
from almasim.services.instrument.primary_beam import primary_beam_fwhm_rad, primary_beam_response
from almasim.services.interferometry.imaging import set_primary_beam

pytestmark = pytest.mark.unit


def _beam(fov_arcsec: float, nphf: int, frequency_hz: float = 230e9, dish: float = 12.0):
    """The beam as Interferometer builds it (core.py's grid, set_primary_beam's response)."""
    imsize = 3 * fov_arcsec
    npix = 2 * nphf
    xx = np.linspace(-imsize / 2, imsize / 2, npix)
    yy = np.ones(npix)
    distmat = -np.outer(xx**2.0, yy) - np.outer(yy, xx**2.0)
    wavelength_km = SPEED_OF_LIGHT_M_S / frequency_hz / 1e3
    modelim = [np.zeros((npix, npix))]
    _, beam = set_primary_beam(
        None, distmat, [0, 0, wavelength_km], [dish, 0], modelim, np.ones((npix, npix))
    )
    return xx, beam[0][npix // 2]


@pytest.mark.parametrize("fov, nphf", [(25.0, 64), (25.0, 256), (60.0, 64)])
def test_the_half_power_width_does_not_depend_on_the_pixel_size(fov, nphf):
    xx, profile = _beam(fov, nphf)
    measured = np.ptp(xx[profile >= 0.5])
    expected = primary_beam_fwhm_rad(12.0, 230e9) * ARCSEC_PER_RAD
    assert measured == pytest.approx(expected, abs=2 * np.diff(xx)[0])


def test_the_beam_is_the_airy_response():
    xx, profile = _beam(25.0, 128)
    # The profile is row npix // 2, half a pixel from the grid's centre.
    radius = np.hypot(xx, xx[len(xx) // 2])
    expected = primary_beam_response(radius / ARCSEC_PER_RAD, 230e9, 12.0)
    np.testing.assert_allclose(profile, expected, rtol=1e-12, atol=1e-15)
    assert profile[0] < 0.05  # near the first null at the edge of a 75" image
