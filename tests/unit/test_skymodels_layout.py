"""The one cube layout every sky model writes and the simulation reads (MADIV-28 T1.1/T1.2)."""

from __future__ import annotations

import astropy.units as U
import numpy as np
import pytest

from almasim.skymodels import (
    GaussianSkyModel,
    PointlikeSkyModel,
    add_channel_first,
    channel_first,
)
from almasim.skymodels.base import _LocalClient

from ._fakes import make_datacube

# As many pixels per side as channels: the shape alone cannot tell the axes apart.
N = 8


def _point(datacube, pos_x, pos_y, flux=1.0):
    return PointlikeSkyModel(
        datacube=datacube,
        continuum=np.full(N, flux),
        line_fluxes=np.array([]),
        pos_x=pos_x,
        pos_y=pos_y,
        pos_z=[],
        fwhm_z=[],
        n_chan=N,
    )


def _gaussian(datacube, pos_x, pos_y, continuum):
    return GaussianSkyModel(
        datacube=datacube,
        continuum=continuum,
        line_fluxes=np.array([]),
        pos_x=pos_x,
        pos_y=pos_y,
        pos_z=[],
        fwhm_z=[],
        fwhm_x=2,
        fwhm_y=2,
        angle=0,
        n_px=N,
        n_chan=N,
        client=_LocalClient(),
    )


@pytest.mark.unit
def test_add_then_read_round_trips_channel_first():
    """A (chan, y, x) cube added to an empty datacube reads back unchanged."""
    datacube = make_datacube(N, N)
    cube = np.random.default_rng(0).random((N, N, N))
    add_channel_first(datacube, cube)
    np.testing.assert_array_equal(channel_first(datacube), cube)


@pytest.mark.unit
def test_add_channel_first_adds_to_what_is_there():
    """Two additions sum: a later source never replaces an earlier one."""
    datacube = make_datacube(N, N)
    add_channel_first(datacube, np.ones((N, N, N)))
    add_channel_first(datacube, np.full((N, N, N), 2.0))
    np.testing.assert_array_equal(channel_first(datacube), np.full((N, N, N), 3.0))


@pytest.mark.unit
def test_add_channel_first_rejects_another_shape():
    """A cube that is not (chan, y, x) for this datacube is refused, not reshaped."""
    datacube = make_datacube(16, 8)
    with pytest.raises(ValueError, match="chan, y, x"):
        add_channel_first(datacube, np.zeros((16, 16, 8)))


@pytest.mark.unit
def test_point_and_gaussian_land_on_the_same_pixel():
    """Both models put a source at (pos_x, pos_y) in the same place of the read cube."""
    pos_x, pos_y = 2, 5
    point = _point(make_datacube(N, N), pos_x, pos_y).insert()
    gauss = _gaussian(make_datacube(N, N), pos_x, pos_y, np.ones(N)).insert()
    for datacube in (point, gauss):
        values = channel_first(datacube)
        assert values.shape == (N, N, N)
        assert np.unravel_index(np.argmax(values[0]), (N, N)) == (pos_y, pos_x)


@pytest.mark.unit
def test_a_second_source_keeps_the_first_ones_flux():
    """The central source's flux survives a serendipitous Gaussian inserted after it."""
    datacube = _point(make_datacube(N, N), 4, 4, flux=0.5).insert()
    datacube = _gaussian(datacube, 1, 1, np.full(N, 0.25)).insert()
    values = channel_first(datacube)
    assert values[:, 4, 4] == pytest.approx(np.full(N, 0.5), rel=1e-3)
    assert values.sum() == pytest.approx(N * (0.5 + 0.25))
    assert datacube._array.unit == U.Jy * U.pix**-2
