"""Unit tests for almasim.skymodels.gaussian."""

from __future__ import annotations

import astropy.units as U
import numpy as np
import pytest

from almasim.skymodels.gaussian import GaussianSkyModel, gaussian2d

from ._fakes import make_datacube, mock_dask_client

# ===========================================================================
# helpers
# ===========================================================================


def _make_model(
    tmp_path=None,
    n_px: int = 16,
    n_chan: int = 8,
    n_lines: int = 2,
) -> GaussianSkyModel:
    datacube = make_datacube(n_px, n_chan)
    client = mock_dask_client(n_px, n_chan)
    return GaussianSkyModel(
        datacube=datacube,
        continuum=np.ones(n_chan) * 0.01,
        line_fluxes=np.array([1.0] * n_lines),
        pos_x=n_px // 2,
        pos_y=n_px // 2,
        pos_z=list(range(n_lines)),
        fwhm_x=4,
        fwhm_y=4,
        fwhm_z=[2.0] * n_lines,
        angle=0,
        n_px=n_px,
        n_chan=n_chan,
        client=client,
    )


# ===========================================================================
# gaussian2d (delayed helper)
# ===========================================================================


# ===========================================================================
# GaussianSkyModel initialization
# ===========================================================================


@pytest.mark.unit
def test_gaussian_skymodel_stores_pos():
    """GaussianSkyModel stores pos_x, pos_y."""
    model = _make_model(n_px=16)
    assert model.pos_x == 8
    assert model.pos_y == 8


@pytest.mark.unit
def test_gaussian_skymodel_stores_fwhm():
    """GaussianSkyModel stores fwhm_x, fwhm_y, angle."""
    model = _make_model(n_px=16)
    assert model.fwhm_x == 4
    assert model.fwhm_y == 4
    assert model.angle == 0


@pytest.mark.unit
def test_gaussian_skymodel_inherits_base():
    """GaussianSkyModel inherits n_px, n_chan from SkyModel."""
    model = _make_model(n_px=16, n_chan=8, n_lines=3)
    assert model.n_px == 16
    assert model.n_chan == 8
    assert len(model.line_fluxes) == 3


@pytest.mark.unit
def test_gaussian_skymodel_default_update_progress_none():
    """update_progress defaults to None."""
    model = _make_model()
    assert model.update_progress is None


# ===========================================================================
# GaussianSkyModel.insert
# ===========================================================================


@pytest.mark.unit
def test_gaussian_insert_returns_datacube():
    """insert() returns the populated datacube."""
    model = _make_model(n_px=16, n_chan=8)
    result = model.insert()
    assert result is not None


@pytest.mark.unit
def test_gaussian_insert_sets_units():
    """Datacube _array has Jy/pix^2 units after insert()."""
    model = _make_model(n_px=16, n_chan=8)
    result = model.insert()
    assert result._array.unit == U.Jy * U.pix**-2


@pytest.mark.unit
def test_gaussian_insert_no_lines():
    """insert() works when there are no emission lines."""
    model = _make_model(n_lines=0)
    model.line_fluxes = np.array([])
    model.pos_z = []
    model.fwhm_z = []
    result = model.insert()
    assert result is not None


@pytest.mark.unit
def test_gaussian_insert_multiple_lines():
    """insert() accumulates multiple emission lines."""
    model = _make_model(n_lines=5)
    result = model.insert()
    assert result is not None


@pytest.mark.unit
def test_gaussian_insert_angled():
    """insert() works with non-zero rotation angle."""
    n_px, n_chan = 16, 8
    datacube = make_datacube(n_px, n_chan)
    client = mock_dask_client(n_px, n_chan)
    model = GaussianSkyModel(
        datacube=datacube,
        continuum=np.ones(n_chan) * 0.01,
        line_fluxes=np.array([1.0]),
        pos_x=n_px // 2,
        pos_y=n_px // 2,
        pos_z=[n_chan // 2],
        fwhm_x=3,
        fwhm_y=6,
        fwhm_z=[2.0],
        angle=30,
        n_px=n_px,
        n_chan=n_chan,
        client=client,
    )
    result = model.insert()
    assert result is not None


@pytest.mark.unit
def test_gaussian2d_peaks_at_its_centre_and_rotates():
    shape = gaussian2d(32, 20, 12, 6, 6, 0)
    assert shape.shape == (32, 32)
    assert np.unravel_index(np.argmax(shape), shape.shape) == (12, 20)
    assert not np.allclose(gaussian2d(32, 16, 16, 4, 8, 0), gaussian2d(32, 16, 16, 4, 8, 45))


@pytest.mark.unit
def test_gaussian_cube_equals_the_per_channel_computation_exactly():
    """Each channel is (flux / sum(shape)) * shape, bit for bit as the old dask loop."""
    from almasim.skymodels import channel_first

    model = _make_model(n_px=16, n_chan=8)
    cube = channel_first(model.insert())
    shape = gaussian2d(16, model.pos_x, model.pos_y, model.fwhm_x, model.fwhm_y, model.angle)
    for z, flux in enumerate(model.spectrum()):
        np.testing.assert_array_equal(cube[z], (flux / np.sum(shape)) * shape)
