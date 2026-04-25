"""Unit tests for almasim.skymodels.gaussian."""

from __future__ import annotations

from unittest.mock import MagicMock

import astropy.units as U
import numpy as np
import pytest
from martini import DataCube

from almasim.skymodels.gaussian import GaussianSkyModel, gaussian2d


# ===========================================================================
# helpers
# ===========================================================================


def _make_datacube(n_px: int = 16, n_chan: int = 8) -> DataCube:
    return DataCube(
        n_px_x=n_px,
        n_px_y=n_px,
        n_channels=n_chan,
        px_size=0.1 * U.arcsec,
        channel_width=0.1 * U.GHz,
        spectral_centre=100.0 * U.GHz,
        ra=0.0 * U.deg,
        dec=0.0 * U.deg,
    )


def _mock_client(n_px: int, n_chan: int) -> MagicMock:
    client = MagicMock()
    client.compute.side_effect = lambda tasks: tasks
    client.gather.side_effect = lambda futures: [
        np.random.rand(n_chan, n_px, n_px).astype(np.float32)
    ]
    return client


def _make_model(
    tmp_path=None,
    n_px: int = 16,
    n_chan: int = 8,
    n_lines: int = 2,
) -> GaussianSkyModel:
    datacube = _make_datacube(n_px, n_chan)
    client = _mock_client(n_px, n_chan)
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


@pytest.mark.unit
def test_gaussian2d_returns_delayed():
    """gaussian2d returns a Delayed object (has .compute())."""
    n_px = 16
    x, y = np.meshgrid(np.arange(n_px), np.arange(n_px))
    result = gaussian2d(1.0, x, y, n_px, n_px // 2, n_px // 2, 4, 4, 0)
    assert hasattr(result, "compute")


@pytest.mark.unit
def test_gaussian2d_peak_at_center():
    """The 2D Gaussian peak is at the specified center pixel."""
    n_px = 32
    cx, cy = 16, 16
    x, y = np.meshgrid(np.arange(n_px), np.arange(n_px))
    result = gaussian2d(1.0, x, y, n_px, cx, cy, 6, 6, 0).compute()
    peak_idx = np.unravel_index(np.argmax(result), result.shape)
    # Peak should be at or very near (cy, cx) — within 1 pixel
    assert abs(peak_idx[0] - cy) <= 1
    assert abs(peak_idx[1] - cx) <= 1


@pytest.mark.unit
def test_gaussian2d_output_shape():
    """gaussian2d output has the same shape as the input x/y grids."""
    n_px = 16
    x, y = np.meshgrid(np.arange(n_px), np.arange(n_px))
    result = gaussian2d(1.0, x, y, n_px, n_px // 2, n_px // 2, 4, 4, 0).compute()
    assert result.shape == (n_px, n_px)


@pytest.mark.unit
def test_gaussian2d_amplitude_scales_sum():
    """Larger amplitude scales total flux proportionally."""
    n_px = 16
    x, y = np.meshgrid(np.arange(n_px), np.arange(n_px))
    r1 = gaussian2d(1.0, x, y, n_px, n_px // 2, n_px // 2, 4, 4, 0).compute()
    r2 = gaussian2d(2.0, x, y, n_px, n_px // 2, n_px // 2, 4, 4, 0).compute()
    np.testing.assert_allclose(np.sum(r2), 2 * np.sum(r1), rtol=1e-5)


@pytest.mark.unit
def test_gaussian2d_angle_rotation():
    """Non-zero rotation angle produces a different (but still valid) Gaussian."""
    n_px = 32
    x, y = np.meshgrid(np.arange(n_px), np.arange(n_px))
    r0 = gaussian2d(1.0, x, y, n_px, n_px // 2, n_px // 2, 4, 8, 0).compute()
    r45 = gaussian2d(1.0, x, y, n_px, n_px // 2, n_px // 2, 4, 8, 45).compute()
    # Both have same total flux but different distribution
    np.testing.assert_allclose(np.sum(r0), np.sum(r45), rtol=1e-4)
    assert not np.allclose(r0, r45)


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
def test_gaussian_insert_calls_dask_client():
    """insert() calls compute and gather on the Dask client."""
    model = _make_model(n_px=16, n_chan=8)
    model.insert()
    model.client.compute.assert_called_once()
    model.client.gather.assert_called_once()


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
    datacube = _make_datacube(n_px, n_chan)
    client = _mock_client(n_px, n_chan)
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
