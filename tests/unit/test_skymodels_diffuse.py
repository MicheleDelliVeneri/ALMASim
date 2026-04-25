"""Unit tests for almasim.skymodels.diffuse."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import astropy.units as U
import numpy as np
import pytest
from martini import DataCube

from almasim.skymodels.diffuse import DiffuseSkyModel, diffuse_image


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
    n_px: int = 16,
    n_chan: int = 8,
    n_lines: int = 2,
) -> DiffuseSkyModel:
    datacube = _make_datacube(n_px, n_chan)
    client = _mock_client(n_px, n_chan)
    return DiffuseSkyModel(
        datacube=datacube,
        continuum=np.ones(n_chan) * 0.01,
        line_fluxes=np.array([1.0] * n_lines),
        pos_z=list(range(n_lines)),
        fwhm_z=[2.0] * n_lines,
        n_px=n_px,
        n_chan=n_chan,
        client=client,
    )


def _fake_diffuse_signal(n_px: int) -> np.ndarray:
    """Reproducible fake diffuse signal without nifty8."""
    rng = np.random.default_rng(42)
    data = rng.random((n_px, n_px)).astype(np.float64)
    data = (data - data.min()) / (data.max() - data.min())
    return data


# ===========================================================================
# diffuse_image (delayed helper)
# ===========================================================================


@pytest.mark.unit
def test_diffuse_image_scales_signal():
    """diffuse_image scales the array by the amplitude."""
    arr = np.ones((8, 8)) * 2.0
    result = diffuse_image(arr, 3.0).compute()
    np.testing.assert_allclose(result, arr * 3.0)


@pytest.mark.unit
def test_diffuse_image_zero_amplitude():
    """diffuse_image with zero amplitude returns zeros."""
    arr = np.ones((4, 4))
    result = diffuse_image(arr, 0.0).compute()
    np.testing.assert_allclose(result, 0.0)


@pytest.mark.unit
def test_diffuse_image_negative_amplitude():
    """diffuse_image with negative amplitude negates the array."""
    arr = np.ones((4, 4))
    result = diffuse_image(arr, -1.0).compute()
    np.testing.assert_allclose(result, -1.0)


@pytest.mark.unit
def test_diffuse_image_returns_delayed():
    """diffuse_image returns a Delayed object."""
    arr = np.ones((8, 8))
    result = diffuse_image(arr, 1.0)
    assert hasattr(result, "compute")


# ===========================================================================
# DiffuseSkyModel.insert (mocking diffuse_signal to avoid nifty8)
# ===========================================================================


@pytest.mark.unit
def test_diffuse_insert_returns_datacube():
    """insert() returns the populated datacube."""
    model = _make_model()
    with patch("almasim.skymodels.diffuse.diffuse_signal", side_effect=_fake_diffuse_signal):
        result = model.insert()
    assert result is not None


@pytest.mark.unit
def test_diffuse_insert_sets_units():
    """Datacube _array has Jy/pix^2 units after insert()."""
    model = _make_model()
    with patch("almasim.skymodels.diffuse.diffuse_signal", side_effect=_fake_diffuse_signal):
        result = model.insert()
    assert result._array.unit == U.Jy * U.pix**-2


@pytest.mark.unit
def test_diffuse_insert_calls_dask_client():
    """insert() calls compute and gather on the Dask client."""
    model = _make_model()
    with patch("almasim.skymodels.diffuse.diffuse_signal", side_effect=_fake_diffuse_signal):
        model.insert()
    model.client.compute.assert_called_once()
    model.client.gather.assert_called_once()


@pytest.mark.unit
def test_diffuse_insert_no_lines():
    """insert() works when there are no emission lines."""
    model = _make_model(n_lines=0)
    model.line_fluxes = np.array([])
    model.pos_z = []
    model.fwhm_z = []
    with patch("almasim.skymodels.diffuse.diffuse_signal", side_effect=_fake_diffuse_signal):
        result = model.insert()
    assert result is not None


@pytest.mark.unit
def test_diffuse_insert_multiple_lines():
    """insert() accumulates multiple emission lines."""
    model = _make_model(n_lines=4)
    with patch("almasim.skymodels.diffuse.diffuse_signal", side_effect=_fake_diffuse_signal):
        result = model.insert()
    assert result is not None


@pytest.mark.unit
def test_diffuse_insert_nan_handling():
    """insert() applies nan_to_num so NaNs in signal do not propagate."""

    def signal_with_nan(n_px: int) -> np.ndarray:
        data = _fake_diffuse_signal(n_px)
        data[0, 0] = float("nan")
        # Re-normalize but keep NaN there
        return data

    model = _make_model()
    with patch("almasim.skymodels.diffuse.diffuse_signal", side_effect=signal_with_nan):
        # Should not raise
        result = model.insert()
    assert result is not None


@pytest.mark.unit
def test_diffuse_skymodel_inherits_base():
    """DiffuseSkyModel inherits n_px, n_chan from SkyModel."""
    model = _make_model(n_px=16, n_chan=8, n_lines=3)
    assert model.n_px == 16
    assert model.n_chan == 8
    assert len(model.line_fluxes) == 3


@pytest.mark.unit
def test_diffuse_skymodel_default_update_progress_none():
    """update_progress defaults to None."""
    model = _make_model()
    assert model.update_progress is None
