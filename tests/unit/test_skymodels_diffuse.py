"""Unit tests for almasim.skymodels.diffuse."""

from __future__ import annotations

from unittest.mock import patch

import astropy.units as U
import numpy as np
import pytest

from almasim.skymodels.diffuse import DiffuseSkyModel

from ._fakes import make_datacube, mock_dask_client

# ===========================================================================
# helpers
# ===========================================================================


def _make_model(
    n_px: int = 16,
    n_chan: int = 8,
    n_lines: int = 2,
) -> DiffuseSkyModel:
    datacube = make_datacube(n_px, n_chan)
    client = mock_dask_client(n_px, n_chan)
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
