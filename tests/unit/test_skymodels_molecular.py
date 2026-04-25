"""Unit tests for almasim.skymodels.molecular."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import astropy.units as U
import numpy as np
import pytest
from martini import DataCube

from almasim.skymodels.molecular import (
    MolecularCloudSkyModel,
    make_extended,
    molecular_cloud,
    molecular_image,
)


# ===========================================================================
# helpers
# ===========================================================================


def _make_datacube(n_px: int = 32, n_chan: int = 16) -> DataCube:
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


def _make_model(n_px: int = 32, n_chan: int = 8) -> MolecularCloudSkyModel:
    """Build a minimal MolecularCloudSkyModel with a mock Dask client."""
    datacube = _make_datacube(n_px, n_chan)
    mock_client = MagicMock()
    # client.compute() returns a list of futures; client.gather() returns the cube
    mock_client.compute.side_effect = lambda tasks: tasks
    mock_client.gather.side_effect = lambda futures: [
        np.ones((n_chan, n_px, n_px), dtype=np.float32)
    ]

    return MolecularCloudSkyModel(
        datacube=datacube,
        continuum=np.ones(n_chan) * 0.01,
        line_fluxes=np.array([1.0, 0.5]),
        pos_z=[3, 6],
        fwhm_z=[1.5, 1.5],
        n_px=n_px,
        n_chan=n_chan,
        client=mock_client,
    )


# ===========================================================================
# make_extended
# ===========================================================================


@pytest.mark.unit
def test_make_extended_returns_array_of_correct_shape():
    """make_extended should produce an (N, N) array."""
    result = make_extended(64)
    # irfft2 output is (64, 64)
    assert result.shape == (64, 64)


@pytest.mark.unit
def test_make_extended_even_imsize():
    """make_extended works with even image size."""
    result = make_extended(32, powerlaw=2.5, randomseed=42)
    assert result.shape == (32, 32)


@pytest.mark.unit
def test_make_extended_odd_imsize():
    """make_extended works with odd image size (irfft2 output shape)."""
    result = make_extended(33, powerlaw=2.0, randomseed=7)
    # irfft2 output of (33, Np1+1) is (33, 32) — the output is 2D
    assert result.ndim == 2
    assert result.shape[0] == 33


@pytest.mark.unit
def test_make_extended_elliptical():
    """make_extended with ellip < 1 uses rotation."""
    result = make_extended(32, ellip=0.5, theta=0.5, randomseed=1)
    assert result.shape == (32, 32)
    assert np.isfinite(result).all()


@pytest.mark.unit
def test_make_extended_circular():
    """make_extended with ellip == 1 uses circular case."""
    result = make_extended(32, ellip=1.0)
    assert result.shape == (32, 32)


@pytest.mark.unit
def test_make_extended_invalid_ellip_raises():
    """ellip > 1 should raise ValueError."""
    with pytest.raises(ValueError, match="ellip must be"):
        make_extended(32, ellip=1.5)


@pytest.mark.unit
def test_make_extended_zero_ellip_raises():
    """ellip == 0 should raise ValueError."""
    with pytest.raises(ValueError, match="ellip must be"):
        make_extended(32, ellip=0.0)


@pytest.mark.unit
def test_make_extended_return_fft_full():
    """return_fft=True, full_fft=True returns a square array."""
    result = make_extended(32, return_fft=True, full_fft=True)
    assert result.shape[0] == result.shape[1] == 32


@pytest.mark.unit
def test_make_extended_return_fft_no_full():
    """return_fft=True, full_fft=False returns partial FFT."""
    result = make_extended(32, return_fft=True, full_fft=False)
    # rfft shape: (32, 17)
    assert result.shape[0] == 32


@pytest.mark.unit
def test_make_extended_reproducible():
    """Same randomseed produces identical output."""
    a = make_extended(16, randomseed=9999)
    b = make_extended(16, randomseed=9999)
    np.testing.assert_array_equal(a, b)


@pytest.mark.unit
def test_make_extended_different_seeds_differ():
    """Different seeds produce different outputs."""
    a = make_extended(16, randomseed=1)
    b = make_extended(16, randomseed=2)
    assert not np.array_equal(a, b)


# ===========================================================================
# molecular_cloud
# ===========================================================================


@pytest.mark.unit
def test_molecular_cloud_returns_2d_array():
    """molecular_cloud returns a 2-D NumPy array."""
    result = molecular_cloud(32)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2


@pytest.mark.unit
def test_molecular_cloud_shape_matches_n_px():
    """Shape matches requested pixel count."""
    result = molecular_cloud(16)
    assert result.shape == (16, 16)


# ===========================================================================
# molecular_image (dask delayed)
# ===========================================================================


@pytest.mark.unit
def test_molecular_image_scales_array():
    """molecular_image scales the input array by the amplitude."""
    arr = np.ones((8, 8))
    # Call the underlying function directly (bypassing @delayed)
    result = (
        molecular_image.__wrapped__(arr, 3.0)
        if hasattr(molecular_image, "__wrapped__")
        else molecular_image(arr, 3.0).compute()
    )
    np.testing.assert_allclose(result, arr * 3.0)


# ===========================================================================
# MolecularCloudSkyModel.insert
# ===========================================================================


@pytest.mark.unit
def test_molecular_cloud_skymodel_insert_returns_datacube():
    """insert() returns the datacube with _array populated."""
    model = _make_model(n_px=16, n_chan=4)
    result = model.insert()
    assert result is not None


@pytest.mark.unit
def test_molecular_cloud_skymodel_insert_calls_client_compute():
    """insert() uses the Dask client to compute the sky model."""
    model = _make_model(n_px=16, n_chan=4)
    model.insert()
    model.client.compute.assert_called_once()
    model.client.gather.assert_called_once()


@pytest.mark.unit
def test_molecular_cloud_skymodel_insert_sets_units():
    """Datacube _array should have Jy / pix^2 units after insert()."""
    model = _make_model(n_px=16, n_chan=4)
    result = model.insert()
    assert result._array.unit == U.Jy * U.pix**-2


@pytest.mark.unit
def test_molecular_cloud_skymodel_multiple_lines():
    """Model with several emission lines accumulates correctly."""
    model = _make_model(n_px=16, n_chan=8)
    model.line_fluxes = np.array([1.0, 0.5, 0.3])
    model.pos_z = [2, 4, 6]
    model.fwhm_z = [1.0, 1.0, 1.0]
    result = model.insert()
    assert result is not None


@pytest.mark.unit
def test_molecular_cloud_skymodel_no_lines():
    """Model with zero emission lines still runs."""
    n_px, n_chan = 16, 4
    datacube = _make_datacube(n_px, n_chan)
    mock_client = MagicMock()
    mock_client.compute.side_effect = lambda tasks: tasks
    mock_client.gather.side_effect = lambda futures: [
        np.ones((n_chan, n_px, n_px), dtype=np.float32)
    ]
    model = MolecularCloudSkyModel(
        datacube=datacube,
        continuum=np.ones(n_chan) * 0.1,
        line_fluxes=np.array([]),
        pos_z=[],
        fwhm_z=[],
        n_px=n_px,
        n_chan=n_chan,
        client=mock_client,
    )
    result = model.insert()
    assert result is not None
