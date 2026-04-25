"""Unit tests for almasim.skymodels.utils."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import astropy.units as U
import numpy as np
import pytest
from martini import DataCube

from almasim.skymodels.utils import gaussian, get_datacube_header, interpolate_array


# ===========================================================================
# interpolate_array
# ===========================================================================


@pytest.mark.unit
def test_interpolate_array_upscale():
    """interpolate_array produces output of target size when upscaling."""
    arr = np.ones((8, 8))
    result = interpolate_array(arr, 16)
    assert result.shape == (16, 16)


@pytest.mark.unit
def test_interpolate_array_downscale():
    """interpolate_array produces output of target size when downscaling."""
    arr = np.ones((32, 32))
    result = interpolate_array(arr, 16)
    assert result.shape == (16, 16)


@pytest.mark.unit
def test_interpolate_array_same_size():
    """interpolate_array with same target size is identity (shape-wise)."""
    arr = np.random.rand(16, 16)
    result = interpolate_array(arr, 16)
    assert result.shape == (16, 16)


@pytest.mark.unit
def test_interpolate_array_non_square_input():
    """interpolate_array works with non-square input."""
    arr = np.ones((8, 16))
    result = interpolate_array(arr, 16)
    assert result.shape == (16, 16)


# ===========================================================================
# gaussian
# ===========================================================================


@pytest.mark.unit
def test_gaussian_peak_at_center():
    """Gaussian peaks at the specified center."""
    x = np.arange(64)
    result = gaussian(x, amp=1.0, cen=32.0, fwhm=5.0)
    assert np.argmax(result) == 32


@pytest.mark.unit
def test_gaussian_total_flux():
    """Gaussian is normalized: np.sum(result) == amp."""
    x = np.arange(128)
    result = gaussian(x, amp=3.0, cen=64.0, fwhm=5.0)
    np.testing.assert_allclose(np.sum(result), 3.0, rtol=1e-5)


@pytest.mark.unit
def test_gaussian_zero_amplitude():
    """Gaussian with amp=0 returns all zeros."""
    x = np.arange(16)
    result = gaussian(x, amp=0.0, cen=8.0, fwhm=2.0)
    np.testing.assert_allclose(result, 0.0)


@pytest.mark.unit
def test_gaussian_width_affects_spread():
    """Wider FWHM produces a wider Gaussian profile."""
    x = np.arange(128)
    narrow = gaussian(x, amp=1.0, cen=64.0, fwhm=3.0)
    wide = gaussian(x, amp=1.0, cen=64.0, fwhm=15.0)
    # Narrow should have higher peak, wide should be flatter
    assert np.max(narrow) > np.max(wide)


@pytest.mark.unit
def test_gaussian_zero_fwhm_doesnt_crash():
    """gaussian with fwhm that makes sum zero uses amp as norm directly."""
    # Very narrow fwhm on coarse grid may yield all-zero gaussian — norm = amp
    x = np.arange(16)
    # Should not raise
    result = gaussian(x, amp=5.0, cen=8.0, fwhm=1e-6)
    assert np.all(np.isfinite(result))


# ===========================================================================
# get_datacube_header
# ===========================================================================


@pytest.mark.unit
def test_get_datacube_header_required_keys():
    """get_datacube_header returns a FITS header with mandatory keys."""
    datacube = DataCube(
        n_px_x=16,
        n_px_y=16,
        n_channels=8,
        px_size=0.1 * U.arcsec,
        channel_width=0.1 * U.GHz,
        spectral_centre=100.0 * U.GHz,
        ra=0.0 * U.deg,
        dec=0.0 * U.deg,
    )
    header = get_datacube_header(datacube, "2024-01-01")
    assert "NAXIS1" in header
    assert "NAXIS2" in header
    assert "NAXIS3" in header
    assert "BUNIT" in header
    assert "OBJECT" in header


@pytest.mark.unit
def test_get_datacube_header_dimensions():
    """Header NAXIS1/2/3 match datacube dimensions."""
    n_px, n_chan = 16, 8
    datacube = DataCube(
        n_px_x=n_px,
        n_px_y=n_px,
        n_channels=n_chan,
        px_size=0.1 * U.arcsec,
        channel_width=0.1 * U.GHz,
        spectral_centre=100.0 * U.GHz,
        ra=0.0 * U.deg,
        dec=0.0 * U.deg,
    )
    header = get_datacube_header(datacube, "2024-01-01")
    assert header["NAXIS1"] == n_px
    assert header["NAXIS2"] == n_px
    assert header["NAXIS3"] == n_chan


@pytest.mark.unit
def test_get_datacube_header_no_bmaj_bmin_before_appended():
    """Header removes any prior BMAJ/BMIN before appending new ones."""
    n_px, n_chan = 16, 8
    datacube = DataCube(
        n_px_x=n_px,
        n_px_y=n_px,
        n_channels=n_chan,
        px_size=0.1 * U.arcsec,
        channel_width=0.1 * U.GHz,
        spectral_centre=100.0 * U.GHz,
        ra=0.0 * U.deg,
        dec=0.0 * U.deg,
    )
    header = get_datacube_header(datacube, "2024-01-01")
    # Only one BMAJ and one BMIN after cleanup
    assert list(header).count("BMAJ") == 1
    assert list(header).count("BMIN") == 1


@pytest.mark.unit
def test_get_datacube_header_obs_date():
    """MJD-OBS key stores the supplied observation date."""
    datacube = DataCube(
        n_px_x=16,
        n_px_y=16,
        n_channels=8,
        px_size=0.1 * U.arcsec,
        channel_width=0.1 * U.GHz,
        spectral_centre=100.0 * U.GHz,
        ra=0.0 * U.deg,
        dec=0.0 * U.deg,
    )
    header = get_datacube_header(datacube, "2024-01-01")
    assert header["MJD-OBS"] == "2024-01-01"
