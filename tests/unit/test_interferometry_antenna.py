"""Unit tests for antenna functions."""

import astropy.units as U
import pytest

from almasim.services.interferometry.antenna import (
    compute_distance,
    estimate_alma_beam_size,
    get_fov_from_band,
)


def test_estimate_alma_beam_size():
    """Test ALMA beam size estimation."""
    central_freq = 100.0 * U.GHz
    max_baseline = 1.0 * U.km
    beam_size = estimate_alma_beam_size(central_freq, max_baseline, return_value=True)
    assert beam_size > 0
    assert isinstance(beam_size, float)


def test_estimate_alma_beam_size_with_units():
    """Test ALMA beam size estimation with Quantity return."""
    central_freq = 100.0 * U.GHz
    max_baseline = 1.0 * U.km
    beam_size = estimate_alma_beam_size(central_freq, max_baseline, return_value=False)
    assert beam_size.value > 0
    assert beam_size.unit == U.arcsec


def test_estimate_alma_beam_size_invalid():
    """Test that invalid inputs raise ValueError."""
    with pytest.raises(ValueError):
        estimate_alma_beam_size(-100.0 * U.GHz, 1.0 * U.km)

    with pytest.raises(ValueError):
        estimate_alma_beam_size(100.0 * U.GHz, -1.0 * U.km)


def test_get_fov_from_band():
    """Test FOV calculation for different bands."""
    for band in range(1, 11):
        fov = get_fov_from_band(band, return_value=True)
        assert fov > 0
        assert isinstance(fov, float)


def test_get_fov_from_band_with_units():
    """Test FOV calculation with Quantity return."""
    fov = get_fov_from_band(6, return_value=False)
    assert fov.value > 0
    assert fov.unit == U.arcsec


def test_compute_distance():
    """Test 3D distance calculation."""
    dist = compute_distance(0, 0, 0, 1, 1, 1)
    assert dist == pytest.approx(1.732, rel=1e-2)  # sqrt(3)

    dist = compute_distance(0, 0, 0, 3, 4, 0)
    assert dist == pytest.approx(5.0, rel=1e-2)  # 3-4-5 triangle
