"""Unit tests for redshift calculations."""

import astropy.units as U
import pytest

from almasim.services.astro.redshift import compute_redshift


def test_compute_redshift_valid():
    """Test redshift calculation with valid inputs."""
    rest_freq = 100.0 * U.GHz
    observed_freq = 90.0 * U.GHz
    redshift = compute_redshift(rest_freq, observed_freq)
    assert redshift == pytest.approx(0.1111, rel=1e-3)


def test_compute_redshift_zero_redshift():
    """Test redshift calculation when rest and observed frequencies are equal."""
    freq = 100.0 * U.GHz
    redshift = compute_redshift(freq, freq)
    assert redshift == pytest.approx(0.0)


def test_compute_redshift_negative_raises():
    """Test that negative frequencies raise ValueError."""
    with pytest.raises(ValueError, match="positive values"):
        compute_redshift(-100.0 * U.GHz, 90.0 * U.GHz)

    with pytest.raises(ValueError, match="positive values"):
        compute_redshift(100.0 * U.GHz, -90.0 * U.GHz)


def test_compute_redshift_observed_greater_than_rest_raises():
    """Test that observed frequency greater than rest frequency raises ValueError."""
    with pytest.raises(ValueError, match="lower than the rest frequency"):
        compute_redshift(90.0 * U.GHz, 100.0 * U.GHz)
