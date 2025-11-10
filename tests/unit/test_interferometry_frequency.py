"""Unit tests for frequency utilities."""
import pytest
import astropy.units as U

from almasim.services.interferometry.frequency import (
    remove_non_numeric,
    freq_supp_extractor,
)


def test_remove_non_numeric():
    """Test removing non-numeric characters."""
    assert remove_non_numeric("123.45") == "123.45"
    assert remove_non_numeric("abc123.45def") == "123.45"
    assert remove_non_numeric("12.34.56") == "12.34.56"
    assert remove_non_numeric("") == ""


def test_freq_supp_extractor():
    """Test frequency support extraction."""
    # Use a valid frequency support format: U[freq_min..freq_max,delta_freq]
    freq_sup = "U[100.0..200.0,0.1]"
    obs_freq = 150.0 * U.GHz
    band_range, central_freq, n_channels, delta_freq = freq_supp_extractor(
        freq_sup, obs_freq
    )
    assert band_range.value == pytest.approx(100.0, rel=1e-1)
    assert central_freq.value == pytest.approx(150.0, rel=1e-1)
    assert n_channels > 0
    assert delta_freq.value > 0


def test_freq_supp_extractor_multiple_windows():
    """Test frequency support extraction with multiple windows."""
    freq_sup = "U[100.0..200.0,0.1]U[200.0..300.0,0.2]"
    obs_freq = 250.0 * U.GHz
    band_range, central_freq, n_channels, delta_freq = freq_supp_extractor(
        freq_sup, obs_freq
    )
    assert central_freq.value == pytest.approx(250.0, rel=1e-1)

