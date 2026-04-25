"""Extended unit tests for interferometry frequency functions."""

import astropy.units as U
import pytest

from almasim.services.interferometry.frequency import (
    freq_supp_extractor,
    remove_non_numeric,
)


@pytest.mark.unit
def test_remove_non_numeric_edge_cases():
    """Test remove_non_numeric with edge cases."""
    assert remove_non_numeric("") == ""
    assert remove_non_numeric("abc") == ""
    assert remove_non_numeric("12.34.56") == "12.34.56"  # Multiple dots allowed
    assert remove_non_numeric("  123  ") == "123"
    assert remove_non_numeric("0.001") == "0.001"


@pytest.mark.unit
def test_freq_supp_extractor_standard_format():
    """Test frequency support extraction with standard ALMA format."""
    freq_sup = "[250.0..252.0GHz,31250.00kHz,2mJy/beam@10km/s,78.5uJy/beam@native, XX YY]"
    obs_freq = 251.0 * U.GHz
    band_range, central_freq, n_channels, freq_d = freq_supp_extractor(freq_sup, obs_freq)

    assert band_range.value == pytest.approx(2.0, rel=1e-3)
    assert central_freq.value == pytest.approx(251.0, rel=1e-3)
    assert n_channels > 0
    assert freq_d.value > 0
    # 31250 kHz = 0.03125 GHz, so channels should be ~64
    assert n_channels == pytest.approx(64, abs=1)


@pytest.mark.unit
def test_freq_supp_extractor_multiple_ranges():
    """Test frequency support extraction with multiple ranges."""
    freq_sup = "[250.0..252.0GHz,31250.00kHz] U [350.0..352.0GHz,31250.00kHz]"

    # Test first range
    obs_freq = 251.0 * U.GHz
    band_range, central_freq, n_channels, freq_d = freq_supp_extractor(freq_sup, obs_freq)
    assert central_freq.value == pytest.approx(251.0, rel=1e-3)

    # Test second range
    obs_freq = 351.0 * U.GHz
    band_range, central_freq, n_channels, freq_d = freq_supp_extractor(freq_sup, obs_freq)
    assert central_freq.value == pytest.approx(351.0, rel=1e-3)


@pytest.mark.unit
def test_freq_supp_extractor_validation():
    """Test frequency support extraction input validation."""
    # Empty string
    with pytest.raises(ValueError, match="cannot be empty"):
        freq_supp_extractor("", 250.0 * U.GHz)

    # Whitespace only
    with pytest.raises(ValueError, match="cannot be empty"):
        freq_supp_extractor("   ", 250.0 * U.GHz)

    # No valid ranges
    with pytest.raises(ValueError, match="No valid frequency ranges"):
        freq_supp_extractor("invalid", 250.0 * U.GHz)

    # Frequency doesn't match any range
    freq_sup = "[250.0..252.0GHz,31250.00kHz]"
    with pytest.raises(ValueError, match="does not match"):
        freq_supp_extractor(freq_sup, 500.0 * U.GHz)


@pytest.mark.unit
def test_freq_supp_extractor_edge_frequencies():
    """Test frequency support extraction at range boundaries."""
    freq_sup = "[250.0..252.0GHz,31250.00kHz]"

    # Frequency at lower bound
    obs_freq = 250.0 * U.GHz
    band_range, central_freq, n_channels, freq_d = freq_supp_extractor(freq_sup, obs_freq)
    assert central_freq.value == pytest.approx(251.0, rel=1e-3)

    # Frequency at upper bound
    obs_freq = 252.0 * U.GHz
    band_range, central_freq, n_channels, freq_d = freq_supp_extractor(freq_sup, obs_freq)
    assert central_freq.value == pytest.approx(251.0, rel=1e-3)


@pytest.mark.unit
def test_freq_supp_extractor_channel_calculation():
    """Test that channel count is correctly calculated."""
    freq_sup = "[250.0..252.0GHz,31250.00kHz]"
    obs_freq = 251.0 * U.GHz
    band_range, central_freq, n_channels, freq_d = freq_supp_extractor(freq_sup, obs_freq)

    # Bandwidth = 2.0 GHz, delta = 0.03125 GHz, so n_channels should be 64
    expected_channels = int(2.0 / 0.03125)
    assert n_channels == expected_channels
    assert freq_d.value == pytest.approx(0.03125, rel=1e-3)


@pytest.mark.unit
def test_freq_supp_extractor_invalid_delta():
    """Test handling of invalid frequency deltas."""
    # This should skip invalid entries but not crash
    freq_sup = "[250.0..252.0GHz,0kHz] U [350.0..352.0GHz,31250.00kHz]"
    obs_freq = 351.0 * U.GHz
    band_range, central_freq, n_channels, freq_d = freq_supp_extractor(freq_sup, obs_freq)
    assert central_freq.value == pytest.approx(351.0, rel=1e-3)
