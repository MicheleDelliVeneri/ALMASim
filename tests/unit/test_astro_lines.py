"""Unit tests for line emission functions."""

import pytest

from almasim.services.astro.lines import (
    read_line_emission_csv,
    get_line_info,
    compute_rest_frequency_from_redshift,
)


def test_read_line_emission_csv(main_dir):
    """Test reading line emission CSV file."""
    csv_path = main_dir / "brightnes" / "calibrated_lines.csv"
    if not csv_path.exists():
        pytest.skip(f"Line emission CSV not found at {csv_path}")

    db = read_line_emission_csv(csv_path, sep=",")
    assert "Line" in db.columns
    assert "freq(GHz)" in db.columns
    assert len(db) > 0


def test_get_line_info(main_dir):
    """Test getting line information."""
    rest_freq, line_names = get_line_info(main_dir)
    assert len(rest_freq) > 0
    assert len(line_names) > 0
    assert len(rest_freq) == len(line_names)


def test_get_line_info_with_indices(main_dir):
    """Test getting line information with specific indices."""
    rest_freq, line_names = get_line_info(main_dir, idxs=[0, 1, 2])
    assert len(rest_freq) == 3
    assert len(line_names) == 3


def test_compute_rest_frequency_from_redshift(main_dir):
    """Test computing rest frequency from redshift."""
    source_freq = 100.0  # GHz
    redshift = 0.1
    rest_freq = compute_rest_frequency_from_redshift(main_dir, source_freq, redshift)
    assert rest_freq > 0
    # Rest frequency should be higher than observed
    assert rest_freq > source_freq
