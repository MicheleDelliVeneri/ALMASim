"""Unit tests for interferometry utilities."""

import numpy as np
import pytest

from almasim.services.interferometry.utils import (
    closest_power_of_2,
    get_channel_wavelength,
    sampling_to_uv_mask,
)


def test_get_channel_wavelength():
    """Test getting channel wavelength."""
    # Wavelengths are in mm, function returns in meters (converts by * 1e-3)
    obs_wavelengths = np.array([[1000.0, 1100.0], [1100.0, 1200.0], [1200.0, 1300.0]])  # in mm
    wavelength = get_channel_wavelength(obs_wavelengths, 0)
    assert len(wavelength) == 3
    assert wavelength[0] == pytest.approx(1.0, rel=1e-3)  # 1000mm = 1.0m
    assert wavelength[1] == pytest.approx(1.1, rel=1e-3)  # 1100mm = 1.1m
    assert wavelength[2] == pytest.approx(1.05, rel=1e-3)  # Average: 1.05m


def test_closest_power_of_2():
    """Test finding closest power of 2."""
    assert closest_power_of_2(100) == 128
    assert closest_power_of_2(64) == 64
    assert closest_power_of_2(65) == 64  # Closer to 64 than 128
    assert closest_power_of_2(96) == 128  # Closer to 128 than 64
    assert closest_power_of_2(1) == 1
    assert closest_power_of_2(2) == 2


def test_sampling_to_uv_mask():
    """Test converting sampling weights into a binary UV mask."""
    sampling = np.array(
        [
            [[0.0, 1.5], [2.0, 0.0]],
            [[0.1, 0.0], [0.0, 3.2]],
        ],
        dtype=np.float32,
    )
    uv_mask = sampling_to_uv_mask(sampling)

    assert uv_mask.dtype == np.uint8
    assert np.array_equal(
        uv_mask,
        np.array(
            [
                [[0, 1], [1, 0]],
                [[1, 0], [0, 1]],
            ],
            dtype=np.uint8,
        ),
    )
