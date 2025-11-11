"""Interferometry utility functions."""
import math
import numpy as np


def get_channel_wavelength(obs_wavelengths: np.ndarray, channel: int) -> list[float]:
    """Get wavelength range for a specific channel."""
    wavelength = list(obs_wavelengths[channel] * 1e-3)
    wavelength.append((wavelength[0] + wavelength[1]) / 2.0)
    return wavelength


def closest_power_of_2(x):
    """Find the closest power of 2 to x."""
    x = int(x)
    if x <= 0:
        return 1
    if x == 1:
        return 1
    bin_str = bin(x)
    # Check if there's more than one '1' bit (not a power of 2)
    if bin_str.count('1') > 1:
        op = math.floor if bin_str[3] != "1" else math.ceil
    else:
        return x  # Already a power of 2
    return 2 ** op(math.log(x, 2))

