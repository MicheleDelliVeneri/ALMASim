"""Frequency support and utility functions for interferometry."""
import numpy as np
import astropy.units as U


def remove_non_numeric(text: str) -> str:
    """Strip characters that are not digits or decimal points."""
    numbers = "0123456789."
    return "".join(char for char in text if char in numbers)


def freq_supp_extractor(freq_sup: str, obs_freq: U.Quantity):
    """Extract frequency support parameters from a formatted string."""
    freq_band, n_channels, freq_mins, freq_maxs, freq_ds = [], [], [], [], []
    freq_sup = freq_sup.split("U")
    for i in range(len(freq_sup)):
        if not freq_sup[i] or not freq_sup[i].strip():
            continue  # Skip empty strings from split
        sup = freq_sup[i][1:-1].split(",")
        sup = [su.split("..") for su in sup][:2]
        if not sup[0] or len(sup[0]) < 2:
            continue  # Skip invalid format
        freq_min = float(remove_non_numeric(sup[0][0]))
        freq_max = float(remove_non_numeric(sup[0][1]))
        freq_d = float(remove_non_numeric(sup[1][0]))
        freq_min = freq_min * U.GHz
        freq_max = freq_max * U.GHz
        freq_d = freq_d * U.kHz
        freq_d = freq_d.to(U.GHz)
        freq_b = freq_max - freq_min
        n_chan = int(freq_b / freq_d)
        freq_band.append(freq_b)
        n_channels.append(n_chan)
        freq_mins.append(freq_min)
        freq_maxs.append(freq_max)
        freq_ds.append(freq_d)
    freq_ranges = np.array(
        [[freq_mins[i].value, freq_maxs[i].value] for i in range(len(freq_mins))]
    )
    idx_ = np.argwhere(
        (obs_freq.value >= freq_ranges[:, 0]) & (obs_freq.value <= freq_ranges[:, 1])
    )[0][0]
    freq_range = freq_ranges[idx_]
    band_range = freq_range[1] - freq_range[0]
    n_channels_val = n_channels[idx_]
    central_freq = freq_range[0] + band_range / 2
    freq_d = freq_ds[idx_]
    return band_range * U.GHz, central_freq * U.GHz, n_channels_val, freq_d

