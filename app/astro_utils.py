import h5py
import os
import numpy as np
import pandas as pd
from random import choices



def compute_redshift(rest_frequency, observed_frequency):
    """Compute the redshift of a source."""
    if rest_frequency <= 0 or observed_frequency <= 0:
        raise ValueError("Rest and observed frequencies must be positive values.")
    if rest_frequency < observed_frequency:
        raise ValueError("Observed frequency must be lower than the rest frequency.")

    return (rest_frequency.value - observed_frequency.value) / observed_frequency.value


def redshift_to_snapshot(redshift):
    """Convert redshift to a TNG snapshot ID."""
    snap_db = {
        0: 20.05,
        1: 14.99,
        2: 11.98,
        3: 10.98,
        4: 10.00,
        5: 9.390,
        6: 9.000,
        7: 8.450,
        8: 8.010,
        9: 7.600,
        10: 7.24,
        11: 7.01,
        12: 6.49,
        13: 6.01,
        14: 5.85,
        15: 5.53,
        16: 5.23,
        17: 5.00,
        18: 4.66,
        19: 4.43,
        20: 4.18,
        21: 4.01,
        22: 3.71,
        23: 3.49,
        24: 3.28,
        25: 3.01,
        26: 2.90,
        27: 2.73,
        28: 2.58,
        29: 2.44,
        30: 2.32,
        31: 2.21,
        32: 2.10,
        33: 2.00,
        34: 1.90,
        35: 1.82,
        36: 1.74,
        37: 1.67,
        38: 1.60,
        39: 1.53,
        40: 1.50,
        41: 1.41,
        42: 1.36,
        43: 1.30,
        44: 1.25,
        45: 1.21,
        46: 1.15,
        47: 1.11,
        48: 1.07,
        49: 1.04,
        50: 1.00,
        51: 0.95,
        52: 0.92,
        53: 0.89,
        54: 0.85,
        55: 0.82,
        56: 0.79,
        57: 0.76,
        58: 0.73,
        59: 0.70,
        60: 0.68,
        61: 0.64,
        62: 0.62,
        63: 0.60,
        64: 0.58,
        65: 0.55,
        66: 0.52,
        67: 0.50,
        68: 0.48,
        69: 0.46,
        70: 0.44,
        71: 0.42,
        72: 0.40,
        73: 0.38,
        74: 0.36,
        75: 0.35,
        76: 0.33,
        77: 0.31,
        78: 0.30,
        79: 0.27,
        80: 0.26,
        81: 0.24,
        82: 0.23,
        83: 0.21,
        84: 0.20,
        85: 0.18,
        86: 0.17,
        87: 0.15,
        88: 0.14,
        89: 0.13,
        90: 0.11,
        91: 0.10,
        92: 0.08,
        93: 0.07,
        94: 0.06,
        95: 0.05,
        96: 0.03,
        97: 0.02,
        98: 0.01,
        99: 0,
    }
    snaps, redshifts = list(snap_db.keys())[::-1], list(snap_db.values())[::-1]
    for i in range(len(redshifts) - 1):
        if redshifts[i] <= redshift < redshifts[i + 1]:
            return snaps[i]


def get_data_from_hdf(file, snapshot):
    """Retrieve data from an HDF5 file."""
    data, column_names = [], []
    with h5py.File(file, "r") as r:
        for key in r.keys():
            if key == f"Snapshot_{snapshot}":
                group = r[key]
                for key2 in group.keys():
                    column_names.append(key2)
                    data.append(group[key2])
    return pd.DataFrame(np.array(data).T, columns=column_names)


def get_subhaloids_from_db(n, main_path, snapshot):
    """Retrieve Subhalo IDs based on morphology."""
    file = os.path.join(main_path, "metadata", "morphologies_deeplearn.hdf5")
    db = get_data_from_hdf(file, snapshot)
    catalogue = db[["SubhaloID", "P_Late", "P_S0", "P_Sab"]].sort_values(by=["P_Late"], ascending=False)

    ellipticals = catalogue[(catalogue["P_Late"] > 0.6) & (catalogue["P_S0"] < 0.5) & (catalogue["P_Sab"] < 0.5)]
    lenticulars = catalogue[(catalogue["P_S0"] > 0.6) & (catalogue["P_Late"] < 0.5) & (catalogue["P_Sab"] < 0.5)]
    spirals = catalogue[(catalogue["P_Sab"] > 0.6) & (catalogue["P_Late"] < 0.5) & (catalogue["P_S0"] < 0.5)]

    ids = np.concatenate([
        choices(ellipticals["SubhaloID"].values, k=n // 3),
        choices(spirals["SubhaloID"].values, k=n // 3),
        choices(lenticulars["SubhaloID"].values, k=n - 2 * (n // 3))
    ]).astype(int)
    return ids[0] if len(ids) == 1 else ids


def read_line_emission_csv(path, sep=";"):
    """Read line emission frequencies from a CSV file."""
    return pd.read_csv(path, sep=sep)


def get_line_info(main_path, idxs=None):
    """Retrieve line information from a CSV file."""
    db_line = read_line_emission_csv(os.path.join(main_path, "brightnes", "calibrated_lines.csv"), sep=",")
    db_line = db_line.sort_values(by="Line")
    rest_frequencies, line_names = db_line["freq(GHz)"].values, db_line["Line"].values
    return (rest_frequencies[idxs], line_names[idxs]) if idxs else (rest_frequencies, line_names)


def compute_rest_frequency_from_redshift(master_path, source_freq, redshift):
    """Compute the rest frequency given a redshift."""
    db_line = read_line_emission_csv(os.path.join(master_path, "brightnes", "calibrated_lines.csv"), sep=",")
    db_line["freq(GHz)"] = db_line["freq(GHz)"].astype(float)
    source_freqs = db_line["freq(GHz)"].values / (1 + redshift)
    closest_freq = min(source_freqs, key=lambda x: abs(x - source_freq))
    line_name = db_line["Line"].values[np.where(source_freqs == closest_freq)][0]
    return db_line[db_line["Line"] == line_name]["freq(GHz)"].values[0]


def write_sim_parameters(path, **kwargs):
    """Write simulation parameters to a file."""
    with open(path, "w") as f:
        f.write("Simulation Parameters:\n")
        for key, value in kwargs.items():
            f.write(f"{key.replace('_', ' ').capitalize()}: {value}\n")
