import h5py
from random import choices
import os
import numpy as np
import pandas as pd


# -------------------- TNG Auxiliary Functions ----------------------------- #


def compute_redshift(rest_frequency, observed_frequency):
    """
    Computes the redshift of a source given the rest frequency and the observed frequency.

    Args:
        rest_frequency (astropy Unit): Rest frequency of the source in GHz.
        observed_frequency (astropy Unit): Observed frequency of the source in GHz.

    Returns:
        float: Redshift of the source.

    Raises:
        ValueError: If either input argument is non-positive.
    """
    # Input validation
    if rest_frequency <= 0 or observed_frequency <= 0:
        raise ValueError("Rest and observed frequencies must be positive values.")
    if rest_frequency < observed_frequency:
        raise ValueError("Observed frequency must be lower than the rest frequency.")

    # Compute redshift
    redshift = (rest_frequency.value - observed_frequency.value) / rest_frequency.value
    return redshift


def redshift_to_snapshot(redshift):
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
        if redshift >= redshifts[i] and redshift < redshifts[i + 1]:
            return snaps[i]


def get_data_from_hdf(file, snapshot):
    data = list()
    column_names = list()
    r = h5py.File(file, "r")
    for key in r.keys():
        if key == f"Snapshot_{snapshot}":
            group = r[key]
            for key2 in group.keys():
                column_names.append(key2)
                data.append(group[key2])
    values = np.array(data)
    r.close()
    db = pd.DataFrame(values.T, columns=column_names)
    return db


def get_subhaloids_from_db(n, main_path, snapshot):
    pd.options.mode.chained_assignment = None
    file = os.path.join(main_path, "metadata", "morphologies_deeplearn.hdf5")
    db = get_data_from_hdf(file, snapshot)
    catalogue = db[["SubhaloID", "P_Late", "P_S0", "P_Sab"]]
    catalogue = catalogue.sort_values(by=["P_Late"], ascending=False)
    ellipticals = catalogue[
        (catalogue["P_Late"] > 0.6)
        & (catalogue["P_S0"] < 0.5)
        & (catalogue["P_Sab"] < 0.5)
    ]
    lenticulars = catalogue[
        (catalogue["P_S0"] > 0.6)
        & (catalogue["P_Late"] < 0.5)
        & (catalogue["P_Sab"] < 0.5)
    ]
    spirals = catalogue[
        (catalogue["P_Sab"] > 0.6)
        & (catalogue["P_Late"] < 0.5)
        & (catalogue["P_S0"] < 0.5)
    ]

    ellipticals["sum"] = ellipticals["P_S0"] + ellipticals["P_Sab"]
    lenticulars["sum"] = lenticulars["P_Late"] + lenticulars["P_Sab"]
    spirals["sum"] = spirals["P_Late"] + spirals["P_S0"]

    ellipticals = ellipticals.sort_values(by=["sum"], ascending=True)
    lenticulars = lenticulars.sort_values(by=["sum"], ascending=True)
    spirals = spirals.sort_values(by=["sum"], ascending=True)
    ellipticals_ids = ellipticals["SubhaloID"].values
    lenticulars_ids = lenticulars["SubhaloID"].values
    spirals_ids = spirals["SubhaloID"].values
    sample_n = n // 3

    n_0 = choices(ellipticals_ids, k=sample_n)
    n_1 = choices(spirals_ids, k=sample_n)
    n_2 = choices(lenticulars_ids, k=n - 2 * sample_n)
    ids = np.concatenate((n_0, n_1, n_2)).astype(int)
    if len(ids) == 1:
        return ids[0]
    return ids


# ---------------- Luminosity Functions ---------------------------------- #


def read_line_emission_csv(path_line_emission_csv, sep=";"):
    """
    Read the csv file in which are stored the line emission's rest frequency.

    Parameter:
    path_line_emission_csv (str): Path to file.csv within there are the line
                                  emission's rest frequency.

    Return:
    pd.DataFrame : Dataframe with line names and rest frequencies.
    """
    db_line = pd.read_csv(path_line_emission_csv, sep=sep)
    return db_line


def get_line_info(main_path, idxs=None):
    path_line_emission_csv = os.path.join(
        main_path, "brightnes", "calibrated_lines.csv"
    )
    db_line = read_line_emission_csv(path_line_emission_csv, sep=",").sort_values(
        by="Line"
    )
    rest_frequencies = db_line["freq(GHz)"].values
    line_names = db_line["Line"].values
    if idxs is not None:
        return rest_frequencies[idxs], line_names[idxs]
    else:
        return rest_frequencies, line_names


def compute_rest_frequency_from_redshift(master_path, source_freq, redshift):
    db_line = read_line_emission_csv(
        os.path.join(master_path, "brightnes", "calibrated_lines.csv"), sep=","
    )
    db_line["freq(GHz)"] = db_line["freq(GHz)"].astype(float)
    source_freqs = db_line["freq(GHz)"].values / (1 + redshift)
    freq_names = db_line["Line"].values
    closest_freq = min(source_freqs, key=lambda x: abs(x - source_freq))
    line_name = freq_names[np.where(source_freqs == closest_freq)][0]
    rest_frequency = db_line[db_line["Line"] == line_name]["freq(GHz)"].values[0]
    return rest_frequency


def write_sim_parameters(
    path,
    ra,
    dec,
    ang_res,
    vel_res,
    int_time,
    band,
    band_range,
    central_freq,
    redshift,
    line_fluxes,
    line_names,
    line_frequencies,
    continum,
    fov,
    beam_size,
    cell_size,
    n_pix,
    n_channels,
    snapshot,
    subhalo,
    lum_infrared,
    fwhm_z,
    source_type,
    fwhm_x=None,
    fwhm_y=None,
    angle=None,
):
    with open(path, "w") as f:
        f.write("Simulation Parameters:\n")
        f.write("RA: {}\n".format(ra))
        f.write("DEC: {}\n".format(dec))
        f.write("Band: {}\n".format(band))
        f.write("Bandwidth {}\n".format(band_range))
        f.write("Band Central Frequency: {}\n".format(central_freq))
        f.write("Pixel size: {}\n".format(cell_size))
        f.write("Beam Size: {}\n".format(beam_size))
        f.write("Fov: {}\n".format(fov))
        f.write("Angular Resolution: {}\n".format(ang_res))
        f.write("Velocity Resolution: {}\n".format(vel_res))
        f.write("Redshift: {}\n".format(redshift))
        f.write("Integration Time: {}\n".format(int_time))
        f.write("Cube Size: {} x {} x {} pixels\n".format(n_pix, n_pix, n_channels))
        f.write("Mean Continum Flux: {}\n".format(np.mean(continum)))
        f.write("Infrared Luminosity: {}\n".format(lum_infrared))
        if source_type == "gaussian":
            f.write("FWHM_x (pixels): {}\n".format(fwhm_x))
            f.write("FWHM_y (pixels): {}\n".format(fwhm_y))
        if (source_type == "gaussian") or (source_type == "extended"):
            f.write("Projection Angle: {}\n".format(angle))
        for i in range(len(line_fluxes)):
            f.write(
                f"Line: {line_names[i]} - Frequency: {line_frequencies[i]} GHz "
                f"- Flux: {line_fluxes[i]} Jy  - Width (Channels): {fwhm_z[i]}\n"
            )
        if snapshot is not None:
            f.write("TNG Snapshot ID: {}\n".format(snapshot))
            f.write("TNG Subhalo ID: {}\n".format(subhalo))
        f.close()


# def get_image_from_ssd(ra, dec, fov):
#    DEF_ACCESS_URL = "https://datalab.noirlab.edu/sia/sdss_dr9"
#    svc_sdss_dr9 = sia.SIAService(DEF_ACCESS_URL)
#    ac.whoAmI()
#    imgTable = svc_sdss_dr9.search(
#        (ra, dec), (fov / np.cos(dec * np.pi / 180), fov), verbosity=2
#    ).to_table()
