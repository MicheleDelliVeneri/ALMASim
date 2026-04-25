"""Line emission functions."""

import os
import numpy as np
import pandas as pd


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
    """Get line emission information from CSV file."""
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
    """Compute rest frequency from redshift and observed frequency."""
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
