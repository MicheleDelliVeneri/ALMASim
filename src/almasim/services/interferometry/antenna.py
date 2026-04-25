"""Antenna configuration and baseline calculation functions."""

import math
import os

import astropy.units as U
import numpy as np
import pandas as pd
from astropy.constants import c
from astropy.units import Quantity


def estimate_alma_beam_size(central_frequency_ghz, max_baseline_km, return_value=True):
    """
    Estimates the beam size of the Atacama Large Millimeter/submillimeter Array (ALMA)
    in arcseconds.

    This function provides an approximation based on the theoretical relationship between
    observing frequency and maximum baseline. The formula used is:
    beam_size = (speed_of_light / central_frequency) / max_baseline * (180 / pi) * 3600
    arcseconds [km]/[s] * [s] / [km] = [radians] * [arcsec /radian] * [arcseconds/degree]

    Args:
        central_frequency_ghz: Central frequency of the observing band in GHz (float).
        max_baseline_km: Maximum baseline of the antenna array in kilometers (float).

    Returns:
        Estimated beam size in arcseconds (float).

    Raises:
        ValueError: If either input argument is non-positive.
    """
    # Input validation
    if central_frequency_ghz <= 0 or max_baseline_km <= 0:
        raise ValueError("Central frequency and maximum baseline must be positive values.")

    if not isinstance(central_frequency_ghz, Quantity):
        central_frequency_ghz = central_frequency_ghz * U.GHz
    if not isinstance(max_baseline_km, Quantity):
        max_baseline_km = max_baseline_km * U.km

    # Speed of light in meters per second
    light_speed = c.to(U.m / U.s).value

    # Convert frequency to Hz
    central_frequency_hz = central_frequency_ghz.to(U.Hz).value

    # Convert baseline to meters
    max_baseline_meters = max_baseline_km.to(U.m).value

    # Theoretical estimate of beam size (radians)
    theta_radians = (light_speed / central_frequency_hz) / max_baseline_meters

    # Convert theta from radians to arcseconds
    beam_size_arcsec = theta_radians * (180 / math.pi) * 3600 * U.arcsec
    if return_value is True:
        return beam_size_arcsec.value
    else:
        return beam_size_arcsec


def get_fov_from_band(band, antenna_diameter: int = 12, return_value=True):
    """
    This function returns the field of view of an ALMA band in arcseconds
    input:
        band number (int): the band number of the ALMA band, between 1 and 10
        antenna_diameter (int): the diameter of the antenna in meters
    output:
        fov (astropy unit): the field of view in arcseconds

    """
    light_speed = c.to(U.m / U.s).value
    if band == 1:
        central_freq = 43 * U.GHz
    elif band == 2:
        central_freq = 67 * U.GHz
    elif band == 3:
        central_freq = 100 * U.GHz
    elif band == 4:
        central_freq = 150 * U.GHz
    elif band == 5:
        central_freq = 217 * U.GHz
    elif band == 6:
        central_freq = 250 * U.GHz
    elif band == 7:
        central_freq = 353 * U.GHz
    elif band == 8:
        central_freq = 545 * U.GHz
    elif band == 9:
        central_freq = 650 * U.GHz
    elif band == 10:
        central_freq = 868.5 * U.GHz
    central_freq = central_freq.to(U.Hz).value
    central_freq_s = 1 / central_freq
    wavelength = light_speed * central_freq_s
    # this is the field of view in Radians
    fov = 1.22 * wavelength / antenna_diameter
    # fov in arcsec
    fov = fov * (180 / math.pi) * 3600 * U.arcsec
    if return_value is True:
        return fov.value
    else:
        return fov


def generate_antenna_config_file_from_antenna_array(antenna_array, master_path, output_dir):
    """Generate antenna configuration file from antenna array string."""
    antenna_coordinates = pd.read_csv(
        os.path.join(master_path, "antenna_config", "antenna_coordinates.csv")
    )
    obs_antennas = antenna_array.split(" ")
    obs_antennas = [antenna.split(":")[0] for antenna in obs_antennas]
    obs_coordinates = antenna_coordinates[antenna_coordinates["name"].isin(obs_antennas)]
    intro_string = "# observatory=ALMA\n# coordsys=LOC (local tangent plane)\n# x y z diam pad#\n"
    with open(os.path.join(output_dir, "antenna.cfg"), "w") as f:
        f.write(intro_string)
        for i in range(len(obs_coordinates)):
            f.write(
                f"{obs_coordinates['x'].values[i]} {obs_coordinates['y'].values[i]} {
                    obs_coordinates['z'].values[i]
                } 12. {obs_coordinates['name'].values[i]}\n"
            )
    f.close()


def compute_distance(x1, y1, z1, x2, y2, z2):
    """Compute Euclidean distance between two 3D points."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


def get_max_baseline_from_antenna_config(update_progress, antenna_config):
    """Get maximum baseline from antenna configuration file."""
    positions = []
    with open(antenna_config, "r") as f:
        lines = f.readlines()
        for line in lines:
            if not line.strip().startswith("#"):
                if "\t" in line:
                    row = [x for x in line.split("\t")][:3]
                else:
                    row = [x for x in line.split(" ")][:3]
                positions.append([float(x) for x in row])
    positions = np.array(positions)
    max_baseline = 0

    for i in range(len(positions)):
        x1, y1, z1 = positions[i]
        for j in range(i + 1, len(positions)):
            x2, y2, z2 = positions[j]
            dist = compute_distance(x1, y1, z1, x2, y2, z2) / 1000
            if dist > max_baseline:
                max_baseline = dist
        if update_progress is not None:
            update_progress.emit((i / len(positions) * 100))

    return max_baseline


def get_max_baseline_from_antenna_array(antenna_array, master_path):
    """Get maximum baseline from antenna array string."""
    antenna_coordinates = pd.read_csv(
        os.path.join(master_path, "antenna_config", "antenna_coordinates.csv")
    )
    obs_antennas = antenna_array.split(" ")
    obs_antennas = [antenna.split(":")[0] for antenna in obs_antennas]
    obs_coordinates = antenna_coordinates[antenna_coordinates["name"].isin(obs_antennas)].values
    max_baseline = 0
    for i in range(len(obs_coordinates)):
        name, x1, y1, z1 = obs_coordinates[i]
        for j in range(i + 1, len(obs_coordinates)):
            name, x2, y2, z2 = obs_coordinates[j]
            dist = compute_distance(x1, y1, z1, x2, y2, z2) / 1000
            if dist > max_baseline:
                max_baseline = dist
    return max_baseline
