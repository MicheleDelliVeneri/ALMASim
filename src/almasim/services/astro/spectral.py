"""Spectral line and SED processing functions."""

from __future__ import annotations

from math import ceil, pi
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import astropy.units as U
from astropy.constants import c
from astropy.cosmology import FlatLambdaCDM

from .redshift import compute_redshift
from .tng import redshift_to_snapshot
from .lines import read_line_emission_csv

LogFn = Optional[Callable[[str], None]]


def _log(logger: LogFn, message: str, remote: bool = False) -> None:
    """Log a message using the logger callback or print."""
    if remote:
        print(message)
    elif logger is not None:
        logger(message)


def sample_given_redshift(
    metadata: pd.DataFrame,
    n: int,
    rest_frequency,
    extended: bool,
    zmax: Optional[float] = None,
    logger: LogFn = None,
) -> pd.DataFrame:
    """Filter and sample metadata entries based on redshift constraints."""
    pd.options.mode.chained_assignment = None
    metadata = metadata.copy()
    if isinstance(rest_frequency, (np.ndarray, list)):
        rest_frequency = np.sort(np.array(rest_frequency))
    else:
        rest_frequency = np.array([rest_frequency])

    freqs = metadata["Freq"].values
    if logger is not None:
        _log(logger, f"Max frequency recorded in metadata: {np.max(freqs)} GHz")
        _log(logger, f"Min frequency recorded in metadata: {np.min(freqs)} GHz")
        _log(logger, "Filtering metadata based on line catalogue...")
        _log(logger, f"Remaining metadata: {len(metadata)}")

    closest_rest_frequencies = []
    for freq in freqs:
        differences = rest_frequency - freq
        differences[differences < 0] = 1e10
        index_min = np.argmin(differences)
        closest_rest_frequencies.append(rest_frequency[index_min])
    rest_frequencies = np.array(closest_rest_frequencies)

    redshifts = [
        compute_redshift(rest_frequency * U.GHz, source_freq * U.GHz)
        for source_freq, rest_frequency in zip(freqs, rest_frequencies)
    ]
    metadata.loc[:, "redshift"] = redshifts
    metadata["rest_frequency"] = rest_frequencies

    n_metadata = 0
    z_save = zmax
    _log(logger, "Computing redshifts")
    while n_metadata < ceil(n / 10):
        s_metadata = n_metadata
        if zmax is not None:
            f_metadata = metadata[
                (metadata["redshift"] <= zmax) & (metadata["redshift"] >= 0)
            ]
        else:
            f_metadata = metadata[metadata["redshift"] >= 0]
        n_metadata = len(f_metadata)
        if n_metadata == s_metadata and zmax is not None:
            zmax += 0.1
    if zmax is not None:
        metadata = metadata[
            (metadata["redshift"] <= zmax) & (metadata["redshift"] >= 0)
        ]
    else:
        metadata = metadata[metadata["redshift"] >= 0]
    if z_save != zmax:
        _log(
            logger,
            "Max redshift has been adjusted fit metadata, "
            f"new max redshift: {round(zmax, 3)}",
        )
    _log(logger, f"Remaining metadata: {len(metadata)}")
    metadata["snapshot"] = [
        redshift_to_snapshot(redshift) for redshift in metadata["redshift"].values
    ]
    if extended:
        metadata = metadata[(metadata["snapshot"] == 99) | (metadata["snapshot"] == 95)]
    if len(metadata) == n:
        return metadata
    if len(metadata) < n:
        a_n = n - len(metadata)
        return pd.concat(
            [metadata, metadata.sample(a_n, replace=True)], ignore_index=True
        )
    return metadata.sample(n, replace=False)


def cont_finder(cont_frequencies, line_frequency):
    """Return the index of the closest continuum frequency to a given line."""
    distances = np.abs(
        cont_frequencies - np.ones(len(cont_frequencies)) * line_frequency
    )
    return int(np.argmin(distances))


def normalize_sed(
    sed: pd.DataFrame,
    lum_infrared: float,
    solid_angle: float,
    cont_sens: float,
    freq_min: float,
    freq_max: float,
    remote: bool = False,
    logger: LogFn = None,
):
    """Normalize the spectral energy distribution."""
    so_to_erg_s = 3.846e33
    lum_infrared_erg_s = lum_infrared * so_to_erg_s
    sed["Jy"] = lum_infrared_erg_s * sed["erg/s/Hz"] * 1e23 / solid_angle
    cont_mask = (sed["GHz"].values >= freq_min) & (sed["GHz"].values <= freq_max)
    if sum(cont_mask) > 0:
        cont_fluxes = sed["Jy"].values[cont_mask]
        min_ = np.min(cont_fluxes)
    else:
        freq_point = np.argmin(np.abs(sed["GHz"].values - freq_min))
        cont_fluxes = sed["Jy"].values[freq_point]
        min_ = cont_fluxes
    _log(logger, f"Minimum continum flux: {min_:.2e}", remote=remote)
    _log(logger, f"Continum sensitivity: {cont_sens:.2e}", remote=remote)
    lum_save = lum_infrared

    if min_ < cont_sens:
        while min_ < cont_sens:
            lum_infrared += 0.1 * lum_infrared
            lum_infrared_erg_s = so_to_erg_s * lum_infrared
            sed["Jy"] = lum_infrared_erg_s * sed["erg/s/Hz"] * 1e23 / solid_angle
            cont_mask = (sed["GHz"] >= freq_min) & (sed["GHz"] <= freq_max)
            if sum(cont_mask) > 0:
                cont_fluxes = sed["Jy"].values[cont_mask]
                min_ = np.min(cont_fluxes)
            else:
                freq_point = np.argmin(np.abs(sed["GHz"].values - freq_min))
                cont_fluxes = sed["Jy"].values[freq_point]
                min_ = cont_fluxes

    if lum_save != lum_infrared:
        _log(
            logger,
            f"To observe the source, luminosity has been set to {lum_infrared:.2e}",
            remote=remote,
        )
        _log(logger, "# ------------------------------------- #\n", remote=remote)
    return sed, lum_infrared_erg_s, lum_infrared


def sed_reading(
    source_type: str,
    path: str | Path,
    cont_sens: float,
    freq_min: float,
    freq_max: float,
    remote: bool,
    lum_infrared: Optional[float] = None,
    redshift: Optional[float] = None,
    logger: LogFn = None,
):
    """Load and normalize SED templates for the requested source type."""
    cosmo = FlatLambdaCDM(H0=70 * U.km / U.s / U.Mpc, Tcmb0=2.725 * U.K, Om0=0.3)
    path = Path(path)
    if source_type in {"extended", "diffuse", "molecular", "galaxy-zoo", "hubble-100"}:
        file_path = path / "SED_low_z_warm_star_forming_galaxy.dat"
        redshift = redshift or 10 ** (-4)
        lum_infrared = lum_infrared or 1e12
    elif source_type in {"point", "gaussian"}:
        file_path = path / "SED_low_z_type2_AGN.dat"
        redshift = redshift or 0.05
        lum_infrared = lum_infrared or 1e12
    else:
        raise ValueError("Not valid source type for SED selection")

    distance_Mpc = cosmo.luminosity_distance(redshift).value
    distance_cm = distance_Mpc * 3.086e24
    solid_angle = 4 * pi * distance_cm**2
    sed = pd.read_csv(file_path, sep=r"\s+")
    sed["GHz"] = sed["um"].apply(
        lambda x: (x * U.um).to(U.GHz, equivalencies=U.spectral()).value
    )
    sed, lum_infrared_erg_s, lum_infrared = normalize_sed(
        sed,
        lum_infrared,
        solid_angle,
        cont_sens,
        freq_min,
        freq_max,
        remote=remote,
        logger=logger,
    )
    flux_infrared = lum_infrared_erg_s * 1e23 / solid_angle
    sed.drop(columns=["um", "erg/s/Hz"], inplace=True)
    sed = sed.sort_values(by="GHz", ascending=True)
    return sed, flux_infrared, lum_infrared


def find_compatible_lines(
    db_line: pd.DataFrame,
    source_freq: float,
    redshift: Optional[float],
    n: int,
    line_names: Optional[Sequence[str]],
    freq_min: float,
    freq_max: float,
    band_range: float,
    line_width_range: Tuple[float, float],
):
    """Return a dataframe with the n spectral lines to simulate."""
    c_km_s = c.to(U.km / U.s)
    min_delta_v, max_delta_v = line_width_range
    db_line = db_line.copy()
    db_line["redshift"] = (db_line["freq(GHz)"].values - source_freq) / source_freq
    db_line = db_line.loc[~((db_line["redshift"] < 0) | (db_line["redshift"] > 20))]
    delta_v = np.random.uniform(min_delta_v, max_delta_v, len(db_line)) * U.km / U.s
    db_line["shifted_freq(GHz)"] = db_line["freq(GHz)"] / (1 + db_line["redshift"])
    fwhms = (
        0.84 * (db_line["shifted_freq(GHz)"].values * (delta_v / c_km_s) * 1e9) * U.Hz
    )
    db_line["delta_v"] = delta_v.value
    fwhms_GHz = fwhms.to(U.GHz).value
    db_line["fwhm_GHz"] = fwhms_GHz
    found_lines = 0
    lines_fitted = []
    if redshift is not None:
        db_line["redshift_distance"] = np.abs(db_line["redshift"] - redshift)
        db_line = db_line.sort_values(by="redshift_distance")
    for i in range(len(db_line)):
        db = db_line.copy()
        first_line = db.iloc[i]
        db["shifted_freq(GHz)"] = db["freq(GHz)"] / (1 + first_line["redshift"])
        db["distance(GHz)"] = abs(
            db["shifted_freq(GHz)"] - first_line["shifted_freq(GHz)"]
        )
        db = db.sort_values(by="distance(GHz)")
        lines = [first_line]
        lines_redshifts = [first_line["redshift"]]
        found_lines = 1
        for j in range(1, len(db)):
            if (
                db.iloc[j]["shifted_freq(GHz)"] >= freq_min
                and db.iloc[j]["shifted_freq(GHz)"] <= freq_max
            ):
                if (
                    db.iloc[j]["shifted_freq(GHz)"] + db.iloc[j]["fwhm_GHz"] / 2
                    <= freq_max
                    and db.iloc[j]["shifted_freq(GHz)"] - db.iloc[j]["fwhm_GHz"] / 2
                    >= freq_min
                ):
                    lines.append(db.iloc[j])
                    lines_redshifts.append(db.iloc[j]["redshift"])
                    found_lines += 1
                if found_lines == n:
                    break
        if found_lines == n:
            lines_fitted = lines
            break
    compatible_lines = pd.DataFrame(lines_fitted)
    if found_lines < n:
        if line_names is not None and len(line_names) > 0:
            raise ValueError(
                "Not enough lines available to fit the requested configuration."
            )
        df_line = db_line.sample(n - found_lines, replace=True)
        df_line["shifted_freq(GHz)"] = np.random.uniform(
            freq_min, freq_max, len(df_line)
        )
        df_line["fwhm_GHz"] = (band_range / n) / 10
        df_line["Line"] = ["Unknown"] * len(df_line)
        compatible_lines = pd.concat(
            (compatible_lines, df_line), ignore_index=True, sort=False
        )
    compatible_lines = compatible_lines.reset_index(drop=True)
    for index, row in compatible_lines.iterrows():
        lower_bound = row["shifted_freq(GHz)"] - row["fwhm_GHz"] / 2
        upper_bound = row["shifted_freq(GHz)"] + row["fwhm_GHz"] / 2
        while lower_bound < freq_min and upper_bound > freq_max:
            row["fwhm_GHz"] -= 0.1
            lower_bound = row["shifted_freq(GHz)"] - row["fwhm_GHz"] / 2
            upper_bound = row["shifted_freq(GHz)"] + row["fwhm_GHz"] / 2
        if row["fwhm_GHz"] != compatible_lines["fwhm_GHz"].iloc[index]:
            compatible_lines.loc[index, "fwhm_GHz"] = row["fwhm_GHz"]
    return compatible_lines


def process_spectral_data(
    source_type: str,
    master_path: str | Path,
    redshift: float,
    central_frequency: float,
    delta_freq: float,
    source_frequency: float,
    n_channels: int,
    lum_infrared: float,
    cont_sens: float,
    line_names: Optional[Sequence[str]] = None,
    n_lines: Optional[int] = None,
    remote: bool = False,
    line_width_range: Tuple[float, float] = (50.0, 300.0),
    logger: LogFn = None,
):
    """Compute continua and emission lines for a simulated observation."""
    master_path = Path(master_path)
    freq_min = central_frequency - delta_freq / 2
    freq_max = central_frequency + delta_freq / 2
    sed, flux_infrared, lum_infrared = sed_reading(
        source_type,
        master_path / "brightnes",
        cont_sens,
        freq_min,
        freq_max,
        remote,
        lum_infrared,
    )
    db_line = read_line_emission_csv(
        master_path / "brightnes" / "calibrated_lines.csv",
        sep=",",
    )
    if line_names is None:
        n = n_lines if n_lines is not None else 1
    else:
        n = len(line_names)
        db_line = db_line[db_line["Line"].isin(line_names)]
    filtered_lines = find_compatible_lines(
        db_line,
        source_frequency,
        redshift,
        n,
        line_names,
        freq_min,
        freq_max,
        delta_freq,
        line_width_range,
    )

    cont_mask = (sed["GHz"] >= freq_min) & (sed["GHz"] <= freq_max)
    if sum(cont_mask) > 0:
        cont_fluxes = sed["Jy"].values[cont_mask]
        cont_frequencies = sed["GHz"].values[cont_mask]
    else:
        freq_point = np.argmin(np.abs(sed["GHz"].values - freq_min))
        cont_fluxes = [sed["Jy"].values[freq_point]]
        cont_frequencies = [sed["GHz"].values[freq_point]]

    line_names = filtered_lines["Line"].values
    cs = filtered_lines["c"].values
    cdeltas = filtered_lines["err_c"].values
    line_ratios = np.array([np.random.normal(c, cd) for c, cd in zip(cs, cdeltas)])
    line_frequencies = filtered_lines["shifted_freq(GHz)"].values
    new_cont_freq = np.linspace(freq_min, freq_max, n_channels)
    if len(cont_fluxes) > 1:
        int_cont_fluxes = np.interp(new_cont_freq, cont_frequencies, cont_fluxes)
    else:
        int_cont_fluxes = np.ones(n_channels) * cont_fluxes[0]
    line_indexes = filtered_lines["shifted_freq(GHz)"].apply(
        lambda x: cont_finder(new_cont_freq, float(x))
    )
    fwhms_GHz = filtered_lines["fwhm_GHz"].values
    freq_steps = (
        np.array(
            [
                new_cont_freq[line_index] + fwhm - new_cont_freq[line_index]
                for fwhm, line_index in zip(fwhms_GHz, line_indexes)
            ]
        )
        * U.GHz
    )
    _log(
        logger,
        f"Line Velocities: {filtered_lines['delta_v'].values} Km/s",
        remote=remote,
    )
    freq_steps = freq_steps.to(U.Hz).value
    line_fluxes = 10 ** (np.log10(flux_infrared) + line_ratios) / freq_steps
    bandwidth = freq_max - freq_min
    freq_support = bandwidth / n_channels
    fwhms = []
    for fwhm in fwhms_GHz / freq_support:
        if fwhm >= 1:
            fwhms.append(fwhm)
        else:
            fwhms.append(1)
    return (
        int_cont_fluxes,
        line_fluxes,
        line_names,
        redshift,
        line_frequencies,
        line_indexes,
        n_channels,
        bandwidth,
        freq_support,
        new_cont_freq,
        fwhms,
        lum_infrared,
    )
