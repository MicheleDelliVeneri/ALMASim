"""One interferometer or total-power run per observing configuration (MADIV-28 T5.7)."""

from __future__ import annotations

from typing import Any

import astropy.units as U
import numpy as np

from .. import interferometry as uin
from .geometry import CubeGeometry
from .params import SimulationParams


def _count_antennas(antenna_array: str) -> int:
    """Count antennas encoded in an ALMA antenna-array string."""
    tokens = [token for token in str(antenna_array).split() if token.strip()]
    return max(len(tokens), 2)


def noise_reference(line_fluxes: Any, beam_area: U.Quantity, effective_snr: float) -> float:
    """The noise the per-channel noise profiles are calibrated to: a tenth of the
    faintest line's peak per beam, over the SNR."""
    line_flux_magnitudes = np.abs(np.asarray(line_fluxes, dtype=float))
    nonzero_line_fluxes = line_flux_magnitudes[line_flux_magnitudes > 0]
    min_line_flux = float(np.min(nonzero_line_fluxes)) if nonzero_line_fluxes.size > 0 else 0.0
    return 0.1 * (min_line_flux / beam_area.value) / effective_snr


def plan_observation_runs(
    params: SimulationParams,
    geometry: CubeGeometry,
    *,
    model_cube: np.ndarray,
    header: Any,
    channel_frequencies_hz: np.ndarray,
    effective_snr: float,
    base_noise_reference: float,
    output_dir_abs: str,
    persist_outputs: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The Interferometer keyword arguments and total-power runs, one per configuration."""
    observation_plan = geometry.observation_plan
    interferometer_runs = []
    total_power_runs = []
    for config in observation_plan.configs:
        raw_noise_profile = uin.compute_channel_noise(
            uin.NoiseModelConfig(
                pwv_mm=params.pwv,
                ground_temperature_k=params.ground_temperature_k,
            ),
            channel_frequencies_hz,
            geometry.channel_width_hz,
            config.total_time_s,
            observation_plan.elevation_deg,
            antenna_diameter_m=config.antenna_diameter_m,
            n_antennas=_count_antennas(config.antenna_array),
        )
        noise_profile = uin.calibrate_noise_profile(
            raw_noise_profile,
            reference_noise=base_noise_reference,
        )
        if config.array_type == "TP":
            total_power_runs.append(
                {
                    "config_name": config.name,
                    "array_type": config.array_type,
                    "antenna_array": config.antenna_array,
                    "antenna_diameter_m": config.antenna_diameter_m,
                    "integration_time_s": config.total_time_s,
                    "noise": noise_profile,
                }
            )
        else:
            interferometer_runs.append(
                {
                    "idx": params.idx,
                    "skymodel": model_cube,
                    "main_dir": params.main_dir,
                    "output_dir": output_dir_abs,
                    "ra": params.ra * U.deg,
                    "dec": params.dec * U.deg,
                    "central_freq": geometry.central_freq,
                    "bandwidth": geometry.band_range,
                    "fov": geometry.fov.value,
                    "antenna_array": config.antenna_array,
                    "noise": noise_profile,
                    "snr": effective_snr,
                    "integration_time": config.total_time_s / 3600.0,
                    "observation_date": params.obs_date,
                    "header": header,
                    "save_mode": params.save_mode,
                    "persist": persist_outputs,
                    "antenna_diameter_m": config.antenna_diameter_m,
                    "config_name": config.name,
                    "array_type": config.array_type,
                    "imaging_algorithm": params.imaging_algorithm,
                }
            )
    return interferometer_runs, total_power_runs
