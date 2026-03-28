"""Interferometry service for ALMA simulations."""
from .core import Interferometer, ProgressSignal
from .baselines import prepare_baselines, set_baselines, set_noise
from .imaging import (
    prepare_2d_arrays,
    _grid_uv,
    set_beam,
    _prepare_model,
    set_primary_beam,
    observe,
    image_channel,
    check_lfac,
    add_thermal_noise,
)
from .utils import get_channel_wavelength, closest_power_of_2, sampling_to_uv_mask
from .antenna import (
    estimate_alma_beam_size,
    get_fov_from_band,
    generate_antenna_config_file_from_antenna_array,
    compute_distance,
    get_max_baseline_from_antenna_config,
    get_max_baseline_from_antenna_array,
)
from .frequency import freq_supp_extractor, remove_non_numeric

__all__ = [
    "Interferometer",
    "ProgressSignal",
    "prepare_baselines",
    "set_baselines",
    "set_noise",
    "prepare_2d_arrays",
    "_grid_uv",
    "set_beam",
    "_prepare_model",
    "set_primary_beam",
    "observe",
    "image_channel",
    "check_lfac",
    "add_thermal_noise",
    "get_channel_wavelength",
    "closest_power_of_2",
    "sampling_to_uv_mask",
    "estimate_alma_beam_size",
    "get_fov_from_band",
    "generate_antenna_config_file_from_antenna_array",
    "compute_distance",
    "get_max_baseline_from_antenna_config",
    "get_max_baseline_from_antenna_array",
    "freq_supp_extractor",
    "remove_non_numeric",
]
