"""Single-dish total-power simulation helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..instrument.constants import ARCSEC_PER_RAD
from ..instrument.primary_beam import primary_beam_fwhm_rad, primary_beam_response
from .combine import normalized_config_weights, single_run_result


def estimate_tp_beam_fwhm_arcsec(
    freqs_hz: np.ndarray,
    antenna_diameter_m: float = 12.0,
) -> np.ndarray:
    """The TP beam FWHM per channel: the primary beam's half-power width."""
    freqs_hz = np.clip(np.asarray(freqs_hz, dtype=float), 1.0, None)
    return primary_beam_fwhm_rad(max(float(antenna_diameter_m), 1e-6), freqs_hz) * ARCSEC_PER_RAD


def _beam_kernel(
    shape: tuple[int, int], cell_size_arcsec: float, frequency_hz: float, dish_diameter_m: float
) -> np.ndarray:
    """The primary beam's power pattern on a centred grid, normalised to unit sum."""
    yy, xx = np.indices(shape, dtype=float)
    cy = (shape[0] - 1) / 2.0
    cx = (shape[1] - 1) / 2.0
    radius_rad = np.hypot(xx - cx, yy - cy) * float(cell_size_arcsec) / ARCSEC_PER_RAD
    kernel = primary_beam_response(radius_rad, frequency_hz, dish_diameter_m)
    total = float(np.sum(kernel))
    if not total > 0.0:
        kernel = np.zeros(shape)
        kernel[shape[0] // 2, shape[1] // 2] = 1.0
        return kernel.astype(np.float32)
    return (kernel / total).astype(np.float32)


def _fft_convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve a 2D image with a kernel using FFTs."""
    image_fft = np.fft.fft2(image)
    kernel_fft = np.fft.fft2(np.fft.ifftshift(kernel), s=image.shape)
    return np.real(np.fft.ifft2(image_fft * kernel_fft)).astype(np.float32)


def simulate_total_power_observation(
    model_cube: np.ndarray,
    *,
    freqs_hz: np.ndarray,
    cell_size_arcsec: float,
    noise_profile: np.ndarray | float,
    antenna_diameter_m: float = 12.0,
    config_name: str | None = None,
    array_type: str = "TP",
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Simulate a TP observation as single-dish imaging smoothed by the primary beam."""
    model_cube = np.asarray(model_cube, dtype=np.float32)
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    if model_cube.ndim != 3:
        raise ValueError("model_cube must have shape (n_channels, ny, nx)")
    if freqs_hz.shape[0] != model_cube.shape[0]:
        raise ValueError("freqs_hz length must match model_cube channels")

    rng = rng or np.random.default_rng(0)
    if np.isscalar(noise_profile):
        noise_profile = np.full(model_cube.shape[0], float(noise_profile), dtype=float)
    else:
        noise_profile = np.asarray(noise_profile, dtype=float)
    if noise_profile.shape[0] != model_cube.shape[0]:
        raise ValueError("noise_profile length must match model_cube channels")

    beam_fwhm_arcsec = estimate_tp_beam_fwhm_arcsec(
        freqs_hz,
        antenna_diameter_m=antenna_diameter_m,
    )

    tp_model_cube = np.zeros_like(model_cube, dtype=np.float32)
    tp_dirty_cube = np.zeros_like(model_cube, dtype=np.float32)
    tp_beam_cube = np.zeros_like(model_cube, dtype=np.float32)
    tp_sampling_cube = np.ones_like(model_cube, dtype=np.float32)

    for channel in range(model_cube.shape[0]):
        kernel = _beam_kernel(
            model_cube.shape[1:],
            max(float(cell_size_arcsec), 1e-6),
            float(np.clip(freqs_hz[channel], 1.0, None)),
            max(float(antenna_diameter_m), 1e-6),
        )
        tp_beam_cube[channel] = kernel / np.max(kernel)
        tp_model_cube[channel] = _fft_convolve2d(model_cube[channel], kernel)
        if noise_profile[channel] > 0.0:
            noise = rng.normal(
                loc=0.0,
                scale=float(noise_profile[channel]),
                size=model_cube[channel].shape,
            ).astype(np.float32)
        else:
            noise = np.zeros_like(model_cube[channel], dtype=np.float32)
        tp_dirty_cube[channel] = tp_model_cube[channel] + noise

    return {
        "tp_model_cube": tp_model_cube,
        "tp_dirty_cube": tp_dirty_cube,
        "tp_beam_cube": tp_beam_cube,
        "tp_sampling_cube": tp_sampling_cube,
        "tp_beam_fwhm_arcsec": beam_fwhm_arcsec.astype(np.float32),
        "config_name": config_name,
        "array_type": array_type,
        "antenna_diameter_m": float(antenna_diameter_m),
    }


def combine_total_power_results(
    per_config_results: list[dict[str, Any]],
    *,
    config_weights: list[float] | None = None,
) -> dict[str, Any]:
    """Combine multiple TP runs into one weighted TP product."""
    single = single_run_result(per_config_results, "TP")
    if single is not None:
        return single
    normalized = normalized_config_weights(len(per_config_results), config_weights)

    def _weighted_stack(key: str) -> np.ndarray:
        stack = np.stack([result[key] for result in per_config_results], axis=0)
        return np.tensordot(normalized, stack, axes=1)

    combined = dict(per_config_results[0])
    combined.update(
        {
            "tp_model_cube": _weighted_stack("tp_model_cube"),
            "tp_dirty_cube": _weighted_stack("tp_dirty_cube"),
            "tp_beam_cube": _weighted_stack("tp_beam_cube"),
            "tp_sampling_cube": np.sum(
                np.stack(
                    [result["tp_sampling_cube"] for result in per_config_results],
                    axis=0,
                ),
                axis=0,
            ),
            "tp_beam_fwhm_arcsec": _weighted_stack("tp_beam_fwhm_arcsec"),
            "per_config_results": per_config_results,
            "combined_config_count": len(per_config_results),
        }
    )
    return combined
