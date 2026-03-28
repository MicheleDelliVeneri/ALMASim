"""Single-dish total-power simulation helpers."""
from __future__ import annotations

from typing import Any

import numpy as np


ARCSEC_PER_RAD = 206264.80624709636


def estimate_tp_beam_fwhm_arcsec(
    freqs_hz: np.ndarray,
    antenna_diameter_m: float = 12.0,
) -> np.ndarray:
    """Estimate a diffraction-limited TP beam FWHM per channel."""
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    wavelengths_m = 299792458.0 / np.clip(freqs_hz, 1.0, None)
    fwhm_rad = 1.13 * wavelengths_m / max(float(antenna_diameter_m), 1e-6)
    return fwhm_rad * ARCSEC_PER_RAD


def _gaussian_kernel(shape: tuple[int, int], sigma_pix: float) -> np.ndarray:
    """Build a centered 2D Gaussian kernel normalized to unit sum."""
    if sigma_pix <= 0.0:
        kernel = np.zeros(shape, dtype=np.float32)
        kernel[shape[0] // 2, shape[1] // 2] = 1.0
        return kernel
    yy, xx = np.indices(shape, dtype=float)
    cy = (shape[0] - 1) / 2.0
    cx = (shape[1] - 1) / 2.0
    rr2 = (xx - cx) ** 2 + (yy - cy) ** 2
    kernel = np.exp(-0.5 * rr2 / max(sigma_pix**2, 1e-12))
    kernel /= np.sum(kernel)
    return kernel.astype(np.float32)


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
    """Simulate a TP observation as beam-smoothed single-dish imaging."""
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
    sigma_pix = beam_fwhm_arcsec / (2.354820045 * max(float(cell_size_arcsec), 1e-6))

    tp_model_cube = np.zeros_like(model_cube, dtype=np.float32)
    tp_dirty_cube = np.zeros_like(model_cube, dtype=np.float32)
    tp_beam_cube = np.zeros_like(model_cube, dtype=np.float32)
    tp_sampling_cube = np.ones_like(model_cube, dtype=np.float32)

    for channel in range(model_cube.shape[0]):
        kernel = _gaussian_kernel(model_cube.shape[1:], float(sigma_pix[channel]))
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
    if not per_config_results:
        raise ValueError("Cannot combine an empty TP result list")
    if len(per_config_results) == 1:
        combined = dict(per_config_results[0])
        combined["per_config_results"] = per_config_results
        combined["combined_config_count"] = 1
        return combined

    if config_weights is None:
        config_weights = [1.0] * len(per_config_results)
    weights = np.asarray(config_weights, dtype=float)
    weights = np.clip(weights, 0.0, None)
    if not np.any(weights):
        weights = np.ones_like(weights)
    normalized = weights / np.sum(weights)

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
