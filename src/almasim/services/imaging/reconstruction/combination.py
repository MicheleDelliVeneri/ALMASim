"""Total-power and interferometric images combined, and the image products (MADIV-28 T4.10)."""

from __future__ import annotations

from typing import Any

import numpy as np

from .cube_ops import regrid_cube_to_match
from .deconvolution import wiener_deconvolve_cube


def match_tp_to_int_flux_scale(
    tp_cube: np.ndarray,
    int_cube: np.ndarray,
    beam_cube: np.ndarray | None = None,
) -> float:
    """Estimate a scalar TP-to-INT flux scale from overlapping weighted flux."""
    tp_cube = np.asarray(tp_cube, dtype=np.float32)
    int_cube = np.asarray(int_cube, dtype=np.float32)
    if tp_cube.shape != int_cube.shape:
        tp_cube = regrid_cube_to_match(tp_cube, int_cube.shape)

    if beam_cube is None:
        weights = np.ones_like(int_cube, dtype=np.float32)
    else:
        beam_cube = np.asarray(beam_cube, dtype=np.float32)
        weights = np.clip(beam_cube, 0.0, None)
        if weights.shape != int_cube.shape:
            weights = regrid_cube_to_match(weights, int_cube.shape)

    tp_flux = float(np.sum(weights * tp_cube))
    int_flux = float(np.sum(weights * int_cube))
    if abs(tp_flux) < 1e-12:
        return 1.0
    return int_flux / tp_flux


def feather_merge_cube(
    int_cube: np.ndarray,
    tp_cube: np.ndarray,
    tp_beam_cube: np.ndarray,
    *,
    int_primary_beam: np.ndarray | None = None,
    flux_scale: float | None = None,
) -> np.ndarray:
    """Merge TP and INT image cubes using a beam-shaped Fourier low-pass weight."""
    int_cube = np.asarray(int_cube, dtype=np.float32)
    tp_cube = np.asarray(tp_cube, dtype=np.float32)
    tp_beam_cube = np.asarray(tp_beam_cube, dtype=np.float32)

    if tp_cube.shape != int_cube.shape:
        tp_cube = regrid_cube_to_match(tp_cube, int_cube.shape)
    if tp_beam_cube.shape != int_cube.shape:
        tp_beam_cube = regrid_cube_to_match(tp_beam_cube, int_cube.shape)
    if int_primary_beam is not None:
        int_primary_beam = np.asarray(int_primary_beam, dtype=np.float32)
        if int_primary_beam.shape != int_cube.shape:
            int_primary_beam = regrid_cube_to_match(int_primary_beam, int_cube.shape)

    if flux_scale is None:
        flux_scale = match_tp_to_int_flux_scale(tp_cube, int_cube, int_primary_beam)

    merged = np.zeros_like(int_cube, dtype=np.float32)
    for channel in range(int_cube.shape[0]):
        tp_fft = np.fft.fft2(tp_cube[channel] * float(flux_scale))
        int_fft = np.fft.fft2(int_cube[channel])
        low_pass = np.abs(np.fft.fft2(np.fft.ifftshift(tp_beam_cube[channel])))
        low_pass /= max(float(np.max(low_pass)), 1e-12)
        low_pass = np.clip(low_pass, 0.0, 1.0)
        merged[channel] = np.real(
            np.fft.ifft2(low_pass * tp_fft + (1.0 - low_pass) * int_fft)
        ).astype(np.float32)
    return merged


def build_image_products(
    *,
    int_results: dict[str, Any] | None,
    tp_results: dict[str, Any] | None,
    reconstruction_epsilon: float = 1e-3,
) -> dict[str, Any]:
    """Build reproducible image-domain products from measurement-stage outputs."""
    products: dict[str, Any] = {
        "int_image_cube": None,
        "tp_image_cube": None,
        "tp_int_image_cube": None,
        "per_config_image_results": [],
    }

    if int_results is not None and "dirty_cube" in int_results and "beam_cube" in int_results:
        products["int_image_cube"] = wiener_deconvolve_cube(
            int_results["dirty_cube"],
            int_results["beam_cube"],
            epsilon=reconstruction_epsilon,
        )
        per_config_results = int_results.get("per_config_results", [])
        products["per_config_image_results"] = [
            {
                "config_name": config_result.get("config_name"),
                "array_type": config_result.get("array_type"),
                "image_cube": wiener_deconvolve_cube(
                    config_result["dirty_cube"],
                    config_result["beam_cube"],
                    epsilon=reconstruction_epsilon,
                ),
            }
            for config_result in per_config_results
            if "dirty_cube" in config_result and "beam_cube" in config_result
        ]

    if tp_results is not None:
        products["tp_image_cube"] = tp_results["tp_dirty_cube"]

    if (
        products["int_image_cube"] is not None
        and products["tp_image_cube"] is not None
        and tp_results is not None
    ):
        products["tp_int_image_cube"] = feather_merge_cube(
            products["int_image_cube"],
            products["tp_image_cube"],
            tp_results["tp_beam_cube"],
            int_primary_beam=int_results.get("beam_cube") if int_results else None,
        )
    elif products["int_image_cube"] is not None:
        products["tp_int_image_cube"] = products["int_image_cube"]
    elif products["tp_image_cube"] is not None:
        products["tp_int_image_cube"] = products["tp_image_cube"]

    return products
