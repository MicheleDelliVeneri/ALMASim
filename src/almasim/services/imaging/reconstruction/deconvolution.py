"""Wiener and Hogbom deconvolution of dirty cubes (MADIV-28 T4.10)."""

from __future__ import annotations

from typing import Any

import numpy as np

from .cube_ops import (
    _crop_channel,
    _estimate_psf_padding,
    _insert_shifted_kernel,
    _pad_channel,
)


def wiener_deconvolve_cube(
    dirty_cube: np.ndarray,
    beam_cube: np.ndarray,
    *,
    epsilon: float = 1e-3,
    clip_negative: bool = True,
) -> np.ndarray:
    """Apply a simple deterministic Wiener-style deconvolution per channel."""
    dirty_cube = np.asarray(dirty_cube, dtype=np.float32)
    beam_cube = np.asarray(beam_cube, dtype=np.float32)
    if dirty_cube.shape != beam_cube.shape:
        raise ValueError("dirty_cube and beam_cube must have the same shape")

    recon = np.zeros_like(dirty_cube, dtype=np.float32)
    for channel in range(dirty_cube.shape[0]):
        psf = beam_cube[channel]
        psf = psf / max(float(np.max(np.abs(psf))), 1e-12)
        psf_fft = np.fft.fft2(np.fft.ifftshift(psf))
        dirty_fft = np.fft.fft2(dirty_cube[channel])
        denom = np.abs(psf_fft) ** 2 + float(epsilon)
        recon_channel = np.real(np.fft.ifft2(np.conj(psf_fft) * dirty_fft / denom))
        if clip_negative:
            recon_channel = np.clip(recon_channel, 0.0, None)
        recon[channel] = recon_channel.astype(np.float32)
    return recon


def _build_clean_beam(psf: np.ndarray) -> np.ndarray:
    """Approximate the central clean beam as a Gaussian from PSF moments."""
    peak_y, peak_x = np.unravel_index(int(np.argmax(np.abs(psf))), psf.shape)
    peak = float(psf[peak_y, peak_x])
    if abs(peak) < 1e-12:
        clean_beam = np.zeros_like(psf, dtype=np.float32)
        clean_beam[peak_y, peak_x] = 1.0
        return clean_beam

    normalized = np.asarray(psf, dtype=np.float32) / peak
    positive = np.clip(normalized, 0.0, None)

    window_radius_y = max(2, psf.shape[0] // 10)
    window_radius_x = max(2, psf.shape[1] // 10)
    y0 = max(0, peak_y - window_radius_y)
    y1 = min(psf.shape[0], peak_y + window_radius_y + 1)
    x0 = max(0, peak_x - window_radius_x)
    x1 = min(psf.shape[1], peak_x + window_radius_x + 1)
    cropped = positive[y0:y1, x0:x1]

    yy, xx = np.indices(cropped.shape, dtype=np.float32)
    yy += float(y0)
    xx += float(x0)
    weights = cropped / max(float(np.sum(cropped)), 1e-12)
    sigma_y = float(np.sqrt(np.sum(weights * (yy - peak_y) ** 2)))
    sigma_x = float(np.sqrt(np.sum(weights * (xx - peak_x) ** 2)))
    sigma_y = max(sigma_y, 1.0)
    sigma_x = max(sigma_x, 1.0)

    full_y, full_x = np.indices(psf.shape, dtype=np.float32)
    clean_beam = np.exp(
        -0.5 * (((full_y - peak_y) / sigma_y) ** 2 + ((full_x - peak_x) / sigma_x) ** 2)
    )
    clean_beam /= max(float(np.max(clean_beam)), 1e-12)
    return clean_beam.astype(np.float32)


def clean_deconvolve_cube(
    dirty_cube: np.ndarray,
    beam_cube: np.ndarray,
    *,
    n_cycles: int = 100,
    gain: float = 0.1,
    threshold: float | None = None,
    clip_negative: bool = False,
    initial_component_cube: np.ndarray | None = None,
    initial_residual_cube: np.ndarray | None = None,
    initial_clean_beam_cube: np.ndarray | None = None,
    initial_cycles_completed: int = 0,
) -> dict[str, Any]:
    """Run a simple Hogbom-style CLEAN loop per channel."""
    dirty_cube = np.asarray(dirty_cube, dtype=np.float32)
    beam_cube = np.asarray(beam_cube, dtype=np.float32)
    if dirty_cube.shape != beam_cube.shape:
        raise ValueError("dirty_cube and beam_cube must have the same shape")
    if dirty_cube.ndim != 3:
        raise ValueError("dirty_cube and beam_cube must have shape (n_channels, ny, nx)")
    if n_cycles < 0:
        raise ValueError("n_cycles must be non-negative")
    if not 0.0 < gain <= 1.0:
        raise ValueError("gain must be in the interval (0, 1]")

    component_cube = (
        np.zeros_like(dirty_cube, dtype=np.float32)
        if initial_component_cube is None
        else np.asarray(initial_component_cube, dtype=np.float32).copy()
    )
    residual_cube = (
        dirty_cube.copy()
        if initial_residual_cube is None
        else np.asarray(initial_residual_cube, dtype=np.float32).copy()
    )
    clean_beam_cube = (
        np.zeros_like(beam_cube, dtype=np.float32)
        if initial_clean_beam_cube is None
        else np.asarray(initial_clean_beam_cube, dtype=np.float32).copy()
    )
    if component_cube.shape != dirty_cube.shape:
        raise ValueError("initial_component_cube must match dirty_cube shape")
    if residual_cube.shape != dirty_cube.shape:
        raise ValueError("initial_residual_cube must match dirty_cube shape")
    if clean_beam_cube.shape != beam_cube.shape:
        raise ValueError("initial_clean_beam_cube must match beam_cube shape")

    restored_cube = np.zeros_like(dirty_cube, dtype=np.float32)
    cycles_added = 0

    for channel in range(dirty_cube.shape[0]):
        psf = np.asarray(beam_cube[channel], dtype=np.float32)
        pad_y, pad_x = _estimate_psf_padding(psf)
        psf = _pad_channel(psf, pad_y, pad_x)
        peak_y, peak_x = np.unravel_index(int(np.argmax(np.abs(psf))), psf.shape)
        psf_peak = float(psf[peak_y, peak_x])
        if abs(psf_peak) < 1e-12:
            psf_normalized = np.zeros_like(psf, dtype=np.float32)
            psf_normalized[peak_y, peak_x] = 1.0
        else:
            psf_normalized = (psf / psf_peak).astype(np.float32)

        existing_clean_beam = np.asarray(clean_beam_cube[channel], dtype=np.float32)
        if np.max(np.abs(existing_clean_beam)) > 0.0:
            clean_beam = _pad_channel(existing_clean_beam, pad_y, pad_x)
        else:
            clean_beam = _build_clean_beam(psf_normalized)

        residual = _pad_channel(residual_cube[channel], pad_y, pad_x)
        components = _pad_channel(component_cube[channel], pad_y, pad_x)
        channel_cycles = 0

        for _ in range(int(n_cycles)):
            peak_index = np.unravel_index(int(np.argmax(np.abs(residual))), residual.shape)
            peak_value = float(residual[peak_index])
            if threshold is not None and abs(peak_value) < float(threshold):
                break
            component_amplitude = gain * peak_value
            components[peak_index] += component_amplitude
            _insert_shifted_kernel(
                residual,
                psf_normalized,
                -component_amplitude,
                int(peak_index[0]),
                int(peak_index[1]),
                int(peak_y),
                int(peak_x),
            )
            channel_cycles += 1

        restored = np.real(
            np.fft.ifft2(np.fft.fft2(components) * np.fft.fft2(np.fft.ifftshift(clean_beam)))
        ).astype(np.float32)
        restored += residual
        if clip_negative:
            restored = np.clip(restored, 0.0, None)
        component_cube[channel] = _crop_channel(components, pad_y, pad_x)
        residual_cube[channel] = _crop_channel(residual, pad_y, pad_x)
        clean_beam_cube[channel] = _crop_channel(clean_beam, pad_y, pad_x)
        restored_cube[channel] = _crop_channel(restored, pad_y, pad_x)
        cycles_added = max(cycles_added, channel_cycles)

    return {
        "restored_cube": restored_cube,
        "component_cube": component_cube,
        "residual_cube": residual_cube,
        "clean_beam_cube": clean_beam_cube,
        "cycles_completed": int(initial_cycles_completed) + cycles_added,
        "cycles_added": cycles_added,
        "gain": float(gain),
        "threshold": None if threshold is None else float(threshold),
    }
