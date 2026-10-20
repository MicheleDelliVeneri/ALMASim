"""Wiener deconvolution of dirty cubes, the interactive preview (MADIV-28 T4.10, T5.12).

The Hogbom ``clean_deconvolve_cube`` was retired in T5.12: WSClean on the
simulated MeasurementSet is the reference imaging.
"""

from __future__ import annotations

import numpy as np

from .cube_ops import convolve_cube_with_beam


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


def clean_beam_cube(beam_cube: np.ndarray) -> np.ndarray:
    """A Gaussian clean beam per channel, fitted to the central lobe of each PSF."""
    beam_cube = np.asarray(beam_cube, dtype=np.float32)
    return np.stack([_build_clean_beam(psf) for psf in beam_cube]).astype(np.float32)


def wiener_preview(
    dirty_cube: np.ndarray,
    beam_cube: np.ndarray,
    *,
    epsilon: float = 1e-3,
) -> dict[str, np.ndarray]:
    """The Wiener reconstruction of a dirty cube, restored like a CLEAN image (MADIV-28 T5.12).

    ``model`` is :func:`wiener_deconvolve_cube`'s image, ``residual`` the
    dirty cube minus the model convolved with the (peak-normalised) PSF,
    ``restored`` the model convolved with a Gaussian clean beam fitted to
    each PSF (``clean_beam``) plus the residual. It is the interactive
    preview; WSClean on the simulated MS is the reference imaging.
    """
    dirty_cube = np.asarray(dirty_cube, dtype=np.float32)
    beam_cube = np.asarray(beam_cube, dtype=np.float32)
    if dirty_cube.shape != beam_cube.shape:
        raise ValueError("dirty_cube and beam_cube must have the same shape")
    if dirty_cube.ndim != 3:
        raise ValueError("dirty_cube and beam_cube must have shape (n_channels, ny, nx)")
    if not epsilon > 0:
        raise ValueError("epsilon must be positive")
    model = wiener_deconvolve_cube(dirty_cube, beam_cube, epsilon=epsilon)
    peaks = np.max(np.abs(beam_cube), axis=(1, 2), keepdims=True)
    psf = beam_cube / np.maximum(peaks, 1e-12)
    residual = dirty_cube - convolve_cube_with_beam(model, psf)
    clean_beam = clean_beam_cube(beam_cube)
    restored = convolve_cube_with_beam(model, clean_beam) + residual
    return {
        "model": model,
        "restored": restored.astype(np.float32),
        "residual": residual.astype(np.float32),
        "clean_beam": clean_beam,
    }
