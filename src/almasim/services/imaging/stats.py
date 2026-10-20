"""Image statistics shared by imaging, quality reports and self-calibration (MADIV-28 T2.8)."""

from __future__ import annotations

import math

import numpy as np


def robust_sigma(values: np.ndarray) -> float:
    """1.4826 x the median absolute deviation of the finite values (nan if none)."""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return math.nan
    return float(1.4826 * np.median(np.abs(finite - np.median(finite))))


def moment_mask(moment: np.ndarray, beam_pixels: float, sigma: float = 3.0) -> np.ndarray:
    """``moment`` above ``sigma`` times its robust noise, grown by one beam (a disk)."""
    noise = robust_sigma(moment)
    mask = moment > sigma * noise if noise > 0 else np.zeros(moment.shape, bool)
    if mask.any() and np.isfinite(beam_pixels) and beam_pixels > 0:
        from scipy import ndimage

        radius = int(np.ceil(beam_pixels))
        yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
        mask = ndimage.binary_dilation(mask, structure=np.hypot(yy, xx) <= beam_pixels)
    return mask
