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
