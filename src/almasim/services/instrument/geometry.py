"""Synthesised beam, field of view, cell and image size (MADIV-28 T2.11).

One model for the CLI imaging planner (``imaging.ms_geometry``) and the
simulator (``simulation.geometry``, with ``instrument_geometry="planner"``).
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np

from .constants import ARCSEC_PER_RAD, SPEED_OF_LIGHT_M_S
from .primary_beam import primary_beam_fwhm_rad

MIN_IMAGE_PIXELS = 16


def max_baseline_m(positions_m: np.ndarray) -> float:
    """The longest baseline (m) between antennas at ``positions_m`` (n, 3)."""
    positions = np.asarray(positions_m, dtype=float)
    i, j = np.triu_indices(positions.shape[0], k=1)
    return float(max(np.linalg.norm(positions[j, :] - positions[i, :], axis=1)))


def synthesized_beam_arcsec(frequency_hz: Any, max_baseline: float) -> Any:
    """The synthesised beam (arcsec), lambda / B_max, at ``frequency_hz``."""
    return SPEED_OF_LIGHT_M_S * ARCSEC_PER_RAD / frequency_hz / max_baseline


def primary_beam_fov_arcsec(dish_diameter_m: float, frequency_hz: Any) -> Any:
    """The field of view (arcsec): the primary beam's half-power width."""
    return primary_beam_fwhm_rad(float(dish_diameter_m), frequency_hz) * ARCSEC_PER_RAD


def cell_arcsec(beam_arcsec: Any, beam_sampling: float) -> Any:
    """The image cell: ``beam_sampling`` pixels across the synthesised beam."""
    return beam_arcsec / beam_sampling


def image_size(
    fov_arcsec: float,
    cell: float,
    *,
    rounding: Literal["ceil", "pow2"] = "ceil",
    minimum: int = MIN_IMAGE_PIXELS,
    min_pixels: float = 0.0,
) -> int:
    """Image side (pixels) covering ``fov_arcsec`` at ``cell``.

    ``ceil`` (the imaging planner) rounds up and keeps at least ``minimum``
    and ``min_pixels`` pixels; ``pow2`` (the simulator's FFT grid) takes the
    power of two closest to the truncated pixel count.
    """
    if rounding == "pow2":
        from ..interferometry.utils import closest_power_of_2

        return int(closest_power_of_2(int(fov_arcsec / cell)))
    return max(minimum, int(math.ceil(fov_arcsec / cell)), int(math.ceil(min_pixels)))
