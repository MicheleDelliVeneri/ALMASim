"""World-coordinate helpers shared by the imaging products (MADIV-28 T2.8)."""

from __future__ import annotations

import math

import numpy as np

from ..instrument.constants import SPEED_OF_LIGHT_KM_S


def beam_pixels(header, bmaj_deg: float | None = None) -> float:
    """The beam major axis in pixels (``bmaj_deg``, else the header BMAJ); nan without a cell."""
    cell = abs(float(header.get("CDELT2", 0.0)))
    bmaj = float(header.get("BMAJ", 0.0)) if bmaj_deg is None else float(bmaj_deg)
    return bmaj / cell if cell else math.nan


def frequency_axis(header, *, fallback_axis: int | None = None) -> tuple[int, np.ndarray]:
    """(FITS axis number, frequency of each plane in Hz) of the image's FREQ axis.

    Without a FREQ axis, ``fallback_axis`` is used as the spectral axis when
    given; otherwise ``ValueError``.
    """
    naxis = int(header.get("NAXIS", 0))
    axis = next(
        (a for a in range(3, naxis + 1) if str(header.get(f"CTYPE{a}", "")).upper() == "FREQ"),
        fallback_axis,
    )
    if axis is None:
        raise ValueError("the image has no FREQ axis")
    n = int(header.get(f"NAXIS{axis}", 0))
    pixels = np.arange(n) + 1.0 - float(header.get(f"CRPIX{axis}", 1.0))
    return axis, float(header.get(f"CRVAL{axis}", 0.0)) + pixels * float(
        header.get(f"CDELT{axis}", 0.0)
    )


def velocity_axis(header, *, fallback_axis: int | None = None) -> tuple[np.ndarray, float]:
    """Channel velocities (km/s, radio convention) and the channel width in km/s.

    The rest frequency is RESTFRQ, else the middle channel's frequency.
    """
    axis, freq = frequency_axis(header, fallback_axis=fallback_axis)
    rest = float(header.get("RESTFRQ", 0.0) or freq[len(freq) // 2] or 1.0)
    velocity = SPEED_OF_LIGHT_KM_S * (rest - freq) / rest
    width = abs(float(header.get(f"CDELT{axis}", 0.0))) / rest * SPEED_OF_LIGHT_KM_S
    return velocity, width
