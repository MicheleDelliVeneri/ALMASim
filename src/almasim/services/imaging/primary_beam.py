"""The primary-beam correction of imaged tasks.

WSClean's images are not primary-beam corrected: a source away from the
pointing centre looks fainter by the antennas' response there. The response
is ALMASim's one beam model, :mod:`almasim.services.instrument.primary_beam`.

The beam is centred on the image's reference pixel: WSClean images every
field around its phase centre, which for an ALMA single pointing is the
pointing centre. Each channel of a cube gets the beam at its own frequency.
As in CASA, the corrected image is blanked (NaN) where the response is below
``PB_CUTOFF``, where noise divided by a small response would dominate.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from ..instrument.constants import SPEED_OF_LIGHT_M_S
from ..instrument.primary_beam import (  # noqa: F401  (re-exported)
    ALMA_AIRY_APERTURES,
    LEGACY_FOV_FACTOR,
    _half_power_radius,
    airy_aperture,
    airy_fwhm_factor,
    dish_diameter_from_fov,
    primary_beam_fwhm_rad,
    primary_beam_response,
)

PB_CUTOFF = 0.2

# The response is tabulated over x = pi D theta / lambda and interpolated:
# a cube needs the beam of every channel on the whole grid.
_TABLE_STEP = 1e-3


def _radius_grid(header) -> np.ndarray:
    """Angular distance (rad) of every pixel from the reference pixel (SIN/TAN)."""
    nx, ny = int(header["NAXIS1"]), int(header["NAXIS2"])
    dx = (np.arange(nx) + 1.0 - float(header["CRPIX1"])) * math.radians(float(header["CDELT1"]))
    dy = (np.arange(ny) + 1.0 - float(header["CRPIX2"])) * math.radians(float(header["CDELT2"]))
    rho = np.hypot(dx[None, :], dy[:, None])
    return np.arcsin(np.clip(rho, 0.0, 1.0))


def _channel_frequencies(header) -> tuple[int, np.ndarray]:
    """(numpy axis, frequency per plane) of the image's FREQ axis."""
    from .fits_wcs import frequency_axis

    try:
        fits_axis, frequencies = frequency_axis(header)
    except ValueError:
        raise ValueError("the image has no FREQ axis to take the beam's frequency from") from None
    return int(header["NAXIS"]) - fits_axis, frequencies


def primary_beam_correct(
    image_path: str | os.PathLike[str],
    pb_path: str | os.PathLike[str],
    pbcor_path: str | os.PathLike[str],
    dish_diameter: float,
    cutoff: float = PB_CUTOFF,
) -> dict[str, Any]:
    """Write the primary beam of ``image_path`` and the image divided by it.

    ``pb_path`` holds the response (1 at the pointing centre) on the image's
    grid, one plane per channel; ``pbcor_path`` the corrected image, NaN
    where the response is below ``cutoff``. Both keep the image's header and
    extensions (a cube's ``BEAMS`` table) and are written through a
    ``.partial`` file, so neither is ever left half-written.
    """
    from astropy.io import fits

    from .fits_io import write_fits_atomic

    if not 0.0 < cutoff < 1.0:
        raise ValueError(f"the primary-beam cutoff must be in (0, 1), not {cutoff}")
    aperture, blockage = airy_aperture(dish_diameter)
    with fits.open(image_path, memmap=True) as hdul:
        header = hdul[0].header.copy()
        data = hdul[0].data
        if data is None or data.ndim < 2:
            raise ValueError(f"{image_path} holds no image")
        extensions = [hdu.copy() for hdu in hdul[1:]]
        freq_axis, frequencies = _channel_frequencies(header)
        radius = _radius_grid(header)
        x_max = math.pi * aperture * float(radius.max()) * frequencies.max() / SPEED_OF_LIGHT_M_S
        table_x = np.arange(0.0, x_max + 2 * _TABLE_STEP, _TABLE_STEP)
        table = primary_beam_response(
            table_x * SPEED_OF_LIGHT_M_S / (math.pi * aperture), 1.0, dish_diameter
        )
        pb = np.empty(data.shape, dtype=np.float32)
        pbcor = np.empty(data.shape, dtype=np.float32)
        for index, frequency in enumerate(frequencies):
            plane: list[slice | int] = [slice(None)] * data.ndim
            plane[freq_axis] = index
            x = math.pi * aperture * radius * frequency / SPEED_OF_LIGHT_M_S
            response = np.interp(x, table_x, table).astype(np.float32)
            response = np.broadcast_to(response, pb[tuple(plane)].shape)
            pb[tuple(plane)] = response
            with np.errstate(divide="ignore", invalid="ignore"):
                pbcor[tuple(plane)] = np.where(
                    response >= cutoff, data[tuple(plane)] / response, np.nan
                )
    model = f"Airy {aperture:g} m aperture, {blockage:g} m blockage"
    for target, values, unit, note in (
        (Path(pb_path), pb, "", "primary-beam response"),
        (Path(pbcor_path), pbcor, header.get("BUNIT", ""), "primary-beam corrected"),
    ):
        out_header = header.copy()
        out_header["BUNIT"] = unit
        out_header["PBMODEL"] = (model, "primary-beam model")
        out_header["PBDISH"] = (float(dish_diameter), "nominal dish diameter (m)")
        out_header["PBCUTOFF"] = (float(cutoff), "corrected image blanked below this response")
        out_header.add_history(f"ALMASim: {note} ({model}) of {Path(image_path).name}")
        write_fits_atomic(target, [fits.PrimaryHDU(values, header=out_header), *extensions])
    centre = frequencies[len(frequencies) // 2]
    fwhm_arcsec = math.degrees(2.0 * _half_power_radius(dish_diameter, centre)) * 3600.0
    return {
        "dish_diameter": float(dish_diameter),
        "model": model,
        "channels": int(len(frequencies)),
        "fwhm_arcsec": round(fwhm_arcsec, 3),
        "cutoff": float(cutoff),
        "blanked_fraction": round(float(np.mean(~np.isfinite(pbcor))), 4),
    }
