"""ALMA primary-beam model and the primary-beam correction of imaged tasks.

WSClean's images are not primary-beam corrected: a source away from the
pointing centre looks fainter by the antennas' response there. The model is
CASA's for ALMA (its ``airy`` voltage pattern): an Airy disk of a 10.7 m
aperture with a 0.75 m central blockage for the 12 m antennas, 6.25 m and
0.75 m for the 7 m ones. Any other diameter is an unblocked Airy disk of
that diameter. The power response at angle ``theta`` and wavelength
``lambda``, with ``x = pi D theta / lambda`` and ``e`` the blockage over the
aperture, is::

    [(jinc(x) - e^2 jinc(e x)) / (1 - e^2)]^2,   jinc(z) = 2 J1(z) / z

Its half-power width is 1.15 lambda / D_dish (12 m) and 1.14 (7 m), close
to the 1.13 ALMA quotes for the measured beams.

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

SPEED_OF_LIGHT_M_S = 299_792_458.0

# Nominal dish diameter -> (effective aperture, central blockage) in metres.
ALMA_AIRY_APERTURES = {12.0: (10.7, 0.75), 7.0: (6.25, 0.75)}
PB_CUTOFF = 0.2

# The response is tabulated over x = pi D theta / lambda and interpolated:
# a cube needs the beam of every channel on the whole grid.
_TABLE_STEP = 1e-3


def airy_aperture(dish_diameter: float) -> tuple[float, float]:
    """(effective aperture, blockage) in metres for a nominal dish diameter."""
    for nominal, aperture in ALMA_AIRY_APERTURES.items():
        if abs(dish_diameter - nominal) < 0.5:
            return aperture
    return float(dish_diameter), 0.0


def _jinc(z: np.ndarray) -> np.ndarray:
    from scipy.special import j1

    z = np.asarray(z, dtype=float)
    safe = np.where(z == 0, 1.0, z)
    return np.where(z == 0, 1.0, 2.0 * j1(safe) / safe)


def primary_beam_response(
    radius_rad: np.ndarray, frequency_hz: float, dish_diameter: float
) -> np.ndarray:
    """Power response (1 on axis) at ``radius_rad`` from the pointing centre."""
    aperture, blockage = airy_aperture(dish_diameter)
    x = math.pi * aperture * np.asarray(radius_rad, dtype=float) * frequency_hz / SPEED_OF_LIGHT_M_S
    e = blockage / aperture
    amplitude = (_jinc(x) - e**2 * _jinc(e * x)) / (1.0 - e**2)
    return amplitude**2


def _radius_grid(header) -> np.ndarray:
    """Angular distance (rad) of every pixel from the reference pixel (SIN/TAN)."""
    nx, ny = int(header["NAXIS1"]), int(header["NAXIS2"])
    dx = (np.arange(nx) + 1.0 - float(header["CRPIX1"])) * math.radians(float(header["CDELT1"]))
    dy = (np.arange(ny) + 1.0 - float(header["CRPIX2"])) * math.radians(float(header["CDELT2"]))
    rho = np.hypot(dx[None, :], dy[:, None])
    return np.arcsin(np.clip(rho, 0.0, 1.0))


def _channel_frequencies(header) -> tuple[int, np.ndarray]:
    """(numpy axis, frequency per plane) of the image's FREQ axis."""
    naxis = int(header["NAXIS"])
    for fits_axis in range(3, naxis + 1):
        if str(header.get(f"CTYPE{fits_axis}", "")).upper() == "FREQ":
            n = int(header[f"NAXIS{fits_axis}"])
            pixels = np.arange(n) + 1.0 - float(header.get(f"CRPIX{fits_axis}", 1.0))
            frequencies = float(header[f"CRVAL{fits_axis}"]) + pixels * float(
                header.get(f"CDELT{fits_axis}", 0.0)
            )
            return naxis - fits_axis, frequencies
    raise ValueError("the image has no FREQ axis to take the beam's frequency from")


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


def _half_power_radius(dish_diameter: float, frequency_hz: float) -> float:
    """Radius (rad) where the response falls to 0.5, by bisection."""
    aperture, _ = airy_aperture(dish_diameter)
    low, high = 0.0, 3.0 * SPEED_OF_LIGHT_M_S / frequency_hz / aperture
    for _ in range(60):
        middle = 0.5 * (low + high)
        if primary_beam_response(np.array(middle), frequency_hz, dish_diameter) > 0.5:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def dish_diameter_from_fov(
    fov_arcsec: float, reference_frequency_hz: float, fov_factor: float
) -> float:
    """Dish diameter (m) behind a ``fov_per_frequency`` of ``fov_factor`` lambda / D."""
    if fov_arcsec <= 0 or reference_frequency_hz <= 0:
        raise ValueError("the field of view and the frequency must be positive")
    return (
        fov_factor * SPEED_OF_LIGHT_M_S / reference_frequency_hz / math.radians(fov_arcsec / 3600.0)
    )
