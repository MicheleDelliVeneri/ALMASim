"""FITS reading and writing shared by the imaging and simulation products (MADIV-28 T2.8)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from astropy.io import fits


def write_fits_atomic(
    target: str | os.PathLike[str], hdus: Sequence[fits.hdu.base.ExtensionHDU]
) -> Path:
    """Write ``hdus`` (primary first) to ``target`` through ``<target>.partial``.

    The rename is atomic, so a reader, or a rerun after a crash, never finds a
    half-written product under its final name.
    """
    target = Path(target)
    partial = target.with_name(target.name + ".partial")
    fits.HDUList(list(hdus)).writeto(partial, overwrite=True)
    partial.replace(target)
    return target


@dataclass(frozen=True)
class ChannelBeam:
    """One channel's restoring beam, in degrees as in a FITS header."""

    bmaj_deg: float
    bmin_deg: float
    bpa_deg: float

    @classmethod
    def from_header(cls, header: fits.Header) -> ChannelBeam:
        return cls(header["BMAJ"], header["BMIN"], header["BPA"])


@dataclass
class FitsCube:
    """A FITS image or cube: the primary data and header, and the per-channel beams if any."""

    data: np.ndarray
    header: fits.Header
    beams: Optional[list[ChannelBeam]] = None


def read_image(path: str | os.PathLike[str], *, memmap: Optional[bool] = None) -> np.ndarray:
    """The image data of a FITS file: the primary HDU's, else the first extension's with data."""
    try:
        data = fits.getdata(path, memmap=memmap)
    except IndexError as exc:  # astropy: "No data in any element of the HDU list"
        raise ValueError(f"{path} does not contain image data") from exc
    if data is None:
        raise ValueError(f"{path} does not contain image data")
    return data


def read_cube(path: str | os.PathLike[str]) -> FitsCube:
    """A FITS cube with its CASA-style ``BEAMS`` table (when present) as :class:`ChannelBeam` s.

    As with ``fits.getdata``, an empty primary HDU gives way to the first
    image extension with data.
    """
    with fits.open(path, memmap=False) as hdus:
        image = next((hdu for hdu in hdus if hdu.is_image and hdu.data is not None), None)
        if image is None:
            raise ValueError(f"{path} does not contain image data")
        beams = None
        if "BEAMS" in hdus:
            table = hdus["BEAMS"].data
            beams = [
                ChannelBeam(float(bmaj) / 3600, float(bmin) / 3600, float(bpa))
                for bmaj, bmin, bpa in zip(table["BMAJ"], table["BMIN"], table["BPA"])
            ]
        return FitsCube(np.asarray(image.data), image.header.copy(), beams)


def beams_table(beams: Sequence[ChannelBeam]) -> fits.BinTableHDU:
    """A CASA-style ``BEAMS`` table: one row per channel, BMAJ/BMIN in arcsec, BPA in degrees."""
    table = fits.BinTableHDU.from_columns(
        [
            fits.Column("BMAJ", "1E", "arcsec", array=[b.bmaj_deg * 3600 for b in beams]),
            fits.Column("BMIN", "1E", "arcsec", array=[b.bmin_deg * 3600 for b in beams]),
            fits.Column("BPA", "1E", "deg", array=[b.bpa_deg for b in beams]),
            fits.Column("CHAN", "1J", array=np.arange(len(beams))),
            fits.Column("POL", "1J", array=np.zeros(len(beams), dtype=int)),
        ],
        name="BEAMS",
    )
    table.header["NCHAN"] = len(beams)
    table.header["NPOL"] = 1
    return table


def write_cube(
    path: str | os.PathLike[str],
    data: np.ndarray,
    header: Optional[fits.Header] = None,
    beams: Optional[Sequence[ChannelBeam]] = None,
) -> Path:
    """Write ``data`` with a copy of ``header``, atomically.

    With ``beams`` (one per channel, which CASA and spectral-cube read), the
    primary header gets the middle channel's beam and ``CASAMBM = T``, and
    the cube a ``BEAMS`` table.
    """
    header = header.copy() if header is not None else fits.Header()
    hdus: list[fits.hdu.base.ExtensionHDU] = []
    if beams:
        middle = beams[len(beams) // 2]
        header["BMAJ"] = middle.bmaj_deg
        header["BMIN"] = middle.bmin_deg
        header["BPA"] = middle.bpa_deg
        header["CASAMBM"] = (True, "one beam per channel in the BEAMS table")
        hdus.append(beams_table(beams))
    return write_fits_atomic(path, [fits.PrimaryHDU(data, header=header), *hdus])
