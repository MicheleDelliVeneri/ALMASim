"""Correction and total-residual columns and images for self-calibrated tasks.

Moved out of archive_imaging.py (MADIV-28 module split).
"""

from __future__ import annotations

import shutil
from pathlib import Path

from almasim.services.imaging.continuum import _line_free_from_ranges, fit_uv_continuum
from almasim.services.imaging.ms.chunks import (
    _default_rows_per_chunk,
    _has_spectrum_weight,
    _read_rows,
)
from almasim.services.imaging.products import (
    SELFCAL_COLUMN,
    SELFCAL_CONTINUUM_COLUMN,
    SELFCAL_LINE_COLUMN,
)


def selfcal_correction_columns(
    ms_path: str,
    *,
    line_ranges: str | None = None,
    order: int = 1,
    rows_per_chunk: int | None = None,
) -> dict[str, object]:
    """Write what self-calibration removed, DATA - CORRECTED_DATA, to new columns.

    Without ``line_ranges`` it goes whole to ``SELFCAL_COLUMN``. With them (the
    visibility-plane continuum subtraction's line channels) it is split the
    way the data were: the per-row continuum fit over the same line-free
    channels and order to ``SELFCAL_CONTINUUM_COLUMN``, the rest to
    ``SELFCAL_LINE_COLUMN``. The fit is linear in the data, so each part of
    the correction plus the matching part of the self-calibrated data is that
    part of DATA.
    """
    from casacore.tables import makecoldesc, table

    ms = table(str(ms_path), readonly=False, ack=False)
    try:
        if "CORRECTED_DATA" not in ms.colnames():
            raise RuntimeError("the MS has no CORRECTED_DATA: nothing was self-calibrated")
        n_rows = ms.nrows()
        if rows_per_chunk is None:
            rows_per_chunk = _default_rows_per_chunk(ms, "DATA")
        split = line_ranges is not None
        names = [SELFCAL_CONTINUUM_COLUMN, SELFCAL_LINE_COLUMN] if split else [SELFCAL_COLUMN]
        for name in names:
            if name in ms.colnames():
                ms.removecols(name)
            ms.addcols(makecoldesc(name, ms.getcoldesc("DATA")))
        spectrum_weight = _has_spectrum_weight(ms)
        line_free = None
        for start in range(0, n_rows, rows_per_chunk):
            n = min(rows_per_chunk, n_rows - start)
            data, weight, flag = _read_rows(ms, "DATA", start, n, spectrum_weight)
            correction = (data - ms.getcol("CORRECTED_DATA", startrow=start, nrow=n)).astype(
                data.dtype
            )
            if not split:
                ms.putcol(SELFCAL_COLUMN, correction, startrow=start, nrow=n)
                continue
            if line_free is None:
                line_free = _line_free_from_ranges(str(line_ranges), data.shape[1])
            continuum = fit_uv_continuum(correction, weight, flag, line_free, order)
            ms.putcol(SELFCAL_CONTINUUM_COLUMN, continuum, startrow=start, nrow=n)
            ms.putcol(
                SELFCAL_LINE_COLUMN,
                (correction - continuum).astype(data.dtype),
                startrow=start,
                nrow=n,
            )
        return {"columns": names, "rows": int(n_rows)}
    finally:
        ms.close()


def add_images(first: Path, second: Path, target: Path, history: str) -> None:
    """Write ``first + second`` to ``target`` plane by plane, with ``first``'s
    header and extensions (a cube's BEAMS table included)."""
    import numpy as np
    from astropy.io import fits

    partial = target.with_name(target.name + ".partial")
    shutil.copyfile(first, partial)
    with (
        fits.open(partial, mode="update", memmap=True) as hdus,
        fits.open(second, memmap=True) as others,
    ):
        data, other = hdus[0].data, others[0].data
        if data.shape != other.shape:
            raise RuntimeError(f"{first.name} {data.shape} and {second.name} {other.shape} differ")
        planes = data.reshape(-1, *data.shape[-2:])
        other_planes = other.reshape(-1, *other.shape[-2:])
        for index in range(planes.shape[0]):
            planes[index] = np.asarray(planes[index], dtype=np.float64) + other_planes[index]
        hdus[0].header["HISTORY"] = history
    partial.replace(target)


def _single_image(out: Path, name: str) -> Path | None:
    """The image a ``-niter 0`` WSClean run wrote: ``-dirty`` or ``-image``."""
    return next(
        (
            path
            for path in (out / f"{name}-dirty.fits", out / f"{name}-image.fits")
            if path.is_file()
        ),
        None,
    )
