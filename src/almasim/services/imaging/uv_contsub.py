"""Subtract the continuum in the uv-plane, in place in the MeasurementSet.

Moved out of archive_imaging.py (MADIV-28 module split).
"""

from __future__ import annotations

import os
import time

from almasim.services.imaging.continuum import _channel_ranges, fit_uv_continuum, line_free_channels
from almasim.services.imaging.ms.chunks import (
    _default_rows_per_chunk,
    _has_spectrum_weight,
    _read_rows,
    _source_spectrum,
)
from almasim.services.imaging.products import CONTINUUM_COLUMN, LINE_COLUMN
from almasim.services.runtime.child import call_in_child
from almasim.services.runtime.env import wsclean_environment


def subtract_uv_continuum(
    ms_path: str | os.PathLike[str],
    *,
    order: int = 1,
    clip_sigma: float = 3.0,
    rows_per_chunk: int | None = None,
    min_line_free_fraction: float = 0.2,
) -> dict[str, object]:
    """Split a single-window MS into continuum and line visibilities.

    The line channels are found on the source spectrum: the visibilities of
    every row averaged (vector, weighted, XX and YY) per channel, its
    amplitude fitted by ``line_free_channels``. Each row is then fitted by
    ``fit_uv_continuum`` over the line-free channels, and the fit and the
    data minus the fit are written to ``CONTINUUM_COLUMN`` and
    ``LINE_COLUMN``. The data are CORRECTED_DATA when the MS has it, DATA
    otherwise (WSClean's own choice).

    Channel-by-channel cleaning stops at the threshold in each channel and
    so leaves a faint source's continuum in the residual (uid___A002_
    X10ded83_X6af4 SPW 16: S/N 8 per channel, 12 % of the line-free image
    left); imaged from the fitted continuum, all line-free channels are
    cleaned together, and the cube holds only the lines.

    Nothing is written when fewer than ``min_line_free_fraction`` of the
    usable channels are line-free (a line forest or a window filled by one
    line); the result then carries ``skipped`` with the reason.

    Rows are read ``rows_per_chunk`` at a time, by default as many as keep a
    chunk near ``_CHUNK_SAMPLES`` samples; rows flagged with FLAG_ROW are
    left out like flagged samples.
    """
    from casacore.tables import makecoldesc, table

    started = time.time()
    ms = table(str(ms_path), readonly=False, ack=False)
    try:
        source = "CORRECTED_DATA" if "CORRECTED_DATA" in ms.colnames() else "DATA"
        n_rows = ms.nrows()
        spectrum_weight = _has_spectrum_weight(ms)
        if rows_per_chunk is None:
            rows_per_chunk = _default_rows_per_chunk(ms, source)
        measured = _source_spectrum(ms, source, rows_per_chunk)
        if measured is None:
            return {"skipped": "the MS has no rows", "source_column": source}
        spectrum, usable = measured
        line_free = line_free_channels(spectrum, usable, order=order, clip_sigma=clip_sigma)
        result: dict[str, object] = {
            "source_column": source,
            "channels": int(usable.size),
            "usable_channels": int(usable.sum()),
            "line_free_channels": int(line_free.sum()),
            "line_channels": int((usable & ~line_free).sum()),
            "line_ranges": _channel_ranges(usable & ~line_free),
            "order": int(order),
        }
        if line_free.sum() < max(order + 2, min_line_free_fraction * usable.sum()):
            result["skipped"] = (
                f"only {int(line_free.sum())} of {int(usable.sum())} usable channels are line-free"
            )
            return result
        for name in (CONTINUUM_COLUMN, LINE_COLUMN):
            if name in ms.colnames():
                ms.removecols(name)
            ms.addcols(makecoldesc(name, ms.getcoldesc(source)))
        for start in range(0, n_rows, rows_per_chunk):
            n = min(rows_per_chunk, n_rows - start)
            data, weight, flag = _read_rows(ms, source, start, n, spectrum_weight)
            continuum = fit_uv_continuum(data, weight, flag, line_free, order)
            ms.putcol(CONTINUUM_COLUMN, continuum, startrow=start, nrow=n)
            ms.putcol(LINE_COLUMN, data - continuum, startrow=start, nrow=n)
        result["seconds"] = round(time.time() - started, 1)
        return result
    finally:
        ms.close()


def subtract_uv_continuum_subprocess(
    ms_path: str | os.PathLike[str],
    *,
    order: int = 1,
    timeout: float | None = None,
    log_path: str | os.PathLike[str] | None = None,
) -> dict[str, object]:
    """Run ``subtract_uv_continuum`` in a child process, for the reason
    ``extract_single_window_ms_subprocess`` gives: casacore holds the GIL."""
    return call_in_child(
        "almasim.services.imaging.archive_imaging:subtract_uv_continuum",
        {"ms_path": str(ms_path), "order": int(order)},
        what="continuum subtraction",
        env=wsclean_environment(1),
        timeout=timeout,
        timeout_noun="task timeout",
        log_path=log_path,
    )


def drop_corrected_data(ms_path: str) -> dict[str, object]:
    """Remove CORRECTED_DATA, so imaging reads DATA again (an undone self-calibration)."""
    from casacore.tables import table

    ms = table(str(ms_path), readonly=False, ack=False)
    try:
        dropped = "CORRECTED_DATA" in ms.colnames()
        if dropped:
            ms.removecols("CORRECTED_DATA")
        return {"dropped": dropped}
    finally:
        ms.close()
