"""Chunked row reads from a MeasurementSet with python-casacore.

Moved out of archive_imaging.py (MADIV-28 module split).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


# Samples (rows x channels x correlations) per chunk read from an MS: with
# complex64 data, float64 weights, flags and the fit's temporaries about
# 1 GB. A fixed 20000 rows was over 5 GB for a 3840-channel window.
_CHUNK_SAMPLES = 20_000_000


def rows_for_chunk(n_channels: int, n_correlations: int, samples: int | None = None) -> int:
    """Rows per chunk keeping it near ``samples`` (default ``_CHUNK_SAMPLES``) samples, >= 1."""
    budget = _CHUNK_SAMPLES if samples is None else samples
    return max(int(budget // max(int(n_channels) * int(n_correlations), 1)), 1)


def _has_spectrum_weight(ms) -> bool:
    return (
        "WEIGHT_SPECTRUM" in ms.colnames()
        and ms.nrows() > 0
        and (ms.iscelldefined("WEIGHT_SPECTRUM", 0))
    )


def _read_rows(
    ms, column: str, start: int, n: int, spectrum_weight: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``column``, its weights (per channel) and flags for ``n`` rows from ``start``.

    The flags combine FLAG with FLAG_ROW: a row flagged as a whole counts as
    flagged in every sample.
    """
    import numpy as np

    data = ms.getcol(column, startrow=start, nrow=n)
    flag = ms.getcol("FLAG", startrow=start, nrow=n)
    if "FLAG_ROW" in ms.colnames():
        flag = flag | ms.getcol("FLAG_ROW", startrow=start, nrow=n)[:, None, None]
    if spectrum_weight:
        weight = ms.getcol("WEIGHT_SPECTRUM", startrow=start, nrow=n)
    else:
        weight = np.broadcast_to(
            ms.getcol("WEIGHT", startrow=start, nrow=n)[:, None, :], data.shape
        )
    return data, np.asarray(weight, dtype=np.float64), flag


def _default_rows_per_chunk(ms, column: str) -> int:
    """``rows_for_chunk`` for the shape of ``column``'s cells (one row for an empty MS)."""
    shape = ms.getcell(column, 0).shape if ms.nrows() else (1, 1)
    return rows_for_chunk(shape[0], shape[-1])


def _parallel_hands(n_corr: int) -> list[int]:
    """XX and YY (or RR and LL): the first and last correlation."""
    return [0, n_corr - 1] if n_corr > 1 else [0]


def _source_spectrum(ms, column: str, rows_per_chunk: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Amplitude of the weighted vector average of every row, per channel, and
    which channels have any unflagged data; None for an empty MS."""
    import numpy as np

    n_rows = ms.nrows()
    spectrum_weight = _has_spectrum_weight(ms)
    num = den = None
    for start in range(0, n_rows, rows_per_chunk):
        data, weight, flag = _read_rows(
            ms, column, start, min(rows_per_chunk, n_rows - start), spectrum_weight
        )
        corrs = _parallel_hands(data.shape[2])
        w = np.where(flag[..., corrs] | ~np.isfinite(data[..., corrs]), 0.0, weight[..., corrs])
        d = np.where(w > 0, data[..., corrs], 0.0)
        if num is None:
            num = np.zeros(data.shape[1], dtype=np.complex128)
            den = np.zeros(data.shape[1])
        num += (w * d).sum(axis=(0, 2))
        den += w.sum(axis=(0, 2))
    if num is None:
        return None
    usable = den > 0
    return np.abs(np.where(usable, num / np.where(usable, den, 1.0), 0.0)), usable
