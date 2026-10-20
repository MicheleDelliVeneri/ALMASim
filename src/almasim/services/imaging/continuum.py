"""Continuum fitting and line-free channel selection, in numpy (no CASA).

Moved out of archive_imaging.py (MADIV-28 module split).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    import numpy as np


def _fit_continuum(
    spectra: np.ndarray,
    usable: np.ndarray,
    order: int,
    clip_sigma: float,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sigma-clipped polynomial continuum of ``spectra`` (channels x pixels).

    Returns ``(continuum, kept)``: the fitted continuum per channel and pixel,
    and which channels the last iteration fitted (the line-free ones).
    """
    import numpy as np

    n_channels = spectra.shape[0]
    x = np.linspace(-1.0, 1.0, n_channels)[:, None] if n_channels > 1 else np.zeros((1, 1))
    y = np.where(usable, spectra, 0.0).astype(np.float64)
    kept = usable.copy()
    continuum = np.zeros_like(y)
    for _ in range(max(int(iterations), 1)):
        w = kept.astype(np.float64)
        s0 = w.sum(axis=0)
        t0 = (w * y).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(s0 > 0, t0 / s0, 0.0)
            if order >= 1:
                s1 = (w * x).sum(axis=0)
                s2 = (w * x * x).sum(axis=0)
                t1 = (w * x * y).sum(axis=0)
                det = s0 * s2 - s1 * s1
                fit = det > 1e-12 * np.maximum(s0 * s2, 1e-300)
                slope = np.where(fit, (s0 * t1 - s1 * t0) / det, 0.0)
                intercept = np.where(fit, (t0 * s2 - t1 * s1) / det, mean)
            else:
                slope, intercept = np.zeros_like(mean), mean
        continuum = intercept[None, :] + slope[None, :] * x
        residual = np.abs(y - continuum)
        sigma = 1.4826 * np.nanmedian(np.where(kept, residual, np.nan), axis=0)
        new_kept = usable & (residual <= clip_sigma * np.nan_to_num(sigma, nan=np.inf))
        # Never clip a pixel down to fewer channels than the fit needs.
        new_kept = np.where(new_kept.sum(axis=0) > order, new_kept, kept)
        if np.array_equal(new_kept, kept):
            break
        kept = new_kept
    return continuum, kept


def subtract_continuum(
    cube_path: str | os.PathLike[str],
    out_path: str | os.PathLike[str],
    *,
    order: int = 1,
    clip_sigma: float = 3.0,
    iterations: int = 10,
    rows_per_chunk: int = 64,
) -> dict[str, float]:
    """Write ``cube_path`` minus its continuum, fitted per pixel along frequency.

    The image-plane counterpart of CASA's ``imcontsub``, with the line-free
    channels found automatically: each pixel's spectrum is fitted with a
    polynomial of ``order`` (0: constant, 1: linear in frequency) by least
    squares, channels further than ``clip_sigma`` robust sigma (1.4826 x
    MAD) from the fit are dropped as line, and the fit is repeated until no
    channel changes (at most ``iterations``). Channels that are all zero or
    NaN (WSClean writes zeros for a fully flagged channel) are left out of
    the fit and stay NaN in the output.

    Emission covering most of a window (a broad line, a line forest) leaves
    too few line-free channels for the clipping to find; such cubes need
    the line-free ranges given explicitly.

    Returns ``{"line_fraction": ...}``: the share of fitted pixel-channels
    clipped as line.
    """
    import numpy as np
    from astropy.io import fits

    if order not in (0, 1):
        raise ValueError(f"continuum order must be 0 or 1, not {order}")
    with fits.open(cube_path, memmap=True) as hdus:
        header = hdus[0].header.copy()
        extensions = [hdu.copy() for hdu in hdus[1:]]
        data = np.asarray(hdus[0].data, dtype=np.float32)
    cube = data.reshape(data.shape[-3:])  # (channels, y, x); Stokes I only
    n_channels, ny, nx = cube.shape
    planes_blank = ~np.any(np.nan_to_num(cube) != 0, axis=(1, 2))
    result = np.full_like(cube, np.nan)
    clipped = fitted = 0
    for y0 in range(0, ny, max(int(rows_per_chunk), 1)):
        block = cube[:, y0 : y0 + rows_per_chunk, :]
        spectra = block.reshape(n_channels, -1)
        usable = np.isfinite(spectra) & ~planes_blank[:, None]
        continuum, kept = _fit_continuum(spectra, usable, order, clip_sigma, iterations)
        subtracted = np.where(usable, spectra - continuum, np.nan).astype(np.float32)
        result[:, y0 : y0 + rows_per_chunk, :] = subtracted.reshape(block.shape)
        fitted += int(usable.sum())
        clipped += int((usable & ~kept).sum())
    header["HISTORY"] = (
        f"ALMASim continuum subtracted: order {order}, {clip_sigma:g} sigma clipping per pixel"
    )
    out = Path(out_path)
    partial = out.with_name(out.name + ".partial")
    fits.HDUList([fits.PrimaryHDU(result.reshape(data.shape), header=header), *extensions]).writeto(
        partial, overwrite=True
    )
    partial.replace(out)
    return {"line_fraction": clipped / fitted if fitted else 0.0}


def line_free_channels(
    spectrum: np.ndarray,
    usable: np.ndarray,
    *,
    order: int = 1,
    clip_sigma: float = 3.0,
    widths: Sequence[int] = (1, 4, 16, 64),
    grow: int | None = None,
    iterations: int = 10,
) -> np.ndarray:
    """Line-free channels of a source spectrum: the ones a continuum fit keeps.

    A polynomial of ``order`` is fitted to the kept channels and its
    residual smoothed with a boxcar of each of ``widths`` channels; a channel
    is line wherever any smoothed residual exceeds ``clip_sigma`` times that
    scale's robust noise (1.4826 x MAD over the kept channels). The fit is
    repeated on the channels left until they stop changing. Each line is
    then widened by ``grow`` channels on both sides (default 1 % of the
    channels, at least 2) so its faint wings stay out of the fit.

    One-channel clipping alone missed the wings of the broad line of
    uid___A002_X10ded83_X6af4 SPW 16: ~0.3 sigma per channel over ~100
    channels, 2.5 sigma once smoothed over them (2026-10-17).
    """
    import numpy as np

    spectrum = np.asarray(spectrum, dtype=np.float64)
    usable = np.asarray(usable, dtype=bool) & np.isfinite(spectrum)
    n = spectrum.size
    x = np.linspace(-1.0, 1.0, n) if n > 1 else np.zeros(1)
    kept = usable.copy()
    scales = [int(k) for k in widths if 1 <= int(k) <= max(n // 8, 1)] or [1]
    # A noiseless (simulated) spectrum has zero spread: then anything above
    # rounding is a line, or the fit would absorb it.
    floor = 1e-9 * float(np.max(np.abs(spectrum[usable]))) if usable.any() else 0.0
    for _ in range(max(int(iterations), 1)):
        if kept.sum() <= order:
            break
        coef = np.polyfit(x[kept], spectrum[kept], order)
        residual = np.where(usable, spectrum - np.polyval(coef, x), 0.0)
        line = np.zeros(n, dtype=bool)
        for k in scales:
            box = np.ones(k)
            counts = np.convolve(usable.astype(float), box, mode="same")
            smooth = np.convolve(residual, box, mode="same") / np.maximum(counts, 1.0)
            # Deviations from the median, not from zero: the first fit still
            # includes the lines, which lifts it and offsets every residual.
            centre = np.median(smooth[kept])
            spread = 1.4826 * np.median(np.abs(smooth[kept] - centre))
            if spread > 0:
                line |= usable & (np.abs(smooth - centre) > clip_sigma * spread)
            elif floor > 0:
                line |= usable & (np.abs(smooth - centre) > floor)
        new_kept = usable & ~line
        if new_kept.sum() <= order or np.array_equal(new_kept, kept):
            break
        kept = new_kept
    line = usable & ~kept
    if grow is None:
        grow = max(2, n // 100)
    if grow > 0 and line.any():
        # "full" and sliced: "same" returns the longer input, which a short
        # spectrum (fewer than 2 * grow + 1 channels) is not.
        line = np.convolve(line, np.ones(2 * grow + 1), mode="full")[grow : grow + n] > 0
    return usable & ~line


def fit_uv_continuum(
    data: np.ndarray,
    weight: np.ndarray,
    flag: np.ndarray,
    line_free: np.ndarray,
    order: int = 1,
) -> np.ndarray:
    """Continuum of each visibility row, fitted over the line-free channels.

    ``data``, ``weight`` and ``flag`` are ``(rows, channels, correlations)``;
    every row and correlation gets a weighted least-squares polynomial of
    ``order`` (0 or 1) in channel, from its unflagged ``line_free`` channels,
    evaluated at every channel: CASA ``uvcontsub``. A row with no usable
    line-free channel gets zero continuum; one whose channels cannot
    constrain a slope gets a constant.
    """
    import numpy as np

    if order not in (0, 1):
        raise ValueError(f"continuum order must be 0 or 1, not {order}")
    n_channels = data.shape[1]
    x = np.linspace(-1.0, 1.0, n_channels)[None, :, None] if n_channels > 1 else np.zeros((1, 1, 1))
    w = np.where(flag | ~np.asarray(line_free, dtype=bool)[None, :, None], 0.0, weight)
    w = np.where(np.isfinite(data), w, 0.0)
    y = np.where(w > 0, data, 0.0)
    s0 = w.sum(axis=1)
    t0 = (w * y).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(s0 > 0, t0 / np.where(s0 > 0, s0, 1.0), 0.0)
        slope = np.zeros_like(mean)
        intercept = mean
        if order == 1:
            s1 = (w * x).sum(axis=1)
            s2 = (w * x * x).sum(axis=1)
            t1 = (w * x * y).sum(axis=1)
            det = s0 * s2 - s1 * s1
            fit = det > 1e-9 * np.maximum(s0 * s2, 1e-300)
            safe = np.where(fit, det, 1.0)
            slope = np.where(fit, (s0 * t1 - s1 * t0) / safe, 0.0)
            intercept = np.where(fit, (t0 * s2 - t1 * s1) / safe, mean)
    return (intercept[:, None, :] + slope[:, None, :] * x).astype(data.dtype)


def _channel_ranges(mask: np.ndarray) -> str:
    """``0~10;20~30``: the runs of True in ``mask``, CASA channel-selection style."""
    import numpy as np

    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(int), [0]))))
    return ";".join(f"{a}~{b - 1}" for a, b in zip(edges[::2], edges[1::2]))


def _line_free_from_ranges(ranges: str, n_channels: int) -> np.ndarray:
    """Line-free channels from ``_channel_ranges`` text ("" when there are no lines)."""
    import numpy as np

    line_free = np.ones(n_channels, dtype=bool)
    for part in filter(None, ranges.split(";")):
        start, end = map(int, part.split("~"))
        line_free[start : end + 1] = False
    return line_free
