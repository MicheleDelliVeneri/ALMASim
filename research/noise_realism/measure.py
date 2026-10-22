"""Noise and model statistics of one imaged window (real or simulated).

    python measure.py TASK_DIR [OUT.json]

Besides the pipeline's own quality.json (S/N, residual shares, residual peak
in sigma, channel noise, model flux), it measures the residuals' noise:
level, Gaussianity (skew, excess kurtosis, 5-sigma tails), spatial
correlation (area of the autocorrelation above half its peak, in beams),
radial trend, and in the cube the spread of the channel noise and the
correlation of adjacent channels. Statistics use the inner part of the image
(radius below 40% of its width), outside the cube's line mask.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

MAD = 1.4826


def _plane_stack(path: Path) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path, memmap=True) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float32)
        header = hdul[0].header
    while data.ndim > 3:
        data = data[0]
    return data.reshape((-1,) + data.shape[-2:]), header


def _robust_sigma(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(MAD * np.median(np.abs(x - np.median(x)))) if x.size else float("nan")


def _moments(x: np.ndarray, sigma: float) -> dict:
    x = x[np.isfinite(x)]
    if not x.size or not sigma > 0:
        return {"skew": None, "kurtosis": None, "tail5": None}
    z = (x - np.median(x)) / sigma
    clipped = z[np.abs(z) < 10]  # the moments of the noise, not of a few bright residuals
    return {
        "skew": float(np.mean(clipped**3)),
        "kurtosis": float(np.mean(clipped**4) - 3),
        "tail5": float(np.mean(np.abs(z) > 5)),
    }


def _corr_area_beams(plane: np.ndarray, beam_pix: float) -> float | None:
    """Area of the residual's autocorrelation above half its peak, in beam areas."""
    p = np.nan_to_num(plane - np.nanmedian(plane))
    if not np.any(p) or beam_pix <= 0:
        return None
    f = np.fft.rfft2(p)
    acf = np.fft.irfft2(np.abs(f) ** 2, s=p.shape)
    acf = np.fft.fftshift(acf) / acf.flat[0]
    return float(np.sum(acf > 0.5) / beam_pix)


def _beam_pix(header: fits.Header) -> float:
    cell = abs(header.get("CDELT2", 0.0))
    bmaj, bmin = header.get("BMAJ"), header.get("BMIN")
    if not (cell and bmaj and bmin):
        return 0.0
    return float(np.pi * bmaj * bmin / (4 * np.log(2)) / cell**2)


def _inner(shape: tuple[int, int], fraction: float = 0.4) -> np.ndarray:
    ny, nx = shape
    y, x = np.mgrid[:ny, :nx]
    r = np.hypot(x - nx / 2, y - ny / 2) / max(nx, ny)
    return r < fraction


def continuum_stats(task: Path) -> dict:
    residual, header = _plane_stack(task / "wsclean-residual.fits")
    plane = residual[0]
    inner = _inner(plane.shape)
    sigma = _robust_sigma(plane[inner])
    ny, nx = plane.shape
    y, x = np.mgrid[:ny, :nx]
    r = np.hypot(x - nx / 2, y - ny / 2) / max(nx, ny)
    rings = [_robust_sigma(plane[(r >= a) & (r < b)]) for a, b in ((0, 0.15), (0.15, 0.3), (0.3, 0.45))]
    cy, cx = ny // 2, nx // 2
    half = max(16, min(ny, nx) // 4)
    image, _ = _plane_stack(task / "wsclean-image.fits")
    peak = np.unravel_index(np.nanargmax(image[0]), image[0].shape)
    return {
        "sigma": sigma,
        **_moments(plane[inner], sigma),
        "corr_area_beams": _corr_area_beams(plane[cy - half : cy + half, cx - half : cx + half], _beam_pix(header)),
        "radial_outer_inner": rings[2] / rings[0] if rings[0] else None,
        "peak_pixel": [int(peak[0]), int(peak[1])],
        "peak": float(np.nanmax(image[0])),
    }


def cube_stats(task: Path) -> dict:
    residual, header = _plane_stack(task / "wsclean-cube-residual.fits")
    nchan, ny, nx = residual.shape
    inner = _inner((ny, nx))
    mask_path = task / "wsclean-cube-linemask.fits"
    keep = np.broadcast_to(inner, residual.shape).copy()
    if mask_path.exists():
        mask, _ = _plane_stack(mask_path)
        if mask.shape[-2:] == (ny, nx):
            keep &= ~(np.nan_to_num(mask[0] if len(mask) == 1 else mask) > 0)
    finite = np.array([np.isfinite(residual[k]).any() and np.nanstd(residual[k]) > 0 for k in range(nchan)])
    usable = np.flatnonzero(finite)
    sigmas = np.array([_robust_sigma(residual[k][keep[k]]) for k in usable])
    adj = []
    for k in usable[:-1]:
        if k + 1 in usable:
            a, b = residual[k][keep[k] & keep[k + 1]], residual[k + 1][keep[k] & keep[k + 1]]
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() > 100:
                adj.append(float(np.corrcoef(a[ok], b[ok])[0, 1]))
    sample = usable[:: max(1, len(usable) // 8)]
    beam = _beam_pix(header)
    half = max(16, min(ny, nx) // 4)
    cy, cx = ny // 2, nx // 2
    areas = [
        _corr_area_beams(residual[k][cy - half : cy + half, cx - half : cx + half], beam) for k in sample
    ]
    pooled = np.concatenate([(residual[k][keep[k]] - np.nanmedian(residual[k][keep[k]])) / s for k, s in zip(usable, sigmas) if s > 0])
    return {
        "channels": int(nchan),
        "sigma_median": float(np.median(sigmas)) if len(sigmas) else None,
        "sigma_spread": float(np.subtract(*np.percentile(sigmas, [84, 16])) / 2 / np.median(sigmas)) if len(sigmas) > 2 else None,
        "adjacent_corr": float(np.median(adj)) if adj else None,
        "corr_area_beams": float(np.median([a for a in areas if a is not None])) if any(a is not None for a in areas) else None,
        **_moments(pooled, 1.0),
    }


def _flatten(prefix: str, value, out: dict) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            _flatten(f"{prefix}.{k}" if prefix else k, v, out)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        out[prefix] = float(value)


def measure(task: Path) -> dict:
    out: dict = {"task": str(task)}
    quality = task / "quality.json"
    if quality.exists():
        q = json.loads(quality.read_text())
        _flatten("q", {"continuum": q.get("continuum", {}), "cube": q.get("cube", {})}, out)
    if (task / "wsclean-residual.fits").exists():
        stats = continuum_stats(task)
        out["cont.peak_y"], out["cont.peak_x"] = stats.pop("peak_pixel")
        _flatten("cont", stats, out)
    if (task / "wsclean-cube-residual.fits").exists():
        _flatten("cube", cube_stats(task), out)
    return out


if __name__ == "__main__":
    task = Path(sys.argv[1])
    result = measure(task)
    text = json.dumps(result, indent=1)
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(text)
    print(text)
