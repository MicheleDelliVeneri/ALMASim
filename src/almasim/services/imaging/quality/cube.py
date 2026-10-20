"""Cube cleaning quality: line and line-free residuals, and moment maps (MADIV-28 T4.10)."""

from __future__ import annotations

import math
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from ..continuum import line_free_channels
from ..fits_io import write_fits_atomic
from ..fits_wcs import beam_pixels, velocity_axis
from ..products import LINE_MASK_FILENAME, cube_product
from ..stats import moment_mask, robust_sigma
from .continuum import _plane, _ratio, _source_geometry

QUALITY_DIRNAME = "quality"
_LINE_RANGES = re.compile(
    r"(?:(\d+) of (\d+) channels line-free(?: \((\d+) in the window\))?, )?"
    r"lines in channels ([\d~;]+|none)"
)


def _logged_line_channels(task_dir: Path, n_channels: int) -> np.ndarray | None:
    """Line channels a visibility-plane continuum subtraction logged, on the cube's channels."""
    log = task_dir.parent / f"{task_dir.name}.log"
    if not log.is_file():
        return None
    matches = _LINE_RANGES.findall(log.read_text(errors="replace"))
    if not matches:
        return None
    _, usable, window, ranges = matches[-1]
    line = np.zeros(n_channels, dtype=bool)
    if ranges == "none":
        return line
    native = [tuple(map(int, part.split("~"))) for part in ranges.split(";")]
    # Cube channels average a whole number of native ones (cube_channels picks
    # divisors). The window's channel count gives that factor exactly; an
    # older log gives only the usable count, which flagged channels make
    # smaller, so it is trusted only when the cube has at least that many
    # channels (no averaging); otherwise the lines are found on the spectrum.
    if window:
        if int(window) % n_channels:
            return None
        factor = int(window) // n_channels
    elif n_channels >= max(int(usable or 0), max(end for _, end in native) + 1):
        factor = 1
    else:
        return None
    for start, end in native:
        line[start // factor : end // factor + 1] = True
    return line


def cube_quality(task_dir: Path, *, write_moments: bool = True) -> dict[str, Any] | None:
    """Cube statistics (and moment maps) of one task, or None without its cubes."""
    from astropy.io import fits

    paths = {kind: task_dir / cube_product(kind) for kind in ("image", "residual", "model")}
    if not (paths["image"].is_file() and paths["residual"].is_file()):
        return None
    with (
        fits.open(paths["image"], memmap=True) as image_hdus,
        fits.open(paths["residual"], memmap=True) as residual_hdus,
    ):
        header = image_hdus[0].header
        image = image_hdus[0].data
        residual = residual_hdus[0].data
        n_channels = image.shape[-3]
        planes = image.reshape(-1, n_channels, *image.shape[-2:])[0]
        residual_planes = residual.reshape(-1, n_channels, *residual.shape[-2:])[0]
        beam = beam_pixels(header)
        stacked_image = np.zeros(planes.shape[1:])
        for c in range(n_channels):
            stacked_image += np.nan_to_num(np.asarray(planes[c], dtype=np.float64))
        sy, sx, r = _source_geometry(stacked_image, beam)
        aperture = r <= 0.5
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN (flagged) channels
            spectrum = np.array(
                [
                    np.nanmean(np.asarray(planes[c], dtype=np.float64)[aperture])
                    for c in range(n_channels)
                ]
            )
        usable = np.array(
            [
                bool(np.any(np.nan_to_num(np.asarray(residual_planes[c], dtype=np.float64))))
                for c in range(n_channels)
            ]
        )
        logged = _logged_line_channels(task_dir, n_channels)
        line = (
            usable & logged
            if logged is not None
            else usable & ~line_free_channels(spectrum, usable)
            if usable.sum() > 2
            else np.zeros(n_channels, bool)
        )
        linefree = usable & ~line
        mask_path = task_dir / LINE_MASK_FILENAME
        off_source = r >= 2.5
        noise = np.array(
            [
                robust_sigma(np.asarray(residual_planes[c], dtype=np.float64)[off_source])
                if usable[c]
                else math.nan
                for c in range(0, n_channels, max(n_channels // 64, 1))
            ]
        )
        velocity, width_km_s = velocity_axis(header, fallback_axis=3)
        sums = {
            name: (np.zeros(planes.shape[1:]), np.zeros(planes.shape[1:]))
            for name in ("line", "linefree")
        }
        peak = np.full(planes.shape[1:], -np.inf)
        weighted_velocity = np.zeros(planes.shape[1:])
        weight = np.zeros(planes.shape[1:])
        channel_noise = float(np.nanmedian(noise)) if np.isfinite(noise).any() else math.nan
        for c in range(n_channels):
            if not (line[c] or linefree[c]):
                continue
            plane = np.nan_to_num(np.asarray(planes[c], dtype=np.float64))
            residual_plane = np.nan_to_num(np.asarray(residual_planes[c], dtype=np.float64))
            name = "line" if line[c] else "linefree"
            sums[name][0][:] += plane
            sums[name][1][:] += residual_plane
            if line[c]:
                peak = np.maximum(peak, plane)
                emitting = plane > 3 * channel_noise
                weighted_velocity += np.where(emitting, plane * velocity[c], 0.0)
                weight += np.where(emitting, plane, 0.0)
    if mask_path.is_file():
        line_mask = _plane(fits.getdata(mask_path)).astype(bool)
    else:
        line_mask = moment_mask(sums["line"][0] if line.any() else stacked_image, beam)
    off_mask = ~line_mask & (r >= 2.5)
    stats: dict[str, Any] = {
        "channels": int(n_channels),
        "usable_channels": int(usable.sum()),
        "line_channels": int(line.sum()),
        "line_free_channels": int(linefree.sum()),
        "line_channels_from": "log" if logged is not None else "source spectrum",
        "line_mask_pixels": int(line_mask.sum()),
        "line_mask_from": "task" if mask_path.is_file() else "moment-0",
        "channel_noise_mjy": channel_noise * 1e3,
    }
    one = r <= 1
    if linefree.any():
        image_sum, residual_sum = sums["linefree"]
        sigma = robust_sigma(residual_sum[off_source])
        stats["linefree_residual_sigma"] = _ratio(np.mean(residual_sum[r <= 0.5]), sigma)
        stats["linefree_residual_share"] = _ratio(np.sum(residual_sum[one]), np.sum(image_sum[one]))
    if line.any() and line_mask.any():
        image_sum, residual_sum = sums["line"]
        sigma = robust_sigma(residual_sum[off_mask])
        stats["line_residual_share"] = _ratio(
            np.sum(residual_sum[line_mask]), np.sum(image_sum[line_mask])
        )
        stats["line_residual_mean_sigma"] = _ratio(np.mean(residual_sum[line_mask]), sigma)
        stats["line_residual_peak_sigma"] = _ratio(np.max(residual_sum[line_mask]), sigma)
    if paths["model"].is_file():
        with fits.open(paths["model"], memmap=True) as model_hdus:
            model = model_hdus[0].data
            stats["model_flux_jy"] = float(
                sum(
                    np.nansum(np.asarray(model[..., c, :, :], dtype=np.float64))
                    for c in range(n_channels)
                )
            )
    if write_moments:
        stats["moments"] = _write_moments(
            task_dir,
            header,
            line_any=bool(line.any()),
            mom0=sums["line"][0] * width_km_s,
            mom1=np.where(
                weight > 0, weighted_velocity / np.where(weight > 0, weight, 1.0), np.nan
            ),
            mom8=np.where(np.isfinite(peak), peak, np.nan),
            residual_line=sums["line"][1] * width_km_s,
            residual_linefree=sums["linefree"][1] * width_km_s,
        )
    return stats


def _write_moments(task_dir: Path, header, *, line_any: bool, **maps: np.ndarray) -> list[str]:
    from astropy.io import fits

    out = task_dir / QUALITY_DIRNAME
    out.mkdir(exist_ok=True)
    units = {
        "mom0": "Jy/beam.km/s",
        "mom1": "km/s",
        "mom8": "Jy/beam",
        "residual_line": "Jy/beam.km/s",
        "residual_linefree": "Jy/beam.km/s",
    }
    plane_header = fits.Header()
    for key in (
        "CTYPE1",
        "CRVAL1",
        "CDELT1",
        "CRPIX1",
        "CUNIT1",
        "CTYPE2",
        "CRVAL2",
        "CDELT2",
        "CRPIX2",
        "CUNIT2",
        "BMAJ",
        "BMIN",
        "BPA",
        "EQUINOX",
        "RADESYS",
        "OBJECT",
    ):
        if key in header:
            plane_header[key] = header[key]
    written = []
    for name, data in maps.items():
        if name in ("mom0", "mom1", "mom8", "residual_line") and not line_any:
            continue
        plane_header["BUNIT"] = units[name]
        path = out / f"cube-{name.replace('_', '-')}.fits"
        write_fits_atomic(
            path, [fits.PrimaryHDU(np.asarray(data, dtype=np.float32), header=plane_header)]
        )
        written.append(path.name)
    return written
