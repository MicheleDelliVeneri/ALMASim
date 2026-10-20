"""Cleaning-quality statistics and moment maps for imaged tasks.

Measures, from the images a WSClean task leaves (``wsclean-*.fits`` and
``wsclean-cube-*.fits``) and its log, how much source signal the clean left
in the residual: the statistics used to tune the MADIV-16 imaging (masked
clean, wider images, continuum restart, visibility-plane continuum
subtraction, self-calibration, the cube line mask).

Continuum, around the brightest pixel of the restored image (the source):

* ``snr``: its peak over the residual noise (1.4826 x MAD, more than 2.5
  beams from the source);
* ``residual_share_1beam`` / ``_2beam``: the residual summed within one or
  two beams of the source over the restored image summed there, the share of
  the source still in the residual;
* ``residual_peak_sigma`` and ``residual_core_sigma``: the largest absolute
  residual within one beam and the mean within half a beam, in noise units;
* ``residual_rings_sigma``: the residual's mean in rings of 0.5 beam out to
  5 beams. A positive core with a negative ring at 1-2 beams is flux the
  clean did not reach (the dirty beam's first sidelobe); a positive ring
  inside a negative one is a shape mismatch, not missing flux;
* ``residual_symmetric_sigma`` / ``residual_antisymmetric_sigma``: the
  residual near the source split under a 180-degree rotation; phase errors
  leave mostly antisymmetric structure;
* ``model_flux_jy`` and ``model_components``;
* ``noise_at_mask_creation_ratio``: the noise WSClean estimated in its first
  major cycle over the one in its last (from the log). Well above 1, the
  auto-mask was built while sidelobes inflated the noise and misses the
  source's fainter parts.

Cube, with the source's spectrum (or the channel ranges a visibility-plane
continuum subtraction logged) split into line and line-free channels:

* ``channel_noise_mjy``: median per-channel residual noise;
* ``linefree_residual_sigma`` / ``linefree_residual_share``: the residual
  stacked over the line-free channels at the source, in units of the stacked
  noise and as a share of the stacked image (continuum left in the cube);
* ``line_residual_share``, ``line_residual_mean_sigma``,
  ``line_residual_peak_sigma``: the residual stacked over the line channels
  inside the line mask (``wsclean-cube-linemask.fits``, or a moment-0 mask
  built the same way), as a share of the stacked image and in noise units;
* ``model_flux_jy``: the cube model's total.

Moment maps (``quality/`` next to the images): moment 0 (Jy/beam km/s),
1 (km/s, intensity weighted over pixels above 3 channel sigma) and 8 (peak,
Jy/beam) of the line channels, and moment 0 of the residual over the line
and over the line-free channels, which show uncleaned emission directly.
"""

from __future__ import annotations

import csv
import json
import math
import re
import shlex
import warnings
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from ..instrument.constants import SPEED_OF_LIGHT_KM_S
from .archive_imaging import (
    CUBE_NAME,
    IMAGE_FILENAME,
    LINE_MASK_FILENAME,
    line_free_channels,
)

QUALITY_FILENAME = "quality.json"
QUALITY_DIRNAME = "quality"
SUMMARY_FILENAME = "clean-quality.csv"
RINGS_BEAMS = ((0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2), (2, 3), (3, 5))
_SIGMA = re.compile(r"Estimated standard deviation of background noise: ([-+.\w]+) (\w*Jy)")
_UNIT = {"Jy": 1.0, "mJy": 1e-3, "µJy": 1e-6, "uJy": 1e-6, "nJy": 1e-9}
_LINE_RANGES = re.compile(
    r"(?:(\d+) of (\d+) channels line-free(?: \((\d+) in the window\))?, )?"
    r"lines in channels ([\d~;]+|none)"
)


def robust_sigma(values: np.ndarray) -> float:
    """1.4826 x the median absolute deviation of the finite values (nan if none)."""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return math.nan
    return float(1.4826 * np.median(np.abs(finite - np.median(finite))))


def _plane(data: np.ndarray) -> np.ndarray:
    return np.asarray(data, dtype=np.float64).reshape(data.shape[-2:])


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else math.nan


def _beam_pixels(header) -> float:
    cell = abs(float(header.get("CDELT2", 0.0)))
    return float(header.get("BMAJ", 0.0)) / cell if cell else math.nan


def _source_geometry(image: np.ndarray, beam_pixels: float) -> tuple[int, int, np.ndarray]:
    """The brightest pixel in the image's inner half and every pixel's distance to it in beams."""
    ny, nx = image.shape
    yy, xx = np.mgrid[:ny, :nx]
    inner = (np.abs(yy - ny / 2) < ny / 4) & (np.abs(xx - nx / 2) < nx / 4)
    sy, sx = np.unravel_index(np.nanargmax(np.where(inner, image, -np.inf)), image.shape)
    distance = np.hypot(yy - sy, xx - sx) / beam_pixels
    return int(sy), int(sx), distance


def noise_history(log_path: Path, name: str) -> list[float]:
    """The noise WSClean estimated at each major cycle of the run that wrote ``name``.

    The task log holds every WSClean run of a task, each after a ``# <command>``
    line; the first run with ``-name .../<name>`` is taken (the continuum's
    first pass, before any restart).
    """
    sigmas: list[float] = []
    current = False
    for line in log_path.read_text(errors="replace").splitlines():
        if line.startswith("# ") and " -name " in line:
            if sigmas:
                break
            # The task logs each command with shlex.join: quoted paths
            # (spaces and the like) only split right with shlex.
            try:
                words = shlex.split(line[2:])
            except ValueError:
                words = line[2:].split()
            target = words[words.index("-name") + 1] if "-name" in words[:-1] else ""
            current = Path(target).name == name
            continue
        match = _SIGMA.search(line)
        if current and match:
            try:
                sigmas.append(float(match.group(1)) * _UNIT.get(match.group(2), 1.0))
            except ValueError:
                continue
    return sigmas


def continuum_quality(task_dir: Path) -> dict[str, Any] | None:
    """Continuum statistics of one task, or None without its three images."""
    from astropy.io import fits

    paths = {kind: task_dir / f"wsclean-{kind}.fits" for kind in ("image", "residual", "model")}
    if not all(path.is_file() for path in paths.values()):
        return None
    header = fits.getheader(paths["image"])
    image = _plane(fits.getdata(paths["image"]))
    residual = _plane(fits.getdata(paths["residual"]))
    model = _plane(fits.getdata(paths["model"]))
    beam = _beam_pixels(header)
    sy, sx, r = _source_geometry(image, beam)
    sigma = robust_sigma(residual[r >= 2.5])
    one, two = r <= 1, r <= 2
    half = int(np.ceil(2 * beam)) if np.isfinite(beam) else 8
    box = residual[max(sy - half, 0) : sy + half + 1, max(sx - half, 0) : sx + half + 1]
    symmetric, antisymmetric = 0.5 * (box + box[::-1, ::-1]), 0.5 * (box - box[::-1, ::-1])
    stats: dict[str, Any] = {
        "peak_jy_beam": float(image[sy, sx]),
        "peak_pixel": [sy, sx],
        "beam_arcsec": [
            round(float(header.get("BMAJ", math.nan)) * 3600, 3),
            round(float(header.get("BMIN", math.nan)) * 3600, 3),
        ],
        "noise_jy_beam": sigma,
        "snr": _ratio(image[sy, sx], sigma),
        "residual_share_1beam": _ratio(np.nansum(residual[one]), np.nansum(image[one])),
        "residual_share_2beam": _ratio(np.nansum(residual[two]), np.nansum(image[two])),
        "residual_peak_sigma": _ratio(np.nanmax(np.abs(residual[one])), sigma),
        "residual_core_sigma": _ratio(np.nanmean(residual[r <= 0.5]), sigma),
        "residual_rings_sigma": {
            f"{lo:g}-{hi:g}": _ratio(np.nanmean(residual[(r >= lo) & (r < hi)]), sigma)
            for lo, hi in RINGS_BEAMS
        },
        "residual_symmetric_sigma": _ratio(robust_sigma(symmetric), sigma),
        "residual_antisymmetric_sigma": _ratio(robust_sigma(antisymmetric), sigma),
        "model_flux_jy": float(np.nansum(model)),
        "model_components": int(np.count_nonzero(np.nan_to_num(model))),
    }
    log = task_dir.parent / f"{task_dir.name}.log"
    history = (
        noise_history(log, Path(IMAGE_FILENAME).stem.rsplit("-", 1)[0]) if log.is_file() else []
    )
    if len(history) >= 2:
        stats["noise_at_mask_creation_ratio"] = _ratio(history[0], history[-1])
    return stats


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


def _moment_mask(moment: np.ndarray, beam_pixels: float, sigma: float = 3.0) -> np.ndarray:
    """``moment`` above ``sigma`` times its robust noise, grown by one beam."""
    noise = robust_sigma(moment)
    mask = moment > sigma * noise if noise > 0 else np.zeros(moment.shape, bool)
    if mask.any() and np.isfinite(beam_pixels) and beam_pixels > 0:
        from scipy import ndimage

        radius = int(np.ceil(beam_pixels))
        yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
        mask = ndimage.binary_dilation(mask, structure=np.hypot(yy, xx) <= beam_pixels)
    return mask


def _velocity_axis(header, n_channels: int) -> tuple[np.ndarray, float]:
    """Channel velocities (km/s, radio convention) and the channel width in km/s."""
    freq = float(header.get("CRVAL3", 0.0)) + (
        np.arange(n_channels) + 1 - float(header.get("CRPIX3", 1.0))
    ) * float(header.get("CDELT3", 0.0))
    rest = float(header.get("RESTFRQ", 0.0) or freq[n_channels // 2] or 1.0)
    velocity = SPEED_OF_LIGHT_KM_S * (rest - freq) / rest
    width = abs(float(header.get("CDELT3", 0.0))) / rest * SPEED_OF_LIGHT_KM_S
    return velocity, width


def cube_quality(task_dir: Path, *, write_moments: bool = True) -> dict[str, Any] | None:
    """Cube statistics (and moment maps) of one task, or None without its cubes."""
    from astropy.io import fits

    paths = {kind: task_dir / f"{CUBE_NAME}-{kind}.fits" for kind in ("image", "residual", "model")}
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
        beam = _beam_pixels(header)
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
        velocity, width_km_s = _velocity_axis(header, n_channels)
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
        line_mask = _moment_mask(sums["line"][0] if line.any() else stacked_image, beam)
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
        fits.PrimaryHDU(np.asarray(data, dtype=np.float32), header=plane_header).writeto(
            path, overwrite=True
        )
        written.append(path.name)
    return written


def find_task_dirs(root: Path) -> list[Path]:
    """Every directory under ``root`` (``root`` included) holding a continuum or cube image."""
    root = Path(root)
    names = (IMAGE_FILENAME, f"{CUBE_NAME}-image.fits")
    found = {path.parent for name in names for path in root.rglob(name)}
    return sorted(found)


def assess_task(task_dir: Path, *, write_moments: bool = True) -> dict[str, Any]:
    """Continuum and cube statistics of one task, also written to its ``quality.json``."""
    task_dir = Path(task_dir)
    result: dict[str, Any] = {"task": str(task_dir)}
    for key, stats in (
        ("continuum", continuum_quality(task_dir)),
        ("cube", cube_quality(task_dir, write_moments=write_moments)),
    ):
        if stats is not None:
            result[key] = stats
    (task_dir / QUALITY_FILENAME).write_text(
        json.dumps(result, indent=2, default=_json_default) + "\n", encoding="utf-8"
    )
    return result


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serialisable")


SUMMARY_COLUMNS = (
    ("continuum", "snr"),
    ("continuum", "residual_share_1beam"),
    ("continuum", "residual_share_2beam"),
    ("continuum", "residual_peak_sigma"),
    ("continuum", "residual_core_sigma"),
    ("continuum", "residual_antisymmetric_sigma"),
    ("continuum", "noise_at_mask_creation_ratio"),
    ("continuum", "model_flux_jy"),
    ("cube", "line_channels"),
    ("cube", "channel_noise_mjy"),
    ("cube", "linefree_residual_sigma"),
    ("cube", "line_residual_share"),
    ("cube", "line_residual_peak_sigma"),
    ("cube", "model_flux_jy"),
)


def summary_rows(results: Iterable[dict[str, Any]], root: Path) -> list[dict[str, Any]]:
    """One flat row per task: its path relative to ``root`` and ``SUMMARY_COLUMNS``."""
    rows = []
    for result in results:
        task = Path(result["task"])
        row: dict[str, Any] = {"task": str(task.relative_to(root)) if task != root else "."}
        for section, key in SUMMARY_COLUMNS:
            row[f"{section}.{key}"] = result.get(section, {}).get(key)
        rows.append(row)
    return rows


def write_summary(rows: Sequence[dict[str, Any]], path: Path) -> None:
    """``rows`` as CSV at ``path``."""
    columns = ["task"] + [f"{section}.{key}" for section, key in SUMMARY_COLUMNS]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
