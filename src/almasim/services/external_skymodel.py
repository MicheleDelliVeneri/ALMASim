"""External sky-model ingestion helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.ndimage import zoom


EXTERNAL_SOURCE_TYPES = {
    "external-fits-image",
    "external-fits-cube",
    "external-components",
}


@dataclass
class ExternalSkyModelPayload:
    """Normalized external sky-model payload."""

    cube: np.ndarray
    header: fits.Header
    metadata: dict[str, Any]


def is_external_source_type(source_type: str) -> bool:
    """Return whether the source type is backed by an external model input."""
    return str(source_type).lower() in EXTERNAL_SOURCE_TYPES


def infer_external_cube_geometry(
    *,
    source_type: str,
    skymodel_path: str | None = None,
    component_table_path: str | None = None,
) -> dict[str, int] | None:
    """Infer basic cube geometry from an external input when possible."""
    source_type = str(source_type).lower()
    if source_type == "external-components":
        if component_table_path is None:
            raise ValueError("external_component_table_path is required for external-components")
        return None

    if skymodel_path is None:
        raise ValueError("external_skymodel_path is required for external FITS inputs")

    data, _header = fits.getdata(skymodel_path, header=True)
    cube = _coerce_fits_to_channel_first(np.asarray(data), source_type=source_type)
    return {
        "n_channels": int(cube.shape[0]),
        "n_pix_y": int(cube.shape[1]),
        "n_pix_x": int(cube.shape[2]),
        "n_pix": int(max(cube.shape[1], cube.shape[2])),
    }


def load_external_sky_model(
    *,
    source_type: str,
    skymodel_path: str | None = None,
    component_table_path: str | None = None,
    target_npix: int | None = None,
    target_nchannels: int | None = None,
    alignment_mode: str = "observation",
    header_mode: str = "observation",
    header_overrides: Mapping[str, Any] | None = None,
    target_header: fits.Header | None = None,
    target_wcs: Any | None = None,
    channel_frequencies_hz: np.ndarray | None = None,
) -> ExternalSkyModelPayload:
    """Load an external skymodel and normalize it to `(chan, y, x)` order."""
    source_type = str(source_type).lower()
    alignment_mode = str(alignment_mode or "observation").lower()
    header_mode = str(header_mode or "observation").lower()
    if source_type not in EXTERNAL_SOURCE_TYPES:
        raise ValueError(f"Unsupported external source type: {source_type}")

    if source_type == "external-components":
        if target_npix is None or target_nchannels is None:
            raise ValueError("target_npix and target_nchannels are required for external-components")
        cube = _load_component_table(
            component_table_path=component_table_path,
            target_npix=int(target_npix),
            target_nchannels=int(target_nchannels),
            target_wcs=target_wcs,
            channel_frequencies_hz=channel_frequencies_hz,
        )
        header = fits.Header() if target_header is None else target_header.copy()
        metadata = {
            "input_kind": "component_table",
            "input_path": str(component_table_path),
            "alignment_mode": alignment_mode,
            "header_mode": header_mode,
            "original_shape": [int(v) for v in cube.shape],
        }
    else:
        if skymodel_path is None:
            raise ValueError("external_skymodel_path is required for external FITS inputs")
        raw_data, input_header = fits.getdata(skymodel_path, header=True)
        cube = _coerce_fits_to_channel_first(np.asarray(raw_data), source_type=source_type)
        metadata = {
            "input_kind": "fits",
            "input_path": str(skymodel_path),
            "alignment_mode": alignment_mode,
            "header_mode": header_mode,
            "original_shape": [int(v) for v in cube.shape],
            "input_bunit": str(input_header.get("BUNIT", "")),
        }
        target_shape = _resolve_target_shape(
            cube_shape=cube.shape,
            target_npix=target_npix,
            target_nchannels=target_nchannels,
            alignment_mode=alignment_mode,
        )
        cube = _align_cube_to_shape(cube, target_shape)
        header = (
            input_header.copy()
            if header_mode == "preserve" or target_header is None
            else target_header.copy()
        )

    if header_overrides:
        for key, value in header_overrides.items():
            header[str(key).upper()] = value

    return ExternalSkyModelPayload(
        cube=np.asarray(cube, dtype=np.float32),
        header=header,
        metadata=metadata,
    )


def _coerce_fits_to_channel_first(data: np.ndarray, *, source_type: str) -> np.ndarray:
    """Normalize FITS image/cube data to `(chan, y, x)` order."""
    source_type = str(source_type).lower()
    if data.ndim == 2:
        return data[np.newaxis, :, :].astype(np.float32)
    if data.ndim != 3:
        raise ValueError(f"Expected a 2D image or 3D cube, got {data.ndim}D data")

    # FITS cubes are commonly stored as (chan, y, x). Keep that by default.
    if source_type == "external-fits-image":
        # Allow a single slice to be selected from a 3D cube-like image.
        return data[:1].astype(np.float32)
    return data.astype(np.float32)


def _resolve_target_shape(
    *,
    cube_shape: tuple[int, int, int],
    target_npix: int | None,
    target_nchannels: int | None,
    alignment_mode: str,
) -> tuple[int, int, int]:
    """Resolve the aligned cube shape."""
    if alignment_mode == "preserve":
        nchan = int(target_nchannels) if target_nchannels is not None else int(cube_shape[0])
        ny = int(target_npix) if target_npix is not None else int(cube_shape[1])
        nx = int(target_npix) if target_npix is not None else int(cube_shape[2])
        return (nchan, ny, nx)
    if target_npix is None or target_nchannels is None:
        raise ValueError(
            "target_npix and target_nchannels are required when alignment_mode='observation'"
        )
    return (int(target_nchannels), int(target_npix), int(target_npix))


def _align_cube_to_shape(cube: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    """Resize or crop/pad an external cube to the requested target shape."""
    if tuple(cube.shape) == tuple(target_shape):
        return cube.astype(np.float32, copy=False)

    factors = tuple(
        target / current if current > 0 else 1.0
        for current, target in zip(cube.shape, target_shape)
    )
    resized = zoom(cube, zoom=factors, order=1)
    resized = np.asarray(resized, dtype=np.float32)
    return _crop_or_pad_cube(resized, target_shape)


def _crop_or_pad_cube(cube: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    """Center-crop or zero-pad a cube to the exact target shape."""
    out = np.zeros(target_shape, dtype=np.float32)
    src_slices = []
    dst_slices = []
    for current, target in zip(cube.shape, target_shape):
        if current >= target:
            start = (current - target) // 2
            src_slices.append(slice(start, start + target))
            dst_slices.append(slice(None))
        else:
            offset = (target - current) // 2
            src_slices.append(slice(None))
            dst_slices.append(slice(offset, offset + current))
    out[tuple(dst_slices)] = cube[tuple(src_slices)]
    return out


def _load_component_table(
    *,
    component_table_path: str | None,
    target_npix: int,
    target_nchannels: int,
    target_wcs: Any | None,
    channel_frequencies_hz: np.ndarray | None,
) -> np.ndarray:
    """Build a cube from a simple component-list-like table."""
    if component_table_path is None:
        raise ValueError("external_component_table_path is required for external-components")

    table = Table.read(component_table_path)
    rows = table.to_pandas()
    columns = {str(col).lower(): col for col in rows.columns}
    cube = np.zeros((target_nchannels, target_npix, target_npix), dtype=np.float32)
    reference_freq = (
        float(np.median(channel_frequencies_hz))
        if channel_frequencies_hz is not None and len(channel_frequencies_hz) > 0
        else 1.0
    )

    for _, row in rows.iterrows():
        x_pix, y_pix = _resolve_component_position(row, columns, target_wcs)
        flux = float(_row_value(row, columns, ["flux_jy", "flux", "intensity"], default=0.0))
        if flux == 0.0:
            continue

        alpha = float(_row_value(row, columns, ["spectral_index", "alpha"], default=0.0))
        ref_freq = float(
            _row_value(row, columns, ["reference_freq_hz", "ref_freq_hz"], default=reference_freq)
        )
        channel = _row_value(row, columns, ["channel", "chan"], default=None)

        if channel is not None:
            chan_idx = int(np.clip(int(channel), 0, target_nchannels - 1))
            cube[chan_idx, np.clip(y_pix, 0, target_npix - 1), np.clip(x_pix, 0, target_npix - 1)] += flux
            continue

        if channel_frequencies_hz is None:
            spectrum = np.full(target_nchannels, flux, dtype=np.float32)
        else:
            nu = np.asarray(channel_frequencies_hz, dtype=float)
            spectrum = flux * (nu / max(ref_freq, 1e-12)) ** alpha
        cube[:, np.clip(y_pix, 0, target_npix - 1), np.clip(x_pix, 0, target_npix - 1)] += spectrum.astype(np.float32)

    return cube


def _resolve_component_position(row, columns: dict[str, str], target_wcs: Any | None) -> tuple[int, int]:
    """Resolve component position from pixel or sky coordinates."""
    if "x_pix" in columns and "y_pix" in columns:
        x_pix = int(round(float(row[columns["x_pix"]])))
        y_pix = int(round(float(row[columns["y_pix"]])))
        return x_pix, y_pix

    if "ra_deg" in columns and "dec_deg" in columns and target_wcs is not None:
        ra = float(row[columns["ra_deg"]])
        dec = float(row[columns["dec_deg"]])
        x_pix, y_pix, _ = target_wcs.sub(3).wcs_world2pix(ra, dec, 1.0, 0)
        return int(round(float(x_pix))), int(round(float(y_pix)))

    raise ValueError(
        "Component rows must provide either x_pix/y_pix or ra_deg/dec_deg columns"
    )


def _row_value(row, columns: dict[str, str], keys: list[str], *, default: Any) -> Any:
    """Fetch a case-insensitive row value with a default."""
    for key in keys:
        actual = columns.get(key)
        if actual is None:
            continue
        value = row[actual]
        if value is None:
            continue
        if isinstance(value, float) and np.isnan(value):
            continue
        return value
    return default
