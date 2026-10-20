"""Load data products and turn them into 2D previews for the web UI.

The visualizer and combination pages show any simulation or imaging product
(``.npz``, FITS, or an MSv2 directory gridded into a dirty visibility cube)
integrated along one axis; the deconvolution page previews its input and
output cubes. Moved out of the backend routers and ``imaging.reconstruction``
(MADIV-28 T6.7) so the library, not the web layer, owns the product formats.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from almasim.services.instrument.constants import SPEED_OF_LIGHT_M_S
from almasim.services.interferometry.utils import closest_power_of_2

NPZ_SUFFIXES = (".npz",)
FITS_SUFFIXES = (".fits", ".fit", ".fts")
MSV2_SUFFIXES = (".ms",)


def classify_product_path(path: Path) -> str | None:
    lower = path.name.lower()
    if path.is_dir() and lower.endswith(MSV2_SUFFIXES):  # lgtm[py/path-injection]
        return "msv2"
    if path.is_file():  # lgtm[py/path-injection]
        if lower.endswith(NPZ_SUFFIXES):
            return "npz"
        if lower.endswith(FITS_SUFFIXES):
            return "fits"
    return None


def iter_supported_products(base_dir: Path) -> list[Path]:
    supported: list[Path] = []
    for pattern in ("*.npz", "*.fits", "*.fit", "*.fts"):
        for p in base_dir.rglob(pattern):  # lgtm[py/path-injection]
            if p.is_file():
                supported.append(p)
    for p in base_dir.rglob("*.ms"):  # lgtm[py/path-injection]
        if p.is_dir():
            supported.append(p)
    return sorted(supported, key=lambda path: path.stat().st_mtime, reverse=True)


def _coerce_regular_cube(data: np.ndarray) -> np.ndarray:
    cube = np.asarray(data)
    while cube.ndim > 3:
        cube = cube[0]
    if cube.ndim == 2:
        cube = cube[np.newaxis, :, :]
    if cube.ndim != 3:
        raise ValueError(f"Expected 2D or 3D data, got {cube.ndim}D array")
    return cube


def _load_npz_cube(path: Path) -> tuple[np.ndarray, str]:
    with np.load(path, allow_pickle=False) as data:
        cube = None
        cube_name = None
        for name in (
            "modelCube",
            "dirtyCube",
            "clean_cube",
            "dirty_cube",
            "dirty_vis_cube",
            "clean_vis_cube",
            "beam_cube",
            "uv_mask_cube",
            "u_cube",
            "v_cube",
        ):
            if name in data:
                cube = data[name]
                cube_name = name
                break
        if cube is None:
            keys = list(data.keys())
            if not keys:
                raise ValueError("No arrays found in .npz file")
            cube_name = keys[0]
            cube = data[cube_name]
    return _coerce_regular_cube(np.asarray(cube)), cube_name or path.name


def _load_fits_cube(path: Path) -> tuple[np.ndarray, str]:
    from astropy.io import fits

    data = fits.getdata(path, memmap=False)
    if data is None:
        raise ValueError("FITS file does not contain image data")
    return _coerce_regular_cube(np.asarray(data)), path.name


def _frequency_to_hz(channel_freq: np.ndarray) -> np.ndarray:
    freq = np.asarray(channel_freq, dtype=np.float64)
    if np.max(np.abs(freq)) > 1.0e6:
        return freq
    return freq * 1.0e9


def _infer_grid_size_from_companions(path: Path) -> int | None:
    parent = path.parent
    for pattern in (
        "dirty-vis-cube_*.npz",
        "clean-vis-cube_*.npz",
        "clean-cube_*.npz",
        "dirty-cube_*.npz",
        "beam-cube_*.npz",
        "dirty-vis-cube_real*.fits",
        "clean-vis-cube_real*.fits",
        "clean-cube_*.fits",
        "dirty-cube_*.fits",
        "beam-cube_*.fits",
    ):
        for candidate in sorted(parent.glob(pattern)):  # lgtm[py/path-injection]
            try:
                if candidate.suffix.lower() == ".npz":
                    cube, _ = _load_npz_cube(candidate)
                else:
                    cube, _ = _load_fits_cube(candidate)
            except Exception:
                continue
            if cube.ndim == 3 and cube.shape[1] == cube.shape[2]:
                return int(cube.shape[1])
    return None


def _infer_msv2_grid_size(path: Path, visibility_table: dict[str, np.ndarray]) -> int:
    companion = _infer_grid_size_from_companions(path)
    if companion is not None:
        return companion
    nrows = int(np.asarray(visibility_table["data"]).shape[0])
    estimate = max(int(np.sqrt(max(nrows / 2.0, 1.0))), 16)
    return int(closest_power_of_2(estimate))


def _grid_visibility_table(
    visibility_table: dict[str, np.ndarray],
    *,
    grid_size: int,
    use_model_data: bool = False,
) -> np.ndarray:
    nphf = grid_size // 2
    uvw_m = np.asarray(visibility_table["uvw_m"], dtype=np.float64)
    data_key = "model_data" if use_model_data else "data"
    data = np.asarray(visibility_table[data_key])
    if data.ndim != 3:
        raise ValueError(
            f"Expected visibility table data with shape (nrows, ncorr, nchan), got {data.shape}"
        )

    nrows, _ncorr, nchan = data.shape
    cube = np.zeros((nchan, grid_size, grid_size), dtype=np.complex64)

    freqs_hz = _frequency_to_hz(np.asarray(visibility_table["channel_freq_hz"], dtype=np.float64))
    c_m_s = SPEED_OF_LIGHT_M_S

    for chan_idx in range(nchan):
        wavelength_m = c_m_s / max(float(freqs_hz[chan_idx]), 1.0)
        u_waves = uvw_m[:, 0] / wavelength_m
        v_waves = uvw_m[:, 1] / wavelength_m
        max_extent = max(float(np.max(np.abs(u_waves))), float(np.max(np.abs(v_waves))), 1.0)
        uv_pixsize = max_extent / max(nphf - 1, 1)

        pix_u = np.rint(u_waves / uv_pixsize).astype(np.int32)
        pix_v = np.rint(v_waves / uv_pixsize).astype(np.int32)
        valid = (np.abs(pix_u) < nphf) & (np.abs(pix_v) < nphf)
        if not np.any(valid):
            continue

        p_u = pix_u[valid] + nphf
        p_v = pix_v[valid] + nphf
        m_u = -pix_u[valid] + nphf
        m_v = -pix_v[valid] + nphf
        samples = data[valid, 0, chan_idx].astype(np.complex64, copy=False)

        channel_grid = cube[chan_idx]
        np.add.at(channel_grid, (p_v, m_u), samples)
        np.add.at(channel_grid, (m_v, p_u), np.conjugate(samples))

    return cube


def _load_msv2_cube(path: Path) -> tuple[np.ndarray, str]:
    from almasim.services.ms.io import read_native_ms

    visibility_table = read_native_ms(path)
    grid_size = _infer_msv2_grid_size(path, visibility_table)
    cube = _grid_visibility_table(visibility_table, grid_size=grid_size)
    return cube, f"{path.name}: dirty visibility cube"


def load_preview_cube(path: Path) -> tuple[np.ndarray, str, str]:
    file_type = classify_product_path(path)
    if file_type == "npz":
        cube, cube_name = _load_npz_cube(path)
    elif file_type == "fits":
        cube, cube_name = _load_fits_cube(path)
    elif file_type == "msv2":
        cube, cube_name = _load_msv2_cube(path)
    else:
        raise ValueError(f"Unsupported file type: {path.name}")
    return cube, cube_name, file_type


def _resolve_integration_axis(cube: np.ndarray, requested_axis: str | int | None) -> int:
    if requested_axis is None or requested_axis == "auto":
        return 0
    axis = int(requested_axis)
    if axis < 0 or axis >= cube.ndim:
        raise ValueError(f"Integration axis {axis} is out of bounds for shape {cube.shape}")
    return axis


def _slice_integration_window(
    cube: np.ndarray,
    *,
    axis: int,
    channel_start: int | None,
    channel_end: int | None,
) -> tuple[np.ndarray, int, int]:
    axis_length = int(cube.shape[axis])
    start = 0 if channel_start is None else int(channel_start)
    end = axis_length - 1 if channel_end is None else int(channel_end)
    if start < 0 or end < 0 or start >= axis_length or end >= axis_length:
        raise ValueError(
            f"Channel range [{start}, {end}] is out of bounds for axis length {axis_length}"
        )
    if start > end:
        raise ValueError("channel_start must be less than or equal to channel_end")
    slicer = [slice(None)] * cube.ndim
    slicer[axis] = slice(start, end + 1)
    return cube[tuple(slicer)], start, end


def project_complex_component(
    cube: np.ndarray,
    *,
    complex_component: str,
) -> np.ndarray:
    if complex_component not in {"real", "imag", "magnitude", "phase"}:
        raise ValueError("complex_component must be one of: real, imag, magnitude, phase")
    if not np.iscomplexobj(cube):
        return np.asarray(cube, dtype=np.float32)
    if complex_component == "real":
        return np.real(cube).astype(np.float32, copy=False)
    if complex_component == "imag":
        return np.imag(cube).astype(np.float32, copy=False)
    if complex_component == "phase":
        return np.angle(cube).astype(np.float32, copy=False)
    return np.abs(cube).astype(np.float32, copy=False)


def _integrate_cube(cube: np.ndarray, method: str, *, axis: int) -> np.ndarray:
    if method == "sum":
        integrated = np.sum(cube, axis=axis)
    else:
        integrated = np.nanmean(cube, axis=axis)
    return np.nan_to_num(integrated, nan=0.0, posinf=0.0, neginf=0.0)


def _normalize_for_display(image: np.ndarray) -> np.ndarray:
    if image.max() > image.min():
        scaled = (image - image.min()) / (image.max() - image.min())
        return (scaled * 255).astype(np.uint8)
    return np.zeros_like(image, dtype=np.uint8)


def integrated_preview(
    path: Path,
    method: str,
    *,
    integration_axis: str | int | None = "auto",
    complex_component: str = "real",
    channel_start: int | None = None,
    channel_end: int | None = None,
) -> dict:
    """Load ``path``, integrate it along one axis and normalise it to a 0-255 image."""
    cube, cube_name, file_type = load_preview_cube(path)
    display_cube = project_complex_component(cube, complex_component=complex_component)
    axis = _resolve_integration_axis(display_cube, integration_axis)
    windowed_cube, start, end = _slice_integration_window(
        display_cube,
        axis=axis,
        channel_start=channel_start,
        channel_end=channel_end,
    )
    integrated = _integrate_cube(windowed_cube, method, axis=axis)
    normalized = _normalize_for_display(integrated)
    stats = {
        "shape": list(display_cube.shape),
        "window_shape": list(windowed_cube.shape),
        "integrated_shape": list(integrated.shape),
        "min": float(integrated.min()),
        "max": float(integrated.max()),
        "mean": float(integrated.mean()),
        "std": float(integrated.std()),
        "cube_name": cube_name,
        "format": file_type,
        "integration_axis": axis,
        "complex_component": complex_component,
        "channel_start": start,
        "channel_end": end,
    }
    return {
        "image": normalized.tolist(),
        "stats": stats,
        "method": method,
    }


# Deconvolution page (from imaging.reconstruction)


def load_cube_from_npz(npz_path: str | Path) -> tuple[np.ndarray, str]:
    """Load a cube from an NPZ file and normalize to `(chan, y, x)` layout."""
    with np.load(npz_path) as data:
        cube = None
        cube_name = None
        for name in [
            "modelCube",
            "dirtyCube",
            "clean_cube",
            "dirty_cube",
            "beam_cube",
            "component_cube",
            "residual_cube",
            "restored_cube",
            "clean_beam_cube",
            "convolved_reference_cube",
        ]:
            if name in data:
                cube = data[name]
                cube_name = name
                break
        if cube is None:
            keys = list(data.keys())
            if not keys:
                raise ValueError("No arrays found in NPZ file")
            cube_name = keys[0]
            cube = data[cube_name]

    cube = np.asarray(cube, dtype=np.float32)
    if cube.ndim != 3:
        raise ValueError(f"Expected 3D datacube, got {cube.ndim}D array")
    if cube.shape[2] <= min(cube.shape[0], cube.shape[1]):
        cube = np.transpose(cube, (2, 0, 1))
    return cube, str(cube_name)


def integrate_cube_preview(
    cube: np.ndarray,
    *,
    method: str = "sum",
    cube_name: str = "cube",
) -> dict[str, Any]:
    """Convert a channel-first cube into a normalized 2D preview payload."""
    cube = np.asarray(cube, dtype=np.float32)
    if cube.ndim != 3:
        raise ValueError(f"Expected 3D datacube, got {cube.ndim}D array")
    if method not in {"sum", "mean"}:
        raise ValueError("method must be 'sum' or 'mean'")

    if method == "sum":
        integrated = np.sum(cube, axis=0)
    else:
        integrated = np.nanmean(cube, axis=0)

    integrated = np.nan_to_num(integrated, nan=0.0, posinf=0.0, neginf=0.0)
    if integrated.max() > integrated.min():
        normalized = (
            (integrated - integrated.min()) / (integrated.max() - integrated.min()) * 255.0
        ).astype(np.uint8)
    else:
        normalized = np.zeros_like(integrated, dtype=np.uint8)

    return {
        "image": normalized.tolist(),
        "stats": {
            "shape": list(cube.shape),
            "integrated_shape": list(integrated.shape),
            "min": float(integrated.min()),
            "max": float(integrated.max()),
            "mean": float(integrated.mean()),
            "std": float(integrated.std()),
            "cube_name": cube_name,
        },
        "method": method,
    }
