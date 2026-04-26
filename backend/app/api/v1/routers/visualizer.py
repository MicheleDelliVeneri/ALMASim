"""Visualizer API endpoints for product inspection."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse

from almasim.services.interferometry.utils import closest_power_of_2
from app.core.config import settings

router = APIRouter()

_NPZ_SUFFIXES = (".npz",)
_FITS_SUFFIXES = (".fits", ".fit", ".fts")
_MSV2_SUFFIXES = (".ms",)


def _classify_visualizer_path(path: Path) -> str | None:
    lower = path.name.lower()
    if path.is_dir() and lower.endswith(_MSV2_SUFFIXES):  # lgtm[py/path-injection]
        return "msv2"
    if path.is_file():  # lgtm[py/path-injection]
        if lower.endswith(_NPZ_SUFFIXES):
            return "npz"
        if lower.endswith(_FITS_SUFFIXES):
            return "fits"
    return None


def _iter_supported_paths(base_dir: Path) -> list[Path]:
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
    c_m_s = 299_792_458.0

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
    from almasim.services.products.ms_io import read_native_ms

    visibility_table = read_native_ms(path)
    grid_size = _infer_msv2_grid_size(path, visibility_table)
    cube = _grid_visibility_table(visibility_table, grid_size=grid_size)
    return cube, f"{path.name}: dirty visibility cube"


def _load_visualizer_cube(path: Path) -> tuple[np.ndarray, str, str]:
    file_type = _classify_visualizer_path(path)
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


def _project_complex_component(
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


def _build_integrated_response(
    path: Path,
    method: str,
    *,
    integration_axis: str | int | None = "auto",
    complex_component: str = "real",
    channel_start: int | None = None,
    channel_end: int | None = None,
) -> JSONResponse:
    cube, cube_name, file_type = _load_visualizer_cube(path)
    display_cube = _project_complex_component(cube, complex_component=complex_component)
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
    return JSONResponse(
        {
            "image": normalized.tolist(),
            "stats": stats,
            "method": method,
        }
    )


@router.get("/files")
async def list_datacube_files(dir: Optional[str] = None) -> JSONResponse:
    """List supported visualizer products in the given directory."""
    base = os.path.realpath(str(settings.OUTPUT_DIR))
    if dir:
        full = os.path.realpath(os.path.join(base, dir))
        if full != base and not full.startswith(base + os.sep):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid directory",
            )
        output_dir = Path(full)
    else:
        output_dir = Path(base)

    if not output_dir.exists():  # lgtm[py/path-injection]
        return JSONResponse(
            {
                "files": [],
                "output_dir": str(output_dir),
                "message": "Output directory does not exist",
            }
        )

    files_info = []
    for file_path in _iter_supported_paths(output_dir):
        try:
            stat = file_path.stat()
            file_type = _classify_visualizer_path(file_path)
            if file_type is None:
                continue
            files_info.append(
                {
                    "name": file_path.name,
                    "path": str(file_path.relative_to(output_dir)),
                    "size": stat.st_size,
                    "modified": stat.st_mtime,
                    "type": file_type,
                }
            )
        except Exception:
            continue

    return JSONResponse(
        {
            "files": files_info,
            "output_dir": str(output_dir),
        }
    )


@router.get("/files/{file_path:path}")
async def get_datacube_file(file_path: str, dir: Optional[str] = None) -> FileResponse:
    """Get a visualizer file from the output directory."""
    base = os.path.realpath(str(settings.OUTPUT_DIR))
    sub = os.path.join(dir, file_path) if dir else file_path
    full = os.path.realpath(os.path.join(base, sub))
    if full != base and not full.startswith(base + os.sep):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file path",
        )
    resolved = Path(full)

    if not resolved.exists() or not resolved.is_file():  # lgtm[py/path-injection]
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found",
        )

    file_type = _classify_visualizer_path(resolved)
    if file_type not in {"npz", "fits"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .npz and FITS files can be downloaded directly",
        )

    return FileResponse(
        path=str(resolved),  # lgtm[py/path-injection]
        filename=resolved.name,
        media_type="application/octet-stream",
    )


@router.post("/integrate")
async def integrate_datacube(
    method: str = Form("sum"),
    integration_axis: str = Form("auto"),
    complex_component: str = Form("real"),
    channel_start: int | None = Form(default=None),
    channel_end: int | None = Form(default=None),
    file: UploadFile | None = File(default=None),
    server_path: str | None = Form(default=None),
    dir: str | None = Form(default=None),
) -> JSONResponse:
    """Load a supported visualizer product and produce a 2D inspection image."""
    if method not in ("sum", "mean"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Method must be 'sum' or 'mean'",
        )
    if file is None and not server_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide either an uploaded file or a server-side file path",
        )
    if file is not None and server_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Choose either uploaded file or server-side file path, not both",
        )

    tmp_path: str | None = None
    try:
        if server_path:
            base = os.path.realpath(str(settings.OUTPUT_DIR))
            sub = os.path.join(dir, server_path) if dir else server_path
            full = os.path.realpath(os.path.join(base, sub))
            if full != base and not full.startswith(base + os.sep):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid file path",
                )
            resolved = Path(full)
            if not resolved.exists():  # lgtm[py/path-injection]
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="File not found",
                )
            if _classify_visualizer_path(resolved) is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Only .npz, FITS, and .ms files are supported",
                )
            return _build_integrated_response(
                resolved,
                method,
                integration_axis=integration_axis,
                complex_component=complex_component,
                channel_start=channel_start,
                channel_end=channel_end,
            )

        assert file is not None
        filename = file.filename or "uploaded-product"
        lower_name = filename.lower()
        if lower_name.endswith(_MSV2_SUFFIXES):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="MSv2 directories must be selected from a server-side path",
            )
        if not lower_name.endswith(_NPZ_SUFFIXES + _FITS_SUFFIXES):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only .npz and FITS uploads are supported",
            )

        contents = await file.read()
        suffix = Path(filename).suffix or ".bin"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(contents)
            tmp_path = tmp_file.name
        return _build_integrated_response(
            Path(tmp_path),
            method,
            integration_axis=integration_axis,
            complex_component=complex_component,
            channel_start=channel_start,
            channel_end=channel_end,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process datacube: {exc}",
        ) from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
