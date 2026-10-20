"""On-disk exporters for cube-shaped ALMASim products."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
from astropy.io import fits

from almasim.services.json_safe import json_safe as _json_safe

from ..imaging.fits_io import write_cube, write_fits_atomic


def write_ml_dataset_shard(
    output_path: Path | str,
    *,
    clean_cube: np.ndarray,
    dirty_cube: np.ndarray,
    dirty_vis: np.ndarray,
    uv_mask_cube: np.ndarray,
    metadata: Mapping[str, Any],
) -> str:
    """Persist a simulation sample as a single HDF5 shard for ML workflows."""
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as h5f:
        h5f.create_dataset("clean_cube", data=clean_cube, compression="gzip")
        h5f.create_dataset("dirty_cube", data=dirty_cube, compression="gzip")
        h5f.create_dataset("dirty_vis", data=dirty_vis, compression="gzip")
        h5f.create_dataset("uv_mask_cube", data=uv_mask_cube, compression="gzip")
        h5f.attrs["metadata_json"] = json.dumps(_json_safe(dict(metadata)))

    return str(output_path)


def save_optional_cube(
    output_dir: str | Path,
    stem: str,
    idx: int,
    cube: np.ndarray | None,
    save_mode: str,
    header: Any,
) -> list[str]:
    """Persist an optional cube using ALMASim's standard save formats."""
    if cube is None:
        return []
    output_dir_path = Path(output_dir)
    cube = np.asarray(cube)
    saved_paths: list[str] = []
    if save_mode in ("npz", "both"):
        npz_path = output_dir_path / f"{stem}_{idx}.npz"
        np.savez_compressed(npz_path, cube)
        saved_paths.append(str(npz_path))
    if save_mode == "h5":
        h5_path = output_dir_path / f"{stem}_{idx}.h5"
        with h5py.File(h5_path, "w") as handle:
            handle.create_dataset(stem.replace("-", "_"), data=cube)
        saved_paths.append(str(h5_path))
    if save_mode in ("fits", "both"):
        fits_path = output_dir_path / f"{stem}_{idx}.fits"
        write_cube(fits_path, cube, header)
        saved_paths.append(str(fits_path))
    return saved_paths


def _write_fits_cube(
    path: Path,
    cube: np.ndarray,
    header: fits.Header | None = None,
) -> None:
    cube = np.asarray(cube)
    fits_header = header.copy() if header is not None else fits.Header()
    fits_header["DATAMAX"] = float(np.max(cube))
    fits_header["DATAMIN"] = float(np.min(cube))
    write_cube(path, cube, fits_header)


def _write_fits_complex_cube(output_dir: Path, stem: str, idx: int, cube: np.ndarray) -> None:
    cube = np.asarray(cube)
    write_fits_atomic(output_dir / f"{stem}_real{idx}.fits", [fits.PrimaryHDU(np.real(cube))])
    write_fits_atomic(output_dir / f"{stem}_imag{idx}.fits", [fits.PrimaryHDU(np.imag(cube))])


# Kinds of interferometry product (write_interferometry_products).
_IMAGE = "image"
_VISIBILITIES = "visibilities"
_OPERATOR = "operator"


def write_interferometry_products(
    output_dir: str | Path,
    *,
    idx: int,
    save_mode: str,
    header: fits.Header | None,
    model_cube: np.ndarray,
    vis_cube: np.ndarray,
    dirty_cube: np.ndarray,
    dirty_vis_cube: np.ndarray,
    beam_cube: np.ndarray,
    totsampling_cube: np.ndarray,
    uv_mask_cube: np.ndarray,
    u_cube: np.ndarray,
    v_cube: np.ndarray,
) -> None:
    """Persist interferometric simulation products using ALMASim's standard formats.

    Each product is a ``(stem, cube, kind)`` row: images get the sky header,
    visibilities are written as real and imaginary FITS, and the measurement
    operator's cubes share one HDF5 file.
    """
    if save_mode not in ("npz", "h5", "fits", "both"):
        raise ValueError(f"Unsupported save_mode for interferometry products: {save_mode}")
    output_dir = Path(output_dir)
    products = [
        ("clean-cube", model_cube, _IMAGE),
        ("dirty-cube", dirty_cube, _IMAGE),
        ("dirty-vis-cube", dirty_vis_cube, _VISIBILITIES),
        ("clean-vis-cube", vis_cube, _VISIBILITIES),
        ("beam-cube", beam_cube, _OPERATOR),
        ("totsampling-cube", totsampling_cube, _OPERATOR),
        ("uv-mask-cube", uv_mask_cube, _OPERATOR),
        ("u-cube", u_cube, _OPERATOR),
        ("v-cube", v_cube, _OPERATOR),
    ]

    if save_mode in ("npz", "both"):
        for stem, cube, _ in products:
            np.savez_compressed(output_dir / f"{stem}_{idx}.npz", cube)

    if save_mode == "h5":
        for stem, cube, kind in products:
            if kind != _OPERATOR:
                with h5py.File(output_dir / f"{stem}_{idx}.h5", "w") as handle:
                    handle.create_dataset(stem.replace("-", "_"), data=cube)
        with h5py.File(output_dir / f"measurement-operator_{idx}.h5", "w") as handle:
            for stem, cube, kind in products:
                if kind == _OPERATOR:
                    handle.create_dataset(stem.replace("-", "_"), data=cube)

    if save_mode in ("fits", "both"):
        for stem, cube, kind in products:
            if kind == _IMAGE:
                _write_fits_cube(output_dir / f"{stem}_{idx}.fits", cube, header)
            elif kind == _VISIBILITIES:
                _write_fits_complex_cube(output_dir, stem, idx, cube)
            else:
                write_fits_atomic(output_dir / f"{stem}_{idx}.fits", [fits.PrimaryHDU(cube)])
