"""On-disk exporters for cube-shaped ALMASim products."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import astropy.units as U
import h5py
import numpy as np
from astropy.io import fits


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, U.Quantity):
        return {"value": value.value, "unit": str(value.unit)}
    if isinstance(value, Path):
        return str(value)
    return value


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
) -> None:
    """Persist an optional cube using ALMASim's standard save formats."""
    if cube is None:
        return
    output_dir = str(output_dir)
    cube = np.asarray(cube)
    if save_mode in ("npz", "both"):
        np.savez_compressed(Path(output_dir) / f"{stem}_{idx}.npz", cube)
    if save_mode == "h5":
        with h5py.File(Path(output_dir) / f"{stem}_{idx}.h5", "w") as handle:
            handle.create_dataset(stem.replace("-", "_"), data=cube)
    if save_mode in ("fits", "both"):
        fits.PrimaryHDU(header=header.copy(), data=cube).writeto(
            Path(output_dir) / f"{stem}_{idx}.fits",
            overwrite=True,
        )


def _write_fits_cube(
    path: Path,
    cube: np.ndarray,
    header: fits.Header | None = None,
) -> None:
    cube = np.asarray(cube)
    fits_header = header.copy() if header is not None else fits.Header()
    fits_header["DATAMAX"] = float(np.max(cube))
    fits_header["DATAMIN"] = float(np.min(cube))
    fits.PrimaryHDU(header=fits_header, data=cube).writeto(path, overwrite=True)


def _write_fits_complex_cube(
    output_dir: Path, stem: str, idx: int, cube: np.ndarray
) -> None:
    cube = np.asarray(cube)
    fits.PrimaryHDU(np.real(cube)).writeto(
        output_dir / f"{stem}_real{idx}.fits",
        overwrite=True,
    )
    fits.PrimaryHDU(np.imag(cube)).writeto(
        output_dir / f"{stem}_imag{idx}.fits",
        overwrite=True,
    )


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
    """Persist interferometric simulation products using ALMASim's standard formats."""
    output_dir = Path(output_dir)

    if save_mode in ("npz", "both"):
        np.savez_compressed(output_dir / f"clean-cube_{idx}.npz", model_cube)
        np.savez_compressed(output_dir / f"dirty-cube_{idx}.npz", dirty_cube)
        np.savez_compressed(output_dir / f"dirty-vis-cube_{idx}.npz", dirty_vis_cube)
        np.savez_compressed(output_dir / f"clean-vis-cube_{idx}.npz", vis_cube)
        np.savez_compressed(output_dir / f"beam-cube_{idx}.npz", beam_cube)
        np.savez_compressed(
            output_dir / f"totsampling-cube_{idx}.npz", totsampling_cube
        )
        np.savez_compressed(output_dir / f"uv-mask-cube_{idx}.npz", uv_mask_cube)
        np.savez_compressed(output_dir / f"u-cube_{idx}.npz", u_cube)
        np.savez_compressed(output_dir / f"v-cube_{idx}.npz", v_cube)
        if save_mode == "npz":
            return

    if save_mode == "h5":
        with h5py.File(output_dir / f"clean-cube_{idx}.h5", "w") as handle:
            handle.create_dataset("clean_cube", data=model_cube)
        with h5py.File(output_dir / f"dirty-cube_{idx}.h5", "w") as handle:
            handle.create_dataset("dirty_cube", data=dirty_cube)
        with h5py.File(output_dir / f"dirty-vis-cube_{idx}.h5", "w") as handle:
            handle.create_dataset("dirty_vis_cube", data=dirty_vis_cube)
        with h5py.File(output_dir / f"clean-vis-cube_{idx}.h5", "w") as handle:
            handle.create_dataset("clean_vis_cube", data=vis_cube)
        with h5py.File(output_dir / f"measurement-operator_{idx}.h5", "w") as handle:
            handle.create_dataset("beam_cube", data=beam_cube)
            handle.create_dataset("totsampling_cube", data=totsampling_cube)
            handle.create_dataset("uv_mask_cube", data=uv_mask_cube)
            handle.create_dataset("u_cube", data=u_cube)
            handle.create_dataset("v_cube", data=v_cube)
        return

    if save_mode in ("fits", "both"):
        _write_fits_cube(output_dir / f"clean-cube_{idx}.fits", model_cube, header)
        _write_fits_cube(output_dir / f"dirty-cube_{idx}.fits", dirty_cube, header)
        _write_fits_complex_cube(output_dir, "dirty-vis-cube", idx, dirty_vis_cube)
        _write_fits_complex_cube(output_dir, "clean-vis-cube", idx, vis_cube)
        fits.PrimaryHDU(beam_cube).writeto(
            output_dir / f"beam-cube_{idx}.fits",
            overwrite=True,
        )
        fits.PrimaryHDU(totsampling_cube).writeto(
            output_dir / f"totsampling-cube_{idx}.fits",
            overwrite=True,
        )
        fits.PrimaryHDU(uv_mask_cube).writeto(
            output_dir / f"uv-mask-cube_{idx}.fits",
            overwrite=True,
        )
        fits.PrimaryHDU(u_cube).writeto(
            output_dir / f"u-cube_{idx}.fits",
            overwrite=True,
        )
        fits.PrimaryHDU(v_cube).writeto(
            output_dir / f"v-cube_{idx}.fits",
            overwrite=True,
        )
        return

    if save_mode not in ("h5",):
        raise ValueError(
            f"Unsupported save_mode for interferometry products: {save_mode}"
        )
