"""Helpers for combining multiple interferometric configurations."""
from __future__ import annotations

from typing import Any
import numpy as np

from .utils import sampling_to_uv_mask
from .visibility import concatenate_visibility_tables


def _combine_baseline_cubes(cubes: list[np.ndarray]) -> np.ndarray:
    """Concatenate baseline cubes when their scan axis is compatible."""
    if not cubes:
        raise ValueError("No baseline cubes provided")
    first = cubes[0]
    if any(cube.ndim != 3 for cube in cubes):
        return first
    reference = first.shape
    if any(cube.shape[0] != reference[0] or cube.shape[2] != reference[2] for cube in cubes[1:]):
        return first
    return np.concatenate(cubes, axis=1)


def combine_interferometric_results(
    per_config_results: list[dict[str, Any]],
    *,
    config_weights: list[float] | None = None,
) -> dict[str, Any]:
    """Combine multiple single-pointing interferometric runs into one INT product."""
    if not per_config_results:
        raise ValueError("Cannot combine an empty result list")
    if len(per_config_results) == 1:
        combined = dict(per_config_results[0])
        combined["per_config_results"] = per_config_results
        combined["combined_config_count"] = 1
        return combined

    if config_weights is None:
        config_weights = [1.0] * len(per_config_results)
    weights = np.asarray(config_weights, dtype=float)
    weights = np.clip(weights, 0.0, None)
    if not np.any(weights):
        weights = np.ones_like(weights)
    normalized = weights / np.sum(weights)

    model_cube = np.tensordot(normalized, np.stack([r["model_cube"] for r in per_config_results], axis=0), axes=1)
    dirty_cube = np.tensordot(normalized, np.stack([r["dirty_cube"] for r in per_config_results], axis=0), axes=1)
    model_vis = np.sum(np.stack([r["model_vis"] for r in per_config_results], axis=0), axis=0)
    dirty_vis = np.sum(np.stack([r["dirty_vis"] for r in per_config_results], axis=0), axis=0)
    totsampling_cube = np.sum(np.stack([r["totsampling_cube"] for r in per_config_results], axis=0), axis=0)
    uv_mask_cube = sampling_to_uv_mask(totsampling_cube)

    beam_components = np.stack([r["beam_cube"] for r in per_config_results], axis=0)
    beam_cube = np.tensordot(normalized, beam_components, axes=1)
    center_x = beam_cube.shape[1] // 2
    center_y = beam_cube.shape[2] // 2
    normalization = np.clip(beam_cube[:, center_x, center_y][:, None, None], 1e-12, None)
    beam_cube = beam_cube / normalization

    u_cube = _combine_baseline_cubes([r["u_cube"] for r in per_config_results])
    v_cube = _combine_baseline_cubes([r["v_cube"] for r in per_config_results])

    combined = dict(per_config_results[0])
    combined.update(
        {
            "model_cube": model_cube,
            "dirty_cube": dirty_cube,
            "model_vis": model_vis,
            "dirty_vis": dirty_vis,
            "beam_cube": beam_cube,
            "totsampling_cube": totsampling_cube,
            "uv_mask_cube": uv_mask_cube,
            "u_cube": u_cube,
            "v_cube": v_cube,
            "beam": beam_cube[beam_cube.shape[0] // 2],
            "totsampling": totsampling_cube[totsampling_cube.shape[0] // 2],
            "uv_mask": uv_mask_cube[uv_mask_cube.shape[0] // 2],
            "u": u_cube[u_cube.shape[0] // 2] if getattr(u_cube, "ndim", 0) == 3 else u_cube,
            "v": v_cube[v_cube.shape[0] // 2] if getattr(v_cube, "ndim", 0) == 3 else v_cube,
            "per_config_results": per_config_results,
            "combined_config_count": len(per_config_results),
        }
    )
    visibility_tables = [
        r["visibility_table"]
        for r in per_config_results
        if "visibility_table" in r and r["visibility_table"] is not None
    ]
    if visibility_tables:
        combined["visibility_table"] = concatenate_visibility_tables(visibility_tables)
    return combined
