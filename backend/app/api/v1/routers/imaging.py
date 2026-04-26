"""Imaging API endpoints for iterative deconvolution previews."""

from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException, status

from almasim.services.imaging import (
    clean_deconvolve_cube,
    convolve_cube_with_beam,
    integrate_cube_preview,
    load_cube_from_npz,
)
from app.core.config import settings
from app.core.path_utils import resolve_safe_path
from app.schemas.imaging import DeconvolutionRequest, DeconvolutionResponse

router = APIRouter()


def _resolve_cube_path(base_dir: Path, file_path: str) -> Path:
    """Resolve an absolute or relative file path under a base directory."""
    resolved = resolve_safe_path(file_path, base_dir, detail="Invalid file path")
    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File not found: {file_path}",
        )
    if resolved.suffix.lower() != ".npz":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .npz files are supported",
        )
    return resolved


def _derive_output_paths(dirty_path: Path) -> dict[str, Path]:
    """Derive deterministic state and preview output paths from the dirty cube name."""
    suffix = dirty_path.stem
    if suffix.startswith("dirty-cube_"):
        suffix = suffix[len("dirty-cube_") :]
    else:
        suffix = dirty_path.stem
    # dirty_path was resolved by _resolve_cube_path so parent is always absolute
    parent = dirty_path.parent
    return {
        "state": parent / f"deconvolution-state_{suffix}.npz",
        "component": parent / f"component-cube_{suffix}.npz",
        "restored": parent / f"restored-cube_{suffix}.npz",
        "residual": parent / f"residual-cube_{suffix}.npz",
        "convolved_reference": parent / f"convolved-clean-reference-cube_{suffix}.npz",
    }


@router.post("/deconvolve", response_model=DeconvolutionResponse)
async def deconvolve_saved_products(
    body: DeconvolutionRequest,
) -> DeconvolutionResponse:
    """Run a CLEAN-style deconvolution on saved dirty/beam cubes and return previews."""
    base_dir = resolve_safe_path(
        body.directory, settings.OUTPUT_DIR, detail="Invalid base directory"
    )
    if not base_dir.exists() or not base_dir.is_dir():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid base directory",
        )

    dirty_path = _resolve_cube_path(base_dir, body.dirty_cube_path)
    beam_path = _resolve_cube_path(base_dir, body.beam_cube_path)
    clean_path = (
        _resolve_cube_path(base_dir, body.clean_cube_path) if body.clean_cube_path else None
    )
    state_path = _resolve_cube_path(base_dir, body.state_path) if body.state_path else None
    output_paths = _derive_output_paths(dirty_path)

    try:
        dirty_cube, dirty_name = load_cube_from_npz(dirty_path)
        beam_cube, beam_name = load_cube_from_npz(beam_path)
        reference_clean = None
        convolved_reference = None
        clean_cube = None
        if clean_path is not None:
            clean_cube, clean_name = load_cube_from_npz(clean_path)
            reference_clean = integrate_cube_preview(
                clean_cube,
                method=body.method,
                cube_name=clean_name,
            )

        initial_component_cube = None
        initial_residual_cube = None
        initial_clean_beam_cube = None
        initial_cycles_completed = 0
        resumed = False
        if state_path is not None:
            with np.load(state_path) as state_data:
                initial_component_cube = np.asarray(state_data["component_cube"], dtype=np.float32)
                initial_residual_cube = np.asarray(state_data["residual_cube"], dtype=np.float32)
                initial_clean_beam_cube = np.asarray(
                    state_data["clean_beam_cube"], dtype=np.float32
                )
                initial_cycles_completed = int(state_data.get("cycles_completed", 0))
            resumed = True

        result = clean_deconvolve_cube(
            dirty_cube,
            beam_cube,
            n_cycles=body.cycles,
            gain=body.gain,
            threshold=body.threshold,
            initial_component_cube=initial_component_cube,
            initial_residual_cube=initial_residual_cube,
            initial_clean_beam_cube=initial_clean_beam_cube,
            initial_cycles_completed=initial_cycles_completed,
        )
        if clean_cube is not None:
            convolved_reference_cube = convolve_cube_with_beam(
                clean_cube,
                result["clean_beam_cube"],
            )
            convolved_reference = integrate_cube_preview(
                convolved_reference_cube,
                method=body.method,
                cube_name="convolved_clean_reference_cube",
            )
        else:
            convolved_reference_cube = None

        np.savez_compressed(
            output_paths["state"],
            component_cube=result["component_cube"],
            residual_cube=result["residual_cube"],
            clean_beam_cube=result["clean_beam_cube"],
            cycles_completed=result["cycles_completed"],
        )
        np.savez_compressed(output_paths["component"], component_cube=result["component_cube"])
        np.savez_compressed(output_paths["restored"], restored_cube=result["restored_cube"])
        np.savez_compressed(output_paths["residual"], residual_cube=result["residual_cube"])
        if convolved_reference_cube is not None:
            np.savez_compressed(
                output_paths["convolved_reference"],
                convolved_reference_cube=convolved_reference_cube,
            )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to deconvolve cube products: {exc}",
        ) from exc

    return DeconvolutionResponse(
        dirty=integrate_cube_preview(dirty_cube, method=body.method, cube_name=dirty_name),
        component_model=integrate_cube_preview(
            result["component_cube"],
            method=body.method,
            cube_name="component_model_cube",
        ),
        restored=integrate_cube_preview(
            result["restored_cube"],
            method=body.method,
            cube_name="restored_cube",
        ),
        residual=integrate_cube_preview(
            result["residual_cube"],
            method=body.method,
            cube_name="residual_cube",
        ),
        reference_clean=reference_clean,
        convolved_reference=convolved_reference,
        metadata={
            "dirty_cube_path": str(dirty_path),
            "beam_cube_path": str(beam_path),
            "beam_cube_name": beam_name,
            "clean_cube_path": None if clean_path is None else str(clean_path),
            "cycles_requested": body.cycles,
            "cycles_completed": result["cycles_completed"],
            "cycles_added": result["cycles_added"],
            "gain": result["gain"],
            "threshold": result["threshold"],
            "state_path": str(output_paths["state"]),
            "component_cube_path": str(output_paths["component"]),
            "restored_cube_path": str(output_paths["restored"]),
            "residual_cube_path": str(output_paths["residual"]),
            "convolved_reference_path": (
                None
                if convolved_reference_cube is None
                else str(output_paths["convolved_reference"])
            ),
            "resumed": resumed,
        },
    )
