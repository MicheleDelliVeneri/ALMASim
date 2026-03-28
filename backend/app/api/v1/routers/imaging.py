"""Imaging API endpoints for iterative deconvolution previews."""

from pathlib import Path

from fastapi import APIRouter, HTTPException, status

from almasim.services.imaging import (
    clean_deconvolve_cube,
    integrate_cube_preview,
    load_cube_from_npz,
)
from app.schemas.imaging import DeconvolutionRequest, DeconvolutionResponse

router = APIRouter()


def _resolve_cube_path(base_dir: Path, file_path: str) -> Path:
    """Resolve an absolute or relative file path under a base directory."""
    candidate = Path(file_path)
    resolved = candidate.resolve() if candidate.is_absolute() else (base_dir / candidate).resolve()
    try:
        resolved.relative_to(base_dir.resolve())
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file path",
        ) from exc
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


@router.post("/deconvolve", response_model=DeconvolutionResponse)
async def deconvolve_saved_products(body: DeconvolutionRequest) -> DeconvolutionResponse:
    """Run a CLEAN-style deconvolution on saved dirty/beam cubes and return previews."""
    base_dir = Path(body.directory).resolve()
    if not base_dir.exists() or not base_dir.is_dir():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid base directory",
        )

    dirty_path = _resolve_cube_path(base_dir, body.dirty_cube_path)
    beam_path = _resolve_cube_path(base_dir, body.beam_cube_path)
    clean_path = _resolve_cube_path(base_dir, body.clean_cube_path) if body.clean_cube_path else None

    try:
        dirty_cube, dirty_name = load_cube_from_npz(dirty_path)
        beam_cube, beam_name = load_cube_from_npz(beam_path)
        reference_clean = None
        if clean_path is not None:
            clean_cube, clean_name = load_cube_from_npz(clean_path)
            reference_clean = integrate_cube_preview(
                clean_cube,
                method=body.method,
                cube_name=clean_name,
            )

        result = clean_deconvolve_cube(
            dirty_cube,
            beam_cube,
            n_cycles=body.cycles,
            gain=body.gain,
            threshold=body.threshold,
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
        deconvolved=integrate_cube_preview(
            result["clean_cube"],
            method=body.method,
            cube_name="deconvolved_clean_cube",
        ),
        residual=integrate_cube_preview(
            result["residual_cube"],
            method=body.method,
            cube_name="residual_cube",
        ),
        reference_clean=reference_clean,
        metadata={
            "dirty_cube_path": str(dirty_path),
            "beam_cube_path": str(beam_path),
            "beam_cube_name": beam_name,
            "clean_cube_path": None if clean_path is None else str(clean_path),
            "cycles_requested": body.cycles,
            "cycles_completed": result["cycles_completed"],
            "gain": result["gain"],
            "threshold": result["threshold"],
        },
    )
