"""Imaging API: a Wiener preview of saved simulation cubes (MADIV-28 T5.12).

The Hogbom CLEAN endpoint was retired: WSClean on the simulated MeasurementSet
(``almasim image image-from-ms``) is the reference imaging.
"""

import os
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException, status

from almasim.services.imaging import convolve_cube_with_beam, wiener_preview
from almasim.services.products.preview import integrate_cube_preview, load_cube_from_npz
from app.core.config import settings
from app.core.path_utils import resolve_safe_path
from app.schemas.imaging import WienerPreviewRequest, WienerPreviewResponse

router = APIRouter()


def _resolve_cube_path(directory: str, file_path: str) -> Path:
    """Resolve an .npz cube file under OUTPUT_DIR, guarding against traversal."""
    sub = os.path.join(directory, file_path) if directory else file_path
    resolved = resolve_safe_path(sub, Path(settings.OUTPUT_DIR), detail="Invalid file path")
    if not resolved.exists() or not resolved.is_file():  # lgtm[py/path-injection]
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
        "model": parent / f"wiener-model-cube_{suffix}.npz",
        "restored": parent / f"restored-cube_{suffix}.npz",
        "residual": parent / f"residual-cube_{suffix}.npz",
        "convolved_reference": parent / f"convolved-clean-reference-cube_{suffix}.npz",
    }


@router.post("/preview", response_model=WienerPreviewResponse)
def wiener_preview_saved_products(
    body: WienerPreviewRequest,
) -> WienerPreviewResponse:
    """Wiener-deconvolve saved dirty/beam cubes and return previews of the results."""
    directory = body.directory or ""
    dirty_path = _resolve_cube_path(directory, body.dirty_cube_path)
    beam_path = _resolve_cube_path(directory, body.beam_cube_path)
    clean_path = (
        _resolve_cube_path(directory, body.clean_cube_path) if body.clean_cube_path else None
    )
    output_paths = _derive_output_paths(dirty_path)

    try:
        dirty_cube, dirty_name = load_cube_from_npz(dirty_path)
        beam_cube, beam_name = load_cube_from_npz(beam_path)
        result = wiener_preview(dirty_cube, beam_cube, epsilon=body.epsilon)
        reference_clean = None
        convolved_reference = None
        convolved_reference_cube = None
        if clean_path is not None:
            clean_cube, clean_name = load_cube_from_npz(clean_path)
            reference_clean = integrate_cube_preview(
                clean_cube, method=body.method, cube_name=clean_name
            )
            convolved_reference_cube = convolve_cube_with_beam(clean_cube, result["clean_beam"])
            convolved_reference = integrate_cube_preview(
                convolved_reference_cube,
                method=body.method,
                cube_name="convolved_clean_reference_cube",
            )
        np.savez_compressed(output_paths["model"], model_cube=result["model"])
        np.savez_compressed(output_paths["restored"], restored_cube=result["restored"])
        np.savez_compressed(output_paths["residual"], residual_cube=result["residual"])
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
            detail=f"Failed to preview cube products: {exc}",
        ) from exc

    def preview(cube: np.ndarray, name: str):
        return integrate_cube_preview(cube, method=body.method, cube_name=name)

    return WienerPreviewResponse(
        dirty=preview(dirty_cube, dirty_name),
        model=preview(result["model"], "wiener_model_cube"),
        restored=preview(result["restored"], "restored_cube"),
        residual=preview(result["residual"], "residual_cube"),
        reference_clean=reference_clean,
        convolved_reference=convolved_reference,
        metadata={
            "dirty_cube_path": str(dirty_path),
            "beam_cube_path": str(beam_path),
            "beam_cube_name": beam_name,
            "clean_cube_path": None if clean_path is None else str(clean_path),
            "epsilon": body.epsilon,
            "model_cube_path": str(output_paths["model"]),
            "restored_cube_path": str(output_paths["restored"]),
            "residual_cube_path": str(output_paths["residual"]),
            "convolved_reference_path": (
                None
                if convolved_reference_cube is None
                else str(output_paths["convolved_reference"])
            ),
        },
    )
