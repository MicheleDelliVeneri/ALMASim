"""Schemas for imaging/deconvolution API endpoints."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class DeconvolutionRequest(BaseModel):
    """Request body for iterative deconvolution from saved cube products."""

    directory: str = Field(..., description="Base directory for relative file paths")
    dirty_cube_path: str = Field(..., description="Relative or absolute path to the dirty cube")
    beam_cube_path: str = Field(..., description="Relative or absolute path to the beam cube")
    clean_cube_path: str | None = Field(
        default=None,
        description="Optional reference clean/model cube for visual comparison",
    )
    cycles: int = Field(default=100, ge=0, le=5000, description="Number of CLEAN cycles")
    gain: float = Field(default=0.1, gt=0.0, le=1.0, description="Loop gain")
    threshold: float | None = Field(
        default=None,
        ge=0.0,
        description="Optional absolute stopping threshold on the residual peak",
    )
    state_path: str | None = Field(
        default=None,
        description=("Optional deconvolution state NPZ path for resuming with additional cycles"),
    )
    method: Literal["sum", "mean"] = Field(
        default="sum",
        description="Integration method for the 2D preview images",
    )


class ImagePreviewPayload(BaseModel):
    """Normalized 2D preview and statistics for a cube product."""

    image: list[list[int]]
    stats: dict[str, Any]
    method: str


class DeconvolutionResponse(BaseModel):
    """Frontend response payload for iterative deconvolution previews."""

    dirty: ImagePreviewPayload
    component_model: ImagePreviewPayload
    restored: ImagePreviewPayload
    residual: ImagePreviewPayload
    reference_clean: ImagePreviewPayload | None = None
    convolved_reference: ImagePreviewPayload | None = None
    metadata: dict[str, Any]
