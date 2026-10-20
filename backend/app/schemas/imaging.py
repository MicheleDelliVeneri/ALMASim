"""Schemas for imaging/deconvolution API endpoints."""

from typing import Literal, Optional

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


class PreviewStats(BaseModel):
    """Statistics of an integrated cube preview (``products.preview``)."""

    shape: list[int]
    integrated_shape: list[int]
    min: float
    max: float
    mean: float
    std: float
    cube_name: str


class ImagePreviewPayload(BaseModel):
    """Normalized 2D preview and statistics for a cube product."""

    image: list[list[int]]
    stats: PreviewStats
    method: str


class DeconvolutionMetadata(BaseModel):
    """Where a deconvolution run read and wrote its cubes, and how far it got."""

    dirty_cube_path: str
    beam_cube_path: str
    beam_cube_name: str
    clean_cube_path: Optional[str] = None
    cycles_requested: int
    cycles_completed: int
    cycles_added: int
    gain: float
    threshold: Optional[float] = None
    state_path: str
    component_cube_path: str
    restored_cube_path: str
    residual_cube_path: str
    convolved_reference_path: Optional[str] = None
    resumed: bool


class DeconvolutionResponse(BaseModel):
    """Frontend response payload for iterative deconvolution previews."""

    dirty: ImagePreviewPayload
    component_model: ImagePreviewPayload
    restored: ImagePreviewPayload
    residual: ImagePreviewPayload
    reference_clean: ImagePreviewPayload | None = None
    convolved_reference: ImagePreviewPayload | None = None
    metadata: DeconvolutionMetadata
