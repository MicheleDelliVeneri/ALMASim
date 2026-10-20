"""Schemas for the imaging preview endpoint (MADIV-28 T5.12)."""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class WienerPreviewRequest(BaseModel):
    """Request body for a Wiener preview of saved cube products (MADIV-28 T5.12)."""

    directory: str = Field(..., description="Base directory for relative file paths")
    dirty_cube_path: str = Field(..., description="Relative or absolute path to the dirty cube")
    beam_cube_path: str = Field(..., description="Relative or absolute path to the beam cube")
    clean_cube_path: str | None = Field(
        default=None,
        description="Optional reference clean/model cube for visual comparison",
    )
    epsilon: float = Field(
        default=1e-3,
        gt=0.0,
        le=1.0,
        description="Wiener regularisation: larger is smoother and less noisy",
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


class WienerPreviewMetadata(BaseModel):
    """Where a Wiener preview read and wrote its cubes."""

    dirty_cube_path: str
    beam_cube_path: str
    beam_cube_name: str
    clean_cube_path: Optional[str] = None
    epsilon: float
    model_cube_path: str
    restored_cube_path: str
    residual_cube_path: str
    convolved_reference_path: Optional[str] = None


class WienerPreviewResponse(BaseModel):
    """Frontend response payload for a Wiener preview."""

    dirty: ImagePreviewPayload
    model: ImagePreviewPayload
    restored: ImagePreviewPayload
    residual: ImagePreviewPayload
    reference_clean: ImagePreviewPayload | None = None
    convolved_reference: ImagePreviewPayload | None = None
    metadata: WienerPreviewMetadata
