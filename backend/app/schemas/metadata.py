"""Metadata-related schemas."""
from typing import Optional, Sequence
from pydantic import BaseModel, Field


class MetadataQuery(BaseModel):
    """Metadata query parameters."""

    science_keyword: Optional[Sequence[str]] = Field(None, description="Science keywords")
    scientific_category: Optional[Sequence[str]] = Field(None, description="Scientific categories")
    bands: Optional[Sequence[int]] = Field(None, description="Band numbers")
    fov_range: Optional[tuple[float, float]] = Field(None, description="FOV range")
    time_resolution_range: Optional[tuple[float, float]] = Field(None, description="Time resolution range")
    frequency_range: Optional[tuple[float, float]] = Field(None, description="Frequency range")


class MetadataResponse(BaseModel):
    """Metadata response."""

    count: int = Field(..., description="Number of records")
    data: list[dict] = Field(..., description="Metadata records")


class MetadataSaveRequest(BaseModel):
    """Request payload for saving metadata on the backend."""

    path: str = Field(..., description="Target path relative to the ALMASim metadata directory")
    data: list[dict] = Field(..., description="Metadata records to persist")


class MetadataSaveResponse(BaseModel):
    """Response returned after saving metadata."""

    path: str = Field(..., description="Resolved path where the data was stored")
    count: int = Field(..., description="Number of records saved")
    message: Optional[str] = Field(default=None, description="Optional status message")

