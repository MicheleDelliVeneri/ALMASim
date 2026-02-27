"""Metadata-related schemas."""
from typing import Optional, Sequence
from pydantic import BaseModel, Field


class MetadataQuery(BaseModel):
    """Metadata query parameters."""

    source_name: Optional[str] = Field(None, description="Source name (partial match)")
    science_keyword: Optional[Sequence[str]] = Field(None, description="Science keywords")
    scientific_category: Optional[Sequence[str]] = Field(None, description="Scientific categories")
    bands: Optional[Sequence[int]] = Field(None, description="Band numbers")
    antenna_arrays: Optional[str] = Field(None, description="Antenna array configuration (partial match)")
    angular_resolution_range: Optional[tuple[float, float]] = Field(None, description="Angular resolution range (arcsec)")
    observation_date_range: Optional[tuple[str, str]] = Field(None, description="Observation date range (ISO date strings)")
    qa2_status: Optional[Sequence[str]] = Field(None, description="QA2 status values (e.g. 'T', 'F')")
    obs_type: Optional[str] = Field(None, description="Observation type (partial match)")
    fov_range: Optional[tuple[float, float]] = Field(None, description="FOV range")
    time_resolution_range: Optional[tuple[float, float]] = Field(None, description="Time resolution range")
    frequency_range: Optional[tuple[float, float]] = Field(None, description="Frequency range")


class MetadataResponse(BaseModel):
    """Metadata response."""

    count: int = Field(..., description="Number of records")
    data: list[dict] = Field(..., description="Metadata records")


class MetadataQueryStartResponse(BaseModel):
    """Response from starting a background TAP query job."""
    query_id: str
    status: str = Field("running", description="Job status: running | completed | failed")


class MetadataPageResponse(BaseModel):
    """One page of results from a background TAP query job."""
    query_id: str
    page: int
    rows: list[dict]
    page_size: int
    total_fetched: int
    done: bool
    error: Optional[str] = None


class MetadataSaveRequest(BaseModel):
    """Request payload for saving metadata on the backend."""

    path: str = Field(..., description="Target path relative to the ALMASim metadata directory")
    data: list[dict] = Field(..., description="Metadata records to persist")


class MetadataSaveResponse(BaseModel):
    """Response returned after saving metadata."""

    path: str = Field(..., description="Resolved path where the data was stored")
    count: int = Field(..., description="Number of records saved")
    message: Optional[str] = Field(default=None, description="Optional status message")

