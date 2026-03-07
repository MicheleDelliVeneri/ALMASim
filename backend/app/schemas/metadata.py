"""Metadata-related schemas."""
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple
from pydantic import BaseModel, Field


@dataclass
class ScienceQueryParams:
    """Internal dataclass bundling all science query parameters for service calls."""

    # Inclusion
    source_name: Optional[str] = None
    science_keyword: Optional[List[str]] = None
    scientific_category: Optional[List[str]] = None
    bands: Optional[List[int]] = None
    antenna_arrays: Optional[str] = None
    array_type: Optional[List[str]] = None
    array_configuration: Optional[List[str]] = None
    angular_resolution_range: Optional[Tuple[float, float]] = None
    observation_date_range: Optional[Tuple[str, str]] = None
    qa2_status: Optional[List[str]] = None
    obs_type: Optional[str] = None
    fov_range: Optional[Tuple[float, float]] = None
    time_resolution_range: Optional[Tuple[float, float]] = None
    frequency_range: Optional[Tuple[float, float]] = None
    # Exclusion
    exclude_science_keyword: Optional[List[str]] = None
    exclude_scientific_category: Optional[List[str]] = None
    exclude_source_name: Optional[List[str]] = None
    exclude_obs_type: Optional[List[str]] = None
    exclude_solar: bool = False
    # Output
    visible_columns: Optional[List[str]] = None


class MetadataQuery(BaseModel):
    """Metadata query parameters."""

    # --- Inclusion filters ---
    source_name: Optional[str] = Field(None, description="Source name (partial match)")
    science_keyword: Optional[Sequence[str]] = Field(None, description="Science keywords to include")
    scientific_category: Optional[Sequence[str]] = Field(None, description="Scientific categories to include")
    bands: Optional[Sequence[int]] = Field(None, description="Band numbers to include")
    antenna_arrays: Optional[str] = Field(None, description="Antenna array raw string filter (partial match)")
    array_type: Optional[Sequence[str]] = Field(
        None,
        description="Array type(s) to include: '12m', '7m', 'TP' (matched via antenna prefix patterns)",
    )
    array_configuration: Optional[Sequence[str]] = Field(
        None,
        description="Array configuration(s) to include, e.g. 'C-1', 'C-2' (matched in schedblock_name)",
    )
    angular_resolution_range: Optional[tuple[float, float]] = Field(None, description="Angular resolution range (arcsec)")
    observation_date_range: Optional[tuple[str, str]] = Field(None, description="Observation date range (ISO date strings)")
    qa2_status: Optional[Sequence[str]] = Field(
        None,
        description="QA2 status values: 'Pass', 'Fail', 'SemiPass', or raw 'T'/'F'/'X'",
    )
    obs_type: Optional[str] = Field(None, description="Observation type to include (partial match)")
    fov_range: Optional[tuple[float, float]] = Field(None, description="FOV range")
    time_resolution_range: Optional[tuple[float, float]] = Field(None, description="Time resolution range")
    frequency_range: Optional[tuple[float, float]] = Field(None, description="Frequency range")

    # --- Exclusion filters ---
    exclude_science_keyword: Optional[Sequence[str]] = Field(None, description="Science keywords to exclude")
    exclude_scientific_category: Optional[Sequence[str]] = Field(None, description="Scientific categories to exclude")
    exclude_source_name: Optional[Sequence[str]] = Field(None, description="Source name substrings to exclude")
    exclude_obs_type: Optional[Sequence[str]] = Field(None, description="Observation type substrings to exclude")
    exclude_solar: bool = Field(False, description="Exclude solar observations (target/keyword/category contain 'sun')")

    # --- Output control ---
    visible_columns: Optional[List[str]] = Field(
        None,
        description="Ordered subset of result columns to return. Pass null/omit for all columns.",
    )

    def to_params(self) -> ScienceQueryParams:
        """Convert to the internal ScienceQueryParams dataclass."""
        return ScienceQueryParams(
            source_name=self.source_name,
            science_keyword=list(self.science_keyword) if self.science_keyword else None,
            scientific_category=list(self.scientific_category) if self.scientific_category else None,
            bands=list(self.bands) if self.bands else None,
            antenna_arrays=self.antenna_arrays,
            array_type=list(self.array_type) if self.array_type else None,
            array_configuration=list(self.array_configuration) if self.array_configuration else None,
            angular_resolution_range=self.angular_resolution_range,
            observation_date_range=self.observation_date_range,
            qa2_status=list(self.qa2_status) if self.qa2_status else None,
            obs_type=self.obs_type,
            fov_range=self.fov_range,
            time_resolution_range=self.time_resolution_range,
            frequency_range=self.frequency_range,
            exclude_science_keyword=list(self.exclude_science_keyword) if self.exclude_science_keyword else None,
            exclude_scientific_category=list(self.exclude_scientific_category) if self.exclude_scientific_category else None,
            exclude_source_name=list(self.exclude_source_name) if self.exclude_source_name else None,
            exclude_obs_type=list(self.exclude_obs_type) if self.exclude_obs_type else None,
            exclude_solar=self.exclude_solar,
            visible_columns=self.visible_columns,
        )


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

