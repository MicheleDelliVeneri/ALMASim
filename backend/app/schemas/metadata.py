"""Metadata-related schemas."""

from dataclasses import dataclass
from typing import List, Optional, Sequence

from pydantic import BaseModel, Field

from almasim.services.metadata.filters import ExclusionFilters, InclusionFilters


@dataclass
class ScienceQuery:
    """A metadata query for the services: the shared filters and the columns to return."""

    include: InclusionFilters
    exclude: ExclusionFilters
    visible_columns: Optional[List[str]] = None


class MetadataQuery(BaseModel):
    """Metadata query parameters."""

    # --- Inclusion filters ---
    source_name: Optional[str] = Field(None, description="Source name (partial match)")
    science_keyword: Optional[Sequence[str]] = Field(
        None, description="Science keywords to include"
    )
    scientific_category: Optional[Sequence[str]] = Field(
        None, description="Scientific categories to include"
    )
    bands: Optional[Sequence[int]] = Field(None, description="Band numbers to include")
    antenna_arrays: Optional[str] = Field(
        None, description="Antenna array raw string filter (partial match)"
    )
    array_type: Optional[Sequence[str]] = Field(
        None,
        description=(
            "Array type(s) to include: '12m', '7m', 'TP' (matched via antenna prefix patterns)"
        ),
    )
    array_configuration: Optional[Sequence[str]] = Field(
        None,
        description=(
            "Array configuration(s) to include, e.g. 'C-1', 'C-2' (matched in schedblock_name)"
        ),
    )
    angular_resolution_range: Optional[tuple[float, float]] = Field(
        None, description="Angular resolution range (arcsec)"
    )
    observation_date_range: Optional[tuple[str, str]] = Field(
        None, description="Observation date range (ISO date strings)"
    )
    qa2_status: Optional[Sequence[str]] = Field(
        None,
        description="QA2 status values: 'Pass', 'Fail', 'SemiPass', or raw 'T'/'F'/'X'",
    )
    obs_type: Optional[Sequence[str]] = Field(
        None, description="Observation type(s) to include (partial match, OR logic)"
    )
    fov_range: Optional[tuple[float, float]] = Field(None, description="FOV range")
    time_resolution_range: Optional[tuple[float, float]] = Field(
        None, description="Time resolution range"
    )
    frequency_range: Optional[tuple[float, float]] = Field(None, description="Frequency range")

    # --- Exclusion filters ---
    exclude_science_keyword: Optional[Sequence[str]] = Field(
        None, description="Science keywords to exclude"
    )
    exclude_scientific_category: Optional[Sequence[str]] = Field(
        None, description="Scientific categories to exclude"
    )
    exclude_source_name: Optional[Sequence[str]] = Field(
        None, description="Source name substrings to exclude"
    )
    exclude_obs_type: Optional[Sequence[str]] = Field(
        None, description="Observation type substrings to exclude"
    )
    exclude_solar: bool = Field(
        False,
        description="Exclude solar observations (target/keyword/category contain 'sun')",
    )

    # --- Cycle / proposal_id filtering ---
    # Cycle N → year 2012+N → proposal_id prefix "{year}."
    # Pass cycle numbers (e.g. [4, 5, 6]) and they are converted to proposal_id prefixes.
    cycles: Optional[Sequence[int]] = Field(
        None,
        description=(
            "ALMA cycle numbers to include (e.g. [4, 5, 6]). "
            "Cycle N → proposal_id prefix '{2012+N}.'"
        ),
    )

    # --- Data rights ---
    public_only: bool = Field(True, description="Only query non-proprietary (public) data")

    # --- Observation type ---
    science_only: bool = Field(True, description="Only query science observations")
    exclude_mosaic: bool = Field(True, description="Exclude mosaic observations")

    # --- Output control ---
    visible_columns: Optional[List[str]] = Field(
        None,
        description=("Ordered subset of result columns to return. Pass null/omit for all columns."),
    )

    def to_filters(self) -> tuple[InclusionFilters, ExclusionFilters]:
        """The query as the filters the TAP and cache queries take (MADIV-28 T6.1)."""

        def listed(values):
            return list(values) if values else None

        include = InclusionFilters(
            science_keyword=listed(self.science_keyword),
            scientific_category=listed(self.scientific_category),
            band=listed(self.bands),
            fov_range=self.fov_range,
            time_resolution_range=self.time_resolution_range,
            frequency_range=self.frequency_range,
            source_name=self.source_name,
            antenna_arrays=self.antenna_arrays,
            array_type=listed(self.array_type),
            array_configuration=listed(self.array_configuration),
            angular_resolution_range=self.angular_resolution_range,
            observation_date_range=self.observation_date_range,
            qa2_status=listed(self.qa2_status),
            obs_type=listed(self.obs_type),
            # Cycle N is proposal_id prefix "{2012 + N}.".
            proposal_id_prefix=[f"{2012 + c}." for c in self.cycles] if self.cycles else None,
            public_only=self.public_only,
            science_only=self.science_only,
            exclude_mosaic=self.exclude_mosaic,
        )
        exclude = ExclusionFilters(
            science_keyword=listed(self.exclude_science_keyword),
            scientific_category=listed(self.exclude_scientific_category),
            source_name=listed(self.exclude_source_name),
            obs_type=listed(self.exclude_obs_type),
            solar=self.exclude_solar,
        )
        return include, exclude

    def to_query(self) -> ScienceQuery:
        """The query for the metadata service."""
        include, exclude = self.to_filters()
        return ScienceQuery(include, exclude, self.visible_columns)


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
    status: str = "running"
    error: Optional[str] = None


class MetadataSaveRequest(BaseModel):
    """Request payload for saving metadata on the backend."""

    path: str = Field(..., description="Target path relative to the ALMASim metadata directory")
    data: list[dict] = Field(..., description="Metadata records to persist")
    format: str = Field(
        default="json",
        description="Output format: 'json' or 'csv'.",
    )


class MetadataSaveResponse(BaseModel):
    """Response returned after saving metadata."""

    path: str = Field(..., description="Resolved path where the data was stored")
    count: int = Field(..., description="Number of records saved")
    message: Optional[str] = Field(default=None, description="Optional status message")


class QueryPresetCreate(BaseModel):
    """Request to save a named query preset."""

    name: str = Field(..., description="Unique preset name")
    description: str = Field("", description="Human-readable description")
    filters: dict = Field(..., description="MetadataQuery-compatible filter dict")
    result_count: int = Field(0, description="Number of results returned by this query")


class QueryPresetResponse(BaseModel):
    """A saved query preset."""

    name: str
    description: str
    filters: dict
    result_count: int
    created_at: str


class QueryPresetsResponse(BaseModel):
    """List of saved query presets."""

    presets: List[QueryPresetResponse]
