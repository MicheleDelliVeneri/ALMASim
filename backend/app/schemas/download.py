"""Download-related schemas."""

from typing import Any, List, Optional

from pydantic import BaseModel, Field


class DataProductInfo(BaseModel):
    """A single downloadable data product."""

    access_url: str
    uid: str = Field(..., description="member_ous_uid")
    filename: str
    content_length: int = Field(..., description="Size in bytes")
    content_type: str = ""
    product_type: str = Field(
        ...,
        description=(
            "Classified type: all, raw, calibration, scripts, weblog, qa_reports, "
            "auxiliary, cubes, continuum, fits, other"
        ),
    )
    size_mb: float = Field(..., description="Size in MB")


class BrowseDirectoryEntry(BaseModel):
    """A single entry in a directory listing."""

    name: str
    path: str
    is_dir: bool


class BrowseDirectoryResponse(BaseModel):
    """Response for browsing server-side directories."""

    current: str = Field(..., description="Resolved absolute path of the directory")
    parent: Optional[str] = Field(None, description="Parent directory path, None if at root")
    entries: List[BrowseDirectoryEntry]


class ResolveProductsRequest(BaseModel):
    """Request to list downloadable products for selected observations."""

    member_ous_uids: List[str] = Field(
        ..., description="List of member_ous_uid values to resolve products for"
    )


class ResolveProductsResponse(BaseModel):
    """Response listing available products and size breakdown."""

    products: List[DataProductInfo]
    total_count: int
    total_size_bytes: int
    total_size_display: str = Field(..., description="Human-friendly size string")
    by_type: dict = Field(
        default_factory=dict,
        description="Breakdown: {type: {count, size_bytes, size_display}}",
    )


class DiskSpaceInfo(BaseModel):
    """Disk space info for a given path."""

    path: str
    total_bytes: int
    used_bytes: int
    free_bytes: int
    total_display: str
    used_display: str
    free_display: str
    sufficient: bool = Field(..., description="Whether free space >= needed bytes")


class CheckDiskSpaceRequest(BaseModel):
    """Request to check disk space for a destination."""

    path: str = Field(..., description="Destination directory path")
    needed_bytes: int = Field(0, description="Bytes that will be downloaded")


class StartDownloadRequest(BaseModel):
    """Request to start downloading selected products."""

    member_ous_uids: List[str] = Field(..., description="List of member_ous_uid values")
    product_filter: str = Field(
        "all",
        description=(
            "Product type filter: all, raw, calibration, scripts, weblog, qa_reports, "
            "auxiliary, cubes, continuum, fits, other"
        ),
    )
    destination: str = Field(..., description="Local directory to save files")
    max_parallel: int = Field(
        3,
        description="Maximum parallel download threads",
        ge=1,
        le=8,
    )
    extract_tar: bool = Field(
        False,
        description="Extract tar archives after download and remove the archives",
    )
    unpack_ms: bool = Field(
        False,
        description=("Create raw MeasurementSets from downloaded/extracted ALMA ASDM products"),
    )
    generate_calibrated_visibilities: bool = Field(
        False,
        description=(
            "Create calibrated split MeasurementSets from raw MS and delivered calibration scripts"
        ),
    )
    clean_intermediate_files: bool = Field(
        False,
        description=(
            "After calibrated MS creation, delete downloaded originals "
            "and intermediate raw MS products"
        ),
    )
    archive_output_root: Optional[str] = Field(
        None,
        description="Optional root directory for raw_ms and calibrated_ms outputs",
    )
    casa_data_root: Optional[str] = Field(
        None,
        description="Optional CASA runtime data directory",
    )
    skip_casa_data_update: bool = Field(
        False,
        description="Skip CASA runtime data updates",
    )
    selected_metadata: Optional[List[dict[str, Any]]] = Field(
        None,
        description="Metadata rows associated with this download job",
    )


class RedownloadRequest(BaseModel):
    """Request to re-download a previous job with new execution options."""

    max_parallel: Optional[int] = Field(
        None,
        description="Maximum parallel download threads",
        ge=1,
        le=8,
    )
    extract_tar: Optional[bool] = Field(
        None,
        description="Extract tar archives after download and remove the archives",
    )
    unpack_ms: Optional[bool] = Field(
        None,
        description="Create raw MeasurementSets after re-download",
    )
    generate_calibrated_visibilities: Optional[bool] = Field(
        None,
        description="Create calibrated split MeasurementSets after re-download",
    )
    clean_intermediate_files: Optional[bool] = Field(
        None,
        description="After calibrated MS creation, keep only split calibrated products",
    )
    archive_output_root: Optional[str] = Field(
        None,
        description="Optional root directory for raw_ms and calibrated_ms outputs",
    )
    casa_data_root: Optional[str] = Field(
        None,
        description="Optional CASA runtime data directory",
    )
    skip_casa_data_update: Optional[bool] = Field(
        None,
        description="Skip CASA runtime data updates",
    )


class StartDownloadResponse(BaseModel):
    """Response after initiating a download job."""

    job_id: str
    status: str
    total_files: int
    total_bytes: int
    total_size_display: str
    destination: str


class FileStatus(BaseModel):
    """Status of a single file in a download job."""

    filename: str
    content_length: int
    bytes_downloaded: int
    status: str
    error: Optional[str] = None
    progress: float = Field(..., description="Download progress 0.0 to 1.0")


class DownloadJobStatus(BaseModel):
    """Status of an entire download job."""

    job_id: str
    status: str
    destination: str
    total_files: int
    files_completed: int
    files_failed: int
    total_bytes: int
    bytes_downloaded: int
    progress: float = Field(..., description="Overall progress 0.0 to 1.0")
    error: Optional[str] = None
    files: List[FileStatus] = []
    raw_measurement_sets: List[str] = []
    calibrated_measurement_sets: List[str] = []
    manifest_path: Optional[str] = None
    has_metadata: bool = False
    metadata_count: int = 0
    unpack_ms: bool = False
    generate_calibrated_visibilities: bool = False
    clean_intermediate_files: bool = False
    archive_output_root: Optional[str] = None
    casa_data_root: Optional[str] = None
    skip_casa_data_update: bool = False


class DownloadJobSummary(BaseModel):
    """Summary of a download job for listing."""

    job_id: str
    status: str
    destination: str
    total_files: int
    files_completed: int
    files_failed: int
    progress: float
    created_at: str
    member_ous_uids: List[str] = []
    product_filter: str = "all"
    total_bytes: int = 0
    bytes_downloaded: int = 0
    error: Optional[str] = None
    raw_measurement_sets: List[str] = []
    calibrated_measurement_sets: List[str] = []
    manifest_path: Optional[str] = None
    has_metadata: bool = False
    metadata_count: int = 0
    unpack_ms: bool = False
    generate_calibrated_visibilities: bool = False
    clean_intermediate_files: bool = False
    archive_output_root: Optional[str] = None
    casa_data_root: Optional[str] = None
    skip_casa_data_update: bool = False
