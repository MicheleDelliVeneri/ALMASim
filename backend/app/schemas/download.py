"""Download-related schemas."""

from typing import Any, List, Optional

from pydantic import BaseModel, Field

from almasim.services.download.jobs import json_list


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
        le=30,
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
        le=30,
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


def _fraction(done: float, total: float) -> float:
    return done / total if total > 0 else 0.0


def _job_fields(job: Any, *, live: bool) -> dict[str, Any]:
    """The fields the job schemas share, from a live job or its database record.

    A live job (``download_service.DownloadJob``) holds lists; its database
    record stores them as JSON text.
    """
    if live:
        rows, uids = job.metadata_rows, job.member_ous_uids
        raw, calibrated = job.raw_measurement_sets, job.calibrated_measurement_sets
    else:
        rows, uids = json_list(job.metadata_json), json_list(job.member_ous_uids)
        raw = json_list(job.raw_measurement_sets)
        calibrated = json_list(job.calibrated_measurement_sets)
    total, done = int(job.total_bytes), int(job.bytes_downloaded)
    return {
        "job_id": str(job.job_id),
        "status": job.status,
        "destination": job.destination,
        "total_files": job.total_files,
        "files_completed": job.files_completed,
        "files_failed": job.files_failed,
        "total_bytes": total,
        "bytes_downloaded": done,
        "progress": _fraction(done, total),
        "error": job.error,
        "member_ous_uids": list(uids),
        "product_filter": job.product_filter or "all",
        "created_at": job.created_at.isoformat(),
        "raw_measurement_sets": list(raw),
        "calibrated_measurement_sets": list(calibrated),
        "manifest_path": job.manifest_path,
        "has_metadata": bool(rows),
        "metadata_count": len(rows),
        "unpack_ms": bool(job.unpack_ms),
        "generate_calibrated_visibilities": bool(job.generate_calibrated_visibilities),
        "clean_intermediate_files": bool(job.clean_intermediate_files),
        "archive_output_root": job.archive_output_root,
        "casa_data_root": job.casa_data_root,
        "skip_casa_data_update": bool(job.skip_casa_data_update),
    }


class FileStatus(BaseModel):
    """Status of a single file in a download job."""

    filename: str
    content_length: int
    bytes_downloaded: int
    status: str
    error: Optional[str] = None
    progress: float = Field(..., description="Download progress 0.0 to 1.0")

    @classmethod
    def from_file(cls, file: Any) -> "FileStatus":
        """From a live file status or its database record."""
        length, done = int(file.content_length), int(file.bytes_downloaded)
        return cls(
            filename=file.filename,
            content_length=length,
            bytes_downloaded=done,
            status=file.status,
            error=file.error,
            progress=_fraction(done, length),
        )


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

    @classmethod
    def from_job(cls, job: Any) -> "DownloadJobStatus":
        """From a live (in-memory) job."""
        files = [FileStatus.from_file(f) for f in job.files]
        return cls.model_validate({**_job_fields(job, live=True), "files": files})

    @classmethod
    def from_record(cls, record: Any) -> "DownloadJobStatus":
        """From a job's database record."""
        files = [FileStatus.from_file(f) for f in record.files]
        return cls.model_validate({**_job_fields(record, live=False), "files": files})


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

    @classmethod
    def from_job(cls, job: Any) -> "DownloadJobSummary":
        """From a live (in-memory) job."""
        return cls.model_validate(_job_fields(job, live=True))

    @classmethod
    def from_record(cls, record: Any) -> "DownloadJobSummary":
        """From a job's database record."""
        return cls.model_validate(_job_fields(record, live=False))
