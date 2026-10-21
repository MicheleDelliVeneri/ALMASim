"""Data classes for ALMA archive products and download progress. (MADIV-28 T6.9)"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DataProduct:
    """A single downloadable file from the ALMA archive."""

    access_url: str
    uid: str
    filename: str
    content_length: int
    content_type: str
    product_type: str
    semantics: str = ""

    @property
    def size_mb(self) -> float:
        return self.content_length / (1024 * 1024)


@dataclass
class FileDownloadStatus:
    """Status for a single downloaded file."""

    filename: str
    access_url: str
    content_length: int
    bytes_downloaded: int = 0
    status: str = "pending"
    error: str | None = None
    sha256: str | None = None


@dataclass
class DownloadSummary:
    """Summary returned by :func:`download_products`."""

    destination: str
    total_files: int
    total_bytes: int
    files_completed: int
    files_failed: int
    files: list[FileDownloadStatus] = field(default_factory=list)
    extracted_files: list[str] = field(default_factory=list)
    extraction_failed: list[str] = field(default_factory=list)
    """Tars that downloaded but could not be extracted; they are kept on disk."""
    raw_measurement_sets: list[str] = field(default_factory=list)
    calibrated_measurement_sets: list[str] = field(default_factory=list)
    manifest_path: str | None = None


PRODUCT_TYPES = {
    "all",
    "raw",
    "calibration",
    "scripts",
    "weblog",
    "qa_reports",
    "auxiliary",
    "cubes",
    "continuum",
    "fits",
    "other",
}


def format_bytes(size: int) -> str:
    """Return a human-friendly byte size string."""
    size_float = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size_float) < 1000:
            return f"{size_float:.1f} {unit}"
        size_float /= 1000.0
    return f"{size_float:.1f} PB"
