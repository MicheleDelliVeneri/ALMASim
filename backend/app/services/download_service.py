"""Backend adapter for ALMA download workflows."""

import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from almasim.services.download import (
    DataProduct,
    FileDownloadStatus,
    download_products,
    filter_products,
    resolve_products,
)

logger = logging.getLogger(__name__)


@dataclass
class DownloadJob:
    """In-memory representation of an active download job."""

    job_id: str
    destination: str
    member_ous_uids: List[str] = field(default_factory=list)
    metadata_rows: List[dict] = field(default_factory=list)
    product_filter: str = "all"
    unpack_ms: bool = False
    generate_calibrated_visibilities: bool = False
    clean_intermediate_files: bool = False
    archive_output_root: Optional[str] = None
    casa_data_root: Optional[str] = None
    skip_casa_data_update: bool = False
    raw_measurement_sets: List[str] = field(default_factory=list)
    calibrated_measurement_sets: List[str] = field(default_factory=list)
    manifest_path: Optional[str] = None
    total_files: int = 0
    total_bytes: int = 0
    bytes_downloaded: int = 0
    files_completed: int = 0
    files_failed: int = 0
    status: str = "pending"
    files: List[FileDownloadStatus] = field(default_factory=list)
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class DownloadStore:
    """Thread-safe store mirroring active jobs in memory and the database."""

    def __init__(self):
        self._active = {}
        self._lock = threading.Lock()

    @staticmethod
    def _get_db():
        from app.db.config import new_session

        return new_session()

    def create(self, job: DownloadJob) -> DownloadJob:
        """Register a new job and persist the initial record to DB."""
        with self._lock:
            self._active[job.job_id] = job

        from app.db.models import DownloadJobRecord

        db = self._get_db()
        try:
            rec = DownloadJobRecord(
                job_id=job.job_id,
                destination=job.destination,
                member_ous_uids=json.dumps(job.member_ous_uids),
                metadata_json=json.dumps(job.metadata_rows),
                product_filter=job.product_filter,
                unpack_ms=job.unpack_ms,
                generate_calibrated_visibilities=job.generate_calibrated_visibilities,
                clean_intermediate_files=job.clean_intermediate_files,
                archive_output_root=job.archive_output_root,
                casa_data_root=job.casa_data_root,
                skip_casa_data_update=job.skip_casa_data_update,
                raw_measurement_sets=json.dumps(job.raw_measurement_sets),
                calibrated_measurement_sets=json.dumps(job.calibrated_measurement_sets),
                manifest_path=job.manifest_path,
                total_files=job.total_files,
                total_bytes=job.total_bytes,
                status=job.status,
            )
            db.add(rec)
            db.commit()
        finally:
            db.close()
        return job

    def get(self, job_id: str) -> Optional[DownloadJob]:
        """Return an active job from memory."""
        with self._lock:
            return self._active.get(job_id)

    def update(self, job_id: str, **kwargs) -> Optional[DownloadJob]:
        """Update an active job in memory."""
        with self._lock:
            job = self._active.get(job_id)
            if not job:
                return None
            for key, value in kwargs.items():
                if hasattr(job, key):
                    setattr(job, key, value)
            job.updated_at = datetime.now()
            return job

    def apply_file_update(self, job_id: str, file_status: FileDownloadStatus) -> None:
        """Record one file's progress and refresh the job's totals.

        Files are matched by filename: the transfer layer rewrites each
        access_url to spread downloads over the ALMA mirrors, so the URL is not
        a stable key (keying on it listed most files twice).
        """
        with self._lock:
            job = self._active.get(job_id)
            if not job:
                return
            for index, current in enumerate(job.files):
                if current.filename == file_status.filename:
                    job.files[index] = file_status
                    break
            else:
                job.files.append(file_status)
            job.bytes_downloaded = sum(f.bytes_downloaded for f in job.files)
            job.files_completed = sum(1 for f in job.files if f.status == "completed")
            job.files_failed = sum(1 for f in job.files if f.status == "failed")
            job.updated_at = datetime.now()

    def remove(self, job_id: str) -> None:
        """Drop a job from memory (its database record is kept)."""
        with self._lock:
            self._active.pop(job_id, None)

    def persist(self, job_id: str) -> None:
        """Flush the current in-memory state to the database."""
        with self._lock:
            job = self._active.get(job_id)
        if not job:
            return

        from app.db.models import DownloadFileRecord, DownloadJobRecord

        db = self._get_db()
        try:
            rec = db.query(DownloadJobRecord).filter(DownloadJobRecord.job_id == job_id).first()
            if not rec:
                return

            rec.total_files = job.total_files
            rec.total_bytes = job.total_bytes
            rec.bytes_downloaded = job.bytes_downloaded
            rec.files_completed = job.files_completed
            rec.files_failed = job.files_failed
            rec.status = job.status
            rec.error = job.error
            rec.raw_measurement_sets = json.dumps(job.raw_measurement_sets)
            rec.calibrated_measurement_sets = json.dumps(job.calibrated_measurement_sets)
            rec.manifest_path = job.manifest_path
            rec.updated_at = datetime.now()

            if job.files:
                db.query(DownloadFileRecord).filter(DownloadFileRecord.job_id == job_id).delete()
                for file_status in job.files:
                    db.add(
                        DownloadFileRecord(
                            job_id=job_id,
                            filename=file_status.filename,
                            access_url=file_status.access_url,
                            content_length=file_status.content_length,
                            bytes_downloaded=file_status.bytes_downloaded,
                            status=file_status.status,
                            error=file_status.error,
                            sha256=file_status.sha256,
                        )
                    )
            db.commit()
        finally:
            db.close()

    def finish(self, job_id: str) -> None:
        """Persist final state and remove the job from memory."""
        self.persist(job_id)
        self.remove(job_id)

    def list_all(self) -> list:
        """Return all persisted jobs from the database."""
        from app.db.models import DownloadJobRecord

        db = self._get_db()
        try:
            return db.query(DownloadJobRecord).order_by(DownloadJobRecord.created_at.desc()).all()
        finally:
            db.close()

    def get_from_db(self, job_id: str):
        """Return one persisted job, including file records."""
        from sqlalchemy.orm import joinedload

        from app.db.models import DownloadJobRecord

        db = self._get_db()
        try:
            return (
                db.query(DownloadJobRecord)
                .options(joinedload(DownloadJobRecord.files))
                .filter(DownloadJobRecord.job_id == job_id)
                .first()
            )
        finally:
            db.close()

    def delete_from_db(self, job_id: str) -> bool:
        """Delete one persisted job from the database."""
        from app.db.models import DownloadJobRecord

        db = self._get_db()
        try:
            rec = db.query(DownloadJobRecord).filter(DownloadJobRecord.job_id == job_id).first()
            if not rec:
                return False
            db.delete(rec)
            db.commit()
            return True
        finally:
            db.close()

    def update_in_db(self, job_id: str, **kwargs) -> bool:
        """Update a persisted job directly in the database."""
        from app.db.models import DownloadJobRecord

        db = self._get_db()
        try:
            rec = db.query(DownloadJobRecord).filter(DownloadJobRecord.job_id == job_id).first()
            if not rec:
                return False
            for key, value in kwargs.items():
                if hasattr(rec, key):
                    setattr(rec, key, value)
            rec.updated_at = datetime.now()
            db.commit()
            return True
        finally:
            db.close()


download_store = DownloadStore()


def run_download_job(
    job_id: str,
    products: List[DataProduct],
    destination: str,
    max_parallel: int = 3,
    rate_limit_sec: float = 0.5,
    extract_tar: bool = False,
    unpack_ms: bool = False,
    generate_calibrated_visibilities: bool = False,
    clean_intermediate_files: bool = False,
    archive_output_root: Optional[str] = None,
    casa_data_root: Optional[str] = None,
    skip_casa_data_update: bool = False,
) -> None:
    """Run a download job by delegating transfer work to shared services."""
    job = download_store.get(job_id)
    if not job:
        return

    file_statuses = [
        FileDownloadStatus(
            filename=product.filename,
            access_url=product.access_url,
            content_length=product.content_length,
        )
        for product in products
    ]
    total_bytes = sum(product.content_length for product in products)
    download_store.update(
        job_id,
        status="running",
        files=file_statuses,
        total_files=len(products),
        total_bytes=total_bytes,
    )
    download_store.persist(job_id)

    def should_cancel() -> bool:
        current_job = download_store.get(job_id)
        return current_job is None or current_job.status == "cancelled"

    def on_update(file_status: FileDownloadStatus) -> None:
        download_store.apply_file_update(job_id, file_status)

    try:
        summary = download_products(
            products,
            destination,
            max_parallel=max_parallel,
            rate_limit_sec=rate_limit_sec,
            extract_tar=extract_tar,
            unpack_ms=unpack_ms,
            generate_calibrated_visibilities=generate_calibrated_visibilities,
            clean_intermediate_files=clean_intermediate_files,
            archive_output_root=archive_output_root,
            casa_data_root=casa_data_root,
            skip_casa_data_update=skip_casa_data_update,
            should_cancel=should_cancel,
            update_callback=on_update,
        )
        current_job = download_store.get(job_id)
        if not current_job:
            return

        final: dict = {
            "files": summary.files,
            "bytes_downloaded": sum(file_status.bytes_downloaded for file_status in summary.files),
            "files_completed": summary.files_completed,
            "files_failed": summary.files_failed,
            "raw_measurement_sets": summary.raw_measurement_sets,
            "calibrated_measurement_sets": summary.calibrated_measurement_sets,
            "manifest_path": summary.manifest_path,
        }
        if current_job.status == "cancelled":
            pass
        elif summary.files_failed == 0:
            final.update(status="completed", error=None)
        elif summary.files_completed > 0:
            final.update(status="completed", error=f"{summary.files_failed} file(s) failed")
        else:
            final.update(status="failed", error="All downloads failed")
        # One update, under the store's lock, instead of attribute writes outside it.
        download_store.update(job_id, **final)
    except Exception as exc:
        logger.error("Download job %s failed: %s", job_id, exc, exc_info=True)
        download_store.update(job_id, status="failed", error=str(exc))
    finally:
        download_store.finish(job_id)


__all__ = [
    "DataProduct",
    "DownloadJob",
    "download_store",
    "filter_products",
    "resolve_products",
    "run_download_job",
]
