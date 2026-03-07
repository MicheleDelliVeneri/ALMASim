"""Download service for fetching ALMA data products via DataLink."""

import logging
import os
import shutil
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from xml.etree import ElementTree

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

# ALMA DataLink mirrors
_DATALINK_MIRRORS = [
    "https://almascience.eso.org",
    "https://almascience.nrao.edu",
    "https://almascience.nao.ac.jp",
]

# VOTable XML namespace
_VOT_NS = "http://www.ivoa.net/xml/VOTable/v1.3"


# ---------------------------------------------------------------------------
# DataLink resolution (replaces astroquery.alma.get_data_info)
# ---------------------------------------------------------------------------

@dataclass
class DataProduct:
    """A single downloadable file from the ALMA archive."""

    access_url: str
    uid: str  # member_ous_uid
    filename: str
    content_length: int  # bytes
    content_type: str
    product_type: str  # "raw", "fits", "auxiliary", "weblog", "other"

    @property
    def size_mb(self) -> float:
        return self.content_length / (1024 * 1024)


def _classify_product(url: str, content_type: str) -> str:
    """Classify a product URL into a human-readable type."""
    lower = url.lower()
    if lower.endswith(".fits"):
        return "fits"
    if "asdm" in lower or lower.endswith(".asdm.sdm.tar"):
        return "raw"
    if "weblog" in lower:
        return "weblog"
    if "auxiliary" in lower or "readme" in lower:
        return "auxiliary"
    if lower.endswith(".tar"):
        # Tar files that aren't asdm are usually pipeline products
        return "fits"
    return "other"


def _parse_votable_results(xml_bytes: bytes, uid: str) -> List[DataProduct]:
    """Parse a VOTable DataLink response into DataProduct list."""
    products: List[DataProduct] = []
    try:
        root = ElementTree.fromstring(xml_bytes)
    except ElementTree.ParseError:
        logger.warning("Failed to parse VOTable XML for uid=%s", uid)
        return products

    # Find TABLEDATA rows.  VOTable can use namespace or not.
    for ns_prefix in [f"{{{_VOT_NS}}}", ""]:
        table = root.find(f".//{ns_prefix}TABLEDATA")
        if table is not None:
            # Discover column order from FIELD elements
            resource = root.find(f".//{ns_prefix}RESOURCE")
            tbl = resource.find(f".//{ns_prefix}TABLE") if resource is not None else None
            fields = tbl.findall(f"{ns_prefix}FIELD") if tbl is not None else []
            col_names = [f.get("name", "").lower() for f in fields]

            def col_idx(name: str) -> int:
                try:
                    return col_names.index(name.lower())
                except ValueError:
                    return -1

            i_url = col_idx("access_url")
            i_size = col_idx("content_length")
            i_type = col_idx("content_type")
            i_id = col_idx("ID")

            for tr in table.findall(f"{ns_prefix}TR"):
                tds = tr.findall(f"{ns_prefix}TD")
                if i_url < 0 or i_url >= len(tds):
                    continue
                url = (tds[i_url].text or "").strip()
                if not url:
                    continue
                size_str = tds[i_size].text if i_size >= 0 and i_size < len(tds) else "0"
                ctype = tds[i_type].text if i_type >= 0 and i_type < len(tds) else ""
                row_uid = tds[i_id].text if i_id >= 0 and i_id < len(tds) else uid

                try:
                    content_length = int(size_str or 0)
                except (ValueError, TypeError):
                    content_length = 0

                filename = url.rsplit("/", 1)[-1] if "/" in url else url

                products.append(DataProduct(
                    access_url=url,
                    uid=row_uid or uid,
                    filename=filename,
                    content_length=content_length,
                    content_type=ctype or "",
                    product_type=_classify_product(url, ctype or ""),
                ))
            break  # found the data, no need to try other namespace

    return products


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _fetch_datalink(uid: str, mirror_base: str = _DATALINK_MIRRORS[0]) -> List[DataProduct]:
    """Fetch the DataLink service for a single member_ous_uid."""
    url = f"{mirror_base}/datalink/sync?ID={uid}"
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
    return _parse_votable_results(resp.content, uid)


def resolve_products(
    member_ous_uids: List[str],
    mirror: str = _DATALINK_MIRRORS[0],
) -> List[DataProduct]:
    """Resolve all data products for a list of member OUS UIDs via DataLink.

    Tries the primary mirror; falls back to alternates on failure.
    """
    all_products: List[DataProduct] = []
    for uid in member_ous_uids:
        last_err: Optional[Exception] = None
        for base in [mirror] + [m for m in _DATALINK_MIRRORS if m != mirror]:
            try:
                products = _fetch_datalink(uid, base)
                all_products.extend(products)
                last_err = None
                break
            except Exception as e:
                last_err = e
                logger.warning("DataLink fetch failed for %s on %s: %s", uid, base, e)
        if last_err:
            logger.error("All mirrors failed for %s: %s", uid, last_err)
    return all_products


def filter_products(
    products: List[DataProduct],
    product_filter: str = "all",
) -> List[DataProduct]:
    """Filter products by type.

    product_filter: "all", "fits", "raw", "auxiliary", "weblog"
    """
    if product_filter == "all":
        return products
    return [p for p in products if p.product_type == product_filter]


# ---------------------------------------------------------------------------
# Download job management
# ---------------------------------------------------------------------------

@dataclass
class FileDownloadStatus:
    """Status of a single file download (in-memory during active download)."""

    filename: str
    access_url: str
    content_length: int
    bytes_downloaded: int = 0
    status: str = "pending"  # pending, downloading, completed, failed
    error: Optional[str] = None


@dataclass
class DownloadJob:
    """In-memory representation of a download job (used during active execution)."""

    job_id: str
    destination: str
    member_ous_uids: List[str] = field(default_factory=list)
    product_filter: str = "all"
    total_files: int = 0
    total_bytes: int = 0
    bytes_downloaded: int = 0
    files_completed: int = 0
    files_failed: int = 0
    status: str = "pending"  # pending, running, completed, failed, cancelled
    files: List[FileDownloadStatus] = field(default_factory=list)
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class DownloadStore:
    """Thread-safe store that mirrors active jobs in-memory and persists to DB."""

    def __init__(self):
        self._active: Dict[str, DownloadJob] = {}
        self._lock = threading.Lock()

    # --- DB helpers ---
    @staticmethod
    def _get_db():
        from database.config import SessionLocal
        return SessionLocal()

    # --- Active-job operations (in-memory, for running downloads) ---

    def create(self, job: DownloadJob) -> DownloadJob:
        """Register a new job in memory and persist the initial record to DB."""
        with self._lock:
            self._active[job.job_id] = job
        # Persist to DB
        from database.models import DownloadJobRecord
        import json
        db = self._get_db()
        try:
            rec = DownloadJobRecord(
                job_id=job.job_id,
                destination=job.destination,
                member_ous_uids=json.dumps(job.member_ous_uids),
                product_filter=job.product_filter,
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
        """Get an active (in-memory) job."""
        with self._lock:
            return self._active.get(job_id)

    def update(self, job_id: str, **kwargs) -> Optional[DownloadJob]:
        """Update an active job in memory."""
        with self._lock:
            job = self._active.get(job_id)
            if not job:
                return None
            for k, v in kwargs.items():
                if hasattr(job, k):
                    setattr(job, k, v)
            job.updated_at = datetime.now()
            return job

    def persist(self, job_id: str) -> None:
        """Flush current in-memory state of a job to the database."""
        with self._lock:
            job = self._active.get(job_id)
        if not job:
            return
        from database.models import DownloadJobRecord, DownloadFileRecord
        db = self._get_db()
        try:
            rec = db.query(DownloadJobRecord).filter(
                DownloadJobRecord.job_id == job_id
            ).first()
            if not rec:
                return
            rec.total_files = job.total_files
            rec.total_bytes = job.total_bytes
            rec.bytes_downloaded = job.bytes_downloaded
            rec.files_completed = job.files_completed
            rec.files_failed = job.files_failed
            rec.status = job.status
            rec.error = job.error
            rec.updated_at = datetime.now()

            # Sync per-file records
            if job.files:
                # Remove old file records and insert fresh ones
                db.query(DownloadFileRecord).filter(
                    DownloadFileRecord.job_id == job_id
                ).delete()
                for fs in job.files:
                    db.add(DownloadFileRecord(
                        job_id=job_id,
                        filename=fs.filename,
                        access_url=fs.access_url,
                        content_length=fs.content_length,
                        bytes_downloaded=fs.bytes_downloaded,
                        status=fs.status,
                        error=fs.error,
                    ))
            db.commit()
        finally:
            db.close()

    def finish(self, job_id: str) -> None:
        """Persist final state and remove from active memory."""
        self.persist(job_id)
        with self._lock:
            self._active.pop(job_id, None)

    # --- History queries (from DB) ---

    def list_all(self) -> list:
        """Return all jobs from DB (newest first)."""
        from database.models import DownloadJobRecord
        db = self._get_db()
        try:
            return db.query(DownloadJobRecord).order_by(
                DownloadJobRecord.created_at.desc()
            ).all()
        finally:
            db.close()

    def get_from_db(self, job_id: str):
        """Get a single job from DB (for history / completed jobs)."""
        from sqlalchemy.orm import joinedload
        from database.models import DownloadJobRecord
        db = self._get_db()
        try:
            return db.query(DownloadJobRecord).options(
                joinedload(DownloadJobRecord.files)
            ).filter(
                DownloadJobRecord.job_id == job_id
            ).first()
        finally:
            db.close()

    def delete_from_db(self, job_id: str) -> bool:
        """Delete a job and its file records from the database."""
        from database.models import DownloadJobRecord
        db = self._get_db()
        try:
            rec = db.query(DownloadJobRecord).filter(
                DownloadJobRecord.job_id == job_id
            ).first()
            if not rec:
                return False
            db.delete(rec)  # cascade deletes file records
            db.commit()
            return True
        finally:
            db.close()

    def update_in_db(self, job_id: str, **kwargs) -> bool:
        """Update a job directly in the database (for jobs no longer in memory)."""
        from database.models import DownloadJobRecord
        db = self._get_db()
        try:
            rec = db.query(DownloadJobRecord).filter(
                DownloadJobRecord.job_id == job_id
            ).first()
            if not rec:
                return False
            for k, v in kwargs.items():
                if hasattr(rec, k):
                    setattr(rec, k, v)
            rec.updated_at = datetime.now()
            db.commit()
            return True
        finally:
            db.close()


# Singleton store
download_store = DownloadStore()


# ---------------------------------------------------------------------------
# Actual download execution (runs in background thread)
# ---------------------------------------------------------------------------

_CHUNK_SIZE = 256 * 1024  # 256 KB


def _check_disk_space(path: str, needed_bytes: int) -> bool:
    """Check that destination has enough free space."""
    try:
        usage = shutil.disk_usage(path)
        return usage.free >= needed_bytes
    except OSError:
        return True  # if we can't check, proceed anyway


def _download_single_file(
    file_status: FileDownloadStatus,
    dest_dir: Path,
    job: DownloadJob,
) -> None:
    """Download a single file, updating status as we go."""
    file_status.status = "downloading"
    dest_path = dest_dir / file_status.filename

    try:
        with httpx.Client(timeout=300, follow_redirects=True) as client:
            with client.stream("GET", file_status.access_url) as resp:
                resp.raise_for_status()
                with open(dest_path, "wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=_CHUNK_SIZE):
                        if job.status == "cancelled":
                            file_status.status = "cancelled"
                            return
                        f.write(chunk)
                        file_status.bytes_downloaded += len(chunk)
                        job.bytes_downloaded += len(chunk)
                        job.updated_at = datetime.now()

        file_status.status = "completed"
        job.files_completed += 1
    except Exception as e:
        file_status.status = "failed"
        file_status.error = str(e)
        job.files_failed += 1
        logger.error("Failed to download %s: %s", file_status.filename, e)
    finally:
        job.updated_at = datetime.now()


def run_download_job(
    job_id: str,
    products: List[DataProduct],
    destination: str,
    max_parallel: int = 3,
) -> None:
    """Execute a download job (designed to run in a background thread)."""
    job = download_store.get(job_id)
    if not job:
        return

    dest_dir = Path(destination)
    dest_dir.mkdir(parents=True, exist_ok=True)

    total_bytes = sum(p.content_length for p in products)
    if not _check_disk_space(str(dest_dir), total_bytes):
        download_store.update(
            job_id,
            status="failed",
            error=f"Insufficient disk space. Need {total_bytes / (1024**3):.1f} GB",
        )
        return

    # Set up per-file status
    file_statuses: List[FileDownloadStatus] = []
    for p in products:
        fs = FileDownloadStatus(
            filename=p.filename,
            access_url=p.access_url,
            content_length=p.content_length,
        )
        file_statuses.append(fs)

    # If already cancelled while we were setting up, bail out
    if job.status == "cancelled":
        download_store.update(
            job_id,
            files=file_statuses,
            total_files=len(products),
            total_bytes=total_bytes,
        )
        download_store.finish(job_id)
        return

    download_store.update(
        job_id,
        status="running",
        files=file_statuses,
        total_files=len(products),
        total_bytes=total_bytes,
    )
    # Persist file records early so they survive cancellation
    download_store.persist(job_id)

    # Download files in parallel batches
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as pool:
        futures = []
        for fs in file_statuses:
            # Check if job was cancelled
            current_job = download_store.get(job_id)
            if current_job and current_job.status == "cancelled":
                break
            futures.append(pool.submit(_download_single_file, fs, dest_dir, job))

        # Wait for all to finish
        for fut in concurrent.futures.as_completed(futures):
            try:
                fut.result()
            except Exception:
                pass  # errors are already recorded per-file

    # Final status
    final_job = download_store.get(job_id)
    if final_job and final_job.status != "cancelled":
        if final_job.files_failed == 0:
            download_store.update(job_id, status="completed")
        elif final_job.files_completed > 0:
            download_store.update(job_id, status="completed",
                                  error=f"{final_job.files_failed} file(s) failed")
        else:
            download_store.update(job_id, status="failed",
                                  error="All downloads failed")

    # Persist final state to DB and release from memory
    download_store.finish(job_id)
