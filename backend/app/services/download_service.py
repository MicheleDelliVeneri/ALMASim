"""Download service for fetching ALMA data products via DataLink."""

import hashlib
import logging
import os
import shutil
import threading
import time
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

# Rate-limit between consecutive file downloads (seconds)
_RATE_LIMIT_SEC = 0.5

# Per-file retry policy
_MAX_FILE_ATTEMPTS = 3
_RETRY_BACKOFF_BASE = 2  # seconds; doubles each attempt


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
    product_type: str  # see PRODUCT_TYPES below
    semantics: str = ""  # raw value from VOTable semantics column

    @property
    def size_mb(self) -> float:
        return self.content_length / (1024 * 1024)


# Supported product_type values (superset of former coarse set)
PRODUCT_TYPES = {
    "all",
    "raw",           # ASDM / raw measurement sets
    "calibration",   # pipeline calibration tables / scripts
    "scripts",       # pipeline scripts only
    "weblog",        # pipeline weblog HTML
    "qa_reports",    # QA2 / QA3 reports
    "auxiliary",     # auxiliary files, READMEs, licence files
    "cubes",         # spectral-line image cubes
    "continuum",     # continuum images
    "fits",          # any FITS product (legacy catch-all)
    "other",
}

# Semantics → product_type mapping (IVOA DataLink + ALMA-specific URIs)
_SEMANTICS_MAP: Dict[str, str] = {
    # IVOA standard
    "#progenitor": "raw",
    "#derivation": "fits",
    "#auxiliary": "auxiliary",
    # ALMA-specific (case-insensitive prefix matching done at runtime)
    "alma#calibration": "calibration",
    "alma#calibratedmeasurementset": "calibration",
    "alma#pipeline_calibration": "calibration",
    "alma#script": "scripts",
    "alma#pipelinescript": "scripts",
    "alma#weblog": "weblog",
    "alma#qa": "qa_reports",
    "alma#qa2report": "qa_reports",
    "alma#imagecube": "cubes",
    "alma#spectralcube": "cubes",
    "alma#continuumimage": "continuum",
    "alma#continuum": "continuum",
    "alma#asdm": "raw",
    "alma#raw": "raw",
    "alma#auxiliary": "auxiliary",
    "alma#readme": "auxiliary",
}


def _classify_product(url: str, content_type: str, semantics: str = "") -> str:
    """Classify a product using VOTable semantics (preferred) then URL patterns."""
    # --- Semantics-based (most accurate) ---
    if semantics:
        sem_lower = semantics.lower().strip()
        # Direct match
        if sem_lower in _SEMANTICS_MAP:
            return _SEMANTICS_MAP[sem_lower]
        # Strip URI prefix: "http://almascience.org/ALMA#Calibration" → "alma#calibration"
        for frag in ("#", "/ALMA#", "/alma#"):
            if frag in sem_lower:
                short = "alma#" + sem_lower.split(frag)[-1]
                if short in _SEMANTICS_MAP:
                    return _SEMANTICS_MAP[short]

    # --- URL-pattern fallback ---
    lower = url.lower()
    if "asdm" in lower or lower.endswith(".asdm.sdm.tar"):
        return "raw"
    if "weblog" in lower:
        return "weblog"
    if "qa" in lower and ("report" in lower or ".pdf" in lower):
        return "qa_reports"
    if "script" in lower:
        return "scripts"
    if "auxiliary" in lower or "readme" in lower or "licence" in lower:
        return "auxiliary"
    if lower.endswith(".fits"):
        return "fits"
    if "cube" in lower:
        return "cubes"
    if "continuum" in lower or "cont_" in lower:
        return "continuum"
    if lower.endswith(".tar") or lower.endswith(".tgz"):
        # Pipeline product tarballs (contain FITS) — classify as "fits"
        # unless they match a more specific category above.
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

    for ns_prefix in [f"{{{_VOT_NS}}}", ""]:
        table = root.find(f".//{ns_prefix}TABLEDATA")
        if table is not None:
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
            i_sem = col_idx("semantics")

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
                semantics = (tds[i_sem].text or "").strip() if i_sem >= 0 and i_sem < len(tds) else ""

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
                    product_type=_classify_product(url, ctype or "", semantics),
                    semantics=semantics,
                ))
            break

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

    product_filter: any value in PRODUCT_TYPES, or "fits" for legacy compat.
    The special value "all" returns every product.
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
    status: str = "pending"  # pending, downloading, completed, failed, cancelled
    error: Optional[str] = None
    sha256: Optional[str] = None  # computed after successful download


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

            if job.files:
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
                        sha256=fs.sha256,
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
            db.delete(rec)
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
        return True


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_single_attempt(
    file_status: FileDownloadStatus,
    dest_dir: Path,
    job: DownloadJob,
    part_path: Path,
) -> bool:
    """Single download attempt. Returns True on success, False on failure.

    Writes to `part_path` (.part staging file); does NOT rename on success —
    the caller handles the atomic rename after all retries.
    """
    existing_bytes = part_path.stat().st_size if part_path.exists() else 0

    headers = {}
    if existing_bytes > 0:
        headers["Range"] = f"bytes={existing_bytes}-"

    bytes_before = job.bytes_downloaded

    try:
        with httpx.Client(timeout=300, follow_redirects=True) as client:
            with client.stream(
                "GET", file_status.access_url, headers=headers
            ) as resp:
                if existing_bytes > 0 and resp.status_code == 200:
                    # Server ignored Range header — restart from scratch
                    resume_from = 0
                    open_mode = "wb"
                else:
                    resume_from = existing_bytes
                    open_mode = "ab" if existing_bytes > 0 else "wb"

                resp.raise_for_status()

                # Credit already-on-disk bytes only on the first attempt
                # (resume_from > 0 means we're resuming a partial file)
                file_status.bytes_downloaded = resume_from
                job.bytes_downloaded = bytes_before + resume_from
                job.updated_at = datetime.now()

                with open(part_path, open_mode) as f:
                    for chunk in resp.iter_bytes(chunk_size=_CHUNK_SIZE):
                        if job.status == "cancelled":
                            file_status.status = "cancelled"
                            return False
                        f.write(chunk)
                        file_status.bytes_downloaded += len(chunk)
                        job.bytes_downloaded += len(chunk)
                        job.updated_at = datetime.now()
        return True

    except Exception as e:
        # Undo any bytes counted during this failed attempt so the next
        # retry (which will resume from the .part file) starts clean.
        job.bytes_downloaded = bytes_before
        job.updated_at = datetime.now()
        raise e


def _download_single_file(
    file_status: FileDownloadStatus,
    dest_dir: Path,
    job: DownloadJob,
) -> None:
    """Download a single file with resume, per-file retry, .part staging, and SHA-256."""
    dest_path = dest_dir / file_status.filename
    part_path = dest_path.with_suffix(dest_path.suffix + ".part")

    # Already fully downloaded — skip
    if dest_path.exists():
        existing = dest_path.stat().st_size
        if file_status.content_length > 0 and existing >= file_status.content_length:
            file_status.bytes_downloaded = existing
            file_status.status = "completed"
            job.bytes_downloaded += existing
            job.files_completed += 1
            job.updated_at = datetime.now()
            return

    file_status.status = "downloading"

    last_error: Optional[Exception] = None
    for attempt in range(1, _MAX_FILE_ATTEMPTS + 1):
        try:
            success = _download_single_attempt(file_status, dest_dir, job, part_path)
            if not success:
                # Cancelled
                return
            last_error = None
            break
        except Exception as e:
            last_error = e
            if attempt < _MAX_FILE_ATTEMPTS:
                backoff = _RETRY_BACKOFF_BASE ** (attempt - 1)
                logger.warning(
                    "Download attempt %d/%d failed for %s: %s — retrying in %ds",
                    attempt, _MAX_FILE_ATTEMPTS, file_status.filename, e, backoff,
                )
                time.sleep(backoff)
            else:
                logger.error(
                    "All %d attempts failed for %s: %s",
                    _MAX_FILE_ATTEMPTS, file_status.filename, e,
                )

    if last_error:
        file_status.status = "failed"
        file_status.error = str(last_error)
        job.files_failed += 1
        job.updated_at = datetime.now()
        return

    # Atomic rename: .part → final destination
    try:
        os.replace(part_path, dest_path)
    except OSError as e:
        file_status.status = "failed"
        file_status.error = f"Rename failed: {e}"
        job.files_failed += 1
        job.updated_at = datetime.now()
        return

    # SHA-256 verification
    try:
        digest = _sha256_file(dest_path)
        file_status.sha256 = digest
        logger.debug("SHA-256 %s  %s", digest, file_status.filename)
    except OSError as e:
        logger.warning("Could not compute SHA-256 for %s: %s", file_status.filename, e)

    file_status.status = "completed"
    job.files_completed += 1
    job.updated_at = datetime.now()


def _extract_tar(tar_path: Path, dest_dir: Path) -> List[str]:
    """Safely extract a tar archive and return list of extracted filenames.

    Validates member paths to prevent directory traversal attacks.
    Removes the tar archive after successful extraction.
    """
    import tarfile

    extracted: List[str] = []
    try:
        with tarfile.open(tar_path, "r:*") as tf:
            for member in tf.getmembers():
                # Security: reject absolute paths and directory traversal
                if member.name.startswith("/") or ".." in member.name:
                    logger.warning("Skipping unsafe tar member: %s", member.name)
                    continue
                resolved = (dest_dir / member.name).resolve()
                if not str(resolved).startswith(str(dest_dir.resolve())):
                    logger.warning("Skipping tar member escaping dest: %s", member.name)
                    continue
            # Re-open to extract (after validation pass)
            tf.extractall(dest_dir, filter="data")
            extracted = [m.name for m in tf.getmembers() if not m.isdir()]
        # Remove the archive after successful extraction
        tar_path.unlink(missing_ok=True)
        logger.info("Extracted %d files from %s", len(extracted), tar_path.name)
    except Exception as e:
        logger.warning("Failed to extract %s: %s", tar_path.name, e)
    return extracted


def run_download_job(
    job_id: str,
    products: List[DataProduct],
    destination: str,
    max_parallel: int = 3,
    rate_limit_sec: float = _RATE_LIMIT_SEC,
    extract_tar: bool = False,
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

    file_statuses: List[FileDownloadStatus] = [
        FileDownloadStatus(
            filename=p.filename,
            access_url=p.access_url,
            content_length=p.content_length,
        )
        for p in products
    ]

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
    download_store.persist(job_id)

    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as pool:
        futures = []
        for i, fs in enumerate(file_statuses):
            current_job = download_store.get(job_id)
            if current_job and current_job.status == "cancelled":
                break
            futures.append(pool.submit(_download_single_file, fs, dest_dir, job))
            # Rate-limit: sleep between submissions (except before the first)
            if i < len(file_statuses) - 1 and rate_limit_sec > 0:
                time.sleep(rate_limit_sec)

        for fut in concurrent.futures.as_completed(futures):
            try:
                fut.result()
            except Exception:
                pass

    # Extract tar archives if requested
    if extract_tar:
        final_job = download_store.get(job_id)
        if final_job and final_job.status != "cancelled":
            for fs in file_statuses:
                if fs.status == "completed" and (
                    fs.filename.endswith(".tar") or fs.filename.endswith(".tgz")
                ):
                    tar_path = dest_dir / fs.filename
                    if tar_path.exists():
                        _extract_tar(tar_path, dest_dir)

    final_job = download_store.get(job_id)
    if final_job and final_job.status != "cancelled":
        if final_job.files_failed == 0:
            download_store.update(job_id, status="completed")
        elif final_job.files_completed > 0:
            download_store.update(
                job_id, status="completed",
                error=f"{final_job.files_failed} file(s) failed",
            )
        else:
            download_store.update(job_id, status="failed", error="All downloads failed")

    download_store.finish(job_id)
