"""Library-first download helpers for ALMA data products."""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional
from xml.etree import ElementTree

import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

_DATALINK_MIRRORS = [
    "https://almascience.eso.org",
    "https://almascience.nrao.edu",
    "https://almascience.nao.ac.jp",
]
_VOT_NS = "http://www.ivoa.net/xml/VOTable/v1.3"
_RATE_LIMIT_SEC = 0.5
_MAX_FILE_ATTEMPTS = 3
_RETRY_BACKOFF_BASE = 2
_CHUNK_SIZE = 256 * 1024


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
    error: Optional[str] = None
    sha256: Optional[str] = None


@dataclass
class DownloadSummary:
    """Summary returned by :func:`download_products`."""

    destination: str
    total_files: int
    total_bytes: int
    files_completed: int
    files_failed: int
    files: List[FileDownloadStatus] = field(default_factory=list)
    extracted_files: List[str] = field(default_factory=list)


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

_SEMANTICS_MAP: Dict[str, str] = {
    "#progenitor": "raw",
    "#derivation": "fits",
    "#auxiliary": "auxiliary",
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


def format_bytes(size: int) -> str:
    """Return a human-friendly byte size string."""
    size_float = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size_float) < 1000:
            return f"{size_float:.1f} {unit}"
        size_float /= 1000.0
    return f"{size_float:.1f} PB"


def _classify_product(url: str, content_type: str, semantics: str = "") -> str:
    """Classify a product using VOTable semantics and URL patterns."""
    if semantics:
        sem_lower = semantics.lower().strip()
        if sem_lower in _SEMANTICS_MAP:
            return _SEMANTICS_MAP[sem_lower]
        for frag in ("#", "/ALMA#", "/alma#"):
            if frag in sem_lower:
                short = "alma#" + sem_lower.split(frag)[-1]
                if short in _SEMANTICS_MAP:
                    return _SEMANTICS_MAP[short]

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
        return "fits"
    return "other"


def _parse_votable_results(xml_bytes: bytes, uid: str) -> List[DataProduct]:
    """Parse a DataLink VOTable response into DataProduct rows."""
    products: List[DataProduct] = []
    try:
        root = ElementTree.fromstring(xml_bytes)
    except ElementTree.ParseError:
        logger.warning("Failed to parse VOTable XML for uid=%s", uid)
        return products

    for ns_prefix in [f"{{{_VOT_NS}}}", ""]:
        table = root.find(f".//{ns_prefix}TABLEDATA")
        if table is None:
            continue

        resource = root.find(f".//{ns_prefix}RESOURCE")
        tbl = resource.find(f".//{ns_prefix}TABLE") if resource is not None else None
        fields = tbl.findall(f"{ns_prefix}FIELD") if tbl is not None else []
        col_names = [field.get("name", "").lower() for field in fields]

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
            except (TypeError, ValueError):
                content_length = 0

            filename = url.rsplit("/", 1)[-1] if "/" in url else url
            products.append(
                DataProduct(
                    access_url=url,
                    uid=row_uid or uid,
                    filename=filename,
                    content_length=content_length,
                    content_type=ctype or "",
                    product_type=_classify_product(url, ctype or "", semantics),
                    semantics=semantics,
                )
            )
        break

    return products


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _fetch_datalink(uid: str, mirror_base: str = _DATALINK_MIRRORS[0]) -> List[DataProduct]:
    """Fetch DataLink results for a single member OUS UID."""
    url = f"{mirror_base}/datalink/sync?ID={uid}"
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
    return _parse_votable_results(response.content, uid)


def resolve_products(
    member_ous_uids: Iterable[str],
    mirror: str = _DATALINK_MIRRORS[0],
) -> List[DataProduct]:
    """Resolve downloadable data products through ALMA DataLink."""
    products: List[DataProduct] = []
    member_uids = [str(uid) for uid in member_ous_uids]
    for uid in member_uids:
        last_error: Optional[Exception] = None
        for base in [mirror] + [candidate for candidate in _DATALINK_MIRRORS if candidate != mirror]:
            try:
                products.extend(_fetch_datalink(uid, base))
                last_error = None
                break
            except Exception as exc:  # pragma: no cover - exercised in integration/network use
                last_error = exc
                logger.warning("DataLink fetch failed for %s on %s: %s", uid, base, exc)
        if last_error:
            logger.error("All DataLink mirrors failed for %s: %s", uid, last_error)
    return products


def filter_products(products: List[DataProduct], product_filter: str = "all") -> List[DataProduct]:
    """Return products matching the requested product type."""
    if product_filter == "all":
        return list(products)
    return [product for product in products if product.product_type == product_filter]


def products_to_dataframe(products: List[DataProduct]) -> pd.DataFrame:
    """Convert DataProduct rows into a DataFrame."""
    rows = []
    for product in products:
        row = asdict(product)
        row["size_mb"] = product.size_mb
        rows.append(row)
    return pd.DataFrame(rows)


def save_products_csv(products: List[DataProduct], output_path: Path | str) -> str:
    """Persist resolved DataLink products as CSV."""
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    products_to_dataframe(products).to_csv(output_path, index=False)
    return str(output_path)


def load_products_csv(input_path: Path | str) -> List[DataProduct]:
    """Load previously saved DataProduct rows from CSV."""
    input_path = Path(input_path).expanduser().resolve()
    dataframe = pd.read_csv(input_path)
    products = []
    for row in dataframe.to_dict(orient="records"):
        products.append(
            DataProduct(
                access_url=str(row["access_url"]),
                uid=str(row["uid"]),
                filename=str(row["filename"]),
                content_length=int(row.get("content_length", 0) or 0),
                content_type=str(row.get("content_type", "") or ""),
                product_type=str(row.get("product_type", "other") or "other"),
                semantics=str(row.get("semantics", "") or ""),
            )
        )
    return products


def _sha256_file(path: Path) -> str:
    hsh = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hsh.update(chunk)
    return hsh.hexdigest()


def _download_single_attempt(
    file_status: FileDownloadStatus,
    part_path: Path,
    *,
    should_cancel: Optional[Callable[[], bool]] = None,
    update_callback: Optional[Callable[[FileDownloadStatus], None]] = None,
) -> bool:
    existing_bytes = part_path.stat().st_size if part_path.exists() else 0
    headers = {}
    if existing_bytes > 0:
        headers["Range"] = f"bytes={existing_bytes}-"

    if should_cancel is not None and should_cancel():
        file_status.status = "cancelled"
        if update_callback is not None:
            update_callback(file_status)
        return False

    with httpx.Client(timeout=300, follow_redirects=True) as client:
        with client.stream("GET", file_status.access_url, headers=headers) as response:
            if existing_bytes > 0 and response.status_code == 200:
                resume_from = 0
                open_mode = "wb"
            else:
                resume_from = existing_bytes
                open_mode = "ab" if existing_bytes > 0 else "wb"

            response.raise_for_status()
            file_status.bytes_downloaded = resume_from
            if update_callback is not None:
                update_callback(file_status)
            with open(part_path, open_mode) as handle:
                for chunk in response.iter_bytes(chunk_size=_CHUNK_SIZE):
                    if should_cancel is not None and should_cancel():
                        file_status.status = "cancelled"
                        if update_callback is not None:
                            update_callback(file_status)
                        return False
                    handle.write(chunk)
                    file_status.bytes_downloaded += len(chunk)
                    if update_callback is not None:
                        update_callback(file_status)
    return True


def _download_single_file(
    file_status: FileDownloadStatus,
    destination: Path,
    *,
    should_cancel: Optional[Callable[[], bool]] = None,
    update_callback: Optional[Callable[[FileDownloadStatus], None]] = None,
) -> None:
    destination_path = destination / file_status.filename
    part_path = destination_path.with_suffix(destination_path.suffix + ".part")

    if destination_path.exists():
        existing_size = destination_path.stat().st_size
        if file_status.content_length <= 0 or existing_size >= file_status.content_length:
            file_status.bytes_downloaded = existing_size
            file_status.status = "completed"
            try:
                file_status.sha256 = _sha256_file(destination_path)
            except OSError:
                file_status.sha256 = None
            if update_callback is not None:
                update_callback(file_status)
            return

    file_status.status = "downloading"
    if update_callback is not None:
        update_callback(file_status)
    last_error: Optional[Exception] = None
    for attempt in range(1, _MAX_FILE_ATTEMPTS + 1):
        try:
            completed = _download_single_attempt(
                file_status,
                part_path,
                should_cancel=should_cancel,
                update_callback=update_callback,
            )
            if not completed:
                return
            last_error = None
            break
        except Exception as exc:
            last_error = exc
            if attempt < _MAX_FILE_ATTEMPTS:
                time.sleep(_RETRY_BACKOFF_BASE ** (attempt - 1))

    if last_error is not None:
        file_status.status = "failed"
        file_status.error = str(last_error)
        if update_callback is not None:
            update_callback(file_status)
        return

    try:
        os.replace(part_path, destination_path)
    except OSError as exc:
        file_status.status = "failed"
        file_status.error = f"Rename failed: {exc}"
        if update_callback is not None:
            update_callback(file_status)
        return

    try:
        file_status.sha256 = _sha256_file(destination_path)
    except OSError:
        file_status.sha256 = None
    file_status.status = "completed"
    if update_callback is not None:
        update_callback(file_status)


def _extract_tar(tar_path: Path, destination: Path) -> List[str]:
    """Safely extract a tar archive into the destination directory."""
    import tarfile

    extracted: List[str] = []
    try:
        with tarfile.open(tar_path, "r:*") as archive:
            members = archive.getmembers()
            for member in members:
                if member.name.startswith("/") or ".." in member.name:
                    logger.warning("Skipping unsafe tar member: %s", member.name)
                    continue
                resolved = (destination / member.name).resolve()
                if not str(resolved).startswith(str(destination.resolve())):
                    logger.warning("Skipping escaping tar member: %s", member.name)
                    continue
            archive.extractall(destination, filter="data")
            extracted = [member.name for member in members if not member.isdir()]
        tar_path.unlink(missing_ok=True)
    except Exception as exc:  # pragma: no cover - exercised via integration use
        logger.warning("Failed to extract %s: %s", tar_path.name, exc)
    return extracted


def download_products(
    products: List[DataProduct],
    destination: Path | str,
    *,
    max_parallel: int = 3,
    rate_limit_sec: float = _RATE_LIMIT_SEC,
    extract_tar: bool = False,
    logger_fn: Optional[Callable[[str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    update_callback: Optional[Callable[[FileDownloadStatus], None]] = None,
) -> DownloadSummary:
    """Download resolved products into a local directory."""
    destination_path = Path(destination).expanduser().resolve()
    destination_path.mkdir(parents=True, exist_ok=True)

    statuses = [
        FileDownloadStatus(
            filename=product.filename,
            access_url=product.access_url,
            content_length=product.content_length,
        )
        for product in products
    ]
    extracted_files: List[str] = []

    with ThreadPoolExecutor(max_workers=max_parallel) as pool:
        futures = []
        for index, status in enumerate(statuses):
            if should_cancel is not None and should_cancel():
                status.status = "cancelled"
                if update_callback is not None:
                    update_callback(status)
                continue
            futures.append(
                pool.submit(
                    _download_single_file,
                    status,
                    destination_path,
                    should_cancel=should_cancel,
                    update_callback=update_callback,
                )
            )
            if index < len(statuses) - 1 and rate_limit_sec > 0:
                time.sleep(rate_limit_sec)

        for future in as_completed(futures):
            future.result()

    if extract_tar:
        for status in statuses:
            if status.status != "completed":
                continue
            if not (status.filename.endswith(".tar") or status.filename.endswith(".tgz")):
                continue
            tar_path = destination_path / status.filename
            if tar_path.exists():
                extracted_files.extend(_extract_tar(tar_path, destination_path))

    files_completed = sum(1 for status in statuses if status.status == "completed")
    files_failed = sum(1 for status in statuses if status.status == "failed")
    total_bytes = sum(product.content_length for product in products)

    if logger_fn is not None:
        logger_fn(
            f"Downloaded {files_completed}/{len(statuses)} files "
            f"to {destination_path} ({format_bytes(total_bytes)})"
        )

    return DownloadSummary(
        destination=str(destination_path),
        total_files=len(products),
        total_bytes=total_bytes,
        files_completed=files_completed,
        files_failed=files_failed,
        files=statuses,
        extracted_files=extracted_files,
    )


__all__ = [
    "DataProduct",
    "FileDownloadStatus",
    "DownloadSummary",
    "PRODUCT_TYPES",
    "format_bytes",
    "resolve_products",
    "filter_products",
    "products_to_dataframe",
    "save_products_csv",
    "load_products_csv",
    "download_products",
]
