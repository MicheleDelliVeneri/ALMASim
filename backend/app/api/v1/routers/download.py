"""Download API endpoints for fetching ALMA data products."""

import shutil
import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException, status

from app.schemas.download import (
    BrowseDirectoryEntry,
    BrowseDirectoryResponse,
    CheckDiskSpaceRequest,
    DataProductInfo,
    DiskSpaceInfo,
    DownloadJobStatus,
    DownloadJobSummary,
    FileStatus,
    ResolveProductsRequest,
    ResolveProductsResponse,
    StartDownloadRequest,
    StartDownloadResponse,
)
from app.services.download_service import (
    DataProduct,
    DownloadJob,
    download_store,
    filter_products,
    resolve_products,
    run_download_job,
)

router = APIRouter()


def _browse_path(resolved_path) -> BrowseDirectoryResponse:
    """Build a BrowseDirectoryResponse for an existing directory path."""
    import os

    parent = str(resolved_path.parent) if resolved_path != resolved_path.parent else None
    entries: list[BrowseDirectoryEntry] = []
    try:
        for entry in sorted(os.scandir(str(resolved_path)), key=lambda e: e.name.lower()):
            if entry.name.startswith("."):
                continue
            try:
                if entry.is_dir(follow_symlinks=False):
                    entries.append(
                        BrowseDirectoryEntry(
                            name=entry.name,
                            path=str(resolved_path / entry.name),
                            is_dir=True,
                        )
                    )
            except PermissionError:
                continue
    except PermissionError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Permission denied: {resolved_path}",
        )

    return BrowseDirectoryResponse(current=str(resolved_path), parent=parent, entries=entries)


@router.get("/browse", response_model=BrowseDirectoryResponse)
async def browse_directory(path: str = "/host_home"):
    """List subdirectories of a server-side path for the destination picker.

    Only directories are returned (no files), and hidden entries (starting
    with '.') are excluded for clarity.  The host home directory is mounted
    at /host_home so users can browse and download to their real filesystem.
    """
    from pathlib import Path

    resolved = Path(path).expanduser().resolve()
    # Walk up to the closest existing directory when the path doesn't exist
    while not resolved.is_dir():
        if resolved.parent == resolved:
            break
        resolved = resolved.parent

    return _browse_path(resolved)


def _format_bytes(size: int) -> str:
    """Human-friendly byte size string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size) < 1000:
            return f"{size:.1f} {unit}"
        size /= 1000
    return f"{size:.1f} PB"


@router.post("/mkdir", response_model=BrowseDirectoryResponse)
async def make_directory(path: str):
    """Create a new directory and return a browse result for it."""
    from pathlib import Path

    target = Path(path)
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot create directory: {e}",
        )

    return _browse_path(target)


@router.post("/resolve", response_model=ResolveProductsResponse)
async def resolve_download_products(body: ResolveProductsRequest):
    """Resolve available data products for selected observations via ALMA DataLink.

    Returns the full product list with sizes and a per-type breakdown so the
    user can choose what to download.
    """
    if not body.member_ous_uids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="member_ous_uids must not be empty",
        )
    try:
        products = resolve_products(body.member_ous_uids)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to contact ALMA DataLink service: {e}",
        )

    # Build per-type breakdown
    by_type: dict = {}
    for p in products:
        entry = by_type.setdefault(p.product_type, {"count": 0, "size_bytes": 0})
        entry["count"] += 1
        entry["size_bytes"] += p.content_length
    for entry in by_type.values():
        entry["size_display"] = _format_bytes(entry["size_bytes"])

    total_bytes = sum(p.content_length for p in products)

    return ResolveProductsResponse(
        products=[
            DataProductInfo(
                access_url=p.access_url,
                uid=p.uid,
                filename=p.filename,
                content_length=p.content_length,
                content_type=p.content_type,
                product_type=p.product_type,
                size_mb=p.size_mb,
            )
            for p in products
        ],
        total_count=len(products),
        total_size_bytes=total_bytes,
        total_size_display=_format_bytes(total_bytes),
        by_type=by_type,
    )


@router.post("/disk-space", response_model=DiskSpaceInfo)
async def check_disk_space(body: CheckDiskSpaceRequest):
    """Check available disk space at a given path."""
    try:
        usage = shutil.disk_usage(body.path)
    except OSError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot access path: {e}",
        )

    return DiskSpaceInfo(
        path=body.path,
        total_bytes=usage.total,
        used_bytes=usage.used,
        free_bytes=usage.free,
        total_display=_format_bytes(usage.total),
        used_display=_format_bytes(usage.used),
        free_display=_format_bytes(usage.free),
        sufficient=usage.free >= body.needed_bytes,
    )


@router.post("/start", response_model=StartDownloadResponse)
async def start_download(body: StartDownloadRequest, background_tasks: BackgroundTasks):
    """Start downloading selected ALMA data products.

    Resolves products, applies the type filter, then launches a background
    download job that streams files in parallel.
    """
    if not body.member_ous_uids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="member_ous_uids must not be empty",
        )

    # Resolve products from DataLink
    try:
        all_products = resolve_products(body.member_ous_uids)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to resolve products: {e}",
        )

    # Apply type filter
    products = filter_products(all_products, body.product_filter)
    if not products:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No products match filter '{body.product_filter}'",
        )

    total_bytes = sum(p.content_length for p in products)

    # Create download job
    job_id = str(uuid.uuid4())
    job = DownloadJob(
        job_id=job_id,
        destination=body.destination,
        member_ous_uids=body.member_ous_uids,
        product_filter=body.product_filter,
        total_files=len(products),
        total_bytes=total_bytes,
    )
    download_store.create(job)

    # Launch download in background
    background_tasks.add_task(
        run_download_job,
        job_id=job_id,
        products=products,
        destination=body.destination,
        max_parallel=body.max_parallel,
    )

    return StartDownloadResponse(
        job_id=job_id,
        status="pending",
        total_files=len(products),
        total_bytes=total_bytes,
        total_size_display=_format_bytes(total_bytes),
        destination=body.destination,
    )


@router.get("/jobs", response_model=list[DownloadJobSummary])
async def list_download_jobs():
    """List all download jobs (active and historical)."""
    records = download_store.list_all()  # returns DB records, newest first
    result = []
    for rec in records:
        job_id_str = str(rec.job_id)
        # For active jobs, prefer the live in-memory data
        active = download_store.get(job_id_str)
        if active:
            total = active.total_bytes or 1
            result.append(DownloadJobSummary(
                job_id=active.job_id,
                status=active.status,
                destination=active.destination,
                total_files=active.total_files,
                files_completed=active.files_completed,
                files_failed=active.files_failed,
                progress=active.bytes_downloaded / total if total > 0 else 0,
                created_at=active.created_at.isoformat(),
                member_ous_uids=active.member_ous_uids,
                product_filter=active.product_filter,
                total_bytes=active.total_bytes,
                bytes_downloaded=active.bytes_downloaded,
                error=active.error,
            ))
        else:
            import json
            total = rec.total_bytes or 1
            try:
                uids = json.loads(rec.member_ous_uids) if rec.member_ous_uids else []
            except (json.JSONDecodeError, TypeError):
                uids = []
            result.append(DownloadJobSummary(
                job_id=str(rec.job_id),
                status=rec.status,
                destination=rec.destination,
                total_files=rec.total_files,
                files_completed=rec.files_completed,
                files_failed=rec.files_failed,
                progress=rec.bytes_downloaded / total if total > 0 else 0,
                created_at=rec.created_at.isoformat(),
                member_ous_uids=uids,
                product_filter=rec.product_filter or "all",
                total_bytes=int(rec.total_bytes),
                bytes_downloaded=int(rec.bytes_downloaded),
                error=rec.error,
            ))
    return result


@router.get("/jobs/{job_id}", response_model=DownloadJobStatus)
async def get_download_job(job_id: str):
    """Get detailed status of a download job including per-file progress."""
    # Try active (in-memory) first for live progress
    job = download_store.get(job_id)
    if job:
        return DownloadJobStatus(
            job_id=job.job_id,
            status=job.status,
            destination=job.destination,
            total_files=job.total_files,
            files_completed=job.files_completed,
            files_failed=job.files_failed,
            total_bytes=job.total_bytes,
            bytes_downloaded=job.bytes_downloaded,
            progress=(job.bytes_downloaded / job.total_bytes) if job.total_bytes > 0 else 0,
            error=job.error,
            files=[
                FileStatus(
                    filename=f.filename,
                    content_length=f.content_length,
                    bytes_downloaded=f.bytes_downloaded,
                    status=f.status,
                    error=f.error,
                    progress=(f.bytes_downloaded / f.content_length) if f.content_length > 0 else 0,
                )
                for f in job.files
            ],
        )

    # Fall back to DB record (historical)
    rec = download_store.get_from_db(job_id)
    if not rec:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Download job {job_id} not found",
        )

    return DownloadJobStatus(
        job_id=str(rec.job_id),
        status=rec.status,
        destination=rec.destination,
        total_files=rec.total_files,
        files_completed=rec.files_completed,
        files_failed=rec.files_failed,
        total_bytes=int(rec.total_bytes),
        bytes_downloaded=int(rec.bytes_downloaded),
        progress=(rec.bytes_downloaded / rec.total_bytes) if rec.total_bytes > 0 else 0,
        error=rec.error,
        files=[
            FileStatus(
                filename=f.filename,
                content_length=int(f.content_length),
                bytes_downloaded=int(f.bytes_downloaded),
                status=f.status,
                error=f.error,
                progress=(f.bytes_downloaded / f.content_length) if f.content_length > 0 else 0,
            )
            for f in rec.files
        ],
    )


@router.post("/jobs/{job_id}/cancel")
async def cancel_download_job(job_id: str):
    """Cancel a running download job."""
    # Try in-memory first (active jobs)
    job = download_store.get(job_id)
    if job:
        if job.status not in ("pending", "running"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Job is already {job.status}",
            )
        download_store.update(job_id, status="cancelled")
        download_store.persist(job_id)
        return {"job_id": job_id, "status": "cancelled"}

    # Fall back to DB (job already finished or server restarted)
    rec = download_store.get_from_db(job_id)
    if not rec:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Download job {job_id} not found",
        )
    if rec.status not in ("pending", "running"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job is already {rec.status}",
        )
    # Update directly in DB
    download_store.update_in_db(job_id, status="cancelled")
    return {"job_id": job_id, "status": "cancelled"}


@router.post("/jobs/{job_id}/redownload", response_model=StartDownloadResponse)
async def redownload_job(
    job_id: str,
    background_tasks: BackgroundTasks,
):
    """Re-download a previous job using the stored file records.

    Uses the original access URLs from the database so UID resolution is not
    needed.  Existing files at the destination are overwritten.
    """
    rec = download_store.get_from_db(job_id)
    if not rec:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Download job {job_id} not found",
        )

    import json
    try:
        uids = json.loads(rec.member_ous_uids) if rec.member_ous_uids else []
    except (json.JSONDecodeError, TypeError):
        uids = []

    if rec.files:
        # Build DataProduct list from stored file records
        products = [
            DataProduct(
                access_url=f.access_url,
                uid="",
                filename=f.filename,
                content_length=int(f.content_length),
                content_type="",
                product_type="other",
            )
            for f in rec.files
        ]
    elif uids:
        # No file records (e.g. early cancellation) – re-resolve from DataLink
        try:
            all_products = resolve_products(uids)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Failed to resolve products: {e}",
            )
        products = filter_products(all_products, rec.product_filter or "all")
        if not products:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No products found when re-resolving from ALMA DataLink",
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file records or UIDs stored for this job – cannot re-download",
        )

    total_bytes = sum(p.content_length for p in products)
    new_job_id = str(uuid.uuid4())
    job = DownloadJob(
        job_id=new_job_id,
        destination=rec.destination,
        member_ous_uids=uids,
        product_filter=rec.product_filter or "all",
        total_files=len(products),
        total_bytes=total_bytes,
    )
    download_store.create(job)

    background_tasks.add_task(
        run_download_job,
        job_id=new_job_id,
        products=products,
        destination=rec.destination,
        max_parallel=3,
    )

    return StartDownloadResponse(
        job_id=new_job_id,
        status="pending",
        total_files=len(products),
        total_bytes=total_bytes,
        total_size_display=_format_bytes(total_bytes),
        destination=rec.destination,
    )


@router.delete("/jobs/{job_id}")
async def delete_download_job(job_id: str):
    """Delete a download job from history.

    Active (running/pending) jobs must be cancelled first.
    """
    active = download_store.get(job_id)
    if active and active.status in ("pending", "running"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete an active job — cancel it first",
        )

    deleted = download_store.delete_from_db(job_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Download job {job_id} not found",
        )
    # Also remove from memory if still lingering
    with download_store._lock:
        download_store._active.pop(job_id, None)
    return {"job_id": job_id, "deleted": True}
