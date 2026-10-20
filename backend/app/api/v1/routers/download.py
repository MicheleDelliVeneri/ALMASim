"""Download API endpoints for fetching ALMA data products."""

import os
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, status

from almasim.services.download import format_bytes
from almasim.services.paths import UnsafePathError, safe_join
from app.core.config import settings
from app.core.path_utils import require_allowed_path, resolve_safe_path
from app.schemas.download import (
    BrowseDirectoryEntry,
    BrowseDirectoryResponse,
    CheckDiskSpaceRequest,
    DataProductInfo,
    DiskSpaceInfo,
    DownloadJobStatus,
    DownloadJobSummary,
    RedownloadRequest,
    ResolveProductsRequest,
    ResolveProductsResponse,
    StartDownloadRequest,
    StartDownloadResponse,
    json_list,
)
from app.schemas.metadata import MetadataResponse
from app.services.download_service import (
    DataProduct,
    DownloadJob,
    download_store,
    filter_products,
    resolve_products,
    run_download_job,
)

router = APIRouter()


def _record_or_404(job_id: str):
    """The job's database record, or a 404."""
    record = download_store.get_from_db(job_id)
    if not record:
        raise _job_not_found(job_id)
    return record


def _job_not_found(job_id: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Download job {job_id} not found",
    )


def _browse_path(resolved_path) -> BrowseDirectoryResponse:
    """Build a BrowseDirectoryResponse for an existing directory path."""

    parent = str(resolved_path.parent) if resolved_path != resolved_path.parent else None
    entries: list[BrowseDirectoryEntry] = []
    try:
        scan_iter = os.scandir(str(resolved_path))  # lgtm[py/path-injection]
        for entry in sorted(scan_iter, key=lambda e: e.name.lower()):
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
def browse_directory(path: str = "/host_home"):
    """List subdirectories of a server-side path for the destination picker.

    Only directories are returned (no files), and hidden entries (starting
    with '.') are excluded for clarity.  The host home directory is mounted
    at /host_home so users can browse and download to their real filesystem.
    """
    resolved = resolve_safe_path(path, Path(settings.HOST_DIR))
    # Walk up to the closest existing directory when the path doesn't exist
    while not resolved.is_dir():  # lgtm[py/path-injection]
        if resolved.parent == resolved:
            break
        resolved = resolved.parent

    return _browse_path(resolved)


@router.post("/mkdir", response_model=BrowseDirectoryResponse)
def make_directory(path: str):
    """Create a new directory and return a browse result for it."""
    target = resolve_safe_path(path, Path(settings.HOST_DIR))
    try:
        target.mkdir(parents=True, exist_ok=True)  # lgtm[py/path-injection]
    except OSError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot create directory: {e}",
        )

    return _browse_path(target)


@router.post("/resolve", response_model=ResolveProductsResponse)
def resolve_download_products(body: ResolveProductsRequest):
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
        entry["size_display"] = format_bytes(entry["size_bytes"])

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
        total_size_display=format_bytes(total_bytes),
        by_type=by_type,
    )


@router.post("/disk-space", response_model=DiskSpaceInfo)
def check_disk_space(body: CheckDiskSpaceRequest):
    """Check available disk space at a given path."""
    candidate = resolve_safe_path(body.path, Path(settings.HOST_DIR))
    try:
        usage = shutil.disk_usage(candidate)  # lgtm[py/path-injection]
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
        total_display=format_bytes(usage.total),
        used_display=format_bytes(usage.used),
        free_display=format_bytes(usage.free),
        sufficient=usage.free >= body.needed_bytes,
    )


@router.post("/start", response_model=StartDownloadResponse)
def start_download(body: StartDownloadRequest, background_tasks: BackgroundTasks):
    """Start downloading selected ALMA data products.

    Resolves products, applies the type filter, then launches a background
    download job that streams files in parallel.
    """
    if not body.member_ous_uids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="member_ous_uids must not be empty",
        )

    # Every server path in the request must lie under an allowed root (T1.9).
    destination = str(require_allowed_path(body.destination, field="destination"))
    archive_output_root = (
        str(require_allowed_path(body.archive_output_root, field="archive_output_root"))
        if body.archive_output_root
        else None
    )
    casa_data_root = (
        str(require_allowed_path(body.casa_data_root, field="casa_data_root"))
        if body.casa_data_root
        else None
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
        destination=destination,
        member_ous_uids=body.member_ous_uids,
        metadata_rows=body.selected_metadata or [],
        product_filter=body.product_filter,
        unpack_ms=body.unpack_ms or body.generate_calibrated_visibilities,
        generate_calibrated_visibilities=body.generate_calibrated_visibilities,
        clean_intermediate_files=body.clean_intermediate_files,
        archive_output_root=archive_output_root,
        casa_data_root=casa_data_root,
        skip_casa_data_update=body.skip_casa_data_update,
        total_files=len(products),
        total_bytes=total_bytes,
    )
    download_store.create(job)

    # Launch download in background
    background_tasks.add_task(
        run_download_job,
        job_id=job_id,
        products=products,
        destination=destination,
        max_parallel=body.max_parallel,
        extract_tar=body.extract_tar or body.unpack_ms or body.generate_calibrated_visibilities,
        unpack_ms=body.unpack_ms or body.generate_calibrated_visibilities,
        generate_calibrated_visibilities=body.generate_calibrated_visibilities,
        clean_intermediate_files=body.clean_intermediate_files,
        archive_output_root=archive_output_root,
        casa_data_root=casa_data_root,
        skip_casa_data_update=body.skip_casa_data_update,
    )

    return StartDownloadResponse(
        job_id=job_id,
        status="pending",
        total_files=len(products),
        total_bytes=total_bytes,
        total_size_display=format_bytes(total_bytes),
        destination=destination,
    )


@router.get("/jobs", response_model=list[DownloadJobSummary])
def list_download_jobs():
    """List all download jobs (active and historical)."""
    records = download_store.list_all()  # returns DB records, newest first
    result = []
    for rec in records:
        # For active jobs, prefer the live in-memory data
        active = download_store.get(str(rec.job_id))
        result.append(
            DownloadJobSummary.from_job(active) if active else DownloadJobSummary.from_record(rec)
        )
    return result


@router.get("/jobs/{job_id}", response_model=DownloadJobStatus)
def get_download_job(job_id: str):
    """Get detailed status of a download job including per-file progress."""
    # Try active (in-memory) first for live progress
    job = download_store.get(job_id)
    if job:
        return DownloadJobStatus.from_job(job)
    # Fall back to DB record (historical)
    return DownloadJobStatus.from_record(_record_or_404(job_id))


@router.get("/jobs/{job_id}/metadata", response_model=MetadataResponse)
def get_download_job_metadata(job_id: str):
    """Load the metadata rows associated with a downloaded data product."""
    job = download_store.get(job_id)
    if job:
        rows = job.metadata_rows
    else:
        rows = json_list(_record_or_404(job_id).metadata_json)

    if not rows:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No metadata stored for download job {job_id}",
        )

    return MetadataResponse(count=len(rows), data=rows)


@router.post("/jobs/{job_id}/cancel")
def cancel_download_job(job_id: str):
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
    rec = _record_or_404(job_id)
    if rec.status not in ("pending", "running"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job is already {rec.status}",
        )
    # Update directly in DB
    download_store.update_in_db(job_id, status="cancelled")
    return {"job_id": job_id, "status": "cancelled"}


@router.post("/jobs/{job_id}/redownload", response_model=StartDownloadResponse)
def redownload_job(
    job_id: str,
    background_tasks: BackgroundTasks,
    body: RedownloadRequest = RedownloadRequest(),
):
    """Re-download a previous job using the stored file records.

    Uses the original access URLs from the database so UID resolution is not
    needed.  Existing files at the destination are overwritten.
    """
    rec = _record_or_404(job_id)

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

    max_parallel = body.max_parallel if body.max_parallel is not None else 3
    unpack_ms = body.unpack_ms if body.unpack_ms is not None else bool(rec.unpack_ms)
    generate_calibrated = (
        body.generate_calibrated_visibilities
        if body.generate_calibrated_visibilities is not None
        else bool(rec.generate_calibrated_visibilities)
    )
    clean_intermediate = (
        body.clean_intermediate_files
        if body.clean_intermediate_files is not None
        else bool(rec.clean_intermediate_files)
    )
    extract_tar = (
        (
            body.extract_tar
            if body.extract_tar is not None
            else bool(rec.unpack_ms) or bool(rec.generate_calibrated_visibilities)
        )
        or unpack_ms
        or generate_calibrated
    )
    archive_output_root = (
        body.archive_output_root
        if body.archive_output_root is not None
        else rec.archive_output_root
    )
    casa_data_root = body.casa_data_root if body.casa_data_root is not None else rec.casa_data_root
    # Checked again: the request may override them, and older jobs predate the check.
    destination = str(require_allowed_path(rec.destination, field="destination"))
    if archive_output_root:
        archive_output_root = str(
            require_allowed_path(archive_output_root, field="archive_output_root")
        )
    if casa_data_root:
        casa_data_root = str(require_allowed_path(casa_data_root, field="casa_data_root"))
    skip_casa_data_update = (
        body.skip_casa_data_update
        if body.skip_casa_data_update is not None
        else bool(rec.skip_casa_data_update)
    )

    total_bytes = sum(p.content_length for p in products)
    new_job_id = str(uuid.uuid4())
    job = DownloadJob(
        job_id=new_job_id,
        destination=destination,
        member_ous_uids=uids,
        metadata_rows=json_list(rec.metadata_json),
        product_filter=rec.product_filter or "all",
        unpack_ms=unpack_ms or generate_calibrated,
        generate_calibrated_visibilities=generate_calibrated,
        clean_intermediate_files=clean_intermediate,
        archive_output_root=archive_output_root,
        casa_data_root=casa_data_root,
        skip_casa_data_update=skip_casa_data_update,
        total_files=len(products),
        total_bytes=total_bytes,
    )
    download_store.create(job)

    background_tasks.add_task(
        run_download_job,
        job_id=new_job_id,
        products=products,
        destination=destination,
        max_parallel=max_parallel,
        extract_tar=extract_tar,
        unpack_ms=unpack_ms or generate_calibrated,
        generate_calibrated_visibilities=generate_calibrated,
        clean_intermediate_files=clean_intermediate,
        archive_output_root=archive_output_root,
        casa_data_root=casa_data_root,
        skip_casa_data_update=skip_casa_data_update,
    )

    return StartDownloadResponse(
        job_id=new_job_id,
        status="pending",
        total_files=len(products),
        total_bytes=total_bytes,
        total_size_display=format_bytes(total_bytes),
        destination=destination,
    )


@router.delete("/jobs/{job_id}")
def delete_download_job(job_id: str):
    """Delete a download job from history and its files on disk.

    Active (running/pending) jobs must be cancelled first.
    """
    from pathlib import Path

    active = download_store.get(job_id)
    if active and active.status in ("pending", "running"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete an active job — cancel it first",
        )

    # Look up the job (with file records) before deleting from DB
    rec = _record_or_404(job_id)

    # Delete downloaded files (and any leftover .part files) from disk. Only
    # paths inside the job's own destination are removed (T1.9): a stored
    # filename or generated path pointing elsewhere is left alone.
    dest = Path(rec.destination)
    files_deleted = 0
    skipped = 0
    for file_rec in rec.files:
        for name in (file_rec.filename, file_rec.filename + ".part"):
            try:
                path = safe_join(dest, name)
            except UnsafePathError:
                skipped += 1
                continue
            try:
                if path.is_file():
                    path.unlink()
                    files_deleted += 1
            except OSError:
                pass  # best-effort; don't fail the delete if a file is gone

    generated_paths = [
        *(json_list(rec.raw_measurement_sets)),
        *(json_list(rec.calibrated_measurement_sets)),
    ]
    if rec.manifest_path:
        generated_paths.append(rec.manifest_path)
    for generated in generated_paths:
        try:
            path = safe_join(dest, generated)
        except UnsafePathError:
            skipped += 1
            continue
        try:
            if path.is_dir():
                shutil.rmtree(path)
                files_deleted += 1
            elif path.is_file():
                path.unlink()
                files_deleted += 1
        except OSError:
            pass

    deleted = download_store.delete_from_db(job_id)
    if not deleted:
        raise _job_not_found(job_id)
    # Also remove from memory if still lingering
    download_store.remove(job_id)
    return {
        "job_id": job_id,
        "deleted": True,
        "files_deleted": files_deleted,
        "skipped_outside_destination": skipped,
    }
