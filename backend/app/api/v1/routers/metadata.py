"""Metadata API endpoints."""

import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy.orm import Session

from almasim.services.metadata.storage import (
    InvalidMetadataPathError,
    UnsupportedMetadataFormatError,
    normalize_metadata_format,
    resolve_metadata_output_path,
    save_metadata_records,
)
from app.core.clock import as_utc
from app.core.config import settings
from app.core.path_utils import resolve_safe_path
from app.db.config import get_db
from app.db.service import DatabaseService
from app.schemas.metadata import (
    MetadataPageResponse,
    MetadataQuery,
    MetadataQueryStartResponse,
    MetadataResponse,
    MetadataSaveRequest,
    MetadataSaveResponse,
    QueryPresetCreate,
    QueryPresetResponse,
    QueryPresetsResponse,
)
from app.services.metadata_service import MetadataService
from app.services.status_store import query_store

router = APIRouter()


@router.get("/science-types", response_model=dict)
def get_science_types(db: Session = Depends(get_db)) -> dict:
    """Get available science types and categories from app.db."""
    try:
        service = MetadataService(db=db)
        keywords, categories = service.get_science_types()
        return {
            "keywords": keywords,
            "categories": categories,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch science types: {str(e)}",
        )


@router.post("/query/start", response_model=MetadataQueryStartResponse)
def start_query(
    query: MetadataQuery,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> MetadataQueryStartResponse:
    """Start a background TAP query job and return a query_id to poll for results."""
    try:
        query_id = str(uuid.uuid4())
        query_store.create(query_id)
        service = MetadataService(db=db)
        background_tasks.add_task(
            service.run_background_query,
            query_id=query_id,
            params=query.to_query(),
        )
        return MetadataQueryStartResponse(query_id=query_id, status="running")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start query: {str(e)}",
        )


@router.get("/query/{query_id}/results", response_model=MetadataPageResponse)
def get_query_results(
    query_id: str,
    page: int = 0,
    page_size: int = 500,
) -> MetadataPageResponse:
    """Poll for a page of results from a background TAP query job."""
    job = query_store.get(query_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Query job {query_id} not found or expired",
        )
    page_data = query_store.get_page(query_id, page, page_size)
    return MetadataPageResponse(**page_data)


@router.delete("/query/{query_id}", status_code=status.HTTP_204_NO_CONTENT)
def cancel_query(query_id: str) -> None:
    """Cancel a running background TAP query job."""
    cancelled = query_store.cancel(query_id)
    if not cancelled:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Query job {query_id} not found or already finished",
        )


@router.get("/load/{file_path:path}", response_model=MetadataResponse)
def load_metadata(file_path: str, db: Session = Depends(get_db)) -> MetadataResponse:
    """Load metadata from a CSV file and cache in database."""
    try:
        # Try DATA_DIR first, fall back to OUTPUT_DIR.
        try:
            resolved = resolve_safe_path(file_path, settings.DATA_DIR)
        except HTTPException:
            resolved = resolve_safe_path(
                file_path, settings.OUTPUT_DIR, detail="Path outside allowed directories"
            )
        service = MetadataService(db=db)
        data = service.load_metadata(resolved)
        return MetadataResponse(
            count=len(data),
            data=data,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load metadata: {str(e)}",
        )


_SAVE_DETAIL = "Path must be within the outputs/query_results directory."


@router.post("/save", response_model=MetadataSaveResponse)
def save_metadata(payload: MetadataSaveRequest) -> MetadataSaveResponse:
    """Persist metadata records to disk within the ALMASim metadata directory."""
    try:
        fmt = normalize_metadata_format(payload.format)
        destination = resolve_metadata_output_path(
            payload.path,
            base_dir=settings.OUTPUT_DIR / "query_results",
            fmt=fmt,
            invalid_path_message=_SAVE_DETAIL,
        )

        save_metadata_records(payload.data, destination, fmt=fmt)
        return MetadataSaveResponse(
            path=str(destination),
            count=len(payload.data),
            message="Metadata saved successfully.",
        )
    except UnsupportedMetadataFormatError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except InvalidMetadataPathError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid path: {exc}",
        ) from exc
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save metadata: {str(e)}",
        )


# ---------------------------------------------------------------------------
# Query preset endpoints
# ---------------------------------------------------------------------------


def _preset_to_response(row: Any) -> QueryPresetResponse:
    params = row.query_params or {}
    return QueryPresetResponse(
        name=row.query_name,
        description=row.description or "",
        filters=params.get("filters", params),
        result_count=row.result_count or 0,
        created_at=as_utc(row.created_at).isoformat() if row.created_at is not None else "",
    )


@router.get("/presets", response_model=QueryPresetsResponse)
def list_query_presets(db: Session = Depends(get_db)) -> QueryPresetsResponse:
    """Return all saved query presets from the database."""
    try:
        svc = DatabaseService(db)
        rows = svc.list_query_results()
        return QueryPresetsResponse(presets=[_preset_to_response(r) for r in rows])
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list presets: {str(e)}",
        )


@router.post("/presets", response_model=QueryPresetResponse, status_code=status.HTTP_201_CREATED)
def save_query_preset(
    payload: QueryPresetCreate, db: Session = Depends(get_db)
) -> QueryPresetResponse:
    """Save a named query preset; saving an existing name replaces it."""
    try:
        row = DatabaseService(db).save_query_preset(
            payload.name,
            payload.filters,
            description=payload.description,
            result_count=payload.result_count,
        )
        return _preset_to_response(row)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save preset: {str(e)}",
        )


@router.delete("/presets/{name}", status_code=status.HTTP_204_NO_CONTENT)
def delete_query_preset(name: str, db: Session = Depends(get_db)) -> None:
    """Delete a saved query preset by name."""
    svc = DatabaseService(db)
    deleted = svc.delete_query_result(name)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Preset '{name}' not found",
        )
