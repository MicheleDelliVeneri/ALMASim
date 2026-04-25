"""Metadata API endpoints."""

import json
import sys
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.config import settings
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

# Import database dependency
backend_dir = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(backend_dir))

from database.config import get_db  # noqa: E402
from database.models import QueryResult  # noqa: E402
from database.service import DatabaseService  # noqa: E402

router = APIRouter()


@router.get("/science-types", response_model=dict)
async def get_science_types(db: Session = Depends(get_db)) -> dict:
    """Get available science types and categories from database."""
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


@router.post("/query", response_model=MetadataResponse)
async def query_metadata(query: MetadataQuery, db: Session = Depends(get_db)) -> MetadataResponse:
    """Query ALMA metadata from database cache or TAP archive."""
    try:
        service = MetadataService(db=db)
        data = service.query_by_science(query.to_params())
        return MetadataResponse(count=len(data), data=data)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to query metadata: {str(e)}",
        )


@router.post("/query/start", response_model=MetadataQueryStartResponse)
async def start_query(
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
            params=query.to_params(),
        )
        return MetadataQueryStartResponse(query_id=query_id, status="running")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start query: {str(e)}",
        )


@router.get("/query/{query_id}/results", response_model=MetadataPageResponse)
async def get_query_results(
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
async def cancel_query(query_id: str) -> None:
    """Cancel a running background TAP query job."""
    cancelled = query_store.cancel(query_id)
    if not cancelled:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Query job {query_id} not found or already finished",
        )


@router.get("/load/{file_path:path}", response_model=MetadataResponse)
async def load_metadata(file_path: str, db: Session = Depends(get_db)) -> MetadataResponse:
    """Load metadata from a CSV file and cache in database."""
    try:
        service = MetadataService(db=db)
        data = service.load_metadata(Path(file_path))
        return MetadataResponse(
            count=len(data),
            data=data,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load metadata: {str(e)}",
        )


def _resolve_metadata_path(raw_path: str, fmt: str = "json") -> Path:
    """Resolve and validate metadata save path within a writable directory."""
    # OUTPUT_DIR is writable in deployments (DATA_DIR is mounted read-only).
    base_dir = (settings.OUTPUT_DIR / "query_results").resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    sanitized = (raw_path or "").strip()
    suffix = ".csv" if fmt.lower() == "csv" else ".json"
    candidate = Path(sanitized) if sanitized else Path(f"metadata-results{suffix}")

    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        parts = list(candidate.parts)
        # Remove common prefixes if present
        if parts and parts[0] in ("data", "metadata", "query_results", "outputs"):
            parts = parts[1:]
        relative = Path(*parts) if parts else Path(f"metadata-results{suffix}")
        resolved = (base_dir / relative).resolve()

    if resolved.suffix.lower() not in (".json", ".csv"):
        resolved = resolved.with_suffix(suffix)
    # Re-align extension with requested format if they disagree.
    if resolved.suffix.lower() != suffix:
        resolved = resolved.with_suffix(suffix)

    try:
        resolved.relative_to(base_dir)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Path must be within the outputs/query_results directory.",
        )

    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _write_metadata_csv(destination: Path, rows: list[dict]) -> None:
    """Write metadata rows to a CSV file using the union of all keys as columns."""
    import csv

    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                columns.append(key)
    with destination.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in columns})


def _write_metadata_json(destination: Path, rows: list[dict]) -> None:
    """Write metadata rows to a JSON file with a count + data envelope."""
    with destination.open("w", encoding="utf-8") as fp:
        json.dump({"count": len(rows), "data": rows}, fp, indent=2)


@router.post("/save", response_model=MetadataSaveResponse)
async def save_metadata(payload: MetadataSaveRequest) -> MetadataSaveResponse:
    """Persist metadata records to disk within the ALMASim metadata directory."""
    try:
        fmt = (payload.format or "json").lower()
        if fmt not in ("json", "csv"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported format. Use 'json' or 'csv'.",
            )
        destination = _resolve_metadata_path(payload.path, fmt=fmt)
        if fmt == "csv":
            _write_metadata_csv(destination, payload.data)
        else:
            _write_metadata_json(destination, payload.data)
        return MetadataSaveResponse(
            path=str(destination),
            count=len(payload.data),
            message="Metadata saved successfully.",
        )
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
        created_at=row.created_at.isoformat() if row.created_at else "",
    )


@router.get("/presets", response_model=QueryPresetsResponse)
async def list_query_presets(db: Session = Depends(get_db)) -> QueryPresetsResponse:
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
async def save_query_preset(
    payload: QueryPresetCreate, db: Session = Depends(get_db)
) -> QueryPresetResponse:
    """Save a named query preset to the database."""
    try:
        row = QueryResult(
            query_name=payload.name,
            query_params={"filters": payload.filters},
            result_count=payload.result_count,
            observation_ids=[],
            description=payload.description,
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return _preset_to_response(row)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save preset: {str(e)}",
        )


@router.get("/presets/{name}", response_model=QueryPresetResponse)
async def get_query_preset(name: str, db: Session = Depends(get_db)) -> QueryPresetResponse:
    """Load a single saved query preset by name."""
    svc = DatabaseService(db)
    row = svc.get_query_result(name)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Preset '{name}' not found",
        )
    return _preset_to_response(row)


@router.delete("/presets/{name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_query_preset(name: str, db: Session = Depends(get_db)) -> None:
    """Delete a saved query preset by name."""
    svc = DatabaseService(db)
    deleted = svc.delete_query_result(name)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Preset '{name}' not found",
        )
