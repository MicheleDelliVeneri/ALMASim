"""Metadata API endpoints."""

import json
import sys
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.schemas.metadata import (
    MetadataQuery,
    MetadataResponse,
    MetadataSaveRequest,
    MetadataSaveResponse,
)
from app.services.metadata_service import MetadataService

# Import database dependency
backend_dir = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(backend_dir))

from database.config import get_db

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
async def query_metadata(
    query: MetadataQuery, db: Session = Depends(get_db)
) -> MetadataResponse:
    """Query ALMA metadata from database cache or TAP archive."""
    try:
        service = MetadataService(db=db)
        data = service.query_by_science(
            source_name=query.source_name,
            science_keyword=query.science_keyword,
            scientific_category=query.scientific_category,
            bands=query.bands,
            antenna_arrays=query.antenna_arrays,
            angular_resolution_range=query.angular_resolution_range,
            observation_date_range=query.observation_date_range,
            qa2_status=query.qa2_status,
            obs_type=query.obs_type,
            fov_range=query.fov_range,
            time_resolution_range=query.time_resolution_range,
            frequency_range=query.frequency_range,
        )
        return MetadataResponse(
            count=len(data),
            data=data,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to query metadata: {str(e)}",
        )


@router.get("/load/{file_path:path}", response_model=MetadataResponse)
async def load_metadata(
    file_path: str, db: Session = Depends(get_db)
) -> MetadataResponse:
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


def _resolve_metadata_path(raw_path: str) -> Path:
    """Resolve and validate metadata save path within the data directory."""
    # Use DATA_DIR for saving query results
    base_dir = (settings.DATA_DIR / "query_results").resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    sanitized = (raw_path or "").strip()
    candidate = Path(sanitized) if sanitized else Path("metadata-results.json")

    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        parts = list(candidate.parts)
        # Remove common prefixes if present
        if parts and parts[0] in ("data", "metadata", "query_results"):
            parts = parts[1:]
        relative = Path(*parts) if parts else Path("metadata-results.json")
        resolved = (base_dir / relative).resolve()

    if resolved.suffix.lower() != ".json":
        resolved = resolved.with_suffix(".json")

    try:
        resolved.relative_to(base_dir)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Path must be within the data/query_results directory.",
        )

    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


@router.post("/save", response_model=MetadataSaveResponse)
async def save_metadata(payload: MetadataSaveRequest) -> MetadataSaveResponse:
    """Persist metadata records to disk within the ALMASim metadata directory."""
    try:
        destination = _resolve_metadata_path(payload.path)
        with destination.open("w", encoding="utf-8") as fp:
            json.dump(payload.data, fp, indent=2)
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
