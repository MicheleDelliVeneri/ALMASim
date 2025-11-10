"""Metadata API endpoints."""
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, status

from app.core.config import settings
from app.schemas.metadata import (
    MetadataQuery,
    MetadataResponse,
    MetadataSaveRequest,
    MetadataSaveResponse,
)
from app.services.metadata_service import MetadataService

router = APIRouter()


@router.get("/science-types", response_model=dict)
async def get_science_types() -> dict:
    """Get available science types and categories."""
    try:
        service = MetadataService()
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
async def query_metadata(query: MetadataQuery) -> MetadataResponse:
    """Query ALMA metadata."""
    try:
        service = MetadataService()
        df = service.query_by_science(
            science_keyword=query.science_keyword,
            scientific_category=query.scientific_category,
            bands=query.bands,
            fov_range=query.fov_range,
            time_resolution_range=query.time_resolution_range,
            frequency_range=query.frequency_range,
        )
        return MetadataResponse(
            count=len(df),
            data=df.to_dict("records"),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to query metadata: {str(e)}",
        )


@router.get("/load/{file_path:path}", response_model=MetadataResponse)
async def load_metadata(file_path: str) -> MetadataResponse:
    """Load metadata from a CSV file."""
    try:
        service = MetadataService()
        df = service.load_metadata(Path(file_path))
        return MetadataResponse(
            count=len(df),
            data=df.to_dict("records"),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load metadata: {str(e)}",
        )


def _resolve_metadata_path(raw_path: str) -> Path:
    """Resolve and validate metadata save path within the ALMASim metadata directory."""
    base_dir = (settings.MAIN_DIR / "metadata").resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    sanitized = (raw_path or "").strip()
    candidate = Path(sanitized) if sanitized else Path("metadata-results.json")

    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        parts = list(candidate.parts)
        root_name = settings.MAIN_DIR.name
        if parts and parts[0] == root_name:
            parts = parts[1:]
        if parts and parts[0] == "metadata":
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
            detail="Path must be within the ALMASim metadata directory.",
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

