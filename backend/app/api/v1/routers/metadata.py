"""Metadata API endpoints."""
from fastapi import APIRouter, HTTPException, status
from pathlib import Path

from app.core.config import settings
from app.schemas.metadata import MetadataQuery, MetadataResponse
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


