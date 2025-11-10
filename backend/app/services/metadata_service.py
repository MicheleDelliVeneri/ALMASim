"""Metadata business logic service."""
from pathlib import Path
from typing import Optional, Sequence, Tuple

from almasim.services.metadata.tap.queries import (
    query_science_types,
    query_metadata_by_science,
    load_metadata,
)


class MetadataService:
    """Service for managing metadata queries."""

    def get_science_types(self) -> Tuple[Sequence[str], Sequence[str]]:
        """Get science types and categories."""
        return query_science_types()

    def query_by_science(
        self,
        science_keyword: Optional[Sequence[str]] = None,
        scientific_category: Optional[Sequence[str]] = None,
        bands: Optional[Sequence[int]] = None,
        fov_range: Optional[Tuple[float, float]] = None,
        time_resolution_range: Optional[Tuple[float, float]] = None,
        frequency_range: Optional[Tuple[float, float]] = None,
        save_to: Optional[Path] = None,
    ):
        """Query metadata by science parameters."""
        return query_metadata_by_science(
            science_keyword=science_keyword,
            scientific_category=scientific_category,
            bands=bands,
            fov_range=fov_range,
            time_resolution_range=time_resolution_range,
            frequency_range=frequency_range,
            save_to=save_to,
        )

    def load_metadata(self, metadata_path: Path):
        """Load metadata from CSV file."""
        return load_metadata(metadata_path)

