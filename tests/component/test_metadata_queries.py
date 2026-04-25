"""Component tests for metadata query integration."""

import pytest
import pandas as pd

from almasim.services.metadata.tap.queries import (
    query_science_types,
    query_metadata_by_science,
    query_metadata_by_targets,
    load_metadata,
)


@pytest.mark.integration
@pytest.mark.network
def test_query_science_types():
    """Test querying science types from TAP service."""
    keywords, categories = query_science_types()
    assert len(keywords) > 0
    assert len(categories) > 0
    assert isinstance(keywords, list)
    assert isinstance(categories, list)


@pytest.mark.integration
@pytest.mark.network
def test_query_metadata_by_science(tmp_path, test_data_dir):
    """Test querying metadata by science parameters."""
    # This is an integration test that requires network access
    df = query_metadata_by_science(
        science_keyword="Galaxies",
        bands=[6],
        save_to=tmp_path / "test_metadata.csv",
    )
    assert isinstance(df, pd.DataFrame)
    if len(df) > 0:
        assert "ALMA_source_name" in df.columns
        assert "Band" in df.columns


@pytest.mark.integration
@pytest.mark.network
def test_query_metadata_by_targets(tmp_path):
    """Test querying metadata by target list."""
    targets = [("NGC253", "uid://A001/X123/X456")]
    df = query_metadata_by_targets(targets, save_to=tmp_path / "test_targets.csv")
    assert isinstance(df, pd.DataFrame)


def test_load_metadata(test_data_dir):
    """Test loading metadata from CSV."""
    metadata_path = test_data_dir / "qso_metadata.csv"
    if not metadata_path.exists():
        pytest.skip(f"Metadata file not found at {metadata_path}")

    df = load_metadata(metadata_path)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "ALMA_source_name" in df.columns or "target_name" in df.columns
