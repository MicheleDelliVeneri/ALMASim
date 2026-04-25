"""Component tests for metadata gathering functionality."""

import pandas as pd
import pytest

from almasim.services.metadata.tap.queries import (
    load_metadata,
    query_metadata_by_science,
    query_metadata_by_targets,
    query_science_types,
)


@pytest.mark.component
@pytest.mark.integration
@pytest.mark.network
def test_query_science_types():
    """Test querying science types from TAP service."""
    keywords, categories = query_science_types()
    assert len(keywords) > 0
    assert len(categories) > 0
    assert isinstance(keywords, list)
    assert isinstance(categories, list)


@pytest.mark.component
@pytest.mark.integration
@pytest.mark.network
def test_query_metadata_by_science(tmp_path):
    """Test querying metadata by science parameters."""
    df = query_metadata_by_science(
        science_keyword=["Galaxies"],
        bands=[6],
        save_to=tmp_path / "test_metadata.csv",
    )
    assert isinstance(df, pd.DataFrame)
    if len(df) > 0:
        assert "ALMA_source_name" in df.columns
        assert "Band" in df.columns


@pytest.mark.component
@pytest.mark.integration
@pytest.mark.network
def test_query_metadata_by_targets(tmp_path):
    """Test querying metadata by target list."""
    targets = [("NGC253", "uid://A001/X123/X456")]
    df = query_metadata_by_targets(targets, save_to=tmp_path / "test_targets.csv")
    assert isinstance(df, pd.DataFrame)


@pytest.mark.component
def test_load_metadata(test_data_dir):
    """Test loading metadata from CSV file."""
    metadata_path = test_data_dir / "qso_metadata.csv"
    if not metadata_path.exists():
        pytest.skip(f"Metadata file not found at {metadata_path}")

    df = load_metadata(metadata_path)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "ALMA_source_name" in df.columns or "target_name" in df.columns


@pytest.mark.component
def test_metadata_normalization(test_data_dir, tmp_path):
    """Test that loaded metadata has expected columns."""
    metadata_path = test_data_dir / "qso_metadata.csv"
    if not metadata_path.exists():
        pytest.skip(f"Metadata file not found at {metadata_path}")

    df = load_metadata(metadata_path)

    # Check for key columns that should be present
    expected_columns = [
        "ALMA_source_name",
        "Band",
        "RA",
        "Dec",
        "Freq",
        "antenna_arrays",
    ]

    for col in expected_columns:
        assert col in df.columns, f"Missing expected column: {col}"
