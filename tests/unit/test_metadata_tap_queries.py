"""Unit tests for almasim.services.metadata.tap.queries — pure helpers."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from almasim.services.metadata.tap.queries import (
    _derive_array_type,
    _normalize_metadata,
    _prepare_save_path,
    load_metadata,
    query_metadata_by_science,
    query_metadata_by_targets,
)

# ===========================================================================
# _derive_array_type (local copy in queries module)
# ===========================================================================


@pytest.mark.unit
def test_derive_array_type_empty():
    """Empty string gives empty type."""
    assert _derive_array_type("") == ""


@pytest.mark.unit
def test_derive_array_type_12m():
    """DA antennas give 12m."""
    assert _derive_array_type("DA41 DA42") == "12m"


@pytest.mark.unit
def test_derive_array_type_7m():
    """CM antennas give 7m."""
    assert _derive_array_type("CM01") == "7m"


@pytest.mark.unit
def test_derive_array_type_tp():
    """PM antennas give TP."""
    assert _derive_array_type("PM01") == "TP"


@pytest.mark.unit
def test_derive_array_type_combined():
    """Mixed antennas give combined type."""
    assert _derive_array_type("DA41 CM01") == "12m+7m"


# ===========================================================================
# _prepare_save_path
# ===========================================================================


@pytest.mark.unit
def test_prepare_save_path_none():
    """None input returns None."""
    assert _prepare_save_path(None) is None


@pytest.mark.unit
def test_prepare_save_path_empty_string():
    """Empty string returns None."""
    assert _prepare_save_path("") is None


@pytest.mark.unit
def test_prepare_save_path_creates_parents(tmp_path):
    """Non-existing parent directory is created."""
    target = tmp_path / "sub" / "dir" / "out.csv"
    result = _prepare_save_path(target)
    assert result == target.resolve()
    assert target.parent.is_dir()


@pytest.mark.unit
def test_prepare_save_path_returns_path_object(tmp_path):
    """Result is a Path object."""
    result = _prepare_save_path(tmp_path / "out.csv")
    assert isinstance(result, Path)


# ===========================================================================
# _normalize_metadata
# ===========================================================================


def _make_df(**extra) -> pd.DataFrame:
    """Minimal dataframe with all required raw columns."""
    data = {
        "member_ous_uid": ["uid://A001/X1"],
        "target_name": ["NGC1234"],
        "band_list": [6],
        "antenna_arrays": ["DA41 DA42"],
        "s_resolution": [0.5],
        "obs_release_date": ["2024-01-01T12:00:00"],
        "proposal_abstract": ["Abstract"],
        "science_keyword": ["galaxies"],
        "scientific_category": ["Galaxy"],
        "qa2_passed": [True],
        "type": ["S"],
        "pwv": [1.2],
        "schedblock_name": ["SB001"],
        "velocity_resolution": [10.0],
        "s_ra": [12.3],
        "s_dec": [-45.6],
        "s_fov": [30.0],
        "t_max": [3600.0],
        "cont_sensitivity_bandwidth": [0.1],
        "sensitivity_10kms": [0.05],
        "bandwidth": [7.5],
        "frequency": [230.0],
        "frequency_support": ["[230..231GHz]"],
        "proposal_id": ["2023.1.X"],
        "group_ous_uid": ["uid://A001/X2"],
    }
    data.update(extra)
    return pd.DataFrame(data)


@pytest.mark.unit
def test_normalize_metadata_empty():
    """Empty DataFrame passes through unchanged."""
    df = pd.DataFrame()
    result = _normalize_metadata(df)
    assert result.empty


@pytest.mark.unit
def test_normalize_metadata_derives_array_type():
    """Array_type column is derived from antenna_arrays."""
    df = _make_df()
    result = _normalize_metadata(df)
    assert "Array_type" in result.columns


@pytest.mark.unit
def test_normalize_metadata_deduplicates():
    """Duplicate member_ous_uid rows are dropped."""
    df = _make_df()
    df2 = df.copy()
    combined = pd.concat([df, df2], ignore_index=True)
    result = _normalize_metadata(combined)
    assert len(result) == 1


@pytest.mark.unit
def test_normalize_metadata_visible_columns_subset():
    """visible_columns restricts output columns."""
    df = _make_df()
    result = _normalize_metadata(df, visible_columns=["ALMA_source_name"])
    assert list(result.columns) == ["ALMA_source_name"]


@pytest.mark.unit
def test_normalize_metadata_obs_date_truncated():
    """ISO datetime in Obs.date is truncated to date only."""
    df = _make_df(**{"obs_release_date": ["2024-01-15T08:30:00"]})
    result = _normalize_metadata(df)
    if "Obs.date" in result.columns:
        assert "T" not in str(result["Obs.date"].iloc[0])


# ===========================================================================
# load_metadata
# ===========================================================================


@pytest.mark.unit
def test_load_metadata_csv(tmp_path):
    """load_metadata reads CSV files."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    csv_path = tmp_path / "meta.csv"
    df.to_csv(csv_path, index=False)
    result = load_metadata(csv_path)
    assert len(result) == 2
    assert list(result.columns) == ["a", "b"]


@pytest.mark.unit
def test_load_metadata_json_list(tmp_path):
    """load_metadata reads JSON list format."""
    records = [{"x": 1}, {"x": 2}]
    json_path = tmp_path / "meta.json"
    json_path.write_text(json.dumps(records))
    result = load_metadata(json_path)
    assert len(result) == 2


@pytest.mark.unit
def test_load_metadata_json_dict(tmp_path):
    """load_metadata reads JSON dict with 'data' key."""
    payload = {"data": [{"x": 1}, {"x": 2}]}
    json_path = tmp_path / "meta.json"
    json_path.write_text(json.dumps(payload))
    result = load_metadata(json_path)
    assert len(result) == 2


@pytest.mark.unit
def test_load_metadata_json_invalid_format(tmp_path):
    """load_metadata raises ValueError for unsupported JSON format."""
    json_path = tmp_path / "meta.json"
    json_path.write_text(json.dumps("just a string"))
    with pytest.raises(ValueError, match="Unsupported"):
        load_metadata(json_path)


# ===========================================================================
# query_metadata_by_science (mocked network call)
# ===========================================================================


@pytest.mark.unit
def test_query_metadata_by_science_empty_result():
    """Returns empty DataFrame when underlying query returns empty."""
    with patch(
        "almasim.services.metadata.tap.queries._query_by_science_type",
        return_value=pd.DataFrame(),
    ):
        result = query_metadata_by_science()
    assert result.empty


@pytest.mark.unit
def test_query_metadata_by_science_saves_csv(tmp_path):
    """When save_to is provided, result is written to CSV."""
    df = _make_df()
    csv_path = tmp_path / "out.csv"
    with patch(
        "almasim.services.metadata.tap.queries._query_by_science_type",
        return_value=df,
    ):
        query_metadata_by_science(save_to=csv_path)
    assert csv_path.exists()


@pytest.mark.unit
def test_query_metadata_by_science_no_save():
    """When save_to is None, no file is created."""
    df = _make_df()
    with patch(
        "almasim.services.metadata.tap.queries._query_by_science_type",
        return_value=df,
    ):
        result = query_metadata_by_science(save_to=None)
    assert isinstance(result, pd.DataFrame)
    assert not result.empty


# ===========================================================================
# query_metadata_by_targets (mocked network call)
# ===========================================================================


@pytest.mark.unit
def test_query_metadata_by_targets_basic():
    """query_metadata_by_targets normalizes and returns result."""
    df = _make_df()
    with patch(
        "almasim.services.metadata.tap.queries._query_all_targets",
        return_value=df,
    ):
        result = query_metadata_by_targets([("NGC1234", "uid://A001/X1")])
    assert isinstance(result, pd.DataFrame)


@pytest.mark.unit
def test_query_metadata_by_targets_saves_csv(tmp_path):
    """query_metadata_by_targets saves result to CSV when save_to is given."""
    df = _make_df()
    csv_path = tmp_path / "targets.csv"
    with patch(
        "almasim.services.metadata.tap.queries._query_all_targets",
        return_value=df,
    ):
        query_metadata_by_targets(
            [("NGC1234", "uid://A001/X1")],
            save_to=csv_path,
        )
    assert csv_path.exists()
