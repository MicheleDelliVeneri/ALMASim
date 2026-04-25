"""Unit tests for almasim.services.metadata.adapters."""

from __future__ import annotations

from datetime import date, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from almasim.services.metadata.adapters import (
    apply_visible_columns,
    build_exclusion_filters,
    build_inclusion_filters,
    derive_array_type,
    observation_to_metadata_record,
    observations_to_metadata_records,
)


# ===========================================================================
# derive_array_type
# ===========================================================================


@pytest.mark.unit
def test_derive_array_type_empty_string():
    """Empty string returns empty string."""
    assert derive_array_type("") == ""


@pytest.mark.unit
def test_derive_array_type_non_string():
    """Non-string input returns empty string."""
    assert derive_array_type(None) == ""
    assert derive_array_type(42) == ""


@pytest.mark.unit
def test_derive_array_type_12m_da():
    """DA prefix gives 12m."""
    assert derive_array_type("DA41 DA42") == "12m"


@pytest.mark.unit
def test_derive_array_type_12m_dv():
    """DV prefix gives 12m."""
    assert derive_array_type("DV01 DV02") == "12m"


@pytest.mark.unit
def test_derive_array_type_7m():
    """CM prefix gives 7m."""
    assert derive_array_type("CM01 CM02") == "7m"


@pytest.mark.unit
def test_derive_array_type_tp():
    """PM prefix gives TP."""
    assert derive_array_type("PM01 PM02") == "TP"


@pytest.mark.unit
def test_derive_array_type_combined():
    """Mixed antennas produce concatenated type string."""
    result = derive_array_type("DA41 CM01 PM01")
    assert result == "12m+7m+TP"


@pytest.mark.unit
def test_derive_array_type_no_match():
    """Unknown antenna codes return empty string."""
    assert derive_array_type("XX01 YY02") == ""


@pytest.mark.unit
def test_derive_array_type_case_insensitive():
    """Detection is case-insensitive."""
    assert derive_array_type("da41 cm01") == "12m+7m"


# ===========================================================================
# build_inclusion_filters
# ===========================================================================


@pytest.mark.unit
def test_build_inclusion_filters_all_none():
    """Params with no relevant attrs yields all-None InclusionFilters."""
    params = SimpleNamespace()
    filters = build_inclusion_filters(params)
    assert filters.science_keyword is None
    assert filters.scientific_category is None
    assert filters.band is None
    assert filters.fov_range is None


@pytest.mark.unit
def test_build_inclusion_filters_public_only_default():
    """Default public_only=True when attr absent."""
    params = SimpleNamespace()
    filters = build_inclusion_filters(params)
    assert filters.public_only is True


@pytest.mark.unit
def test_build_inclusion_filters_science_only_default():
    """Default science_only=True when attr absent."""
    params = SimpleNamespace()
    filters = build_inclusion_filters(params)
    assert filters.science_only is True


@pytest.mark.unit
def test_build_inclusion_filters_reads_attrs():
    """Attributes from params are mapped to filter fields."""
    params = SimpleNamespace(
        science_keyword=["galaxies"],
        bands=[3, 6],
        fov_range=(1.0, 10.0),
        public_only=False,
    )
    filters = build_inclusion_filters(params)
    assert filters.science_keyword == ["galaxies"]
    assert filters.band == [3, 6]
    assert filters.fov_range == (1.0, 10.0)
    assert filters.public_only is False


# ===========================================================================
# build_exclusion_filters
# ===========================================================================


@pytest.mark.unit
def test_build_exclusion_filters_defaults():
    """Exclusion filters with no attrs default to None/False."""
    params = SimpleNamespace()
    filters = build_exclusion_filters(params)
    assert filters.science_keyword is None
    assert filters.solar is False


@pytest.mark.unit
def test_build_exclusion_filters_reads_attrs():
    """Exclusion filter attributes mapped from params."""
    params = SimpleNamespace(
        exclude_science_keyword=["solar"],
        exclude_solar=True,
    )
    filters = build_exclusion_filters(params)
    assert filters.science_keyword == ["solar"]
    assert filters.solar is True


# ===========================================================================
# observation_to_metadata_record
# ===========================================================================


def _make_obs(**kwargs):
    """Build a minimal mock observation object."""
    kw = MagicMock()
    kw.keyword = "galaxies"
    obs = MagicMock()
    obs.target_name = "NGC1234"
    obs.band = 6
    obs.antenna_arrays = "DA41 DA42"
    obs.spatial_resolution = 0.5
    obs.obs_release_date = None
    obs.proposal_abstract = "Abstract"
    obs.science_keywords = [kw]
    obs.scientific_category = None
    obs.qa2_passed = True
    obs.obs_type = "S"
    obs.pwv = 1.2
    obs.schedblock_name = "SB001"
    obs.velocity_resolution = 10.0
    obs.ra = 12.3
    obs.dec = -45.6
    obs.s_fov = 30.0
    obs.t_max = 3600.0
    obs.cont_sensitivity_bandwidth = 0.1
    obs.sensitivity_10kms = 0.05
    obs.bandwidth = 7.5
    obs.frequency = 230.0
    obs.frequency_support = "[230.0..231.0GHz]"
    obs.proposal_id = "2023.1.00001.S"
    obs.member_ous_uid = "uid://A001/X1"
    obs.group_ous_uid = "uid://A001/X2"
    for k, v in kwargs.items():
        setattr(obs, k, v)
    return obs


@pytest.mark.unit
def test_observation_to_metadata_record_basic_keys():
    """observation_to_metadata_record returns all expected keys."""
    obs = _make_obs()
    record = observation_to_metadata_record(obs)
    assert "ALMA_source_name" in record
    assert "Band" in record
    assert "Array_type" in record
    assert "proposal_id" in record


@pytest.mark.unit
def test_observation_to_metadata_record_array_type_derived():
    """Array_type is derived from antenna_arrays."""
    obs = _make_obs(antenna_arrays="DA41 DA42")
    record = observation_to_metadata_record(obs)
    assert record["Array_type"] == "12m"


@pytest.mark.unit
def test_observation_to_metadata_record_obs_date_none():
    """obs_release_date=None gives None in Obs.date."""
    obs = _make_obs(obs_release_date=None)
    record = observation_to_metadata_record(obs)
    assert record["Obs.date"] is None


@pytest.mark.unit
def test_observation_to_metadata_record_obs_date_isoformat():
    """obs_release_date with .isoformat() is called."""
    fake_date = MagicMock()
    fake_date.isoformat.return_value = "2024-01-01"
    obs = _make_obs(obs_release_date=fake_date)
    record = observation_to_metadata_record(obs)
    assert record["Obs.date"] == "2024-01-01"


@pytest.mark.unit
def test_observation_to_metadata_record_science_keywords_joined():
    """science_keyword field joins multiple keywords with comma+space."""
    kw1, kw2 = MagicMock(), MagicMock()
    kw1.keyword = "galaxies"
    kw2.keyword = "AGN"
    obs = _make_obs(science_keywords=[kw1, kw2])
    record = observation_to_metadata_record(obs)
    assert record["science_keyword"] == "galaxies, AGN"


@pytest.mark.unit
def test_observation_to_metadata_record_scientific_category_none():
    """scientific_category=None gives None in record."""
    obs = _make_obs(scientific_category=None)
    record = observation_to_metadata_record(obs)
    assert record["scientific_category"] is None


@pytest.mark.unit
def test_observation_to_metadata_record_scientific_category_present():
    """scientific_category is read from .category attribute."""
    cat = MagicMock()
    cat.category = "Galaxies"
    obs = _make_obs(scientific_category=cat)
    record = observation_to_metadata_record(obs)
    assert record["scientific_category"] == "Galaxies"


# ===========================================================================
# apply_visible_columns
# ===========================================================================


@pytest.mark.unit
def test_apply_visible_columns_no_filter():
    """No visible_columns uses ALL_COLUMNS ordering, keeps matching keys."""
    from almasim.services.metadata.tap.queries import ALL_COLUMNS

    records = [{"ALMA_source_name": "NGC1", "Band": 6, "extra": "drop"}]
    result = apply_visible_columns(records)
    # All ALL_COLUMNS keys present in record appear; extra keys dropped
    assert "ALMA_source_name" in result[0]
    assert "Band" in result[0]
    assert "extra" not in result[0]


@pytest.mark.unit
def test_apply_visible_columns_subset():
    """Passing visible_columns restricts output to those columns."""
    records = [{"ALMA_source_name": "NGC1", "Band": 6, "RA": 12.0}]
    result = apply_visible_columns(records, visible_columns=["ALMA_source_name", "RA"])
    assert list(result[0].keys()) == ["ALMA_source_name", "RA"]


@pytest.mark.unit
def test_apply_visible_columns_missing_col_skipped():
    """Columns absent from the record are silently skipped."""
    records = [{"Band": 6}]
    result = apply_visible_columns(records, visible_columns=["ALMA_source_name", "Band"])
    assert "ALMA_source_name" not in result[0]
    assert result[0]["Band"] == 6


@pytest.mark.unit
def test_apply_visible_columns_empty_records():
    """Empty records list returns empty list."""
    result = apply_visible_columns([])
    assert result == []


# ===========================================================================
# observations_to_metadata_records
# ===========================================================================


@pytest.mark.unit
def test_observations_to_metadata_records_length():
    """observations_to_metadata_records returns one record per observation."""
    obs_list = [_make_obs(), _make_obs()]
    records = observations_to_metadata_records(obs_list)
    assert len(records) == 2


@pytest.mark.unit
def test_observations_to_metadata_records_visible_columns():
    """visible_columns kwarg is forwarded correctly."""
    obs_list = [_make_obs()]
    records = observations_to_metadata_records(obs_list, visible_columns=["Band"])
    assert list(records[0].keys()) == ["Band"]
