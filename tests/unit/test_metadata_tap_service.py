"""Unit tests for TAP service query building and helper functions."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from almasim.services.metadata.tap.service import (
    ExclusionFilters,
    InclusionFilters,
    _build_exclusion_conditions,
    _build_inclusion_conditions,
    _like_or_clause,
    get_science_types,
    get_tap_service,
    query_all_targets,
    query_by_science_type,
    query_observations,
    query_products_for_members,
    search_with_retry,
)

# ---------------------------------------------------------------------------
# _like_or_clause
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_like_or_clause_single_value():
    clause = _like_or_clause("science_keyword", ["Galaxies"])
    assert "science_keyword LIKE '%Galaxies%'" in clause


@pytest.mark.unit
def test_like_or_clause_multiple_values():
    clause = _like_or_clause("band_list", ["6", "7"])
    assert "OR" in clause
    assert "LIKE '%6%'" in clause
    assert "LIKE '%7%'" in clause


@pytest.mark.unit
def test_like_or_clause_negate():
    clause = _like_or_clause("type", ["CAL", "FLUX"], negate=True)
    assert "NOT LIKE" in clause
    assert "AND" in clause


# ---------------------------------------------------------------------------
# _build_inclusion_conditions
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_inclusion_conditions_empty():
    conds = _build_inclusion_conditions(
        InclusionFilters(public_only=False, science_only=False, exclude_mosaic=False)
    )
    assert conds == []


@pytest.mark.unit
def test_inclusion_conditions_science_keyword():
    f = InclusionFilters(
        science_keyword=["Galaxies", "ISM"],
        public_only=False,
        science_only=False,
        exclude_mosaic=False,
    )
    conds = _build_inclusion_conditions(f)
    assert any("science_keyword" in c for c in conds)
    assert any("Galaxies" in c for c in conds)


@pytest.mark.unit
def test_inclusion_conditions_band():
    f = InclusionFilters(band=[6, 7], public_only=False, science_only=False, exclude_mosaic=False)
    conds = _build_inclusion_conditions(f)
    assert any("band_list" in c for c in conds)


@pytest.mark.unit
def test_inclusion_conditions_source_name():
    f = InclusionFilters(
        source_name="NGC253", public_only=False, science_only=False, exclude_mosaic=False
    )
    conds = _build_inclusion_conditions(f)
    assert any("NGC253" in c for c in conds)


@pytest.mark.unit
def test_inclusion_conditions_fov_range():
    f = InclusionFilters(
        fov_range=(0.01, 0.1), public_only=False, science_only=False, exclude_mosaic=False
    )
    conds = _build_inclusion_conditions(f)
    assert any("s_fov BETWEEN" in c for c in conds)


@pytest.mark.unit
def test_inclusion_conditions_frequency_range():
    f = InclusionFilters(
        frequency_range=(200.0, 300.0), public_only=False, science_only=False, exclude_mosaic=False
    )
    conds = _build_inclusion_conditions(f)
    assert any("frequency BETWEEN" in c for c in conds)


@pytest.mark.unit
def test_inclusion_conditions_angular_resolution():
    f = InclusionFilters(
        angular_resolution_range=(0.1, 1.0),
        public_only=False,
        science_only=False,
        exclude_mosaic=False,
    )
    conds = _build_inclusion_conditions(f)
    assert any("spatial_resolution BETWEEN" in c for c in conds)


@pytest.mark.unit
def test_inclusion_conditions_observation_date_range():
    f = InclusionFilters(
        observation_date_range=("2020-01-01", "2021-01-01"),
        public_only=False,
        science_only=False,
        exclude_mosaic=False,
    )
    conds = _build_inclusion_conditions(f)
    assert any("t_max" in c for c in conds)


@pytest.mark.unit
def test_inclusion_conditions_qa2_status_pass():
    f = InclusionFilters(
        qa2_status=["Pass"], public_only=False, science_only=False, exclude_mosaic=False
    )
    conds = _build_inclusion_conditions(f)
    assert any("qa2_passed IN" in c for c in conds)
    assert any("'T'" in c for c in conds)


@pytest.mark.unit
def test_inclusion_conditions_qa2_status_raw():
    f = InclusionFilters(
        qa2_status=["T", "X"], public_only=False, science_only=False, exclude_mosaic=False
    )
    conds = _build_inclusion_conditions(f)
    assert any("qa2_passed IN" in c for c in conds)


@pytest.mark.unit
def test_inclusion_conditions_array_type_12m():
    f = InclusionFilters(
        array_type=["12m"], public_only=False, science_only=False, exclude_mosaic=False
    )
    conds = _build_inclusion_conditions(f)
    assert any("%DA%" in c or "%DV%" in c for c in conds)


@pytest.mark.unit
def test_inclusion_conditions_array_type_7m():
    f = InclusionFilters(
        array_type=["7m"], public_only=False, science_only=False, exclude_mosaic=False
    )
    conds = _build_inclusion_conditions(f)
    assert any("%CM%" in c for c in conds)


@pytest.mark.unit
def test_inclusion_conditions_array_type_tp():
    f = InclusionFilters(
        array_type=["TP"], public_only=False, science_only=False, exclude_mosaic=False
    )
    conds = _build_inclusion_conditions(f)
    assert any("%PM%" in c for c in conds)


@pytest.mark.unit
def test_inclusion_conditions_array_configuration():
    f = InclusionFilters(
        array_configuration=["C-1", "C-2"],
        public_only=False,
        science_only=False,
        exclude_mosaic=False,
    )
    conds = _build_inclusion_conditions(f)
    assert any("schedblock_name" in c for c in conds)


@pytest.mark.unit
def test_inclusion_conditions_proposal_id_prefix():
    f = InclusionFilters(
        proposal_id_prefix=["2016.", "2017."],
        public_only=False,
        science_only=False,
        exclude_mosaic=False,
    )
    conds = _build_inclusion_conditions(f)
    assert any("proposal_id" in c for c in conds)


@pytest.mark.unit
def test_inclusion_conditions_defaults_add_public_science_mosaic():
    f = InclusionFilters()
    conds = _build_inclusion_conditions(f)
    assert any("data_rights = 'Public'" in c for c in conds)
    assert any("science_observation = 'T'" in c for c in conds)
    assert any("is_mosaic = 'F'" in c for c in conds)


@pytest.mark.unit
def test_inclusion_conditions_obs_type():
    f = InclusionFilters(
        obs_type=["SCIENCE"], public_only=False, science_only=False, exclude_mosaic=False
    )
    conds = _build_inclusion_conditions(f)
    assert any("type" in c for c in conds)


@pytest.mark.unit
def test_inclusion_conditions_time_resolution():
    f = InclusionFilters(
        time_resolution_range=(1.0, 10.0),
        public_only=False,
        science_only=False,
        exclude_mosaic=False,
    )
    conds = _build_inclusion_conditions(f)
    assert any("t_resolution BETWEEN" in c for c in conds)


@pytest.mark.unit
def test_inclusion_conditions_antenna_arrays():
    f = InclusionFilters(
        antenna_arrays="DA41", public_only=False, science_only=False, exclude_mosaic=False
    )
    conds = _build_inclusion_conditions(f)
    assert any("antenna_arrays" in c for c in conds)


# ---------------------------------------------------------------------------
# _build_exclusion_conditions
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_exclusion_conditions_empty():
    conds = _build_exclusion_conditions(ExclusionFilters())
    assert conds == []


@pytest.mark.unit
def test_exclusion_conditions_science_keyword():
    f = ExclusionFilters(science_keyword=["Solar", "Planetary"])
    conds = _build_exclusion_conditions(f)
    assert len([c for c in conds if "science_keyword NOT LIKE" in c]) == 2


@pytest.mark.unit
def test_exclusion_conditions_scientific_category():
    f = ExclusionFilters(scientific_category=["Solar system"])
    conds = _build_exclusion_conditions(f)
    assert any("scientific_category NOT LIKE" in c for c in conds)


@pytest.mark.unit
def test_exclusion_conditions_source_name():
    f = ExclusionFilters(source_name=["Sun", "Moon"])
    conds = _build_exclusion_conditions(f)
    assert any("LOWER(target_name) NOT LIKE" in c for c in conds)
    assert len([c for c in conds if "target_name" in c]) == 2


@pytest.mark.unit
def test_exclusion_conditions_obs_type():
    f = ExclusionFilters(obs_type=["CAL"])
    conds = _build_exclusion_conditions(f)
    assert any("type NOT LIKE '%CAL%'" in c for c in conds)


@pytest.mark.unit
def test_exclusion_conditions_solar():
    f = ExclusionFilters(solar=True)
    conds = _build_exclusion_conditions(f)
    assert any("target_name" in c and "sun" in c for c in conds)
    assert any("science_keyword" in c and "sun" in c for c in conds)
    assert any("scientific_category" in c and "sun" in c for c in conds)


# ---------------------------------------------------------------------------
# Network-dependent functions (mocked)
# ---------------------------------------------------------------------------


def _make_mock_service(df=None):
    svc = MagicMock()
    if df is None:
        df = pd.DataFrame(
            {"science_keyword": ["Galaxies"], "scientific_category": ["Galaxy evolution"]}
        )
    result = MagicMock()
    result.to_table.return_value.to_pandas.return_value = df
    svc.search.return_value = result
    return svc


@pytest.mark.unit
def test_search_with_retry_calls_service():
    svc = _make_mock_service()
    df = search_with_retry(svc, "SELECT TOP 1 * FROM ivoa.obscore")
    assert isinstance(df, pd.DataFrame)
    svc.search.assert_called_once()


@pytest.mark.unit
def test_get_tap_service_returns_on_first_success():
    mock_svc = MagicMock()
    with patch("almasim.services.metadata.tap.service.pyvo") as mock_pyvo:
        mock_pyvo.dal.TAPService.return_value = mock_svc
        result = get_tap_service()
    assert result is mock_svc


@pytest.mark.unit
def test_get_science_types_parses_keywords():
    df = pd.DataFrame(
        {
            "science_keyword": ["Galaxies, ISM", "Star formation", ""],
            "scientific_category": ["Galaxy evolution", "Star formation", ""],
        }
    )
    mock_svc = MagicMock()
    mock_svc.search.return_value.to_table.return_value.to_pandas.return_value = df

    with patch("almasim.services.metadata.tap.service.get_tap_service", return_value=mock_svc):
        keywords, categories = get_science_types()

    assert "Galaxies" in keywords
    assert "ISM" in keywords
    assert "Star formation" in keywords
    assert "" not in keywords
    assert "Galaxy evolution" in categories


@pytest.mark.unit
def test_query_observations_calls_search():
    df = pd.DataFrame({"member_ous_uid": ["uid://A001/X1/X1"]})
    mock_svc = MagicMock()
    mock_svc.search.return_value.to_table.return_value.to_pandas.return_value = df

    with patch("almasim.services.metadata.tap.service.get_tap_service", return_value=mock_svc):
        result = query_observations("uid://A001/X1/X1", "MyTarget")

    assert isinstance(result, pd.DataFrame)
    mock_svc.search.assert_called_once()
    query_str = mock_svc.search.call_args[0][0]
    assert "uid://A001/X1/X1" in query_str
    assert "MyTarget" in query_str


@pytest.mark.unit
def test_query_all_targets_concatenates_results():
    df1 = pd.DataFrame({"member_ous_uid": ["uid://A001/X1/X1"]})
    df2 = pd.DataFrame({"member_ous_uid": ["uid://A001/X2/X2"]})
    call_count = [0]

    def fake_query(uid, name):
        r = df1 if call_count[0] == 0 else df2
        call_count[0] += 1
        return r

    with patch("almasim.services.metadata.tap.service.query_observations", side_effect=fake_query):
        result = query_all_targets([("T1", "uid://A001/X1/X1"), ("T2", "uid://A001/X2/X2")])

    assert len(result) == 2


@pytest.mark.unit
def test_query_by_science_type_no_filters():
    df = pd.DataFrame({"target_name": ["NGC253"]})
    mock_svc = MagicMock()
    mock_svc.search.return_value.to_table.return_value.to_pandas.return_value = df

    with patch("almasim.services.metadata.tap.service.get_tap_service", return_value=mock_svc):
        result = query_by_science_type()

    assert isinstance(result, pd.DataFrame)
    query_str = mock_svc.search.call_args[0][0]
    assert "is_mosaic = 'F'" in query_str
    assert "science_observation = 'T'" in query_str


@pytest.mark.unit
def test_query_by_science_type_with_include_and_exclude():
    df = pd.DataFrame({"target_name": ["NGC253"]})
    mock_svc = MagicMock()
    mock_svc.search.return_value.to_table.return_value.to_pandas.return_value = df

    inc = InclusionFilters(band=[6], public_only=True, science_only=True, exclude_mosaic=True)
    exc = ExclusionFilters(solar=True)

    with patch("almasim.services.metadata.tap.service.get_tap_service", return_value=mock_svc):
        query_by_science_type(include=inc, exclude=exc)

    query_str = mock_svc.search.call_args[0][0]
    assert "band_list" in query_str
    assert "sun" in query_str.lower()


@pytest.mark.unit
def test_query_products_for_members_single_uid():
    df = pd.DataFrame(
        {"member_ous_uid": ["uid://A001/X1/X1"], "access_url": ["http://example.com"]}
    )
    mock_svc = MagicMock()
    mock_svc.search.return_value.to_table.return_value.to_pandas.return_value = df

    with patch("almasim.services.metadata.tap.service.get_tap_service", return_value=mock_svc):
        result = query_products_for_members("uid://A001/X1/X1")

    assert isinstance(result, pd.DataFrame)
    query_str = mock_svc.search.call_args[0][0]
    assert "uid://A001/X1/X1" in query_str


@pytest.mark.unit
def test_query_products_for_members_list_of_uids():
    df = pd.DataFrame({"member_ous_uid": ["uid://A001/X1/X1", "uid://A001/X2/X2"]})
    mock_svc = MagicMock()
    mock_svc.search.return_value.to_table.return_value.to_pandas.return_value = df

    with patch("almasim.services.metadata.tap.service.get_tap_service", return_value=mock_svc):
        query_products_for_members(["uid://A001/X1/X1", "uid://A001/X2/X2"])

    query_str = mock_svc.search.call_args[0][0]
    assert "uid://A001/X1/X1" in query_str
    assert "uid://A001/X2/X2" in query_str
