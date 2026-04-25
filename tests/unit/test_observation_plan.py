"""Unit tests for almasim.services.observation_plan."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from almasim.services.observation_plan import (
    ALMA_LATITUDE_DEG,
    ObservationConfig,
    SinglePointingObservationPlan,
    _coerce_observation_config,
    build_single_pointing_observation_plan,
    derive_array_type,
    estimate_transit_elevation,
    infer_antenna_diameter_m,
    normalize_observation_configs,
    split_antenna_array_by_type,
)


# ===========================================================================
# derive_array_type
# ===========================================================================


@pytest.mark.unit
def test_derive_array_type_12m():
    """DA-only antennas give 12m."""
    assert derive_array_type("DA41 DA42") == "12m"


@pytest.mark.unit
def test_derive_array_type_7m():
    """CM-only antennas give 7m."""
    assert derive_array_type("CM01 CM02") == "7m"


@pytest.mark.unit
def test_derive_array_type_tp():
    """PM-only antennas give TP."""
    assert derive_array_type("PM01 PM02") == "TP"


@pytest.mark.unit
def test_derive_array_type_12m_7m():
    """DA+CM gives 12m+7m."""
    assert derive_array_type("DA41 CM01") == "12m+7m"


@pytest.mark.unit
def test_derive_array_type_12m_tp():
    """DA+PM gives 12m+TP."""
    assert derive_array_type("DA41 PM01") == "12m+TP"


@pytest.mark.unit
def test_derive_array_type_7m_tp():
    """CM+PM gives 7m+TP."""
    assert derive_array_type("CM01 PM01") == "7m+TP"


@pytest.mark.unit
def test_derive_array_type_empty_defaults_12m():
    """Empty/unknown string defaults to 12m."""
    assert derive_array_type("") == "12m"
    assert derive_array_type("UNKNOWN") == "12m"


# ===========================================================================
# infer_antenna_diameter_m
# ===========================================================================


@pytest.mark.unit
def test_infer_antenna_diameter_7m():
    """7m array type gives 7.0 m dish diameter."""
    assert infer_antenna_diameter_m("7m") == 7.0


@pytest.mark.unit
def test_infer_antenna_diameter_12m():
    """12m and unknown types give 12.0 m dish diameter."""
    assert infer_antenna_diameter_m("12m") == 12.0
    assert infer_antenna_diameter_m("TP") == 12.0
    assert infer_antenna_diameter_m("") == 12.0


# ===========================================================================
# split_antenna_array_by_type
# ===========================================================================


@pytest.mark.unit
def test_split_antenna_array_mixed():
    """Mixed array strings are split into groups by type."""
    result = split_antenna_array_by_type("DA41 CM01 PM01")
    types = [g[0] for g in result]
    assert "12m" in types
    assert "7m" in types
    assert "TP" in types


@pytest.mark.unit
def test_split_antenna_array_12m_only():
    """12m-only string gives single group."""
    result = split_antenna_array_by_type("DA41 DA42")
    assert len(result) == 1
    assert result[0][0] == "12m"


@pytest.mark.unit
def test_split_antenna_array_empty():
    """Empty string gives empty list."""
    result = split_antenna_array_by_type("")
    assert result == []


# ===========================================================================
# estimate_transit_elevation
# ===========================================================================


@pytest.mark.unit
def test_estimate_transit_elevation_uses_alma_lat():
    """With default latitude, a dec near ALMA latitude gives high elevation."""
    el = estimate_transit_elevation(dec_deg=ALMA_LATITUDE_DEG)
    assert el == pytest.approx(90.0)


@pytest.mark.unit
def test_estimate_transit_elevation_clamped_low():
    """Elevation is at least 5 degrees."""
    el = estimate_transit_elevation(dec_deg=80.0)  # Very far from ALMA
    assert el >= 5.0


@pytest.mark.unit
def test_estimate_transit_elevation_clamped_high():
    """Elevation is at most 90 degrees."""
    el = estimate_transit_elevation(dec_deg=ALMA_LATITUDE_DEG)
    assert el <= 90.0


# ===========================================================================
# ObservationConfig
# ===========================================================================


@pytest.mark.unit
def test_observation_config_as_dict():
    """ObservationConfig.as_dict() returns all fields."""
    cfg = ObservationConfig(
        name="cfg0",
        array_type="12m",
        antenna_array="DA41 DA42",
        total_time_s=3600.0,
    )
    d = cfg.as_dict()
    assert d["name"] == "cfg0"
    assert d["array_type"] == "12m"
    assert d["total_time_s"] == 3600.0


# ===========================================================================
# SinglePointingObservationPlan
# ===========================================================================


@pytest.mark.unit
def test_single_pointing_plan_as_dict():
    """as_dict() serializes plan including configs."""
    cfg = ObservationConfig(name="c0", array_type="12m", antenna_array="DA41", total_time_s=3600.0)
    plan = SinglePointingObservationPlan(
        phase_center_ra_deg=10.0,
        phase_center_dec_deg=-23.0,
        fov_arcsec=30.0,
        obs_date="2024-01-01",
        pwv_mm=1.0,
        elevation_deg=70.0,
        primary_beam_model="gaussian",
        primary_beam_reference_diameter_m=12.0,
        configs=[cfg],
    )
    d = plan.as_dict()
    assert d["phase_center_ra_deg"] == 10.0
    assert len(d["configs"]) == 1
    assert d["configs"][0]["name"] == "c0"


# ===========================================================================
# _coerce_observation_config
# ===========================================================================


@pytest.mark.unit
def test_coerce_observation_config_passthrough():
    """ObservationConfig is returned unchanged."""
    cfg = ObservationConfig(name="c0", array_type="12m", antenna_array="DA41", total_time_s=3600.0)
    result = _coerce_observation_config(
        cfg, default_time_s=1800.0, default_correlator=None, index=0
    )
    assert result is cfg


@pytest.mark.unit
def test_coerce_observation_config_from_string():
    """String antenna array produces correct ObservationConfig."""
    result = _coerce_observation_config(
        "DA41 DA42", default_time_s=3600.0, default_correlator="TDM", index=0
    )
    assert result.array_type == "12m"
    assert result.total_time_s == 3600.0
    assert result.correlator == "TDM"


@pytest.mark.unit
def test_coerce_observation_config_from_dict():
    """Dict config is coerced correctly."""
    raw = {"antenna_array": "CM01 CM02", "total_time_s": 1800.0}
    result = _coerce_observation_config(
        raw, default_time_s=3600.0, default_correlator=None, index=1
    )
    assert result.array_type == "7m"
    assert result.total_time_s == 1800.0


@pytest.mark.unit
def test_coerce_observation_config_dict_missing_antenna_raises():
    """Dict without antenna_array raises ValueError."""
    with pytest.raises(ValueError, match="antenna_array"):
        _coerce_observation_config({}, default_time_s=3600.0, default_correlator=None, index=0)


@pytest.mark.unit
def test_coerce_observation_config_bad_type_raises():
    """Non-string/dict/ObservationConfig raises TypeError."""
    with pytest.raises(TypeError):
        _coerce_observation_config(42, default_time_s=3600.0, default_correlator=None, index=0)


# ===========================================================================
# normalize_observation_configs
# ===========================================================================


@pytest.mark.unit
def test_normalize_configs_empty_splits_default():
    """No raw configs splits default_antenna_array into groups."""
    result = normalize_observation_configs(
        None,
        default_antenna_array="DA41 CM01",
        default_time_s=3600.0,
    )
    types = [c.array_type for c in result]
    assert "12m" in types
    assert "7m" in types


@pytest.mark.unit
def test_normalize_configs_list_passthrough():
    """Existing ObservationConfig list is returned normalized."""
    cfg = ObservationConfig(name="c0", array_type="12m", antenna_array="DA41", total_time_s=3600.0)
    result = normalize_observation_configs(
        [cfg], default_antenna_array="DA41", default_time_s=3600.0
    )
    assert len(result) == 1
    assert result[0] is cfg


@pytest.mark.unit
def test_normalize_configs_from_strings():
    """String configs are coerced correctly."""
    result = normalize_observation_configs(
        ["DA41 DA42", "CM01"],
        default_antenna_array="DA41",
        default_time_s=7200.0,
    )
    assert len(result) == 2
    assert result[0].array_type == "12m"
    assert result[1].array_type == "7m"


# ===========================================================================
# build_single_pointing_observation_plan
# ===========================================================================


@pytest.mark.unit
def test_build_plan_basic():
    """build_single_pointing_observation_plan returns a valid plan."""
    params = SimpleNamespace(
        antenna_array="DA41 DA42",
        int_time=3600.0,
        ra=10.0,
        dec=-23.0,
        fov=1.0 / 3600.0,  # 1 arcsec in degrees
        obs_date="2024-01-01",
        pwv=1.0,
        observation_configs=None,
    )
    plan = build_single_pointing_observation_plan(params)
    assert isinstance(plan, SinglePointingObservationPlan)
    assert plan.phase_center_ra_deg == 10.0
    assert plan.fov_arcsec == pytest.approx(1.0)
    assert len(plan.configs) >= 1


@pytest.mark.unit
def test_build_plan_elevation_from_params():
    """Explicit elevation_deg in params is used directly."""
    params = SimpleNamespace(
        antenna_array="DA41",
        int_time=3600.0,
        ra=0.0,
        dec=-23.0,
        fov=1.0 / 3600.0,
        obs_date="2024-01-01",
        pwv=1.0,
        observation_configs=None,
        elevation_deg=55.0,
    )
    plan = build_single_pointing_observation_plan(params)
    assert plan.elevation_deg == 55.0


@pytest.mark.unit
def test_build_plan_tp_excluded_from_primary_beam():
    """TP-only config does not contribute to primary_beam_reference_diameter_m."""
    cfg_tp = ObservationConfig(
        name="tp",
        array_type="TP",
        antenna_array="PM01",
        total_time_s=3600.0,
        antenna_diameter_m=12.0,
    )
    params = SimpleNamespace(
        antenna_array="PM01",
        int_time=3600.0,
        ra=0.0,
        dec=-23.0,
        fov=1.0 / 3600.0,
        obs_date="2024-01-01",
        pwv=1.0,
        observation_configs=[cfg_tp],
    )
    plan = build_single_pointing_observation_plan(params)
    # No interferometric configs → defaults to 12 m
    assert plan.primary_beam_reference_diameter_m == 12.0
