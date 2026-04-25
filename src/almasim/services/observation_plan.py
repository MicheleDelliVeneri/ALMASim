"""Observation planning helpers for single-pointing ALMA simulations."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Iterable, Mapping, Optional

ALMA_LATITUDE_DEG = -23.028


def derive_array_type(antenna_array: str) -> str:
    """Derive a coarse ALMA array type from an antenna string."""
    upper = (antenna_array or "").upper()
    has_12m = ("DA" in upper) or ("DV" in upper)
    has_7m = "CM" in upper
    has_tp = "PM" in upper

    if has_12m and not has_7m and not has_tp:
        return "12m"
    if has_7m and not has_12m and not has_tp:
        return "7m"
    if has_tp and not has_12m and not has_7m:
        return "TP"
    if has_12m and has_7m:
        return "12m+7m"
    if has_12m and has_tp:
        return "12m+TP"
    if has_7m and has_tp:
        return "7m+TP"
    if has_12m and has_7m and has_tp:
        return "12m+7m+TP"
    return "12m"


def infer_antenna_diameter_m(array_type: str) -> float:
    """Infer a representative dish diameter for an ALMA array type."""
    normalized = (array_type or "12m").lower()
    if normalized == "7m":
        return 7.0
    return 12.0


def split_antenna_array_by_type(antenna_array: str) -> list[tuple[str, str]]:
    """Split a raw ALMA antenna-array string into inferred ALMA array groups."""
    tokens = [token for token in str(antenna_array or "").split() if token.strip()]
    groups: dict[str, list[str]] = {"12m": [], "7m": [], "TP": []}
    for token in tokens:
        upper = token.upper()
        if "CM" in upper:
            groups["7m"].append(token)
        elif "PM" in upper:
            groups["TP"].append(token)
        else:
            groups["12m"].append(token)

    ordered_groups: list[tuple[str, str]] = []
    for array_type in ("12m", "7m", "TP"):
        if groups[array_type]:
            ordered_groups.append((array_type, " ".join(groups[array_type])))
    return ordered_groups


def estimate_transit_elevation(
    dec_deg: float, site_latitude_deg: float = ALMA_LATITUDE_DEG
) -> float:
    """Estimate source elevation at transit for a single-pointing plan."""
    elevation = 90.0 - abs(site_latitude_deg - dec_deg)
    return max(5.0, min(90.0, elevation))


@dataclass
class ObservationConfig:
    """A single interferometric or total-power observing configuration."""

    name: str
    array_type: str
    antenna_array: str
    total_time_s: float
    correlator: Optional[str] = None
    antenna_diameter_m: float = 12.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SinglePointingObservationPlan:
    """Explicit single-pointing observing plan shared by simulation stages."""

    phase_center_ra_deg: float
    phase_center_dec_deg: float
    fov_arcsec: float
    obs_date: str
    pwv_mm: float
    elevation_deg: float
    primary_beam_model: str
    primary_beam_reference_diameter_m: float
    configs: list[ObservationConfig]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["configs"] = [cfg.as_dict() for cfg in self.configs]
        return payload


def _coerce_observation_config(
    raw_config: Any,
    *,
    default_time_s: float,
    default_correlator: Optional[str],
    index: int,
) -> ObservationConfig:
    """Normalize a user-supplied configuration mapping into a dataclass."""
    if isinstance(raw_config, ObservationConfig):
        return raw_config

    if isinstance(raw_config, str):
        antenna_array = raw_config
        array_type = derive_array_type(antenna_array)
        return ObservationConfig(
            name=f"config_{index}",
            array_type=array_type,
            antenna_array=antenna_array,
            total_time_s=float(default_time_s),
            correlator=default_correlator,
            antenna_diameter_m=infer_antenna_diameter_m(array_type),
        )

    if not isinstance(raw_config, Mapping):
        raise TypeError(
            "Observation config entries must be dict-like, strings, "
            "or ObservationConfig objects"
        )

    antenna_array = str(
        raw_config.get("antenna_array") or raw_config.get("antennalist") or ""
    )
    if not antenna_array:
        raise ValueError("Observation config is missing 'antenna_array'")

    array_type = str(raw_config.get("array_type") or derive_array_type(antenna_array))
    antenna_diameter_m = float(
        raw_config.get("antenna_diameter_m") or infer_antenna_diameter_m(array_type)
    )
    return ObservationConfig(
        name=str(raw_config.get("name") or f"config_{index}"),
        array_type=array_type,
        antenna_array=antenna_array,
        total_time_s=float(raw_config.get("total_time_s") or default_time_s),
        correlator=raw_config.get("correlator") or default_correlator,
        antenna_diameter_m=antenna_diameter_m,
    )


def normalize_observation_configs(
    raw_configs: Optional[Iterable[Any]],
    *,
    default_antenna_array: str,
    default_time_s: float,
    default_correlator: Optional[str] = None,
) -> list[ObservationConfig]:
    """Build a normalized config list while preserving current single-config behavior."""
    if not raw_configs:
        split_groups = split_antenna_array_by_type(default_antenna_array)
        if not split_groups:
            array_type = derive_array_type(default_antenna_array)
            split_groups = [(array_type, default_antenna_array)]
        return [
            ObservationConfig(
                name=f"config_{index}_{array_type}",
                array_type=array_type,
                antenna_array=antenna_group,
                total_time_s=float(default_time_s),
                correlator=default_correlator,
                antenna_diameter_m=infer_antenna_diameter_m(array_type),
            )
            for index, (array_type, antenna_group) in enumerate(split_groups)
        ]

    return [
        _coerce_observation_config(
            raw_config,
            default_time_s=default_time_s,
            default_correlator=default_correlator,
            index=index,
        )
        for index, raw_config in enumerate(raw_configs)
    ]


def build_single_pointing_observation_plan(
    params: Any,
) -> SinglePointingObservationPlan:
    """Construct an explicit single-pointing observation plan from simulation params."""
    configs = normalize_observation_configs(
        getattr(params, "observation_configs", None),
        default_antenna_array=params.antenna_array,
        default_time_s=params.int_time,
        default_correlator=getattr(params, "correlator", None),
    )
    int_like_configs = [cfg for cfg in configs if cfg.array_type != "TP"]
    primary_beam_diameter_m = max(
        (cfg.antenna_diameter_m for cfg in int_like_configs),
        default=12.0,
    )
    return SinglePointingObservationPlan(
        phase_center_ra_deg=float(params.ra),
        phase_center_dec_deg=float(params.dec),
        fov_arcsec=float(params.fov * 3600.0),
        obs_date=str(params.obs_date),
        pwv_mm=float(params.pwv),
        elevation_deg=float(
            getattr(params, "elevation_deg", None)
            or estimate_transit_elevation(float(params.dec))
        ),
        primary_beam_model="gaussian",
        primary_beam_reference_diameter_m=float(primary_beam_diameter_m),
        configs=configs,
    )
