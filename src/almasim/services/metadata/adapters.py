"""Shared metadata adapter helpers for API and backend layers."""

from __future__ import annotations

from typing import Any, Optional, Sequence

from .tap.queries import ALL_COLUMNS
from .tap.service import ExclusionFilters, InclusionFilters


def derive_array_type(antenna_arrays_str: object) -> str:
    """Derive a human-readable array type from the raw antenna array string."""
    if not isinstance(antenna_arrays_str, str) or not antenna_arrays_str:
        return ""
    upper = antenna_arrays_str.upper()
    types = []
    if "DA" in upper or "DV" in upper:
        types.append("12m")
    if "CM" in upper:
        types.append("7m")
    if "PM" in upper:
        types.append("TP")
    return "+".join(types) if types else ""


def build_inclusion_filters(params: Any) -> InclusionFilters:
    """Convert a parameter object into shared InclusionFilters."""
    return InclusionFilters(
        science_keyword=getattr(params, "science_keyword", None),
        scientific_category=getattr(params, "scientific_category", None),
        band=getattr(params, "bands", None),
        fov_range=getattr(params, "fov_range", None),
        time_resolution_range=getattr(params, "time_resolution_range", None),
        frequency_range=getattr(params, "frequency_range", None),
        source_name=getattr(params, "source_name", None),
        antenna_arrays=getattr(params, "antenna_arrays", None),
        array_type=getattr(params, "array_type", None),
        array_configuration=getattr(params, "array_configuration", None),
        angular_resolution_range=getattr(params, "angular_resolution_range", None),
        observation_date_range=getattr(params, "observation_date_range", None),
        qa2_status=getattr(params, "qa2_status", None),
        obs_type=getattr(params, "obs_type", None),
        proposal_id_prefix=getattr(params, "proposal_id_prefix", None),
        public_only=getattr(params, "public_only", True),
        science_only=getattr(params, "science_only", True),
        exclude_mosaic=getattr(params, "exclude_mosaic", True),
    )


def build_exclusion_filters(params: Any) -> ExclusionFilters:
    """Convert a parameter object into shared ExclusionFilters."""
    return ExclusionFilters(
        science_keyword=getattr(params, "exclude_science_keyword", None),
        scientific_category=getattr(params, "exclude_scientific_category", None),
        source_name=getattr(params, "exclude_source_name", None),
        obs_type=getattr(params, "exclude_obs_type", None),
        solar=getattr(params, "exclude_solar", False),
    )


def observation_to_metadata_record(observation: Any) -> dict[str, Any]:
    """Serialize one observation-like object into canonical metadata columns."""
    return {
        "ALMA_source_name": observation.target_name,
        "Band": observation.band,
        "Array_type": derive_array_type(observation.antenna_arrays),
        "antenna_arrays": observation.antenna_arrays,
        "Ang.res.": observation.spatial_resolution,
        "Obs.date": (
            observation.obs_release_date.isoformat() if observation.obs_release_date else None
        ),
        "Project_abstract": observation.proposal_abstract,
        "science_keyword": ", ".join(keyword.keyword for keyword in observation.science_keywords),
        "scientific_category": (
            observation.scientific_category.category if observation.scientific_category else None
        ),
        "QA2_status": observation.qa2_passed,
        "Type": observation.obs_type,
        "PWV": observation.pwv,
        "SB_name": observation.schedblock_name,
        "Vel.res.": observation.velocity_resolution,
        "RA": observation.ra,
        "Dec": observation.dec,
        "FOV": observation.s_fov,
        "Int.Time": observation.t_max,
        "Cont_sens_mJybeam": observation.cont_sensitivity_bandwidth,
        "Line_sens_10kms_mJybeam": observation.sensitivity_10kms,
        "Bandwidth": observation.bandwidth,
        "Freq": observation.frequency,
        "Freq.sup.": observation.frequency_support,
        "proposal_id": observation.proposal_id,
        "member_ous_uid": observation.member_ous_uid,
        "group_ous_uid": observation.group_ous_uid,
    }


def apply_visible_columns(
    records: Sequence[dict[str, Any]],
    visible_columns: Optional[Sequence[str]] = None,
) -> list[dict[str, Any]]:
    """Restrict records to a visible column subset while preserving order."""
    if not visible_columns:
        ordered_columns = ALL_COLUMNS
    else:
        ordered_columns = list(visible_columns)

    filtered_records = []
    for record in records:
        filtered_records.append(
            {column: record[column] for column in ordered_columns if column in record}
        )
    return filtered_records


def observations_to_metadata_records(
    observations: Sequence[Any],
    visible_columns: Optional[Sequence[str]] = None,
) -> list[dict[str, Any]]:
    """Serialize many observations to canonical metadata records."""
    records = [observation_to_metadata_record(observation) for observation in observations]
    return apply_visible_columns(records, visible_columns=visible_columns)


__all__ = [
    "derive_array_type",
    "build_inclusion_filters",
    "build_exclusion_filters",
    "observation_to_metadata_record",
    "observations_to_metadata_records",
    "apply_visible_columns",
]
