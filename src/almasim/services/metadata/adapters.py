"""Shared metadata adapter helpers for API and backend layers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..instrument.array import derive_array_type
from .tap.queries import ALL_COLUMNS


def _integration_time(observation: Any) -> Any:
    """Int.Time of a cached observation.

    It is stored in ``t_resolution``; rows cached before MADIV-28 T1.12 hold
    it in ``t_max`` instead.
    """
    if observation.t_resolution is not None:
        return observation.t_resolution
    return observation.t_max


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
        "Int.Time": _integration_time(observation),
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
    visible_columns: Sequence[str] | None = None,
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
    visible_columns: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Serialize many observations to canonical metadata records."""
    records = [observation_to_metadata_record(observation) for observation in observations]
    return apply_visible_columns(records, visible_columns=visible_columns)


__all__ = [
    "derive_array_type",
    "observation_to_metadata_record",
    "observations_to_metadata_records",
    "apply_visible_columns",
]
