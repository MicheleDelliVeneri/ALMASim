"""The one metadata filter model, for the TAP and the cache queries (MADIV-28 T6.1).

:class:`InclusionFilters` and :class:`ExclusionFilters` describe a query; the
TAP builder (``tap.service``) and the backend's cache query translate the same
objects, with the constants below shared between them.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

# Map human-friendly QA2 labels to ALMA TAP stored values
QA2_STATUS_MAP = {
    "Pass": "T",
    "Fail": "F",
    "SemiPass": "X",
    # Pass-through for raw values already in TAP form
    "T": "T",
    "F": "F",
    "X": "X",
}

# Antenna prefix patterns used to identify array type from antenna_arrays column
# 12m arrays: DA / DV antennas; 7m arrays: CM antennas; TP: PM antennas
ARRAY_TYPE_ANTENNA_PATTERNS = {
    "12m": ["%DA%", "%DV%"],
    "7m": ["%CM%"],
    "TP": ["%PM%"],
}


def qa2_codes(statuses: Iterable[str]) -> list[str]:
    """QA2 labels ('Pass', 'Fail', 'SemiPass') as the archive stores them ('T', 'F', 'X')."""
    return [QA2_STATUS_MAP.get(status, status) for status in statuses]


@dataclass
class InclusionFilters:
    """Positive (include-only) filters for ALMA TAP science queries."""

    science_keyword: list[str] | None = None
    scientific_category: list[str] | None = None
    band: list[int] | None = None
    fov_range: tuple[float, float] | None = None
    time_resolution_range: tuple[float, float] | None = None
    frequency_range: tuple[float, float] | None = None
    source_name: str | None = None
    antenna_arrays: str | None = None
    array_type: list[str] | None = None  # e.g. ['12m', '7m', 'TP']
    array_configuration: list[str] | None = None  # e.g. ['C-1', 'C-2']
    angular_resolution_range: tuple[float, float] | None = None
    observation_date_range: tuple[str, str] | None = None
    qa2_status: list[str] | None = None
    obs_type: list[str] | None = None
    proposal_id_prefix: list[str] | None = None  # e.g. ['2016.', '2017.']
    proposal_id: list[str] | None = None  # exact codes e.g. ['2019.1.00001.S']
    public_only: bool = True  # default: only query non-proprietary data
    science_only: bool = True  # default: only science observations
    exclude_mosaic: bool = True  # default: exclude mosaic observations


@dataclass
class ExclusionFilters:
    """Negative (exclude) filters for ALMA TAP science queries."""

    science_keyword: list[str] | None = None
    scientific_category: list[str] | None = None
    source_name: list[str] | None = None
    obs_type: list[str] | None = None
    solar: bool = False  # exclude observations related to the Sun
