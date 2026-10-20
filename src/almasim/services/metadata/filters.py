"""The one metadata filter model, for the TAP and the cache queries (MADIV-28 T6.1).

:class:`InclusionFilters` and :class:`ExclusionFilters` describe a query; the
TAP builder (``tap.service``) and the backend's cache query translate the same
objects, with the constants below shared between them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

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

    science_keyword: Optional[List[str]] = None
    scientific_category: Optional[List[str]] = None
    band: Optional[List[int]] = None
    fov_range: Optional[Tuple[float, float]] = None
    time_resolution_range: Optional[Tuple[float, float]] = None
    frequency_range: Optional[Tuple[float, float]] = None
    source_name: Optional[str] = None
    antenna_arrays: Optional[str] = None
    array_type: Optional[List[str]] = None  # e.g. ['12m', '7m', 'TP']
    array_configuration: Optional[List[str]] = None  # e.g. ['C-1', 'C-2']
    angular_resolution_range: Optional[Tuple[float, float]] = None
    observation_date_range: Optional[Tuple[str, str]] = None
    qa2_status: Optional[List[str]] = None
    obs_type: Optional[List[str]] = None
    proposal_id_prefix: Optional[List[str]] = None  # e.g. ['2016.', '2017.']
    proposal_id: Optional[List[str]] = None  # exact codes e.g. ['2019.1.00001.S']
    public_only: bool = True  # default: only query non-proprietary data
    science_only: bool = True  # default: only science observations
    exclude_mosaic: bool = True  # default: exclude mosaic observations


@dataclass
class ExclusionFilters:
    """Negative (exclude) filters for ALMA TAP science queries."""

    science_keyword: Optional[List[str]] = None
    scientific_category: Optional[List[str]] = None
    source_name: Optional[List[str]] = None
    obs_type: Optional[List[str]] = None
    solar: bool = False  # exclude observations related to the Sun
