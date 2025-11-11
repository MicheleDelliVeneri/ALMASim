"""High-level query functions for ALMA metadata with normalization."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import pandas as pd

from .service import (
    get_science_types,
    query_by_science_type as _query_by_science_type,
    query_all_targets as _query_all_targets,
)

# Columns exposed to consumers of the metadata service.
_COLUMN_RENAMES = {
    "target_name": "ALMA_source_name",
    "pwv": "PWV",
    "schedblock_name": "SB_name",
    "velocity_resolution": "Vel.res.",
    "spatial_resolution": "Ang.res.",
    "s_ra": "RA",
    "s_dec": "Dec",
    "s_fov": "FOV",
    "t_resolution": "Int.Time",
    "cont_sensitivity_bandwidth": "Cont_sens_mJybeam",
    "sensitivity_10kms": "Line_sens_10kms_mJybeam",
    "obs_release_date": "Obs.date",
    "band_list": "Band",
    "bandwidth": "Bandwidth",
    "frequency": "Freq",
    "frequency_support": "Freq.sup.",
}

_COLUMN_ORDER = [
    "ALMA_source_name",
    "Band",
    "PWV",
    "SB_name",
    "Vel.res.",
    "Ang.res.",
    "RA",
    "Dec",
    "FOV",
    "Int.Time",
    "Cont_sens_mJybeam",
    "Line_sens_10kms_mJybeam",
    "Obs.date",
    "Bandwidth",
    "Freq",
    "Freq.sup.",
    "antenna_arrays",
    "proposal_id",
    "member_ous_uid",
    "group_ous_uid",
]


def _prepare_save_path(save_to: Optional[Path | str]) -> Optional[Path]:
    """Prepare a save path, creating parent directories if needed."""
    if save_to is None or save_to == "":
        return None
    path = Path(save_to).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _normalize_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize metadata DataFrame with column renaming and ordering."""
    if df.empty:
        return df
    normalized = df.drop_duplicates(subset="member_ous_uid").copy()
    normalized.rename(columns=_COLUMN_RENAMES, inplace=True)
    normalized = normalized[_COLUMN_ORDER]
    if "Obs.date" in normalized:
        normalized.loc[:, "Obs.date"] = normalized["Obs.date"].apply(
            lambda x: x.split("T")[0] if isinstance(x, str) and "T" in x else x
        )
    return normalized


def query_science_types() -> Tuple[Sequence[str], Sequence[str]]:
    """Return cached science keywords and categories from the ALMA TAP."""
    return get_science_types()


def query_metadata_by_science(
    science_keyword: Optional[Sequence[str]] = None,
    scientific_category: Optional[Sequence[str]] = None,
    bands: Optional[Sequence[int]] = None,
    fov_range: Optional[Tuple[float, float]] = None,
    time_resolution_range: Optional[Tuple[float, float]] = None,
    frequency_range: Optional[Tuple[float, float]] = None,
    save_to: Optional[Path | str] = None,
) -> pd.DataFrame:
    """Query ALMA obscore metadata filtered by science configuration."""
    df = _query_by_science_type(
        science_keyword,
        scientific_category,
        bands,
        fov_range,
        time_resolution_range,
        frequency_range,
    )
    df = df[df.get("science_keyword", "") != ""]
    normalized = _normalize_metadata(df)
    save_path = _prepare_save_path(save_to)
    if save_path is not None:
        normalized.to_csv(save_path, index=False)
    return normalized


def query_metadata_by_targets(
    targets: Iterable[Tuple[str, str]],
    save_to: Optional[Path | str] = None,
) -> pd.DataFrame:
    """Query metadata for a pre-defined list of (target_name, member_ous_uid)."""
    df = _query_all_targets(list(targets))
    normalized = _normalize_metadata(df)
    save_path = _prepare_save_path(save_to)
    if save_path is not None:
        normalized.to_csv(save_path, index=False)
    return normalized


def load_metadata(metadata_path: Path | str) -> pd.DataFrame:
    """Load a previously saved metadata CSV."""
    path = Path(metadata_path).expanduser().resolve()
    return pd.read_csv(path)

