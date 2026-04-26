"""High-level query functions for ALMA metadata with normalization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from .service import (
    ExclusionFilters,
    InclusionFilters,
    get_science_types,
)
from .service import (
    query_all_targets as _query_all_targets,
)
from .service import (
    query_by_science_type as _query_by_science_type,
)
from .service import (
    query_products_for_members as _query_products_for_members,
)

# Columns exposed to consumers of the metadata service.
_COL_OBS_DATE = "Obs.date"

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
    "obs_release_date": _COL_OBS_DATE,
    "band_list": "Band",
    "bandwidth": "Bandwidth",
    "frequency": "Freq",
    "frequency_support": "Freq.sup.",
    "proposal_abstract": "Project_abstract",
    "qa2_passed": "QA2_status",
    "type": "Type",
}

# All available output columns in display order.  Consumers can pass a subset
# to query_metadata_by_science via `visible_columns` to restrict the result.
ALL_COLUMNS = [
    "ALMA_source_name",
    "Band",
    "Array_type",
    "antenna_arrays",
    "Ang.res.",
    _COL_OBS_DATE,
    "Project_abstract",
    "science_keyword",
    "scientific_category",
    "QA2_status",
    "Type",
    "PWV",
    "SB_name",
    "Vel.res.",
    "RA",
    "Dec",
    "FOV",
    "Int.Time",
    "Cont_sens_mJybeam",
    "Line_sens_10kms_mJybeam",
    "Bandwidth",
    "Freq",
    "Freq.sup.",
    "proposal_id",
    "member_ous_uid",
    "group_ous_uid",
]


def _derive_array_type(antenna_arrays_str: object) -> str:
    """Derive a human-readable array type string from the antenna_arrays field.

    Returns a '+'-joined combination of the types present, e.g. '12m', '7m',
    'TP', '12m+7m', '12m+7m+TP', etc.  DA/DV → 12m, CM → 7m, PM → TP.
    """
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


def _prepare_save_path(save_to: Optional[Path | str]) -> Optional[Path]:
    """Prepare a save path, creating parent directories if needed."""
    if save_to is None or save_to == "":
        return None
    path = Path(save_to).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _normalize_metadata(
    df: pd.DataFrame,
    visible_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Normalize metadata DataFrame with column renaming and ordering.

    Parameters:
    visible_columns: Ordered subset of ALL_COLUMNS to keep.  When None all
                     columns are returned.
    """
    if df.empty:
        return df
    normalized = df.drop_duplicates(subset="member_ous_uid").copy()
    # Derive Array_type before renaming so antenna_arrays is still raw
    normalized["Array_type"] = normalized["antenna_arrays"].apply(_derive_array_type)
    normalized.rename(columns=_COLUMN_RENAMES, inplace=True)
    column_order = list(visible_columns) if visible_columns else ALL_COLUMNS
    # Only keep columns that actually exist in the DataFrame
    column_order = [c for c in column_order if c in normalized.columns]
    normalized = normalized[column_order]
    if _COL_OBS_DATE in normalized.columns:
        normalized.loc[:, _COL_OBS_DATE] = normalized[_COL_OBS_DATE].apply(
            lambda x: x.split("T")[0] if isinstance(x, str) and "T" in x else x
        )
    return normalized


def query_science_types() -> Tuple[Sequence[str], Sequence[str]]:
    """Return cached science keywords and categories from the ALMA TAP."""
    return get_science_types()


def query_metadata_by_science(
    include: Optional[InclusionFilters] = None,
    exclude: Optional[ExclusionFilters] = None,
    visible_columns: Optional[Sequence[str]] = None,
    save_to: Optional[Path | str] = None,
) -> pd.DataFrame:
    """Query ALMA obscore metadata filtered by inclusion and exclusion criteria.

    Parameters:
    include: InclusionFilters specifying what to keep (science_keyword,
             scientific_category, band, fov_range, time_resolution_range,
             frequency_range, source_name, antenna_arrays, array_type,
             array_configuration, angular_resolution_range,
             observation_date_range, qa2_status, obs_type).
    exclude: ExclusionFilters specifying what to remove (science_keyword,
             scientific_category, source_name, obs_type, solar).
    visible_columns: Ordered list of column names (from ALL_COLUMNS) to include
                     in the returned DataFrame.  Pass None to get all columns.
    save_to: Optional CSV file path to write results.

    Returns:
    pandas.DataFrame of normalized metadata.
    """
    df = _query_by_science_type(include=include, exclude=exclude)
    if df.empty:
        return df
    normalized = _normalize_metadata(df, visible_columns=visible_columns)
    save_path = _prepare_save_path(save_to)
    if save_path is not None:
        normalized.to_csv(save_path, index=False)
    return normalized


def query_metadata_by_targets(
    targets: Iterable[Tuple[str, str]],
    visible_columns: Optional[Sequence[str]] = None,
    save_to: Optional[Path | str] = None,
) -> pd.DataFrame:
    """Query metadata for a pre-defined list of (target_name, member_ous_uid)."""
    df = _query_all_targets(list(targets))
    normalized = _normalize_metadata(df, visible_columns=visible_columns)
    save_path = _prepare_save_path(save_to)
    if save_path is not None:
        normalized.to_csv(save_path, index=False)
    return normalized


def query_products(
    member_ous_uids: List[str] | str,
) -> pd.DataFrame:
    """Return available data products for the given member OUS UID(s).

    Each row is a single data product (visibility data, cube, continuum image,
    etc.) associated with the given member_ous_uid(s).  Unlike the main science
    query the result is NOT deduplicated, so every product appears.

    The returned DataFrame includes: member_ous_uid, target_name,
    dataproduct_type, calib_level, access_url, obs_publisher_did,
    band_list, qa2_passed, type.
    """
    return _query_products_for_members(member_ous_uids)


def load_metadata(metadata_path: Path | str) -> pd.DataFrame:
    """Load previously saved metadata from CSV or JSON."""
    raw = str(metadata_path)
    if "\x00" in raw or ".." in raw:
        raise ValueError(f"Invalid metadata path: {metadata_path!r}")
    path = Path(metadata_path).expanduser().resolve()  # lgtm[py/path-injection]
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        if isinstance(payload, dict):
            records = payload.get("data", [])
        elif isinstance(payload, list):
            records = payload
        else:
            raise ValueError(f"Unsupported metadata JSON format: {path}")
        return pd.DataFrame.from_records(records)
    return pd.read_csv(path)
