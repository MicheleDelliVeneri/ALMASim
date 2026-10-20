"""ADQL for the ALMA TAP queries: escaped literals and the filter conditions (MADIV-28 T6.10)."""

from __future__ import annotations

import math

from astropy.time import Time

from ..filters import ARRAY_TYPE_ANTENNA_PATTERNS, ExclusionFilters, InclusionFilters, qa2_codes

_SCIENCE_COLUMNS = [
    "target_name",
    "member_ous_uid",
    "group_ous_uid",
    "pwv",
    "schedblock_name",
    "velocity_resolution",
    "spatial_resolution",
    "s_ra",
    "s_dec",
    "s_fov",
    "t_resolution",
    "proposal_id",
    "cont_sensitivity_bandwidth",
    "sensitivity_10kms",
    "obs_release_date",
    "band_list",
    "bandwidth",
    "frequency",
    "frequency_support",
    "science_keyword",
    "scientific_category",
    "antenna_arrays",
    "t_max",
    "proposal_abstract",
    "qa2_passed",
    "type",
    "data_rights",
]


def _adql_str(value: object) -> str:
    """An ADQL string literal for *value*: quoted, with embedded quotes doubled.

    User text goes into the WHERE clause, so a quote in it must not end the
    literal (``O'Brien`` broke the query; ``x' OR 'a'='a`` rewrote it).
    """
    return "'" + str(value).replace("'", "''") + "'"


def _adql_in(values) -> str:
    """An ADQL ``('a', 'b')`` list of string literals."""
    return "(" + ", ".join(_adql_str(value) for value in values) + ")"


def _adql_num(value: object) -> str:
    """*value* written into ADQL, after checking it is a finite number."""
    number = float(value)  # type: ignore[arg-type]
    if not math.isfinite(number):
        raise ValueError(f"Not a finite number: {value!r}")
    return str(value)


def _like_or_clause(column: str, values: list, negate: bool = False) -> str:
    """Build a (col LIKE '%v1%' OR col LIKE '%v2%') clause."""
    op = "NOT LIKE" if negate else "LIKE"
    join_op = " AND " if negate else " OR "
    clauses = [f"{column} {op} {_adql_str(f'%{v}%')}" for v in values]
    return f"({join_op.join(clauses)})"


def _build_inclusion_conditions(f: InclusionFilters) -> list:
    """Translate an InclusionFilters object into a list of ADQL condition strings."""
    conds = []

    if f.science_keyword:
        conds.append(_like_or_clause("science_keyword", f.science_keyword))

    if f.scientific_category:
        conds.append(_like_or_clause("scientific_category", f.scientific_category))

    if f.band:
        conds.append(_like_or_clause("band_list", [str(b) for b in f.band]))

    if f.source_name:
        conds.append(f"target_name LIKE {_adql_str(f'%{f.source_name}%')}")

    if f.antenna_arrays:
        conds.append(f"antenna_arrays LIKE {_adql_str(f'%{f.antenna_arrays}%')}")

    # Array type: 12m → DA/DV antennas, 7m → CM antennas, TP → PM antennas
    if f.array_type:
        type_clauses = [
            f"antenna_arrays LIKE {_adql_str(pattern)}"
            for atype in f.array_type
            for pattern in ARRAY_TYPE_ANTENNA_PATTERNS.get(atype, [f"%{atype}%"])
        ]
        conds.append(f"({' OR '.join(type_clauses)})")

    # Array configuration matched against schedblock_name (e.g. 'C-1', 'C-2')
    if f.array_configuration:
        conf_clauses = [
            f"schedblock_name LIKE {_adql_str(f'%{c}%')}" for c in f.array_configuration
        ]
        conds.append(f"({' OR '.join(conf_clauses)})")

    if f.angular_resolution_range:
        lo, hi = f.angular_resolution_range
        conds.append(f"spatial_resolution BETWEEN {_adql_num(lo)} AND {_adql_num(hi)}")

    if f.observation_date_range:
        mjd_min = Time(f.observation_date_range[0], format="iso", scale="utc").mjd
        mjd_max = Time(f.observation_date_range[1], format="iso", scale="utc").mjd
        conds.append(f"t_max >= {mjd_min} AND t_max <= {mjd_max}")

    # QA2 status: map human-friendly labels to TAP-stored single-char values
    if f.qa2_status:
        mapped = qa2_codes(f.qa2_status)
        conds.append(f"qa2_passed IN {_adql_in(mapped)}")

    if f.obs_type:
        clauses = " OR ".join(f"type LIKE {_adql_str(f'%{t}%')}" for t in f.obs_type)
        conds.append(f"({clauses})")

    if f.fov_range:
        lo, hi = f.fov_range
        conds.append(f"s_fov BETWEEN {_adql_num(lo)} AND {_adql_num(hi)}")

    if f.time_resolution_range:
        lo, hi = f.time_resolution_range
        conds.append(f"t_resolution BETWEEN {_adql_num(lo)} AND {_adql_num(hi)}")

    if f.frequency_range:
        lo, hi = f.frequency_range
        conds.append(f"frequency BETWEEN {_adql_num(lo)} AND {_adql_num(hi)}")

    if f.proposal_id_prefix:
        prefix_clauses = [f"proposal_id LIKE {_adql_str(f'{p}%')}" for p in f.proposal_id_prefix]
        conds.append(f"({' OR '.join(prefix_clauses)})")

    if f.proposal_id:
        conds.append(f"proposal_id IN {_adql_in(f.proposal_id)}")

    if f.public_only:
        conds.append("data_rights = 'Public'")

    if f.science_only:
        conds.append("science_observation = 'T'")

    if f.exclude_mosaic:
        conds.append("is_mosaic = 'F'")

    return conds


def _build_exclusion_conditions(f: ExclusionFilters) -> list:
    """Translate an ExclusionFilters object into a list of ADQL NOT conditions."""
    conds = []

    if f.science_keyword:
        for kw in f.science_keyword:
            conds.append(f"science_keyword NOT LIKE {_adql_str(f'%{kw}%')}")

    if f.scientific_category:
        for cat in f.scientific_category:
            conds.append(f"scientific_category NOT LIKE {_adql_str(f'%{cat}%')}")

    if f.source_name:
        for name in f.source_name:
            conds.append(f"LOWER(target_name) NOT LIKE LOWER({_adql_str(f'%{name}%')})")

    if f.obs_type:
        for t in f.obs_type:
            conds.append(f"type NOT LIKE {_adql_str(f'%{t}%')}")

    # Solar exclusion: all three fields must not mention 'sun' (case-insensitive)
    if f.solar:
        conds.append("LOWER(target_name) NOT LIKE '%sun%'")
        conds.append("LOWER(science_keyword) NOT LIKE '%sun%'")
        conds.append("LOWER(scientific_category) NOT LIKE '%sun%'")

    return conds
