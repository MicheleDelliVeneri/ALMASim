"""Flagging and visibility summary of calibrated MeasurementSets.

One row per (field, spectral window) of an MS: how much of it is flagged, the
shape and extent of the visibilities (channels, correlations, scans, antennas,
time on source) and their weights, including how many unflagged rows still
carry an implausible WEIGHT. That last number is what exposed the Cycle-11
first pass, which applied ALMA's caltables without restoring the
Pipeline_Final flags: the rows the pipeline had flagged came out with WEIGHT
~1e7 and dominated every image. After a correct calibration it is zero.

Everything comes from grouped TaQL queries over FLAG, FLAG_ROW, WEIGHT and the
scalar columns; DATA is never read, so a few-GB MS takes well under a second.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

DEFAULT_MAX_WEIGHT = 1e4  # the imaging step's --max-weight
CALIBRATED_SUFFIX = ".ms.split.cal"

SUMMARY_COLUMNS = [
    "uid",
    "field_id",
    "field_name",
    "intents",
    "spectral_window_id",
    "channels",
    "correlations",
    "scans",
    "antennas",
    "start_time_mjd_s",
    "end_time_mjd_s",
    "on_source_s",
    "rows",
    "flagged_rows",
    "flagged_rows_pct",
    "visibilities",
    "flagged_visibilities",
    "flagged_visibilities_pct",
    "median_unflagged_weight",
    "max_unflagged_weight",
    "high_weight_rows",
    "high_weight_rows_pct",
]

# A row is flagged when FLAG_ROW is set or every channel and correlation is.
_FLAGGED = "(FLAG_ROW || all(FLAG))"
_KEYS = ["field_id", "spectral_window_id"]


def uid_of(ms_path: str | os.PathLike[str]) -> str:
    """``.../uid___A002_X1_X2.ms.split.cal`` -> ``uid___A002_X1_X2``."""
    name = Path(ms_path).name
    return name[: -len(CALIBRATED_SUFFIX)] if name.endswith(CALIBRATED_SUFFIX) else Path(name).stem


def _intent_names(obs_mode: str) -> set[str]:
    """``CALIBRATE_PHASE#ON_SOURCE,CALIBRATE_WVR#ON_SOURCE`` -> its intents without sub-intent."""
    return {token.split("#", 1)[0].strip() for token in str(obs_mode).split(",") if token.strip()}


def _query(main, columns: list[str], select: str, where: str = "", group: str = "") -> pd.DataFrame:
    from casacore.tables import taql

    result = taql(
        f"select {select} from $1 {f'where {where}' if where else ''} groupby {group}",
        tables=[main],
    )
    try:
        if result.nrows() == 0:
            return pd.DataFrame(columns=[c.lower() for c in columns])
        return pd.DataFrame({c.lower(): result.getcol(c) for c in columns})
    finally:
        result.close()


def flag_summary(
    ms_path: str | os.PathLike[str], *, max_weight: float = DEFAULT_MAX_WEIGHT
) -> pd.DataFrame:
    """Flagging and visibility properties of ``ms_path``, one row per (field, SPW).

    * ``intents``: the scan intents the field's rows carry in that window,
      without sub-intents (``CALIBRATE_PHASE``, ``OBSERVE_TARGET``).
    * ``scans``, ``antennas``: distinct SCAN_NUMBER and antennas in the rows;
      ``on_source_s`` sums each scan's span (first to last integration, plus
      one integration); start/end are the first and last TIME (MJD seconds).
    * ``flagged_rows``: rows with FLAG_ROW or every channel and correlation
      flagged; ``visibilities`` count channel x correlation values.
    * ``median_unflagged_weight`` / ``max_unflagged_weight``: over unflagged
      rows, of each row's largest WEIGHT; ``high_weight_rows`` counts the
      unflagged rows above ``max_weight``.
    """
    from casacore.tables import table

    path = str(ms_path)
    lock = {"ack": False, "readonly": True, "lockoptions": "usernoread"}
    with table(f"{path}/DATA_DESCRIPTION", **lock) as dd_table:
        spw_of_dd = [int(x) for x in dd_table.getcol("SPECTRAL_WINDOW_ID")]
    with table(f"{path}/FIELD", **lock) as field_table:
        field_names = [str(x) for x in field_table.getcol("NAME")]
    with table(f"{path}/STATE", **lock) as state_table:
        obs_modes = list(state_table.getcol("OBS_MODE")) if state_table.nrows() else []

    with table(path, **lock) as main:
        if main.nrows() == 0:
            return pd.DataFrame(columns=SUMMARY_COLUMNS)
        by_state = _query(
            main,
            [
                "FIELD_ID",
                "DATA_DESC_ID",
                "STATE_ID",
                "ROWS",
                "FLAGGED_ROWS",
                "FLAGGED_VIS",
                "VIS",
                "HIGH",
                "MAX_W",
                "CHANNELS",
                "CORRELATIONS",
                "T0",
                "T1",
            ],
            f"""FIELD_ID, DATA_DESC_ID, STATE_ID, gcount() as ROWS,
                gntrue({_FLAGGED}) as FLAGGED_ROWS, gntrue(FLAG) as FLAGGED_VIS,
                gsum(nelements(FLAG)) as VIS,
                gntrue(!{_FLAGGED} && max(WEIGHT) > {float(max_weight)!r}) as HIGH,
                gmax(iif({_FLAGGED}, 0.0, max(WEIGHT))) as MAX_W,
                gmax(shape(FLAG)[0]) as CHANNELS, gmax(shape(FLAG)[1]) as CORRELATIONS,
                gmin(TIME) as T0, gmax(TIME) as T1""",
            group="FIELD_ID, DATA_DESC_ID, STATE_ID",
        )
        median = _query(
            main,
            ["FIELD_ID", "DATA_DESC_ID", "MEDIAN_W"],
            "FIELD_ID, DATA_DESC_ID, gmedian(max(WEIGHT)) as MEDIAN_W",
            where=f"!{_FLAGGED}",
            group="FIELD_ID, DATA_DESC_ID",
        )
        scans = _query(
            main,
            ["FIELD_ID", "DATA_DESC_ID", "SCAN_NUMBER", "SPAN"],
            "FIELD_ID, DATA_DESC_ID, SCAN_NUMBER, gmax(TIME) - gmin(TIME) + gmax(INTERVAL) as SPAN",
            group="FIELD_ID, DATA_DESC_ID, SCAN_NUMBER",
        )
        antennas = pd.concat(
            [
                _query(
                    main,
                    ["FIELD_ID", "DATA_DESC_ID", column],
                    f"FIELD_ID, DATA_DESC_ID, {column}",
                    group=f"FIELD_ID, DATA_DESC_ID, {column}",
                ).rename(columns={column.lower(): "antenna"})
                for column in ("ANTENNA1", "ANTENNA2")
            ]
        )

    def with_spw(frame: pd.DataFrame) -> pd.DataFrame:
        frame["spectral_window_id"] = [spw_of_dd[int(dd)] for dd in frame["data_desc_id"]]
        return frame

    by_state = with_spw(by_state)
    by_state["intent_set"] = [
        _intent_names(obs_modes[int(s)]) if 0 <= int(s) < len(obs_modes) else set()
        for s in by_state["state_id"]
    ]
    out = (
        by_state.groupby(_KEYS, sort=True)
        .agg(
            intents=("intent_set", lambda sets: ",".join(sorted(set().union(*sets)))),
            channels=("channels", "max"),
            correlations=("correlations", "max"),
            start_time_mjd_s=("t0", "min"),
            end_time_mjd_s=("t1", "max"),
            rows=("rows", "sum"),
            flagged_rows=("flagged_rows", "sum"),
            visibilities=("vis", "sum"),
            flagged_visibilities=("flagged_vis", "sum"),
            max_unflagged_weight=("max_w", "max"),
            high_weight_rows=("high", "sum"),
        )
        .reset_index()
    )
    scan_totals = (
        with_spw(scans)
        .groupby(_KEYS)
        .agg(scans=("scan_number", "nunique"), on_source_s=("span", "sum"))
        .reset_index()
    )
    antenna_totals = (
        with_spw(antennas).groupby(_KEYS).agg(antennas=("antenna", "nunique")).reset_index()
    )
    medians = (
        with_spw(median).groupby(_KEYS).agg(median_unflagged_weight=("median_w", "max"))
        if len(median)
        else pd.DataFrame(columns=_KEYS + ["median_unflagged_weight"]).set_index(_KEYS)
    ).reset_index()
    for extra in (scan_totals, antenna_totals, medians):
        out = out.merge(extra, on=_KEYS, how="left")

    out["uid"] = uid_of(ms_path)
    out["field_name"] = [field_names[int(f)] for f in out["field_id"]]
    out["on_source_s"] = out["on_source_s"].round(3)
    out["flagged_rows_pct"] = (100 * out["flagged_rows"] / out["rows"]).round(3)
    out["flagged_visibilities_pct"] = (
        100 * out["flagged_visibilities"] / out["visibilities"].where(out["visibilities"] > 0)
    ).round(3)
    out["high_weight_rows_pct"] = (100 * out["high_weight_rows"] / out["rows"]).round(3)
    return out[SUMMARY_COLUMNS]
