"""Native ALMASim logical model for MeasurementSet v2 content."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class MeasurementSetTable:
    """Logical representation of one MSv2 table."""

    name: str
    nrows: int
    columns: dict[str, Any]


@dataclass(frozen=True)
class MeasurementSetModel:
    """Logical representation of an MSv2 dataset before serialization."""

    main_keywords: dict[str, Any]
    main: MeasurementSetTable
    subtables: dict[str, MeasurementSetTable]


def build_measurement_set_model(
    *,
    visibility_table: Mapping[str, Any],
    project_name: str,
    source_name: str,
    telescope_name: str = "ALMA",
) -> MeasurementSetModel:
    """Build ALMASim's native logical MSv2 model from row-wise visibilities."""
    data = np.asarray(visibility_table["data"], dtype=np.complex64)
    model_data = np.asarray(visibility_table["model_data"], dtype=np.complex64)
    flag = np.asarray(visibility_table["flag"], dtype=np.bool_)
    uvw_m = np.asarray(visibility_table["uvw_m"], dtype=np.float64)
    antenna1 = np.asarray(visibility_table["antenna1"], dtype=np.int32)
    antenna2 = np.asarray(visibility_table["antenna2"], dtype=np.int32)
    time_mjd_s = np.asarray(visibility_table["time_mjd_s"], dtype=np.float64)
    interval_s = np.asarray(visibility_table["interval_s"], dtype=np.float64)
    exposure_s = np.asarray(visibility_table["exposure_s"], dtype=np.float64)
    weight = np.asarray(visibility_table["weight"], dtype=np.float32)
    sigma = np.asarray(visibility_table["sigma"], dtype=np.float32)
    channel_freq_hz = np.asarray(visibility_table["channel_freq_hz"], dtype=np.float64)
    antenna_names = list(visibility_table["antenna_names"])
    antenna_positions_m = np.asarray(
        visibility_table["antenna_positions_m"], dtype=np.float64
    )
    field_ra_rad = float(visibility_table["field_ra_rad"])
    field_dec_rad = float(visibility_table["field_dec_rad"])
    nrows, ncorr, nchan = data.shape
    time_range = np.array([time_mjd_s.min(), time_mjd_s.max()], dtype=np.float64)

    main_columns = {
        "TIME": time_mjd_s,
        "TIME_CENTROID": time_mjd_s,
        "INTERVAL": interval_s,
        "EXPOSURE": exposure_s,
        "UVW": uvw_m,
        "ANTENNA1": antenna1,
        "ANTENNA2": antenna2,
        "FEED1": np.zeros(nrows, dtype=np.int32),
        "FEED2": np.zeros(nrows, dtype=np.int32),
        "DATA_DESC_ID": np.zeros(nrows, dtype=np.int32),
        "FIELD_ID": np.zeros(nrows, dtype=np.int32),
        "ARRAY_ID": np.zeros(nrows, dtype=np.int32),
        "OBSERVATION_ID": np.zeros(nrows, dtype=np.int32),
        "PROCESSOR_ID": np.full(nrows, -1, dtype=np.int32),
        "SCAN_NUMBER": np.ones(nrows, dtype=np.int32),
        "STATE_ID": np.zeros(nrows, dtype=np.int32),
        "DATA": data,
        "MODEL_DATA": model_data,
        "FLAG": flag,
        "FLAG_ROW": np.all(flag, axis=(1, 2)),
        "WEIGHT": weight,
        "SIGMA": sigma,
    }

    chan_width = (
        np.gradient(channel_freq_hz).astype(np.float64)
        if len(channel_freq_hz) > 1
        else np.array([1.0], dtype=np.float64)
    )
    direction_3d = np.array([[[field_ra_rad, field_dec_rad]]], dtype=np.float64)
    direction_2d = np.array([field_ra_rad, field_dec_rad], dtype=np.float64)

    subtables = {
        "ANTENNA": MeasurementSetTable(
            name="ANTENNA",
            nrows=len(antenna_names),
            columns={
                "NAME": np.asarray(antenna_names, dtype=str),
                "STATION": np.asarray(antenna_names, dtype=str),
                "TYPE": np.asarray(["GROUND-BASED"] * len(antenna_names), dtype=str),
                "MOUNT": np.asarray(["alt-az"] * len(antenna_names), dtype=str),
                "POSITION": antenna_positions_m,
                "OFFSET": np.zeros((len(antenna_names), 3), dtype=np.float64),
                "DISH_DIAMETER": np.full(len(antenna_names), 12.0, dtype=np.float64),
                "FLAG_ROW": np.zeros(len(antenna_names), dtype=np.bool_),
            },
        ),
        "SPECTRAL_WINDOW": MeasurementSetTable(
            name="SPECTRAL_WINDOW",
            nrows=1,
            columns={
                "NUM_CHAN": int(nchan),
                "CHAN_FREQ": channel_freq_hz.astype(np.float64),
                "CHAN_WIDTH": chan_width,
                "EFFECTIVE_BW": np.abs(chan_width),
                "RESOLUTION": np.abs(chan_width),
                "REF_FREQUENCY": float(np.median(channel_freq_hz)),
                "TOTAL_BANDWIDTH": float(np.sum(np.abs(chan_width))),
                "NAME": "ALMASim SPW 0",
                "MEAS_FREQ_REF": 5,
                "NET_SIDEBAND": 1,
                "FLAG_ROW": False,
            },
        ),
        "POLARIZATION": MeasurementSetTable(
            name="POLARIZATION",
            nrows=1,
            columns={
                "NUM_CORR": int(ncorr),
                "CORR_TYPE": np.array([9], dtype=np.int32),
                "CORR_PRODUCT": np.array([[0], [0]], dtype=np.int32),
                "FLAG_ROW": False,
            },
        ),
        "DATA_DESCRIPTION": MeasurementSetTable(
            name="DATA_DESCRIPTION",
            nrows=1,
            columns={
                "SPECTRAL_WINDOW_ID": 0,
                "POLARIZATION_ID": 0,
                "FLAG_ROW": False,
            },
        ),
        "FIELD": MeasurementSetTable(
            name="FIELD",
            nrows=1,
            columns={
                "NAME": source_name,
                "CODE": "",
                "NUM_POLY": 0,
                "DELAY_DIR": direction_3d,
                "PHASE_DIR": direction_3d,
                "REFERENCE_DIR": direction_3d,
                "SOURCE_ID": 0,
                "TIME": float(np.mean(time_range)),
                "FLAG_ROW": False,
            },
        ),
        "OBSERVATION": MeasurementSetTable(
            name="OBSERVATION",
            nrows=1,
            columns={
                "TELESCOPE_NAME": telescope_name,
                "OBSERVER": "ALMASim",
                "PROJECT": project_name,
                "SCHEDULE_TYPE": "ALMASim",
                "TIME_RANGE": time_range.astype(np.float64),
                "RELEASE_DATE": float(time_range.max()),
                "FLAG_ROW": False,
            },
        ),
        "SOURCE": MeasurementSetTable(
            name="SOURCE",
            nrows=1,
            columns={
                "SOURCE_ID": 0,
                "TIME": float(np.mean(time_range)),
                "INTERVAL": float(time_range[1] - time_range[0]),
                "SPECTRAL_WINDOW_ID": -1,
                "NUM_LINES": 0,
                "NAME": source_name,
                "CODE": "",
                "CALIBRATION_GROUP": 0,
                "DIRECTION": direction_2d,
            },
        ),
        "STATE": MeasurementSetTable(
            name="STATE",
            nrows=1,
            columns={
                "SIG": True,
                "REF": False,
                "CAL": 0.0,
                "LOAD": 0.0,
                "SUB_SCAN": 0,
                "OBS_MODE": "ON_SOURCE",
                "FLAG_ROW": False,
            },
        ),
        "HISTORY": MeasurementSetTable(
            name="HISTORY",
            nrows=1,
            columns={
                "TIME": float(np.mean(time_range)),
                "OBSERVATION_ID": 0,
                "MESSAGE": "ALMASim MSv2 export",
                "PRIORITY": "NORMAL",
                "ORIGIN": "ALMASim",
                "OBJECT_ID": 0,
                "APPLICATION": "ALMASim",
                "CLI_COMMAND": "export_native_ms",
                "APP_PARAMS": project_name,
            },
        ),
    }

    return MeasurementSetModel(
        main_keywords={
            "type": "Measurement Set",
            "subType": "ALMASim",
            "MS_VERSION": 2.0,
        },
        main=MeasurementSetTable(name="MAIN", nrows=nrows, columns=main_columns),
        subtables=subtables,
    )
