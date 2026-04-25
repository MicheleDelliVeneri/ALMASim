"""MSv2 writer/reader backed by casatools or python-casacore (fallback)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

_SUBTABLE_NAMES = (
    "ANTENNA",
    "DATA_DESCRIPTION",
    "FIELD",
    "HISTORY",
    "OBSERVATION",
    "POLARIZATION",
    "SOURCE",
    "SPECTRAL_WINDOW",
    "STATE",
)

# ---------------------------------------------------------------------------
# Value-type helper
# ---------------------------------------------------------------------------


def _vtype(value) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, np.integer)):
        return "int"
    if isinstance(value, np.float32):
        return "float"
    if isinstance(value, (float, np.floating)):
        return "double"
    if isinstance(value, (complex, np.complexfloating)):
        return "complex"
    return "string"


def _sca_desc(name: str, value) -> tuple[str, dict]:
    return name, {
        "valueType": _vtype(value),
        "dataManagerType": "StandardStMan",
        "dataManagerGroup": "",
        "keywords": {},
        "maxlen": 0,
        "option": 0,
        "comment": "",
    }


def _arr_desc(name: str, value, ndim: int, shape: list | None = None) -> tuple[str, dict]:
    d: dict = {
        "valueType": _vtype(value),
        "dataManagerType": "StandardStMan",
        "dataManagerGroup": "",
        "keywords": {},
        "maxlen": 0,
        "ndim": ndim,
        "option": 0,
        "comment": "",
    }
    if shape is not None:
        d["shape"] = np.array(shape)
        d["option"] = 5  # FixedShape
    return name, d


def _make_tabledesc(coldesc_list: list[tuple[str, dict]]) -> dict:
    desc: dict = {
        "_define_hypercolumn_": {},
        "_keywords_": {},
        "_private_keywords_": {},
    }
    for name, col_desc in coldesc_list:
        desc[name] = col_desc
    return desc


# ---------------------------------------------------------------------------
# Backend abstraction
# ---------------------------------------------------------------------------


class _CasatoolsBackend:
    def __init__(self):
        import casatools as _ct

        self._ct = _ct

    def create(self, path: str, coldesc_list: list, nrow: int):
        desc = _make_tabledesc(coldesc_list)
        tb = self._ct.table()
        tb.create(path, desc, nrow=nrow)
        return tb

    def open(self, path: str, readonly: bool = True):
        tb = self._ct.table()
        tb.open(path, nomodify=readonly)
        return tb

    @staticmethod
    def sca(name, value, **_kw):
        return _sca_desc(name, value)

    @staticmethod
    def arr(name, value, ndim, shape=None, **_kw):
        return _arr_desc(name, value, ndim, shape)


class _CasacoreBackend:
    def __init__(self):
        from casacore import tables as _  # noqa: F401 — fail fast if not installed

    def create(self, path: str, coldesc_list: list, nrow: int):
        from casacore.tables import table

        desc = _make_tabledesc(coldesc_list)
        return table(path, tabledesc=desc, nrow=nrow, readonly=False)

    def open(self, path: str, readonly: bool = True):
        from casacore.tables import table

        return table(path, readonly=readonly)

    @staticmethod
    def sca(name, value, **_kw):
        return _sca_desc(name, value)

    @staticmethod
    def arr(name, value, ndim, shape=None, **_kw):
        return _arr_desc(name, value, ndim, shape)


def _get_backend():
    for cls in (_CasatoolsBackend, _CasacoreBackend):
        try:
            return cls()
        except Exception:
            continue
    raise ImportError(
        "MS export requires casatools or python-casacore.\n"
        "  pip install casatools        (requires Python ≤ 3.12)\n"
        "  pip install python-casacore  (broader Python support)\n"
        "See https://casadocs.readthedocs.io/en/stable/api/casatools.html"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def export_native_ms(
    *,
    ms_path: str | Path,
    visibility_table: dict[str, Any],
    project_name: str,
    source_name: str,
    telescope_name: str = "ALMA",
) -> str:
    """Write a CASA MSv2 to *ms_path* using casatools or python-casacore."""
    backend = _get_backend()
    ms_path = str(Path(ms_path).expanduser().resolve())
    vt = visibility_table

    data = np.asarray(vt["data"], dtype=np.complex64)
    channel_freq_hz = np.asarray(vt["channel_freq_hz"], dtype=np.float64)
    antenna_names = list(vt["antenna_names"])
    antenna_positions_m = np.asarray(vt["antenna_positions_m"], dtype=np.float64)
    field_ra_rad = float(vt["field_ra_rad"])
    field_dec_rad = float(vt["field_dec_rad"])
    time_mjd_s = np.asarray(vt["time_mjd_s"], dtype=np.float64)

    nrows, ncorr, nchan = data.shape
    nant = len(antenna_names)
    time_range = np.array([time_mjd_s.min(), time_mjd_s.max()])

    # Main table must be created first — casatools creates the MS root directory.
    _write_main(backend, ms_path, nrows, vt)
    _write_antenna(backend, ms_path, antenna_names, antenna_positions_m, nant)
    _write_spectral_window(backend, ms_path, channel_freq_hz, nchan)
    _write_polarization(backend, ms_path, ncorr)
    _write_data_description(backend, ms_path)
    _write_field(backend, ms_path, source_name, field_ra_rad, field_dec_rad, time_range)
    _write_observation(backend, ms_path, telescope_name, project_name, time_range)
    _write_source(backend, ms_path, source_name, field_ra_rad, field_dec_rad, time_range)
    _write_state(backend, ms_path)
    _write_history(backend, ms_path, project_name, time_range)
    return ms_path


def read_native_ms(ms_path: str | Path) -> dict[str, Any]:
    """Read a CASA MSv2 back into a visibility-table dict."""
    backend = _get_backend()
    ms_path = str(Path(ms_path).expanduser().resolve())

    tb = backend.open(ms_path, readonly=True)
    # casatools/casacore both return (col_shape, nrows) — transpose to (nrows, ...)
    data = tb.getcol("DATA").transpose(2, 0, 1).astype(np.complex64)
    model_data = tb.getcol("MODEL_DATA").transpose(2, 0, 1).astype(np.complex64)
    flag = tb.getcol("FLAG").transpose(2, 0, 1)
    uvw_m = tb.getcol("UVW").T.astype(np.float64)
    antenna1 = tb.getcol("ANTENNA1").astype(np.int32)
    antenna2 = tb.getcol("ANTENNA2").astype(np.int32)
    time_mjd_s = tb.getcol("TIME").astype(np.float64)
    interval_s = tb.getcol("INTERVAL").astype(np.float64)
    exposure_s = tb.getcol("EXPOSURE").astype(np.float64)
    weight = tb.getcol("WEIGHT").T.astype(np.float32)
    sigma = tb.getcol("SIGMA").T.astype(np.float32)
    tb.close()

    spw = backend.open(os.path.join(ms_path, "SPECTRAL_WINDOW"), readonly=True)
    channel_freq_hz = np.asarray(spw.getcell("CHAN_FREQ", 0), dtype=np.float64)
    spw.close()

    ant = backend.open(os.path.join(ms_path, "ANTENNA"), readonly=True)
    antenna_names = list(ant.getcol("NAME"))
    antenna_positions_m = ant.getcol("POSITION").T.astype(np.float64)
    ant.close()

    fld = backend.open(os.path.join(ms_path, "FIELD"), readonly=True)
    phase_dir = fld.getcell("PHASE_DIR", 0)
    field_ra_rad = float(np.asarray(phase_dir).flat[0])
    field_dec_rad = float(np.asarray(phase_dir).flat[1])
    source_name = str(fld.getcell("NAME", 0))
    fld.close()

    obs = backend.open(os.path.join(ms_path, "OBSERVATION"), readonly=True)
    telescope_name = str(obs.getcell("TELESCOPE_NAME", 0))
    project_name = str(obs.getcell("PROJECT", 0))
    obs.close()

    return {
        "uvw_m": uvw_m,
        "antenna1": antenna1,
        "antenna2": antenna2,
        "time_mjd_s": time_mjd_s,
        "interval_s": interval_s,
        "exposure_s": exposure_s,
        "data": data,
        "model_data": model_data,
        "flag": flag,
        "weight": weight,
        "sigma": sigma,
        "channel_freq_hz": channel_freq_hz,
        "antenna_names": antenna_names,
        "antenna_positions_m": antenna_positions_m,
        "field_ra_rad": field_ra_rad,
        "field_dec_rad": field_dec_rad,
        "source_name": source_name,
        "telescope_name": telescope_name,
        "project_name": project_name,
    }


# ---------------------------------------------------------------------------
# Subtable writers
# ---------------------------------------------------------------------------


def _write_antenna(backend, ms_path, antenna_names, antenna_positions_m, nant):
    tb = backend.create(
        os.path.join(ms_path, "ANTENNA"),
        [
            backend.sca("NAME", ""),
            backend.sca("STATION", ""),
            backend.sca("TYPE", ""),
            backend.sca("MOUNT", ""),
            backend.sca("DISH_DIAMETER", 0.0),
            backend.sca("FLAG_ROW", False),
            backend.arr("POSITION", 0.0, 1, shape=[3]),
            backend.arr("OFFSET", 0.0, 1, shape=[3]),
        ],
        nrow=nant,
    )
    tb.putcol("NAME", list(antenna_names))
    tb.putcol("STATION", list(antenna_names))
    tb.putcol("TYPE", ["GROUND-BASED"] * nant)
    tb.putcol("MOUNT", ["alt-az"] * nant)
    tb.putcol("DISH_DIAMETER", np.full(nant, 12.0))
    tb.putcol("FLAG_ROW", np.zeros(nant, dtype=bool))
    tb.putcol("POSITION", antenna_positions_m.T)  # (3, nant)
    tb.putcol("OFFSET", np.zeros((3, nant)))
    tb.flush()
    tb.close()


def _write_spectral_window(backend, ms_path, channel_freq_hz, nchan):
    chan_width = np.gradient(channel_freq_hz).astype(np.float64) if nchan > 1 else np.array([1.0])
    tb = backend.create(
        os.path.join(ms_path, "SPECTRAL_WINDOW"),
        [
            backend.sca("NUM_CHAN", 0),
            backend.sca("REF_FREQUENCY", 0.0),
            backend.sca("TOTAL_BANDWIDTH", 0.0),
            backend.sca("NET_SIDEBAND", 0),
            backend.sca("MEAS_FREQ_REF", 0),
            backend.sca("FLAG_ROW", False),
            backend.sca("NAME", ""),
            backend.arr("CHAN_FREQ", 0.0, 1),
            backend.arr("CHAN_WIDTH", 0.0, 1),
            backend.arr("EFFECTIVE_BW", 0.0, 1),
            backend.arr("RESOLUTION", 0.0, 1),
        ],
        nrow=1,
    )
    tb.putcell("NUM_CHAN", 0, int(nchan))
    tb.putcell("CHAN_FREQ", 0, channel_freq_hz)
    tb.putcell("CHAN_WIDTH", 0, chan_width)
    tb.putcell("EFFECTIVE_BW", 0, np.abs(chan_width))
    tb.putcell("RESOLUTION", 0, np.abs(chan_width))
    tb.putcell("REF_FREQUENCY", 0, float(np.median(channel_freq_hz)))
    tb.putcell("TOTAL_BANDWIDTH", 0, float(np.sum(np.abs(chan_width))))
    tb.putcell("NAME", 0, "ALMASim SPW 0")
    tb.putcell("MEAS_FREQ_REF", 0, 5)
    tb.putcell("NET_SIDEBAND", 0, 1)
    tb.putcell("FLAG_ROW", 0, False)
    tb.flush()
    tb.close()


def _write_polarization(backend, ms_path, ncorr):
    tb = backend.create(
        os.path.join(ms_path, "POLARIZATION"),
        [
            backend.sca("NUM_CORR", 0),
            backend.sca("FLAG_ROW", False),
            backend.arr("CORR_TYPE", 0, 1),
            backend.arr("CORR_PRODUCT", 0, 2),
        ],
        nrow=1,
    )
    tb.putcell("NUM_CORR", 0, int(ncorr))
    tb.putcell("CORR_TYPE", 0, np.array([9], dtype=np.int32))  # Stokes I
    tb.putcell("CORR_PRODUCT", 0, np.array([[0], [0]], dtype=np.int32))
    tb.putcell("FLAG_ROW", 0, False)
    tb.flush()
    tb.close()


def _write_data_description(backend, ms_path):
    tb = backend.create(
        os.path.join(ms_path, "DATA_DESCRIPTION"),
        [
            backend.sca("SPECTRAL_WINDOW_ID", 0),
            backend.sca("POLARIZATION_ID", 0),
            backend.sca("FLAG_ROW", False),
        ],
        nrow=1,
    )
    tb.putcell("SPECTRAL_WINDOW_ID", 0, 0)
    tb.putcell("POLARIZATION_ID", 0, 0)
    tb.putcell("FLAG_ROW", 0, False)
    tb.flush()
    tb.close()


def _write_field(backend, ms_path, source_name, ra_rad, dec_rad, time_range):
    direction_3d = np.array([[ra_rad, dec_rad]])  # shape (1, 2): one polynomial term
    tb = backend.create(
        os.path.join(ms_path, "FIELD"),
        [
            backend.sca("NAME", ""),
            backend.sca("CODE", ""),
            backend.sca("NUM_POLY", 0),
            backend.sca("SOURCE_ID", 0),
            backend.sca("TIME", 0.0),
            backend.sca("FLAG_ROW", False),
            backend.arr("DELAY_DIR", 0.0, 2),
            backend.arr("PHASE_DIR", 0.0, 2),
            backend.arr("REFERENCE_DIR", 0.0, 2),
        ],
        nrow=1,
    )
    tb.putcell("NAME", 0, source_name)
    tb.putcell("CODE", 0, "")
    tb.putcell("NUM_POLY", 0, 0)
    tb.putcell("SOURCE_ID", 0, 0)
    tb.putcell("TIME", 0, float(np.mean(time_range)))
    tb.putcell("FLAG_ROW", 0, False)
    tb.putcell("DELAY_DIR", 0, direction_3d)
    tb.putcell("PHASE_DIR", 0, direction_3d)
    tb.putcell("REFERENCE_DIR", 0, direction_3d)
    tb.flush()
    tb.close()


def _write_observation(backend, ms_path, telescope_name, project_name, time_range):
    tb = backend.create(
        os.path.join(ms_path, "OBSERVATION"),
        [
            backend.sca("TELESCOPE_NAME", ""),
            backend.sca("OBSERVER", ""),
            backend.sca("PROJECT", ""),
            backend.sca("SCHEDULE_TYPE", ""),
            backend.sca("RELEASE_DATE", 0.0),
            backend.sca("FLAG_ROW", False),
            backend.arr("TIME_RANGE", 0.0, 1, shape=[2]),
        ],
        nrow=1,
    )
    tb.putcell("TELESCOPE_NAME", 0, telescope_name)
    tb.putcell("OBSERVER", 0, "ALMASim")
    tb.putcell("PROJECT", 0, project_name)
    tb.putcell("SCHEDULE_TYPE", 0, "ALMASim")
    tb.putcell("RELEASE_DATE", 0, float(time_range.max()))
    tb.putcell("FLAG_ROW", 0, False)
    tb.putcell("TIME_RANGE", 0, time_range.astype(np.float64))
    tb.flush()
    tb.close()


def _write_source(backend, ms_path, source_name, ra_rad, dec_rad, time_range):
    tb = backend.create(
        os.path.join(ms_path, "SOURCE"),
        [
            backend.sca("SOURCE_ID", 0),
            backend.sca("TIME", 0.0),
            backend.sca("INTERVAL", 0.0),
            backend.sca("SPECTRAL_WINDOW_ID", 0),
            backend.sca("NUM_LINES", 0),
            backend.sca("NAME", ""),
            backend.sca("CODE", ""),
            backend.sca("CALIBRATION_GROUP", 0),
            backend.arr("DIRECTION", 0.0, 1, shape=[2]),
        ],
        nrow=1,
    )
    tb.putcell("SOURCE_ID", 0, 0)
    tb.putcell("TIME", 0, float(np.mean(time_range)))
    tb.putcell("INTERVAL", 0, float(time_range[1] - time_range[0]))
    tb.putcell("SPECTRAL_WINDOW_ID", 0, -1)
    tb.putcell("NUM_LINES", 0, 0)
    tb.putcell("NAME", 0, source_name)
    tb.putcell("CODE", 0, "")
    tb.putcell("CALIBRATION_GROUP", 0, 0)
    tb.putcell("DIRECTION", 0, np.array([ra_rad, dec_rad]))
    tb.flush()
    tb.close()


def _write_state(backend, ms_path):
    tb = backend.create(
        os.path.join(ms_path, "STATE"),
        [
            backend.sca("SIG", True),
            backend.sca("REF", False),
            backend.sca("CAL", 0.0),
            backend.sca("LOAD", 0.0),
            backend.sca("SUB_SCAN", 0),
            backend.sca("OBS_MODE", ""),
            backend.sca("FLAG_ROW", False),
        ],
        nrow=1,
    )
    tb.putcell("SIG", 0, True)
    tb.putcell("REF", 0, False)
    tb.putcell("CAL", 0, 0.0)
    tb.putcell("LOAD", 0, 0.0)
    tb.putcell("SUB_SCAN", 0, 0)
    tb.putcell("OBS_MODE", 0, "ON_SOURCE")
    tb.putcell("FLAG_ROW", 0, False)
    tb.flush()
    tb.close()


def _write_history(backend, ms_path, project_name, time_range):
    tb = backend.create(
        os.path.join(ms_path, "HISTORY"),
        [
            backend.sca("TIME", 0.0),
            backend.sca("OBSERVATION_ID", 0),
            backend.sca("MESSAGE", ""),
            backend.sca("PRIORITY", ""),
            backend.sca("ORIGIN", ""),
            backend.sca("OBJECT_ID", 0),
            backend.sca("APPLICATION", ""),
            backend.arr("CLI_COMMAND", "", 1),
            backend.arr("APP_PARAMS", "", 1),
        ],
        nrow=1,
    )
    tb.putcell("TIME", 0, float(np.mean(time_range)))
    tb.putcell("OBSERVATION_ID", 0, 0)
    tb.putcell("MESSAGE", 0, "ALMASim MSv2 export")
    tb.putcell("PRIORITY", 0, "NORMAL")
    tb.putcell("ORIGIN", 0, "ALMASim")
    tb.putcell("OBJECT_ID", 0, 0)
    tb.putcell("APPLICATION", 0, "ALMASim")
    tb.putcell("CLI_COMMAND", 0, np.array(["export_native_ms"]))
    tb.putcell("APP_PARAMS", 0, np.array([project_name]))
    tb.flush()
    tb.close()


def _write_main(backend, ms_path, nrows, vt):
    data = np.asarray(vt["data"], dtype=np.complex64)
    model_data = np.asarray(vt["model_data"], dtype=np.complex64)
    flag = np.asarray(vt["flag"], dtype=bool)
    uvw_m = np.asarray(vt["uvw_m"], dtype=np.float64)
    antenna1 = np.asarray(vt["antenna1"], dtype=np.int32)
    antenna2 = np.asarray(vt["antenna2"], dtype=np.int32)
    time_mjd_s = np.asarray(vt["time_mjd_s"], dtype=np.float64)
    interval_s = np.asarray(vt["interval_s"], dtype=np.float64)
    exposure_s = np.asarray(vt["exposure_s"], dtype=np.float64)
    weight = np.asarray(vt["weight"], dtype=np.float32)
    sigma = np.asarray(vt["sigma"], dtype=np.float32)

    tb = backend.create(
        ms_path,
        [
            backend.sca("TIME", 0.0),
            backend.sca("TIME_CENTROID", 0.0),
            backend.sca("INTERVAL", 0.0),
            backend.sca("EXPOSURE", 0.0),
            backend.sca("ANTENNA1", 0),
            backend.sca("ANTENNA2", 0),
            backend.sca("FEED1", 0),
            backend.sca("FEED2", 0),
            backend.sca("DATA_DESC_ID", 0),
            backend.sca("FIELD_ID", 0),
            backend.sca("ARRAY_ID", 0),
            backend.sca("OBSERVATION_ID", 0),
            backend.sca("PROCESSOR_ID", 0),
            backend.sca("SCAN_NUMBER", 0),
            backend.sca("STATE_ID", 0),
            backend.sca("FLAG_ROW", False),
            backend.arr("UVW", 0.0, 1, shape=[3]),
            backend.arr("DATA", np.complex64(0), 2),
            backend.arr("MODEL_DATA", np.complex64(0), 2),
            backend.arr("FLAG", False, 2),
            backend.arr("WEIGHT", np.float32(0), 1),
            backend.arr("SIGMA", np.float32(0), 1),
        ],
        nrow=nrows,
    )

    # Scalar columns: (nrows,)
    tb.putcol("TIME", time_mjd_s)
    tb.putcol("TIME_CENTROID", time_mjd_s)
    tb.putcol("INTERVAL", interval_s)
    tb.putcol("EXPOSURE", exposure_s)
    tb.putcol("ANTENNA1", antenna1)
    tb.putcol("ANTENNA2", antenna2)
    tb.putcol("FEED1", np.zeros(nrows, dtype=np.int32))
    tb.putcol("FEED2", np.zeros(nrows, dtype=np.int32))
    tb.putcol("DATA_DESC_ID", np.zeros(nrows, dtype=np.int32))
    tb.putcol("FIELD_ID", np.zeros(nrows, dtype=np.int32))
    tb.putcol("ARRAY_ID", np.zeros(nrows, dtype=np.int32))
    tb.putcol("OBSERVATION_ID", np.zeros(nrows, dtype=np.int32))
    tb.putcol("PROCESSOR_ID", np.full(nrows, -1, dtype=np.int32))
    tb.putcol("SCAN_NUMBER", np.ones(nrows, dtype=np.int32))
    tb.putcol("STATE_ID", np.zeros(nrows, dtype=np.int32))
    tb.putcol("FLAG_ROW", np.all(flag, axis=(1, 2)))

    # Array columns: CASA convention is (col_shape..., nrows)
    tb.putcol("UVW", uvw_m.T)  # (3, nrows)
    tb.putcol("DATA", data.transpose(1, 2, 0))  # (ncorr, nchan, nrows)
    tb.putcol("MODEL_DATA", model_data.transpose(1, 2, 0))
    tb.putcol("FLAG", flag.transpose(1, 2, 0))
    tb.putcol("WEIGHT", weight.T)  # (ncorr, nrows)
    tb.putcol("SIGMA", sigma.T)

    tb.putkeyword("MS_VERSION", 2.0)
    tb.putinfo({"type": "Measurement Set", "subType": "ALMASim"})

    for name in _SUBTABLE_NAMES:
        tb.putkeyword(name, "Table: " + os.path.join(ms_path, name))

    tb.flush()
    tb.close()
