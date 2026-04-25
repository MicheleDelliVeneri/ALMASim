"""Unit tests for MSv2 I/O utilities."""

import os
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from almasim.services.products.ms_io import (
    _arr_desc,
    _get_backend,
    _make_tabledesc,
    _sca_desc,
    _vtype,
    _write_antenna,
    _write_data_description,
    _write_field,
    _write_history,
    _write_main,
    _write_observation,
    _write_polarization,
    _write_source,
    _write_spectral_window,
    _write_state,
    export_native_ms,
    read_native_ms,
)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_vtype_bool():
    assert _vtype(True) == "boolean"
    assert _vtype(False) == "boolean"


@pytest.mark.unit
def test_vtype_int():
    assert _vtype(1) == "int"
    assert _vtype(np.int32(5)) == "int"
    assert _vtype(np.int64(5)) == "int"


@pytest.mark.unit
def test_vtype_float32():
    assert _vtype(np.float32(1.0)) == "float"


@pytest.mark.unit
def test_vtype_double():
    assert _vtype(1.0) == "double"
    assert _vtype(np.float64(1.0)) == "double"


@pytest.mark.unit
def test_vtype_complex():
    assert _vtype(1 + 2j) == "complex"
    assert _vtype(np.complex64(1 + 2j)) == "complex"


@pytest.mark.unit
def test_vtype_string():
    assert _vtype("hello") == "string"
    assert _vtype(None) == "string"


@pytest.mark.unit
def test_sca_desc_structure():
    name, desc = _sca_desc("MY_COL", 3.14)
    assert name == "MY_COL"
    assert desc["valueType"] == "double"
    assert desc["dataManagerType"] == "StandardStMan"
    assert "keywords" in desc
    assert desc["option"] == 0


@pytest.mark.unit
def test_arr_desc_no_shape():
    name, desc = _arr_desc("DATA", np.complex64(0), ndim=2)
    assert name == "DATA"
    assert desc["valueType"] == "complex"
    assert desc["ndim"] == 2
    assert "shape" not in desc
    assert desc["option"] == 0


@pytest.mark.unit
def test_arr_desc_with_shape():
    name, desc = _arr_desc("UVW", 0.0, ndim=1, shape=[3])
    assert name == "UVW"
    assert desc["option"] == 5  # FixedShape
    np.testing.assert_array_equal(desc["shape"], [3])


@pytest.mark.unit
def test_make_tabledesc_structure():
    cols = [
        _sca_desc("TIME", 0.0),
        _arr_desc("DATA", np.complex64(0), 2),
    ]
    desc = _make_tabledesc(cols)
    assert "_define_hypercolumn_" in desc
    assert "_keywords_" in desc
    assert "_private_keywords_" in desc
    assert "TIME" in desc
    assert "DATA" in desc


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_get_backend_casatools_preferred():
    mock_ct = MagicMock()
    with patch.dict("sys.modules", {"casatools": mock_ct}):
        backend = _get_backend()
        assert backend is not None


@pytest.mark.unit
def test_get_backend_raises_when_nothing_available():
    with (
        patch(
            "almasim.services.products.ms_io._CasatoolsBackend.__init__",
            side_effect=ImportError,
        ),
        patch(
            "almasim.services.products.ms_io._CasacoreBackend.__init__",
            side_effect=ImportError,
        ),
    ):
        with pytest.raises(ImportError, match="MS export requires"):
            _get_backend()


# ---------------------------------------------------------------------------
# Subtable writers (mocked backend)
# ---------------------------------------------------------------------------


def _mock_backend():
    """Build a mock backend that records calls."""
    b = MagicMock()
    tb = MagicMock()
    b.create.return_value = tb
    b.open.return_value = tb
    b.sca.side_effect = _sca_desc
    b.arr.side_effect = _arr_desc
    return b, tb


@pytest.mark.unit
def test_write_antenna(tmp_path):
    b, tb = _mock_backend()
    names = ["DA41", "DA42", "DV01"]
    positions = np.zeros((3, 3))
    _write_antenna(b, str(tmp_path / "test.ms"), names, positions, 3)
    assert b.create.called
    tb.putcol.assert_any_call("NAME", names)
    tb.flush.assert_called_once()
    tb.close.assert_called_once()


@pytest.mark.unit
def test_write_spectral_window_single_channel(tmp_path):
    b, tb = _mock_backend()
    freq = np.array([230e9])
    _write_spectral_window(b, str(tmp_path / "test.ms"), freq, 1)
    tb.putcell.assert_any_call("NUM_CHAN", 0, 1)
    tb.flush.assert_called_once()


@pytest.mark.unit
def test_write_spectral_window_multi_channel(tmp_path):
    b, tb = _mock_backend()
    freq = np.linspace(230e9, 232e9, 32)
    _write_spectral_window(b, str(tmp_path / "test.ms"), freq, 32)
    tb.putcell.assert_any_call("NUM_CHAN", 0, 32)


@pytest.mark.unit
def test_write_polarization(tmp_path):
    b, tb = _mock_backend()
    _write_polarization(b, str(tmp_path / "test.ms"), 1)
    tb.putcell.assert_any_call("NUM_CORR", 0, 1)
    tb.flush.assert_called_once()


@pytest.mark.unit
def test_write_data_description(tmp_path):
    b, tb = _mock_backend()
    _write_data_description(b, str(tmp_path / "test.ms"))
    tb.putcell.assert_any_call("SPECTRAL_WINDOW_ID", 0, 0)
    tb.putcell.assert_any_call("POLARIZATION_ID", 0, 0)


@pytest.mark.unit
def test_write_field(tmp_path):
    b, tb = _mock_backend()
    time_range = np.array([5e9, 5.1e9])
    _write_field(b, str(tmp_path / "test.ms"), "J0001+0001", 0.1, -0.2, time_range)
    tb.putcell.assert_any_call("NAME", 0, "J0001+0001")


@pytest.mark.unit
def test_write_observation(tmp_path):
    b, tb = _mock_backend()
    time_range = np.array([5e9, 5.1e9])
    _write_observation(b, str(tmp_path / "test.ms"), "ALMA", "2021.1.00001.S", time_range)
    tb.putcell.assert_any_call("TELESCOPE_NAME", 0, "ALMA")
    tb.putcell.assert_any_call("PROJECT", 0, "2021.1.00001.S")


@pytest.mark.unit
def test_write_source(tmp_path):
    b, tb = _mock_backend()
    time_range = np.array([5e9, 5.1e9])
    _write_source(b, str(tmp_path / "test.ms"), "MySource", 1.0, -0.5, time_range)
    tb.putcell.assert_any_call("NAME", 0, "MySource")


@pytest.mark.unit
def test_write_state(tmp_path):
    b, tb = _mock_backend()
    _write_state(b, str(tmp_path / "test.ms"))
    tb.putcell.assert_any_call("OBS_MODE", 0, "ON_SOURCE")


@pytest.mark.unit
def test_write_history(tmp_path):
    b, tb = _mock_backend()
    time_range = np.array([5e9, 5.1e9])
    _write_history(b, str(tmp_path / "test.ms"), "my_project", time_range)
    tb.putcell.assert_any_call("MESSAGE", 0, "ALMASim MSv2 export")


@pytest.mark.unit
def test_write_main(tmp_path):
    nrows, ncorr, nchan = 10, 1, 8
    vt = {
        "data": np.ones((nrows, ncorr, nchan), dtype=np.complex64),
        "model_data": np.zeros((nrows, ncorr, nchan), dtype=np.complex64),
        "flag": np.zeros((nrows, ncorr, nchan), dtype=bool),
        "uvw_m": np.zeros((nrows, 3)),
        "antenna1": np.zeros(nrows, dtype=np.int32),
        "antenna2": np.ones(nrows, dtype=np.int32),
        "time_mjd_s": np.linspace(5e9, 5.1e9, nrows),
        "interval_s": np.full(nrows, 10.0),
        "exposure_s": np.full(nrows, 10.0),
        "weight": np.ones((nrows, ncorr), dtype=np.float32),
        "sigma": np.ones((nrows, ncorr), dtype=np.float32),
    }
    b, tb = _mock_backend()
    _write_main(b, str(tmp_path / "test.ms"), nrows, vt)
    tb.putcol.assert_any_call("ANTENNA1", vt["antenna1"])
    tb.putcol.assert_any_call("ANTENNA2", vt["antenna2"])
    tb.putkeyword.assert_any_call("MS_VERSION", 2.0)


# ---------------------------------------------------------------------------
# export_native_ms end-to-end (fully mocked backend)
# ---------------------------------------------------------------------------


def _sample_vt(nrows=5, ncorr=1, nchan=4):
    return {
        "data": np.ones((nrows, ncorr, nchan), dtype=np.complex64),
        "model_data": np.zeros((nrows, ncorr, nchan), dtype=np.complex64),
        "flag": np.zeros((nrows, ncorr, nchan), dtype=bool),
        "uvw_m": np.zeros((nrows, 3)),
        "antenna1": np.zeros(nrows, dtype=np.int32),
        "antenna2": np.ones(nrows, dtype=np.int32),
        "time_mjd_s": np.linspace(5e9, 5.1e9, nrows),
        "interval_s": np.full(nrows, 10.0),
        "exposure_s": np.full(nrows, 10.0),
        "weight": np.ones((nrows, ncorr), dtype=np.float32),
        "sigma": np.ones((nrows, ncorr), dtype=np.float32),
        "channel_freq_hz": np.linspace(230e9, 231e9, nchan),
        "antenna_names": ["DA41", "DA42"],
        "antenna_positions_m": np.zeros((2, 3)),
        "field_ra_rad": 0.1,
        "field_dec_rad": -0.2,
    }


@pytest.mark.unit
def test_export_native_ms_calls_all_writers(tmp_path):
    vt = _sample_vt()
    b, tb = _mock_backend()

    with patch("almasim.services.products.ms_io._get_backend", return_value=b):
        result = export_native_ms(
            ms_path=tmp_path / "out.ms",
            visibility_table=vt,
            project_name="test_proj",
            source_name="J0000+0000",
        )

    assert result.endswith("out.ms")
    # create called once per subtable + main
    assert b.create.call_count >= 10


@pytest.mark.unit
def test_export_native_ms_default_telescope(tmp_path):
    vt = _sample_vt()
    b, tb = _mock_backend()

    with patch("almasim.services.products.ms_io._get_backend", return_value=b):
        export_native_ms(
            ms_path=tmp_path / "out.ms",
            visibility_table=vt,
            project_name="p",
            source_name="S",
        )

    # TELESCOPE_NAME cell should receive "ALMA"
    write_obs_calls = [c for c in tb.putcell.call_args_list if c[0][0] == "TELESCOPE_NAME"]
    assert len(write_obs_calls) == 1
    assert write_obs_calls[0][0][2] == "ALMA"


# ---------------------------------------------------------------------------
# read_native_ms (mocked backend)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_read_native_ms_returns_expected_keys(tmp_path):
    nrows, ncorr, nchan = 3, 1, 4
    main_tb = MagicMock()
    main_tb.getcol.side_effect = lambda col: {
        "DATA": np.ones((ncorr, nchan, nrows), dtype=np.complex64),
        "MODEL_DATA": np.zeros((ncorr, nchan, nrows), dtype=np.complex64),
        "FLAG": np.zeros((ncorr, nchan, nrows), dtype=bool),
        "UVW": np.zeros((3, nrows)),
        "ANTENNA1": np.zeros(nrows, dtype=np.int32),
        "ANTENNA2": np.ones(nrows, dtype=np.int32),
        "TIME": np.linspace(5e9, 5.1e9, nrows),
        "INTERVAL": np.full(nrows, 10.0),
        "EXPOSURE": np.full(nrows, 10.0),
        "WEIGHT": np.ones((ncorr, nrows), dtype=np.float32),
        "SIGMA": np.ones((ncorr, nrows), dtype=np.float32),
    }[col]

    spw_tb = MagicMock()
    spw_tb.getcell.return_value = np.linspace(230e9, 231e9, nchan)

    ant_tb = MagicMock()
    ant_tb.getcol.side_effect = lambda col: {
        "NAME": ["DA41", "DA42"],
        "POSITION": np.zeros((3, 2)),
    }[col]

    fld_tb = MagicMock()
    fld_tb.getcell.side_effect = lambda col, row: {
        "PHASE_DIR": np.array([[0.1, -0.2]]),
        "NAME": "J0000",
    }[col]

    obs_tb = MagicMock()
    obs_tb.getcell.side_effect = lambda col, row: {
        "TELESCOPE_NAME": "ALMA",
        "PROJECT": "2021.1.S",
    }[col]

    open_call_count = [0]
    subtables = [spw_tb, ant_tb, fld_tb, obs_tb]

    def fake_open(path, readonly=True):
        if path.endswith("SPECTRAL_WINDOW"):
            return spw_tb
        if path.endswith("ANTENNA"):
            return ant_tb
        if path.endswith("FIELD"):
            return fld_tb
        if path.endswith("OBSERVATION"):
            return obs_tb
        return main_tb

    b = MagicMock()
    b.open.side_effect = fake_open

    with patch("almasim.services.products.ms_io._get_backend", return_value=b):
        result = read_native_ms(tmp_path / "test.ms")

    expected_keys = {
        "data",
        "model_data",
        "flag",
        "uvw_m",
        "antenna1",
        "antenna2",
        "time_mjd_s",
        "interval_s",
        "exposure_s",
        "weight",
        "sigma",
        "channel_freq_hz",
        "antenna_names",
        "antenna_positions_m",
        "field_ra_rad",
        "field_dec_rad",
        "source_name",
        "telescope_name",
        "project_name",
    }
    assert expected_keys == set(result.keys())
    assert result["telescope_name"] == "ALMA"
    assert result["data"].shape == (nrows, ncorr, nchan)
