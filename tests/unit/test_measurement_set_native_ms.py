"""Unit tests for the native `.ms` writer."""

import numpy as np

from almasim.services.products.ms_io import export_native_ms


def _sample_visibility_table():
    return {
        "uvw_m": np.array([[0.0, 0.0, 0.0], [10.0, 5.0, 1.0]], dtype=np.float64),
        "antenna1": np.array([0, 0], dtype=np.int32),
        "antenna2": np.array([1, 1], dtype=np.int32),
        "time_mjd_s": np.array([5.0e9, 5.0e9 + 6.0], dtype=np.float64),
        "interval_s": np.array([6.0, 6.0], dtype=np.float64),
        "exposure_s": np.array([6.0, 6.0], dtype=np.float64),
        "data": np.zeros((2, 1, 4), dtype=np.complex64),
        "model_data": np.zeros((2, 1, 4), dtype=np.complex64),
        "flag": np.zeros((2, 1, 4), dtype=bool),
        "weight": np.ones((2, 1), dtype=np.float32),
        "sigma": np.ones((2, 1), dtype=np.float32),
        "channel_freq_hz": np.array([1.0e11, 1.001e11, 1.002e11, 1.003e11], dtype=np.float64),
        "antenna_names": ["DA01", "DA02"],
        "antenna_positions_m": np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]], dtype=np.float64),
        "field_ra_rad": 1.0,
        "field_dec_rad": 0.5,
        "observation_date": "2020-01-01",
    }


def test_native_ms_layout_writer_creates_standard_tree(tmp_path):
    ms_path = tmp_path / "demo.ms"
    result_path = export_native_ms(
        ms_path=ms_path,
        visibility_table=_sample_visibility_table(),
        project_name="demo_project",
        source_name="demo_source",
    )

    assert result_path == str(ms_path.resolve())
    assert ms_path.is_dir()
    assert (ms_path / "table.info").exists()
    assert (ms_path / "table.lock").exists()
    assert (ms_path / "table.dat").exists()
    assert (ms_path / "table.f0").exists()
    assert (ms_path / "ANTENNA" / "table.info").exists()
    assert (ms_path / "SPECTRAL_WINDOW" / "table.dat").exists()
    assert (ms_path / "FIELD" / "table.f0").exists()

    table_info = (ms_path / "table.info").read_text()
    assert "Type = MeasurementSet" in table_info
    assert "SubType = ALMASim" in table_info

    table_lock = (ms_path / "table.lock").read_bytes()
    assert len(table_lock) == 325
    assert table_lock[:260] == bytes(260)
    assert table_lock[264:268] == bytes.fromhex("be be be be")

    table_dat = (ms_path / "table.dat").read_bytes()
    assert b"PlainTable" in table_dat
    assert b"TableDesc" in table_dat
    assert b"StandardStMan" in table_dat

    stman_file = (ms_path / "table.f0").read_bytes()
    assert len(stman_file) >= 512
    assert b"StandardStMan" in stman_file[:512]
