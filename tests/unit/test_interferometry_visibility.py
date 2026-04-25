"""Unit tests for almasim.services.interferometry.visibility."""

from __future__ import annotations

import numpy as np
import pytest

from almasim.services.interferometry.visibility import (
    VisibilityTable,
    assemble_visibility_table,
    concatenate_visibility_tables,
)

# ===========================================================================
# helpers
# ===========================================================================


def _make_channel_rows(nrows: int = 6) -> dict:
    """Build minimal channel-row dict."""
    return {
        "uvw_m": np.zeros((nrows, 3), dtype=np.float64),
        "antenna1": np.zeros(nrows, dtype=np.int32),
        "antenna2": np.ones(nrows, dtype=np.int32),
        "time_index": np.arange(nrows, dtype=np.int32),
        "valid": np.ones(nrows, dtype=np.bool_),
        "model_data": np.ones(nrows, dtype=np.complex64),
        "data": np.ones(nrows, dtype=np.complex64),
        "weight": np.ones(nrows, dtype=np.float32),
        "sigma": np.ones(nrows, dtype=np.float32) * 0.1,
    }


def _make_visibility_table(nrows: int = 6, nchan: int = 4) -> VisibilityTable:
    """Build a VisibilityTable dataclass."""
    return VisibilityTable(
        uvw_m=np.zeros((nrows, 3)),
        antenna1=np.zeros(nrows, dtype=np.int32),
        antenna2=np.ones(nrows, dtype=np.int32),
        time_mjd_s=np.arange(nrows, dtype=float),
        interval_s=np.ones(nrows),
        exposure_s=np.ones(nrows),
        data=np.zeros((nrows, 1, nchan), dtype=np.complex64),
        model_data=np.zeros((nrows, 1, nchan), dtype=np.complex64),
        flag=np.ones((nrows, 1, nchan), dtype=bool),
        weight=np.ones((nrows, 1), dtype=np.float32),
        sigma=np.ones((nrows, 1), dtype=np.float32),
        channel_freq_hz=np.linspace(100e9, 104e9, nchan),
        antenna_names=["A1", "A2"],
        antenna_positions_m=np.zeros((2, 3)),
        source_name="TEST",
        field_ra_rad=0.0,
        field_dec_rad=0.0,
        observation_date="2024-01-01",
    )


# ===========================================================================
# VisibilityTable.as_dict
# ===========================================================================


@pytest.mark.unit
def test_visibility_table_as_dict_keys():
    """as_dict() returns all expected keys."""
    vt = _make_visibility_table()
    d = vt.as_dict()
    for key in (
        "uvw_m",
        "antenna1",
        "antenna2",
        "time_mjd_s",
        "data",
        "model_data",
        "flag",
        "weight",
        "sigma",
        "channel_freq_hz",
        "antenna_names",
        "source_name",
        "field_ra_rad",
        "field_dec_rad",
        "observation_date",
    ):
        assert key in d


@pytest.mark.unit
def test_visibility_table_as_dict_optional_fields():
    """as_dict() includes config_name and array_type (may be None)."""
    vt = _make_visibility_table()
    vt.config_name = "C43"
    vt.array_type = "12m"
    d = vt.as_dict()
    assert d["config_name"] == "C43"
    assert d["array_type"] == "12m"


# ===========================================================================
# assemble_visibility_table
# ===========================================================================


@pytest.mark.unit
def test_assemble_visibility_table_basic():
    """assemble_visibility_table returns a VisibilityTable."""
    rows = [_make_channel_rows(6) for _ in range(4)]
    freqs = np.linspace(100e9, 104e9, 4)
    vt = assemble_visibility_table(
        channel_rows=rows,
        channel_freq_hz=freqs,
        scan_time_s=10.0,
        observation_date="2024-01-01",
        antenna_names=["A1", "A2"],
        antenna_positions_m=np.zeros((2, 3)),
        source_name="TEST",
        field_ra_rad=0.0,
        field_dec_rad=0.0,
    )
    assert isinstance(vt, VisibilityTable)


@pytest.mark.unit
def test_assemble_visibility_table_data_shape():
    """Assembled data has shape (nrows, 1, nchan)."""
    nrows, nchan = 6, 4
    rows = [_make_channel_rows(nrows) for _ in range(nchan)]
    freqs = np.linspace(100e9, 104e9, nchan)
    vt = assemble_visibility_table(
        channel_rows=rows,
        channel_freq_hz=freqs,
        scan_time_s=10.0,
        observation_date="2024-01-01",
        antenna_names=["A1", "A2"],
        antenna_positions_m=np.zeros((2, 3)),
        source_name="TEST",
        field_ra_rad=0.0,
        field_dec_rad=0.0,
    )
    assert vt.data.shape == (nrows, 1, nchan)
    assert vt.model_data.shape == (nrows, 1, nchan)
    assert vt.flag.shape == (nrows, 1, nchan)


@pytest.mark.unit
def test_assemble_visibility_table_empty_rows_raises():
    """Empty channel_rows raises ValueError."""
    with pytest.raises(ValueError, match="channel_rows must not be empty"):
        assemble_visibility_table(
            channel_rows=[],
            channel_freq_hz=np.array([]),
            scan_time_s=10.0,
            observation_date="2024-01-01",
            antenna_names=["A1"],
            antenna_positions_m=np.zeros((1, 3)),
            source_name="TEST",
            field_ra_rad=0.0,
            field_dec_rad=0.0,
        )


@pytest.mark.unit
def test_assemble_visibility_table_config_name():
    """config_name and array_type propagate to the output VisibilityTable."""
    rows = [_make_channel_rows(4)]
    vt = assemble_visibility_table(
        channel_rows=rows,
        channel_freq_hz=np.array([100e9]),
        scan_time_s=10.0,
        observation_date="2024-01-01",
        antenna_names=["A1"],
        antenna_positions_m=np.zeros((1, 3)),
        source_name="FIELD",
        field_ra_rad=1.0,
        field_dec_rad=-0.5,
        config_name="C43",
        array_type="12m",
    )
    assert vt.config_name == "C43"
    assert vt.array_type == "12m"
    assert vt.source_name == "FIELD"


@pytest.mark.unit
def test_assemble_visibility_table_times():
    """time_mjd_s is computed from observation_date + time_index * scan_time_s."""
    nrows = 4
    rows = [_make_channel_rows(nrows)]
    vt = assemble_visibility_table(
        channel_rows=rows,
        channel_freq_hz=np.array([100e9]),
        scan_time_s=60.0,
        observation_date="2024-01-01",
        antenna_names=["A1"],
        antenna_positions_m=np.zeros((1, 3)),
        source_name="S",
        field_ra_rad=0.0,
        field_dec_rad=0.0,
    )
    # time_mjd_s should have nrows entries and be monotonically increasing
    assert vt.time_mjd_s.shape == (nrows,)
    assert np.all(np.diff(vt.time_mjd_s) >= 0)


# ===========================================================================
# concatenate_visibility_tables
# ===========================================================================


@pytest.mark.unit
def test_concatenate_empty_raises():
    """Empty list raises ValueError."""
    with pytest.raises(ValueError, match="tables must not be empty"):
        concatenate_visibility_tables([])


@pytest.mark.unit
def test_concatenate_single_returns_dict():
    """Single table (dict) is returned as dict."""
    vt = _make_visibility_table().as_dict()
    result = concatenate_visibility_tables([vt])
    assert isinstance(result, dict)
    assert "uvw_m" in result


@pytest.mark.unit
def test_concatenate_two_tables_row_count():
    """Two tables concatenated have combined row count."""
    vt1 = _make_visibility_table(nrows=4, nchan=3)
    vt2 = _make_visibility_table(nrows=6, nchan=3)
    result = concatenate_visibility_tables([vt1.as_dict(), vt2.as_dict()])
    assert result["uvw_m"].shape[0] == 10


@pytest.mark.unit
def test_concatenate_two_tables_channel_from_first():
    """channel_freq_hz comes from the first table."""
    vt1 = _make_visibility_table(nrows=4, nchan=3)
    vt2 = _make_visibility_table(nrows=4, nchan=3)
    vt1_dict = vt1.as_dict()
    vt2_dict = vt2.as_dict()
    vt1_dict["channel_freq_hz"] = np.array([100e9, 200e9, 300e9])
    vt2_dict["channel_freq_hz"] = np.array([999e9, 888e9, 777e9])
    result = concatenate_visibility_tables([vt1_dict, vt2_dict])
    np.testing.assert_array_equal(result["channel_freq_hz"], vt1_dict["channel_freq_hz"])
