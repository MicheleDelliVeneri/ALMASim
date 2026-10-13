"""Flagging and visibility summary of MeasurementSets, on small real casacore tables."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

casacore_tables = pytest.importorskip("casacore.tables")

from almasim import cli  # noqa: E402
from almasim.services.archive.flag_summary import (  # noqa: E402
    SUMMARY_COLUMNS,
    flag_summary,
    uid_of,
)

runner = CliRunner()

# (field, dd, state, scan, time, antenna1, antenna2, FLAG all?, FLAG_ROW, WEIGHT):
# field 0 is the phase calibrator, field 1 the target; dd 0 is SPW 25, dd 1 SPW 27.
_ROWS = [
    (0, 0, 0, 1, 100.0, 0, 1, False, False, [1.0, 1.0]),
    (0, 0, 0, 1, 106.0, 0, 1, True, False, [1e7, 1e7]),  # flagged: high weight ignored
    (1, 0, 1, 2, 200.0, 0, 1, False, False, [5e5, 1.0]),  # unflagged high weight
    (1, 0, 1, 2, 200.0, 1, 2, False, True, [1.0, 1.0]),  # FLAG_ROW set
    (1, 0, 2, 3, 300.0, 0, 2, False, False, [2.0, 2.0]),  # target under the atmosphere intent
    (1, 1, 1, 2, 200.0, 0, 1, False, False, [3.0, 3.0]),
]


def _make_ms(path, rows=_ROWS, partial_flag_row=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    ms = casacore_tables.default_ms(str(path))
    ms.addrows(len(rows))
    for i, row in enumerate(rows):
        all_flagged, flag_row, weight = row[7:]
        flag = np.full((4, 2), all_flagged)  # 4 channels, 2 correlations
        if partial_flag_row == i:
            flag[0, 0] = True
        ms.putcell("FLAG", i, flag)
        ms.putcell("FLAG_ROW", i, flag_row)
        ms.putcell("WEIGHT", i, np.array(weight, dtype=float))
    for index, column in enumerate(
        ["FIELD_ID", "DATA_DESC_ID", "STATE_ID", "SCAN_NUMBER", "TIME", "ANTENNA1", "ANTENNA2"]
    ):
        ms.putcol(column, np.array([r[index] for r in rows]))
    ms.putcol("INTERVAL", np.full(len(rows), 6.0))
    ms.close()
    for sub, column, values in (
        ("FIELD", "NAME", ["J0552+0313", "HOPS-358"]),
        (
            "STATE",
            "OBS_MODE",
            [
                "CALIBRATE_PHASE#ON_SOURCE,CALIBRATE_WVR#ON_SOURCE",
                "OBSERVE_TARGET#ON_SOURCE",
                "CALIBRATE_ATMOSPHERE#ON_SOURCE",
            ],
        ),
        ("DATA_DESCRIPTION", "SPECTRAL_WINDOW_ID", [25, 27]),
    ):
        sub_table = casacore_tables.table(f"{path}/{sub}", readonly=False, ack=False)
        sub_table.addrows(len(values))
        sub_table.putcol(column, np.array(values))
        sub_table.close()
    return path


@pytest.mark.unit
def test_uid_of_strips_calibrated_suffix(tmp_path):
    assert uid_of(tmp_path / "uid___A002_X1_X2.ms.split.cal") == "uid___A002_X1_X2"
    assert uid_of("x/uid___A002_X1_X2.ms") == "uid___A002_X1_X2"


@pytest.mark.unit
def test_flag_summary_flags_and_weights_per_field_and_spw(tmp_path):
    ms = _make_ms(tmp_path / "uid___A002_X1_X2.ms.split.cal", partial_flag_row=0)

    out = flag_summary(ms)

    assert list(out.columns) == SUMMARY_COLUMNS
    assert list(zip(out["field_name"], out["spectral_window_id"])) == [
        ("J0552+0313", 25),
        ("HOPS-358", 25),
        ("HOPS-358", 27),
    ]
    cal, target, target27 = (out.iloc[i] for i in range(3))
    assert set(out["uid"]) == {"uid___A002_X1_X2"}

    assert (cal.rows, cal.flagged_rows, cal.high_weight_rows) == (2, 1, 0)
    assert cal.intents == "CALIBRATE_PHASE,CALIBRATE_WVR"
    assert cal.visibilities == 16 and cal.flagged_visibilities == 9  # 8 + 1 partial
    assert cal.max_unflagged_weight == 1.0, "the flagged 1e7 row is ignored"
    assert cal.median_unflagged_weight == 1.0

    assert (target.rows, target.flagged_rows, target.high_weight_rows) == (3, 1, 1)
    assert target.intents == "CALIBRATE_ATMOSPHERE,OBSERVE_TARGET"
    assert target.flagged_rows_pct == pytest.approx(100 / 3, abs=1e-3)
    assert target.max_unflagged_weight == 5e5
    # TaQL's gmedian takes the lower middle value of an even count, no interpolation.
    assert target.median_unflagged_weight == 2.0
    assert target.flagged_visibilities == 0, "FLAG_ROW alone flags the row, not FLAG"

    assert (target27.rows, target27.flagged_rows, target27.intents) == (1, 0, "OBSERVE_TARGET")


@pytest.mark.unit
def test_flag_summary_visibility_properties(tmp_path):
    out = flag_summary(_make_ms(tmp_path / "a.ms.split.cal")).set_index(
        ["field_id", "spectral_window_id"]
    )

    cal, target = out.loc[(0, 25)], out.loc[(1, 25)]
    assert (cal.channels, cal.correlations) == (4, 2)
    assert (cal.scans, cal.antennas) == (1, 2)
    assert cal.on_source_s == 12.0, "one scan, 100 s to 106 s plus one 6 s integration"
    assert (cal.start_time_mjd_s, cal.end_time_mjd_s) == (100.0, 106.0)
    assert (target.scans, target.antennas) == (2, 3)
    assert target.on_source_s == 12.0, "two single-integration scans"


@pytest.mark.unit
def test_flag_summary_max_weight_threshold(tmp_path):
    ms = _make_ms(tmp_path / "a.ms.split.cal")
    assert flag_summary(ms, max_weight=1e6)["high_weight_rows"].sum() == 0
    assert flag_summary(ms, max_weight=2.5)["high_weight_rows"].sum() == 2  # 5e5 and 3.0


@pytest.mark.unit
def test_flag_summary_cli_takes_an_ms_or_a_folder(tmp_path):
    folder = tmp_path / "calibrated"
    one = _make_ms(folder / "uid___A002_X1_X2.ms.split.cal")
    _make_ms(folder / "uid___A002_X1_X3.ms.split.cal")
    single_csv, folder_csv = tmp_path / "one.csv", tmp_path / "all.csv"

    result = runner.invoke(
        cli.app, ["visibilities", "flag-summary", str(one), "-o", str(single_csv)]
    )
    assert result.exit_code == 0, result.output
    assert set(pd.read_csv(single_csv)["uid"]) == {"uid___A002_X1_X2"}
    assert "unflagged with WEIGHT > 10000: 1 in 1 MS(s)" in result.output
    assert (
        "--compare-root"
        not in runner.invoke(cli.app, ["visibilities", "flag-summary", "--help"]).output
    )

    (folder / "broken.ms.split.cal").mkdir()  # not a table
    result = runner.invoke(
        cli.app, ["visibilities", "flag-summary", str(folder), "-o", str(folder_csv)]
    )
    assert result.exit_code == 1, "the unreadable MS is reported at the end"
    assert set(pd.read_csv(folder_csv)["uid"]) == {"uid___A002_X1_X2", "uid___A002_X1_X3"}
    assert "broken.ms.split.cal" in (tmp_path / "all.csv.failed.tsv").read_text()
