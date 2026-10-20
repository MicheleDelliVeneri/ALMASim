"""SimulationParams coerces its own fields (MADIV-28 T5.2)."""

from __future__ import annotations

import math
import os

import pandas as pd
import pytest

from almasim.services.simulation import SimulationParams

pytestmark = pytest.mark.unit

ROW = {
    "ALMA_source_name": "J0842",
    "member_ous_uid": "uid://A001/X1/X1",
    "Band": 6,
    "RA": "130.6",
    "Dec": 12.3,
    "Ang.res.": 1.0,
    "Vel.res.": 10.0,
    "FOV": 25.0,
    "Obs.date": "2017-03-14",
    "PWV": 2.1,
    "Int.Time": 450.0,
    "Bandwidth": 1.875,
    "Freq": 269.7,
    "Freq.sup.": "[251.05..252.92GHz,1938.48kHz]",
    "Cont_sens_mJybeam": 0.045,
    "antenna_arrays": "A001:DA41 A002:DA42",
    "redshift": 0.5,
}


def _paths(tmp_path):
    return {
        name: tmp_path / name
        for name in ("main_dir", "output_dir", "tng_dir", "galaxy_zoo_dir", "hubble_dir")
    }


def test_construction_coerces_numbers_paths_and_cpu_count(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    params = SimulationParams.from_metadata_row(
        ROW, idx="2", project_name=7, **{**_paths(tmp_path), "output_dir": "out"}
    )
    assert params.idx == 2 and params.ra == 130.6 and params.project_name == "7"
    assert params.output_dir == str(tmp_path / "out")
    assert params.ncpu == (os.cpu_count() or 1)
    assert params.source_type == "point" and params.save_mode == "fits" and params.persist


def test_explicit_values_beat_the_row_and_nan_counts_as_missing(tmp_path):
    row = {**ROW, "lum_infrared": math.nan}
    params = SimulationParams.from_metadata_row(
        pd.Series(row), idx=0, project_name="p", redshift=1.5, **_paths(tmp_path)
    )
    assert params.redshift == 1.5
    assert params.lum_infrared is None
    assert (
        SimulationParams.from_metadata_row(
            row, idx=0, project_name="p", **_paths(tmp_path)
        ).redshift
        == 0.5
    )


def test_a_missing_required_column_is_named(tmp_path):
    row = {key: value for key, value in ROW.items() if key != "FOV"}
    with pytest.raises(KeyError, match="FOV"):
        SimulationParams.from_metadata_row(row, idx=0, project_name="p", **_paths(tmp_path))


def test_system_locations_are_refused(tmp_path):
    with pytest.raises(ValueError, match="restricted system location"):
        SimulationParams.from_metadata_row(
            ROW, idx=0, project_name="p", **{**_paths(tmp_path), "output_dir": "/proc/self"}
        )
