"""Antenna configuration, baselines, beam and FOV helpers (MADIV-28 T8.2: moved from tests/)."""

import os
from pathlib import Path

import astropy.units as U
import pandas as pd
import pytest

from almasim.services.interferometry import antenna as alma_antenna

pytestmark = pytest.mark.unit


def test_alma_functions(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    main_dir = repo_root / "src" / "almasim"
    metadata = pd.read_csv(repo_root / "data" / "qso_metadata.csv").iloc[0]
    antenna_array = metadata["antenna_arrays"]
    central_freq = metadata["Freq"] * U.GHz
    alma_antenna.generate_antenna_config_file_from_antenna_array(
        antenna_array, str(main_dir), str(tmp_path)
    )
    assert os.path.isfile(tmp_path / "antenna.cfg")
    max_baseline = alma_antenna.get_max_baseline_from_antenna_array(antenna_array, str(main_dir))
    assert max_baseline > 0
    beam_size = alma_antenna.estimate_alma_beam_size(central_freq, max_baseline, return_value=False)
    assert beam_size > 0
    beam_size = alma_antenna.estimate_alma_beam_size(central_freq, max_baseline, return_value=True)
    assert beam_size > 0

    fov = alma_antenna.get_fov_from_band(6, return_value=False)
    assert fov > 0
    fov = alma_antenna.get_fov_from_band(6, return_value=True)
    assert fov > 0
    for band in range(1, 11):
        fov = alma_antenna.get_fov_from_band(band)
        assert fov > 0
