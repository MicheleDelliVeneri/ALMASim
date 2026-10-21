"""Line catalogue, redshift and snapshot helpers (MADIV-28 T8.2: moved from tests/)."""

from pathlib import Path

import astropy.units as U
import pandas as pd
import pytest

from almasim.services import astro

pytestmark = pytest.mark.unit


def test_luminosity_functions():
    repo_root = Path(__file__).resolve().parents[2]
    line_path = repo_root / "src" / "almasim"
    rest_freq, line_names = astro.get_line_info(line_path)
    assert len(rest_freq) > 0
    assert len(line_names) > 0
    rest_freq, line_names = astro.get_line_info(line_path, [0])
    assert len(rest_freq) > 0
    assert len(line_names) > 0
    metadata_path = repo_root / "data" / "qso_metadata.csv"
    metadata = pd.read_csv(metadata_path).iloc[0]
    source_freq = metadata["Freq"]
    redshift = 0.01
    rest_freq = astro.compute_rest_frequency_from_redshift(line_path, source_freq, redshift)
    assert rest_freq > 0
    redshift = astro.compute_redshift(rest_freq * U.GHz, source_freq * U.GHz)
    snapshot = astro.redshift_to_snapshot(redshift)
    assert snapshot != -1
