import faulthandler
import inspect
import os
from pathlib import Path

import astropy.units as U
import pandas as pd
import pytest

from almasim.services import astro

faulthandler.enable()
os.environ["LC_ALL"] = "C"


def test_luminosity_functions():
    repo_root = Path(inspect.getfile(inspect.currentframe())).resolve().parents[1]
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


if __name__ == "__main__":
    pytest.main(["-v", __file__])
