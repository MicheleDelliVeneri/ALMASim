import pytest
import almasim.astro as astro
from pathlib import Path
import inspect
import faulthandler
import os
import pandas as pd
import astropy.units as U

faulthandler.enable()
os.environ["LC_ALL"] = "C"

def test_luminosity_functions():
    main_path = os.path.sep + os.path.join(
        *str(Path(inspect.getfile(inspect.currentframe())).resolve()).split(
            os.path.sep
        )[:-2]
    )
    line_path = os.path.join(main_path, "almasim")
    rest_freq, line_names = astro.get_line_info(line_path)
    assert len(rest_freq) > 0
    assert len(line_names) > 0
    rest_freq, line_names = astro.get_line_info(line_path, [0])
    assert len(rest_freq) > 0
    assert len(line_names) > 0
    metadata_path = os.path.join(main_path, "almasim", "metadata", "qso_metadata.csv")
    metadata = pd.read_csv(metadata_path).iloc[0]
    source_freq = metadata["Freq"]
    redshift = 0.01
    rest_freq = astro.compute_rest_frequency_from_redshift(
        line_path, source_freq, redshift
    )
    assert rest_freq > 0
    redshift = astro.compute_redshift(rest_freq * U.GHz, source_freq * U.GHz)
    snapshot = astro.redshift_to_snapshot(redshift)
    assert snapshot != -1


if __name__ == "__main__":
    pytest.main(["-v", __file__])
