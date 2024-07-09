import pytest
import almasim.alma as alma
from pathlib import Path
import inspect
import faulthandler
import os
import pandas as pd
import astropy.units as U

faulthandler.enable()


def test_query():
    main_path = os.path.sep + os.path.join(
        *str(Path(inspect.getfile(inspect.currentframe())).resolve()).split(
            os.path.sep
        )[:-2]
    )
    science_keywods, _ = alma.get_science_types()
    science_keyword = science_keywods[0]
    results = alma.query_by_science_type(science_keyword, band=[6])
    target_list = pd.read_csv(
        os.path.join(main_path, "almasim", "metadata", "targets_qso.csv")
    ).values.tolist()
    data = alma.query_all_targets(target_list)
    assert not data.empty
    assert not results.empty


def test_alma_functions():
    main_path = os.path.sep + os.path.join(
        *str(Path(inspect.getfile(inspect.currentframe())).resolve()).split(
            os.path.sep
        )[:-2]
    )
    metadata_path = os.path.join(main_path, "almasim", "metadata", "qso_metadata.csv")
    metadata = pd.read_csv(metadata_path).iloc[0]
    antenna_array = metadata["antenna_arrays"]
    central_freq = metadata["Freq"] * U.GHz
    alma.generate_antenna_config_file_from_antenna_array(
        antenna_array, os.path.join(main_path, "almasim"), main_path
    )
    antennalist = os.path.join(main_path, "antenna.cfg")
    assert os.path.isfile(antennalist)
    max_baseline = alma.get_max_baseline_from_antenna_config(None, antennalist) * U.km
    assert max_baseline > 0
    max_baseline = alma.get_max_baseline_from_antenna_array(
        antenna_array, os.path.join(main_path, "almasim")
    )
    assert max_baseline > 0
    beam_size = alma.estimate_alma_beam_size(
        central_freq, max_baseline, return_value=False
    )
    assert beam_size > 0
    beam_size = alma.estimate_alma_beam_size(
        central_freq, max_baseline, return_value=True
    )
    assert beam_size > 0
    os.remove(os.path.join(main_path, "antenna.cfg"))

    fov = astro.get_fov_from_band(6, return_value=False)
    assert fov > 0
    fov = astro.get_fov_from_band(6, return_value=True)
    assert fov > 0


if __name__ == "__main__":
    pytest.main(["-v", __file__])
