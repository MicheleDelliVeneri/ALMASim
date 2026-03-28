import pytest
from almasim.services.metadata.tap import service as alma
from almasim.services.interferometry import antenna as alma_antenna
from pathlib import Path
import inspect
import faulthandler
import os
import pandas as pd
import astropy.units as U

faulthandler.enable()
os.environ["LC_ALL"] = "C"


@pytest.mark.integration
@pytest.mark.network
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
        os.path.join(main_path, "data", "targets_qso.csv")
    ).values.tolist()
    data = alma.query_all_targets(target_list)
    assert not data.empty
    assert not results.empty


def test_alma_functions():
    repo_root = Path(inspect.getfile(inspect.currentframe())).resolve().parents[1]
    main_dir = repo_root / "src" / "almasim"
    metadata_path = repo_root / "data" / "qso_metadata.csv"
    metadata = pd.read_csv(metadata_path).iloc[0]
    antenna_array = metadata["antenna_arrays"]
    central_freq = metadata["Freq"] * U.GHz
    alma_antenna.generate_antenna_config_file_from_antenna_array(
        antenna_array, str(main_dir), str(repo_root)
    )
    antennalist = repo_root / "antenna.cfg"
    assert os.path.isfile(antennalist)
    max_baseline = alma_antenna.get_max_baseline_from_antenna_config(None, antennalist) * U.km
    assert max_baseline > 0
    max_baseline = alma_antenna.get_max_baseline_from_antenna_array(
        antenna_array, str(main_dir)
    )
    assert max_baseline > 0
    beam_size = alma_antenna.estimate_alma_beam_size(
        central_freq, max_baseline, return_value=False
    )
    assert beam_size > 0
    beam_size = alma_antenna.estimate_alma_beam_size(
        central_freq, max_baseline, return_value=True
    )
    assert beam_size > 0
    os.remove(antennalist)

    fov = alma_antenna.get_fov_from_band(6, return_value=False)
    assert fov > 0
    fov = alma_antenna.get_fov_from_band(6, return_value=True)
    assert fov > 0
    for band in range(1, 11):
        fov = alma_antenna.get_fov_from_band(band)
        assert fov > 0


if __name__ == "__main__":
    pytest.main(["-v", __file__])
