import pytest
import almasim.alma as alma
from pathlib import Path
import inspect
import faulthandler
import os
import pandas as pd

faulthandler.enable()


def test_query():
    main_path = os.path.sep + os.path.join(
        *str(Path(inspect.getfile(inspect.currentframe())).resolve()).split(
            os.path.sep
        )[:-2]
    )
    print(main_path)
    science_keywods, _ = alma.get_science_types()
    science_keyword = science_keywods[0]
    results = alma.query_by_science_type(science_keyword, band=6)
    target_list = pd.read_csv(
        os.path.join(main_path, "almasim", "metadata", "targets_qso.csv")
    ).values.tolist()
    data = alma.query_all_targets(target_list)
    assert not data.empty
    assert not results.empty


if __name__ == "__main__":
    pytest.main(["-v", __file__])
