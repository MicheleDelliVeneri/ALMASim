"""A live ALMA TAP query (MADIV-28 T8.2: moved from tests/)."""

from pathlib import Path

import pandas as pd
import pytest

from almasim.services.metadata.tap import service as alma


@pytest.mark.integration
@pytest.mark.network
def test_query():
    repo_root = Path(__file__).resolve().parents[2]
    science_keywords, _ = alma.get_science_types()
    results = alma.query_by_science_type(science_keywords[0], band=[6])
    target_list = pd.read_csv(repo_root / "data" / "targets_qso.csv").values.tolist()
    data = alma.query_all_targets(target_list)
    assert not data.empty
    assert not results.empty
