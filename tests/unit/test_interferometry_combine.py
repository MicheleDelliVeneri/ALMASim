"""Shared steps of combining per-configuration runs (MADIV-28 T5.9)."""

from __future__ import annotations

import numpy as np
import pytest

from almasim.services.interferometry.combine import normalized_config_weights, single_run_result

pytestmark = pytest.mark.unit


def test_weights_are_normalised_with_negatives_clipped():
    np.testing.assert_allclose(normalized_config_weights(3, [2.0, -1.0, 2.0]), [0.5, 0.0, 0.5])
    np.testing.assert_allclose(normalized_config_weights(2, None), [0.5, 0.5])
    np.testing.assert_allclose(normalized_config_weights(2, [0.0, 0.0]), [0.5, 0.5])


def test_a_single_run_is_its_own_combination():
    run = {"model_cube": 1}
    combined = single_run_result([run], "TP")
    assert combined == {"model_cube": 1, "per_config_results": [run], "combined_config_count": 1}
    assert single_run_result([run, run], "TP") is None
    with pytest.raises(ValueError, match="empty TP result list"):
        single_run_result([], "TP")
