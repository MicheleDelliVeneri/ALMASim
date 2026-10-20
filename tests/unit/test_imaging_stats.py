"""Shared image statistics (MADIV-28 T2.8)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from almasim.services.imaging import quality, stats

pytestmark = pytest.mark.unit


def test_robust_sigma_is_1_4826_mad_of_the_finite_values():
    values = np.array([1.0, 2.0, 3.0, 4.0, 100.0, np.nan, np.inf])
    assert stats.robust_sigma(values) == pytest.approx(1.4826 * 1.0)


def test_robust_sigma_of_nothing_finite_is_nan():
    assert math.isnan(stats.robust_sigma(np.array([np.nan, np.inf])))


def test_quality_reexports_the_shared_function():
    assert quality.robust_sigma is stats.robust_sigma
