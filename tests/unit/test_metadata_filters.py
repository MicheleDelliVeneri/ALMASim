"""One filter model for the TAP queries (MADIV-28 T6.1)."""

from __future__ import annotations

import pytest

from almasim.services.metadata import filters
from almasim.services.metadata.tap import service as tap_service

pytestmark = pytest.mark.unit


def test_the_tap_service_uses_the_shared_filter_classes():
    assert tap_service.InclusionFilters is filters.InclusionFilters
    assert tap_service.ExclusionFilters is filters.ExclusionFilters


def test_qa2_codes_maps_labels_and_keeps_codes():
    assert filters.qa2_codes(["Pass", "Fail", "SemiPass", "T", "?"]) == ["T", "F", "X", "T", "?"]
