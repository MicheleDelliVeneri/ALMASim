"""One filter model for the TAP and the cache queries (MADIV-28 T6.1)."""

from __future__ import annotations

import importlib

import pytest

from almasim.services.metadata import filters
from almasim.services.metadata.tap import service as tap_service

pytestmark = pytest.mark.unit


def test_the_tap_service_uses_the_shared_filter_classes():
    assert tap_service.InclusionFilters is filters.InclusionFilters
    assert tap_service.ExclusionFilters is filters.ExclusionFilters


def test_qa2_codes_maps_labels_and_keeps_codes():
    assert filters.qa2_codes(["Pass", "Fail", "SemiPass", "T", "?"]) == ["T", "F", "X", "T", "?"]


def test_metadata_query_to_filters(backend_db):
    schemas = importlib.import_module("app.schemas.metadata")
    query = schemas.MetadataQuery(
        source_name="NGC",
        bands=[6, 7],
        cycles=[4, 10],
        qa2_status=["Pass"],
        exclude_source_name=["Sun"],
        exclude_solar=True,
        public_only=False,
        visible_columns=["Band"],
    )
    include, exclude = query.to_filters()
    assert include.source_name == "NGC" and include.band == [6, 7]
    assert include.proposal_id_prefix == ["2016.", "2022."]
    assert include.qa2_status == ["Pass"] and include.public_only is False
    assert exclude.source_name == ["Sun"] and exclude.solar is True
    science = query.to_query()
    assert (science.include, science.exclude) == (include, exclude)
    assert science.visible_columns == ["Band"]


def test_an_empty_query_has_the_default_filters(backend_db):
    schemas = importlib.import_module("app.schemas.metadata")
    include, exclude = schemas.MetadataQuery().to_filters()
    assert include == filters.InclusionFilters()
    assert exclude == filters.ExclusionFilters()
