"""The metadata cache answers with the same filters as TAP (MADIV-28 T1.12).

Runs DatabaseService and MetadataService against an in-memory SQLite cache
(the ``backend_db`` fixture in conftest.py).
"""

from __future__ import annotations

import importlib
from datetime import datetime

import pytest

from almasim.services.metadata.filters import ExclusionFilters, InclusionFilters

pytestmark = pytest.mark.unit


@pytest.fixture
def cache(backend_db):
    """The SQLite cache plus a helper that adds an observation as the importer would."""
    csv = importlib.import_module("app.db.csv_importer").CSVImporter(backend_db.session)

    def add(uid, target="NGC 1068", keywords=("Galaxies",), category="Active galaxies", **row):
        observation = csv.parse_csv_row(
            {"member_ous_uid": uid, "ALMA_source_name": target, "Band": "6", **row},
            "test",
        )
        for keyword in keywords:
            observation.science_keywords.append(csv.get_or_create_keyword(keyword))
        observation.scientific_category = csv.get_or_create_category(category)
        backend_db.session.add(observation)
        backend_db.session.commit()
        return observation

    backend_db.add = add
    return backend_db


def _uids(observations):
    return sorted(observation.member_ous_uid for observation in observations)


def test_int_time_is_stored_as_t_resolution_and_filtered(cache):
    cache.add("uid://A/short", **{"Int.Time": "60"})
    cache.add("uid://A/long", **{"Int.Time": "3600"})
    legacy = cache.add("uid://A/legacy")
    legacy.t_max = 600.0  # cached before T1.12: Int.Time in t_max
    cache.session.commit()

    assert cache.session.get(cache.models.Observation, 1).t_resolution == 60.0
    include = InclusionFilters(time_resolution_range=(100, 1000))
    assert _uids(cache.db.query_observations(include=include)) == ["uid://A/legacy"]

    from almasim.services.metadata.adapters import observation_to_metadata_record

    assert observation_to_metadata_record(legacy)["Int.Time"] == 600.0


def test_solar_exclusion_checks_name_keyword_and_category(cache):
    cache.add("uid://A/galaxy")
    cache.add("uid://A/name", target="Sun")
    cache.add("uid://A/keyword", target="AR 1234", keywords=("Sun",))
    cache.add("uid://A/category", target="Quiet region", category="Sun")

    exclude = ExclusionFilters(solar=True)
    assert _uids(cache.db.query_observations(exclude=exclude)) == ["uid://A/galaxy"]


def _metadata_service(cache):
    module = importlib.import_module("app.services.metadata_service")
    return module.MetadataService(db=cache.session)


def _params(**overrides):
    """The service's query for a MetadataQuery with ``overrides``, as the router builds it."""
    from app.schemas.metadata import MetadataQuery

    return MetadataQuery(**overrides).to_query()


def test_cache_keeps_included_keywords_and_applies_exclusions(cache):
    cache.add("uid://A/agn")
    cache.add("uid://A/disk", target="HL Tau", keywords=("Disks around low-mass stars",))
    cache.add("uid://A/sun", target="Sun", keywords=("Galaxies",))
    service = _metadata_service(cache)

    rows = service._try_cache_query(_params(science_keyword=["Galaxies"], exclude_solar=True))
    # The keyword used to land in the exclusions too (empty result), and
    # exclude_solar was dropped.
    assert [row["member_ous_uid"] for row in rows] == ["uid://A/agn"]

    rows = service._try_cache_query(_params(exclude_source_name=["HL"]))
    assert sorted(row["member_ous_uid"] for row in rows) == ["uid://A/agn", "uid://A/sun"]


def test_cache_misses_on_empty_result_and_on_date_filters(cache):
    cache.add("uid://A/agn", **{"Obs.date": datetime(2020, 1, 1).isoformat()})
    service = _metadata_service(cache)

    assert service._try_cache_query(_params(bands=[3])) is None
    dated = _params(observation_date_range=("2019-01-01", "2021-01-01"))
    assert service._try_cache_query(dated) is None
    assert service._try_cache_query(_params(bands=[6])) is not None


def test_qa2_labels_and_cycles_reach_the_cache_query(cache):
    """The cache maps QA2 labels itself, like TAP, and takes cycles as proposal prefixes."""
    passed = cache.add("uid://A/pass", proposal_id="2019.1.00001.S")
    passed.qa2_passed = "T"
    failed = cache.add("uid://A/fail", proposal_id="2018.1.00002.S")
    failed.qa2_passed = "F"
    cache.session.commit()

    query = _params(qa2_status=["Pass"])
    assert _uids(cache.db.query_observations(query.include, query.exclude)) == ["uid://A/pass"]
    query = _params(cycles=[6])
    assert _uids(cache.db.query_observations(query.include, query.exclude)) == ["uid://A/fail"]
    exact = InclusionFilters(proposal_id=["2019.1.00001.S"])
    assert _uids(cache.db.query_observations(exact)) == ["uid://A/pass"]
