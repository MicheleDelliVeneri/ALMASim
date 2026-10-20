"""Saving a preset twice replaces it (MADIV-28 T1.10).

POST /presets used to insert a row every time; the lookups by name then raised
MultipleResultsFound, so GET and DELETE answered 500.
"""

from __future__ import annotations

from datetime import datetime

import pytest

pytestmark = pytest.mark.unit


def test_saving_a_name_twice_keeps_one_preset(backend_db):
    backend_db.db.save_query_preset("band6", {"bands": [6]}, description="first", result_count=3)
    saved = backend_db.db.save_query_preset("band6", {"bands": [6, 7]}, description="second")

    rows = backend_db.session.query(backend_db.models.QueryResult).all()
    assert len(rows) == 1
    assert rows[0].id == saved.id
    preset = backend_db.db.get_query_result("band6")
    assert preset.query_params == {"filters": {"bands": [6, 7]}}
    assert (preset.description, preset.result_count) == ("second", 0)


def test_duplicates_saved_before_the_fix_are_readable_and_folded(backend_db):
    QueryResult = backend_db.models.QueryResult
    for day, band in ((1, 3), (2, 6)):
        backend_db.session.add(
            QueryResult(
                query_name="old",
                query_params={"filters": {"bands": [band]}},
                result_count=0,
                observation_ids=[],
                created_at=datetime(2025, 1, day),
            )
        )
    backend_db.session.commit()

    assert backend_db.db.get_query_result("old").query_params == {"filters": {"bands": [6]}}
    assert [row.query_name for row in backend_db.db.list_query_results()] == ["old"]

    backend_db.db.save_query_preset("old", {"bands": [7]})
    assert backend_db.session.query(QueryResult).count() == 1

    assert backend_db.db.delete_query_result("old") is True
    assert backend_db.db.get_query_result("old") is None
    assert backend_db.db.delete_query_result("old") is False


def test_list_is_newest_first_and_limited(backend_db):
    for name in ("a", "b", "c"):
        backend_db.db.save_query_preset(name, {})
    assert [row.query_name for row in backend_db.db.list_query_results(limit=2)] == ["c", "b"]
