"""init_db's upgrade of older databases (MADIV-28 T6.8)."""

from __future__ import annotations

import pytest

pytest.importorskip("sqlalchemy")

from sqlalchemy import create_engine, inspect, text


@pytest.mark.unit
def test_add_missing_columns_upgrades_old_tables_once(backend_db):
    from app.db.schema_upgrade import ADDED_COLUMNS, add_missing_columns

    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE observations (id INTEGER PRIMARY KEY)"))
        conn.execute(
            text("CREATE TABLE download_jobs (id INTEGER PRIMARY KEY, manifest_path TEXT)")
        )

    added = add_missing_columns(engine)

    assert "observations.obs_type" in added
    assert "download_jobs.manifest_path" not in added, "a present column is left alone"
    insp = inspect(engine)
    for table, columns in ADDED_COLUMNS.items():
        present = {c["name"] for c in insp.get_columns(table)}
        assert {name for name, _ in columns} <= present
    assert add_missing_columns(engine) == []


@pytest.mark.unit
def test_add_missing_columns_skips_absent_tables(backend_db):
    from app.db.schema_upgrade import add_missing_columns

    assert add_missing_columns(create_engine("sqlite://")) == []
