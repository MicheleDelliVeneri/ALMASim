"""The backend database schema is managed by Alembic (MADIV-28 T6.3)."""

from __future__ import annotations

import importlib
import io
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit

alembic = pytest.importorskip("alembic")


@pytest.fixture
def migrate(backend_db):
    return importlib.import_module("app.db.migrate")


def test_one_chain_of_revisions(migrate):
    from alembic.script import ScriptDirectory

    script = ScriptDirectory.from_config(migrate.alembic_config())
    assert script.get_heads() == ["0002"]
    assert [rev.revision for rev in script.walk_revisions()] == ["0002", migrate.BASELINE]


def test_the_baseline_creates_every_model_table(migrate, backend_db):
    """The baseline, rendered as PostgreSQL SQL without a database, creates every table."""
    from alembic import command

    config = migrate.alembic_config()
    config.set_main_option("sqlalchemy.url", "postgresql://")
    config.output_buffer = io.StringIO()
    command.upgrade(config, migrate.BASELINE, sql=True)
    sql = config.output_buffer.getvalue()
    for table in backend_db.models.Base.metadata.tables:
        assert f"CREATE TABLE {table} " in sql, table


def _sqlite_engine(directory):
    from sqlalchemy import create_engine

    directory.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{directory / 'almasim.db'}")


def test_a_database_made_before_alembic_is_stamped_then_upgraded(migrate, tmp_path, monkeypatch):
    from sqlalchemy import text

    engine = _sqlite_engine(tmp_path)
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE observations (id INTEGER PRIMARY KEY, ra FLOAT)"))
    calls = []
    fake = MagicMock()
    fake.stamp.side_effect = lambda config, revision: calls.append(("stamp", revision))
    fake.upgrade.side_effect = lambda config, revision: calls.append(("upgrade", revision))
    monkeypatch.setattr(migrate, "command", fake)

    migrate.upgrade_database(engine)
    assert calls == [("stamp", "0001"), ("upgrade", "head")]

    calls.clear()
    migrate.upgrade_database(_sqlite_engine(tmp_path / "new"))  # no tables: a new database
    assert calls == [("upgrade", "head")]


def test_revision_0002_adds_the_columns_older_databases_lack(migrate, tmp_path):
    """End to end on SQLite: stamped at the baseline, 0002 adds the missing columns."""
    from sqlalchemy import inspect, text

    engine = _sqlite_engine(tmp_path)
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE observations (id INTEGER PRIMARY KEY, ra FLOAT)"))
        conn.execute(text("CREATE TABLE download_jobs (id INTEGER PRIMARY KEY)"))

    migrate.upgrade_database(engine)

    inspector = inspect(engine)
    assert {"proposal_abstract", "qa2_passed", "obs_type"} <= {
        c["name"] for c in inspector.get_columns("observations")
    }
    assert "manifest_path" in {c["name"] for c in inspector.get_columns("download_jobs")}
    with engine.connect() as conn:
        assert conn.execute(text("SELECT version_num FROM alembic_version")).scalar() == "0002"


def test_a_new_database_is_built_and_torn_down_by_the_revisions(migrate, backend_db, tmp_path):
    """The whole chain on a new SQLite database, through the online environment."""
    from alembic import command
    from sqlalchemy import inspect

    engine = _sqlite_engine(tmp_path)
    migrate.upgrade_database(engine)
    tables = set(inspect(engine).get_table_names())
    assert set(backend_db.models.Base.metadata.tables) <= tables

    config = migrate.alembic_config()
    config.set_main_option("sqlalchemy.url", str(engine.url))  # no connection: env.py's engine
    command.downgrade(config, "base")
    assert set(inspect(engine).get_table_names()) == {"alembic_version"}


def test_init_db_upgrades_the_application_database(backend_db, monkeypatch):
    config = importlib.import_module("app.db.config")
    migrate = importlib.import_module("app.db.migrate")
    engine = object()
    seen = []
    monkeypatch.setattr(config, "get_engine", lambda: engine)
    monkeypatch.setattr(migrate, "upgrade_database", seen.append)
    config.init_db()
    assert seen == [engine]
