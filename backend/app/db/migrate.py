"""Upgrade the backend database with Alembic (MADIV-28 T6.3)."""

from __future__ import annotations

import logging
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, inspect

logger = logging.getLogger(__name__)

ALEMBIC_INI = Path(__file__).resolve().parents[2] / "alembic.ini"
# The revision whose tables the application created with create_all before
# it used Alembic.
BASELINE = "0001"


def alembic_config() -> Config:
    return Config(str(ALEMBIC_INI))


def upgrade_database(engine: Engine) -> None:
    """Bring the database at ``engine`` to the latest revision.

    A database made before Alembic (tables but no ``alembic_version``) is
    first stamped at the baseline, so only the later revisions run on it.
    """
    config = alembic_config()
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        tables = set(inspect(connection).get_table_names())
        if "alembic_version" not in tables and "observations" in tables:
            logger.info("Database made before Alembic: stamping it at %s", BASELINE)
            command.stamp(config, BASELINE)
        command.upgrade(config, "head")
