"""Bring a database created before Alembic up to the baseline (MADIV-28 T6.3).

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-21

A database the application made with ``create_all`` before Alembic is
stamped at 0001 (``app.db.migrate``) and then needs what ``init_db`` and the
startup used to do on every start: the columns added to existing tables later
(``schema_upgrade.ADDED_COLUMNS``) and the NOT NULL constraints TAP rows
cannot meet dropped. On a database 0001 created both are no-ops.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from app.db.schema_upgrade import ADDED_COLUMNS

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Observation columns an incomplete TAP row may leave empty.
NULLABLE_OBSERVATION_COLUMNS = ("ra", "dec", "target_name")


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    for table, columns in ADDED_COLUMNS.items():
        if not inspector.has_table(table):
            continue
        present = {column["name"] for column in inspector.get_columns(table)}
        for name, ddl in columns:
            if name not in present:
                op.execute(sa.text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"))
    if bind.dialect.name == "postgresql" and inspector.has_table("observations"):
        not_null = {
            column["name"]
            for column in inspector.get_columns("observations")
            if not column["nullable"]
        }
        for name in NULLABLE_OBSERVATION_COLUMNS:
            if name in not_null:
                op.alter_column("observations", name, nullable=True)


def downgrade() -> None:
    """Nothing to undo: the columns are part of the baseline's tables."""
