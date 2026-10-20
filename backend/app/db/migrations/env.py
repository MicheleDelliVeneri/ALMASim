"""Alembic environment: the backend's engine and models (MADIV-28 T6.3)."""

from alembic import context
from app.db.models import Base
from sqlalchemy import engine_from_config, pool

config = context.config
target_metadata = Base.metadata


def _connectable():
    """The connection the caller passed (app.db.migrate), else an engine for the settings' URL."""
    connection = config.attributes.get("connection")
    if connection is not None:
        return connection
    url = config.get_main_option("sqlalchemy.url")
    if not url:
        from app.core.config import settings

        url = settings.DATABASE_URL
    return engine_from_config(
        {"sqlalchemy.url": url}, prefix="sqlalchemy.", poolclass=pool.NullPool
    )


def run_migrations_offline() -> None:
    from app.core.config import settings

    context.configure(
        url=config.get_main_option("sqlalchemy.url") or settings.DATABASE_URL,
        target_metadata=target_metadata,
        literal_binds=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = _connectable()
    if hasattr(connectable, "connect"):
        with connectable.connect() as connection:
            context.configure(connection=connection, target_metadata=target_metadata)
            with context.begin_transaction():
                context.run_migrations()
    else:
        context.configure(connection=connectable, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
