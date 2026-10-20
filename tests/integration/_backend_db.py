"""Whether the backend's configured database can be reached from the tests."""

from __future__ import annotations

import importlib.util


def database_driver_missing() -> str | None:
    """A skip reason when the configured database needs a driver that is not installed.

    The engine is created lazily (MADIV-28 T6.2), so the app imports without
    one; without this check the requests would fail instead of skipping.
    """
    from app.core.config import settings

    if settings.DATABASE_URL.startswith("postgresql") and not importlib.util.find_spec("psycopg2"):
        return "the PostgreSQL driver (psycopg2) is not installed"
    return None
