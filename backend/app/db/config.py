"""Database configuration and session management."""

from contextlib import contextmanager
from functools import lru_cache
from typing import Generator

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """The database engine, created on first use (at application startup)."""
    from app.core.config import settings

    url = settings.DATABASE_URL
    connect_args = {"connect_timeout": 1} if url.startswith("postgresql") else {}
    return create_engine(
        url,
        pool_pre_ping=True,  # Verify connections before using them
        echo=False,  # Set to True for SQL query logging
        connect_args=connect_args,
    )


@lru_cache(maxsize=1)
def _session_factory() -> sessionmaker:
    return sessionmaker(autocommit=False, autoflush=False, bind=get_engine())


def new_session() -> Session:
    """A new database session (the caller closes it)."""
    return _session_factory()()


def init_db():
    """Create or upgrade the database's tables (Alembic, MADIV-28 T6.3)."""
    from .migrate import upgrade_database

    upgrade_database(get_engine())


def get_db() -> Generator[Session, None, None]:
    """
    Get database session for dependency injection.

    Usage in FastAPI:
        @app.get("/items")
        def read_items(db: Session = Depends(get_db)):
            return db.query(Item).all()
    """
    db = new_session()
    try:
        yield db
    except Exception:
        # Roll back any in-flight transaction so the connection is returned
        # to the pool clean and Postgres does not log
        # "unexpected EOF on client connection with an open transaction".
        db.rollback()
        raise
    finally:
        db.close()


@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """
    Get database session as context manager.

    Usage:
        with get_db_context() as db:
            db.query(Item).all()
    """
    db = new_session()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
