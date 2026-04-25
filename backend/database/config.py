"""Database configuration and session management."""

import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from .models import Base


def get_database_url() -> str:
    """Get database URL from settings or environment."""
    # Try to import settings, fall back to env variable
    try:
        from app.core.config import settings

        return settings.DATABASE_URL
    except ImportError:
        return os.getenv(
            "DATABASE_URL",
            "postgresql://almasim:almasim_dev_password@localhost:5432/almasim",
        )


# Get database URL
DATABASE_URL = get_database_url()

_connect_args = {}
if DATABASE_URL.startswith("postgresql"):
    _connect_args["connect_timeout"] = 1

# Create engine
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,  # Verify connections before using them
    echo=False,  # Set to True for SQL query logging
    connect_args=_connect_args,
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)
    # Add columns that may be missing from older schemas
    from sqlalchemy import inspect, text

    insp = inspect(engine)
    if insp.has_table("download_jobs"):
        cols = {c["name"] for c in insp.get_columns("download_jobs")}
        with engine.begin() as conn:
            if "member_ous_uids" not in cols:
                conn.execute(
                    text(
                        "ALTER TABLE download_jobs "
                        "ADD COLUMN member_ous_uids TEXT NOT NULL DEFAULT ''"
                    )
                )
            if "product_filter" not in cols:
                conn.execute(
                    text(
                        "ALTER TABLE download_jobs "
                        "ADD COLUMN product_filter VARCHAR(20) NOT NULL DEFAULT 'all'"
                    )
                )
            if "metadata_json" not in cols:
                conn.execute(
                    text(
                        "ALTER TABLE download_jobs "
                        "ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '[]'"
                    )
                )
            if "unpack_ms" not in cols:
                conn.execute(
                    text(
                        "ALTER TABLE download_jobs "
                        "ADD COLUMN unpack_ms BOOLEAN NOT NULL DEFAULT FALSE"
                    )
                )
            if "generate_calibrated_visibilities" not in cols:
                conn.execute(
                    text(
                        "ALTER TABLE download_jobs "
                        "ADD COLUMN generate_calibrated_visibilities "
                        "BOOLEAN NOT NULL DEFAULT FALSE"
                    )
                )
            if "clean_intermediate_files" not in cols:
                conn.execute(
                    text(
                        "ALTER TABLE download_jobs "
                        "ADD COLUMN clean_intermediate_files "
                        "BOOLEAN NOT NULL DEFAULT FALSE"
                    )
                )
            if "archive_output_root" not in cols:
                conn.execute(text("ALTER TABLE download_jobs ADD COLUMN archive_output_root TEXT"))
            if "casa_data_root" not in cols:
                conn.execute(text("ALTER TABLE download_jobs ADD COLUMN casa_data_root TEXT"))
            if "skip_casa_data_update" not in cols:
                conn.execute(
                    text(
                        "ALTER TABLE download_jobs "
                        "ADD COLUMN skip_casa_data_update "
                        "BOOLEAN NOT NULL DEFAULT FALSE"
                    )
                )
            if "raw_measurement_sets" not in cols:
                conn.execute(
                    text(
                        "ALTER TABLE download_jobs "
                        "ADD COLUMN raw_measurement_sets TEXT NOT NULL DEFAULT '[]'"
                    )
                )
            if "calibrated_measurement_sets" not in cols:
                conn.execute(
                    text(
                        "ALTER TABLE download_jobs "
                        "ADD COLUMN calibrated_measurement_sets "
                        "TEXT NOT NULL DEFAULT '[]'"
                    )
                )
            if "manifest_path" not in cols:
                conn.execute(text("ALTER TABLE download_jobs ADD COLUMN manifest_path TEXT"))


def get_db() -> Generator[Session, None, None]:
    """
    Get database session for dependency injection.

    Usage in FastAPI:
        @app.get("/items")
        def read_items(db: Session = Depends(get_db)):
            return db.query(Item).all()
    """
    db = SessionLocal()
    try:
        yield db
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
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
