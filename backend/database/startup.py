"""Database initialization on application startup."""

import logging
from pathlib import Path

from sqlalchemy import text

from .config import SessionLocal, engine, init_db
from .csv_importer import initialize_database_from_csv
from .models import Observation

logger = logging.getLogger(__name__)


def check_database_initialized() -> bool:
    """
    Check if the database has been initialized with data.

    Returns:
        True if database has observations, False otherwise
    """
    try:
        with SessionLocal() as db:
            result = db.execute(text("SELECT COUNT(*) FROM observations"))
            count = result.scalar()
            return count > 0
    except Exception as e:
        logger.warning(f"Could not check database initialization status: {e}")
        return False


def initialize_database_on_startup(data_dir: Path) -> None:
    """
    Initialize database on application startup.

    This function:
    1. Creates all tables if they don't exist
    2. Checks if database is already populated
    3. If empty, imports data from CSV/JSON files in data directory

    Args:
        data_dir: Path to directory containing CSV/JSON files
    """
    logger.info("Initializing database...")

    try:
        # Create tables
        logger.info("Creating database tables...")
        init_db()
        logger.info("Database tables created successfully")

        # Check if database is already initialized
        if check_database_initialized():
            logger.info("Database already contains data, skipping CSV import")
            return

        # Import data from CSV/JSON files
        logger.info(f"Database is empty, importing data from {data_dir}...")

        if not data_dir.exists():
            logger.warning(f"Data directory not found: {data_dir}")
            logger.info("Database initialized with empty tables")
            return

        with SessionLocal() as db:
            results = initialize_database_from_csv(db, data_dir)

            total_imported = sum(results.values())
            logger.info(
                f"Database initialization complete. Imported {total_imported} total observations"
            )

            for filename, count in results.items():
                if count > 0:
                    logger.info(f"  - {filename}: {count} observations")

    except Exception as e:
        logger.error(f"Error during database initialization: {e}", exc_info=True)
        raise
