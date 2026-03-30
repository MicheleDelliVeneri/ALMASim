"""One-time migration: add proposal_abstract, qa2_passed, obs_type to observations table."""

import logging
import sys
from pathlib import Path

from sqlalchemy import text

# Ensure backend directory is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.config import engine  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

MIGRATIONS = [
    (
        "proposal_abstract",
        "ALTER TABLE observations ADD COLUMN IF NOT EXISTS proposal_abstract TEXT",
    ),
    (
        "qa2_passed",
        "ALTER TABLE observations ADD COLUMN IF NOT EXISTS qa2_passed VARCHAR(10)",
    ),
    (
        "obs_type",
        "ALTER TABLE observations ADD COLUMN IF NOT EXISTS obs_type VARCHAR(100)",
    ),
]


def run():
    with engine.connect() as conn:
        for col_name, sql in MIGRATIONS:
            try:
                conn.execute(text(sql))
                conn.commit()
                logger.info(f"Added column '{col_name}' to observations table.")
            except Exception as e:
                logger.error(f"Failed to add column '{col_name}': {e}")
                raise
    logger.info("Migration complete.")


if __name__ == "__main__":
    run()
