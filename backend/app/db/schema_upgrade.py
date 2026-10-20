"""Bring an older database up to the current tables (until T6.3 adopts Alembic)."""

from __future__ import annotations

# Columns added to a table after it was first created, as (name, SQL type and
# default). init_db adds the missing ones to an older database; Base's models
# already have them, so a new database gets them from create_all.
ADDED_COLUMNS: dict[str, list[tuple[str, str]]] = {
    "observations": [
        ("proposal_abstract", "TEXT"),
        ("qa2_passed", "VARCHAR(10)"),
        ("obs_type", "VARCHAR(100)"),
    ],
    "download_jobs": [
        ("member_ous_uids", "TEXT NOT NULL DEFAULT ''"),
        ("product_filter", "VARCHAR(20) NOT NULL DEFAULT 'all'"),
        ("metadata_json", "TEXT NOT NULL DEFAULT '[]'"),
        ("unpack_ms", "BOOLEAN NOT NULL DEFAULT FALSE"),
        ("generate_calibrated_visibilities", "BOOLEAN NOT NULL DEFAULT FALSE"),
        ("clean_intermediate_files", "BOOLEAN NOT NULL DEFAULT FALSE"),
        ("archive_output_root", "TEXT"),
        ("casa_data_root", "TEXT"),
        ("skip_casa_data_update", "BOOLEAN NOT NULL DEFAULT FALSE"),
        ("raw_measurement_sets", "TEXT NOT NULL DEFAULT '[]'"),
        ("calibrated_measurement_sets", "TEXT NOT NULL DEFAULT '[]'"),
        ("manifest_path", "TEXT"),
    ],
}


def add_missing_columns(bind, added: dict[str, list[tuple[str, str]]] = ADDED_COLUMNS) -> list[str]:
    """Add the ``added`` columns an existing table lacks; returns ``table.column`` names added."""
    from sqlalchemy import inspect, text

    insp = inspect(bind)
    done: list[str] = []
    for table, columns in added.items():
        if not insp.has_table(table):
            continue
        present = {c["name"] for c in insp.get_columns(table)}
        with bind.begin() as conn:
            for name, ddl in columns:
                if name not in present:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"))
                    done.append(f"{table}.{name}")
    return done
