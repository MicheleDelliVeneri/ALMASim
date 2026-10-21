"""Saved metadata-query presets.

Query presets live in query_results, one row per name. Rows saved before
MADIV-28 T1.10 may repeat a name: reads take the newest, writes fold the rest.
"""

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.clock import db_now

from ..models import QueryResult


class PresetRepository:
    """Query presets by name."""

    def __init__(self, db: Session):
        self.db = db

    def _query_results_named(self, query_name: str) -> list[QueryResult]:
        stmt = (
            select(QueryResult)
            .where(QueryResult.query_name == query_name)
            .order_by(QueryResult.created_at.desc(), QueryResult.id.desc())
        )
        return list(self.db.execute(stmt).scalars().all())

    def save_query_preset(
        self,
        query_name: str,
        filters: dict[str, Any],
        *,
        description: str | None = None,
        result_count: int = 0,
    ) -> QueryResult:
        """Create the preset, or replace the one already saved under this name."""
        rows = self._query_results_named(query_name)
        if rows:
            row, duplicates = rows[0], rows[1:]
            for duplicate in duplicates:
                self.db.delete(duplicate)
            row.created_at = db_now()
        else:
            row = QueryResult(query_name=query_name, observation_ids=[])
            self.db.add(row)
        row.query_params = {"filters": filters}
        row.result_count = result_count
        row.description = description
        self.db.commit()
        self.db.refresh(row)
        return row

    def get_query_result(self, query_name: str) -> QueryResult | None:
        """Get a saved query result by name (the newest if the name repeats)."""
        rows = self._query_results_named(query_name)
        return rows[0] if rows else None

    def list_query_results(self, limit: int = 50) -> list[QueryResult]:
        """List saved query results, newest first, one per name."""
        stmt = select(QueryResult).order_by(QueryResult.created_at.desc(), QueryResult.id.desc())
        newest: dict[str, QueryResult] = {}
        for row in self.db.execute(stmt).scalars():
            newest.setdefault(row.query_name, row)
            if len(newest) == limit:
                break
        return list(newest.values())

    def delete_query_result(self, query_name: str) -> bool:
        """Delete every saved query result with this name. Returns True if any existed."""
        rows = self._query_results_named(query_name)
        for row in rows:
            self.db.delete(row)
        if rows:
            self.db.commit()
        return bool(rows)
