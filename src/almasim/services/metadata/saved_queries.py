"""Persist and recall named query filter presets.

Works standalone via JSON files (no backend required) and can optionally
mirror saves to / loads from a :class:`DatabaseService` instance when
running inside the backend.

JSON file format::

    {
        "name": "galaxies_band6",
        "description": "Public Band 6 galaxy observations",
        "created_at": "2025-04-25T12:34:56",
        "filters": {
            "science_keyword": "Galaxies",
            "band": "Band 6",
            "row_limit": 10,
            "member_limit": 1,
            "output_dir": "examples/output/archive_pipeline",
        },
        "result_count": 42,
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

_FILE_SUFFIX = ".query.json"


@dataclass
class QueryPreset:
    """A named, serialisable query filter preset."""

    name: str
    filters: dict[str, Any]
    description: str = ""
    result_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "filters": self.filters,
            "result_count": self.result_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QueryPreset":
        return cls(
            name=data["name"],
            filters=data.get("filters", {}),
            description=data.get("description", ""),
            result_count=data.get("result_count", 0),
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
        )


# ------------------------------------------------------------------
# JSON file persistence
# ------------------------------------------------------------------


def _preset_path(directory: Path, name: str) -> Path:
    safe = name.replace("/", "_").replace(" ", "_")
    return directory / f"{safe}{_FILE_SUFFIX}"


def save_preset(preset: QueryPreset, directory: Path | str) -> Path:
    """Write *preset* to ``<directory>/<name>.query.json``.

    Creates *directory* if it does not exist.  Overwrites any existing file
    with the same name.
    """
    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    path = _preset_path(directory, preset.name)
    path.write_text(json.dumps(preset.to_dict(), indent=2), encoding="utf-8")
    return path


def load_preset(name: str, directory: Path | str) -> QueryPreset:
    """Load a preset by *name* from *directory*.

    Raises :exc:`FileNotFoundError` if the file does not exist.
    """
    directory = Path(directory).expanduser().resolve()
    path = _preset_path(directory, name)
    data = json.loads(path.read_text(encoding="utf-8"))
    return QueryPreset.from_dict(data)


def load_preset_from_path(path: Path | str) -> QueryPreset:
    """Load a preset directly from a file path."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return QueryPreset.from_dict(data)


def list_presets(directory: Path | str) -> list[QueryPreset]:
    """Return all presets found in *directory*, sorted by creation time."""
    directory = Path(directory).expanduser().resolve()
    if not directory.exists():
        return []
    presets = []
    for p in sorted(directory.glob(f"*{_FILE_SUFFIX}")):
        try:
            presets.append(load_preset_from_path(p))
        except Exception:
            pass
    presets.sort(key=lambda q: q.created_at, reverse=True)
    return presets


def delete_preset(name: str, directory: Path | str) -> bool:
    """Delete a preset file.  Returns *True* if deleted, *False* if not found."""
    path = _preset_path(Path(directory).expanduser().resolve(), name)
    if path.exists():
        path.unlink()
        return True
    return False


# ------------------------------------------------------------------
# Optional database bridge (requires backend DatabaseService)
# ------------------------------------------------------------------


def save_preset_to_db(preset: QueryPreset, db_service: Any) -> None:  # type: ignore[type-arg]
    """Persist *preset* to the database via *db_service* (a DatabaseService).

    Does nothing if *db_service* is None so callers don't need to guard.
    """
    if db_service is None:
        return
    db_service.save_query_result(
        query_name=preset.name,
        observation_ids=[],
        query_params=preset.to_dict(),
        description=preset.description,
    )


def load_preset_from_db(name: str, db_service: Any) -> QueryPreset | None:  # type: ignore[type-arg]
    """Load a preset from the database.  Returns *None* if not found."""
    if db_service is None:
        return None
    row = db_service.get_query_result(name)
    if row is None:
        return None
    params = row.query_params or {}
    return (
        QueryPreset.from_dict(params)
        if "name" in params
        else QueryPreset(
            name=row.query_name,
            filters=params,
            description=row.description or "",
            result_count=row.result_count,
            created_at=row.created_at.isoformat()
            if row.created_at
            else datetime.utcnow().isoformat(),
        )
    )


def list_presets_from_db(db_service: Any) -> list[QueryPreset]:  # type: ignore[type-arg]
    """Return all query presets stored in the database."""
    if db_service is None:
        return []
    rows = db_service.list_query_results()
    presets = []
    for row in rows:
        params = row.query_params or {}
        presets.append(
            QueryPreset.from_dict(params)
            if "name" in params
            else QueryPreset(
                name=row.query_name,
                filters=params,
                description=row.description or "",
                result_count=row.result_count,
                created_at=(
                    row.created_at.isoformat() if row.created_at else datetime.utcnow().isoformat()
                ),
            )
        )
    return presets
