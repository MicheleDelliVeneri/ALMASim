"""Shared helpers for persisting metadata query results."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

SUPPORTED_METADATA_FORMATS = {"json", "csv"}
# Common UI prefixes currently emitted by ALMASim frontend path forms; stripping these normalizes
# values like data/... and outputs/query_results/... while keeping writes rooted at base_dir.
# This also hardens path handling by reducing user-controlled prefixes before boundary checks.
# The list is intentionally conservative and can be extended if new UI prefixes are introduced.
STRIPPABLE_PATH_PREFIXES = {"data", "metadata", "query_results", "outputs"}


class MetadataStorageError(ValueError):
    """Base exception for metadata storage helper failures."""


class UnsupportedMetadataFormatError(MetadataStorageError):
    """Raised when metadata output format is not supported."""


class InvalidMetadataPathError(MetadataStorageError):
    """Raised when resolved metadata output path is outside the allowed base."""


def normalize_metadata_format(fmt: str | None) -> str:
    """Normalize and validate metadata serialization format."""
    normalized = (fmt or "json").strip().lower()
    if normalized not in SUPPORTED_METADATA_FORMATS:
        raise UnsupportedMetadataFormatError(
            f"Unsupported format: {normalized!r}. Use 'json' or 'csv'."
        )
    return normalized


def resolve_metadata_output_path(
    raw_path: str | None,
    *,
    base_dir: Path,
    fmt: str,
    default_name: str = "metadata-results",
    invalid_path_message: str = "Invalid metadata output path.",
) -> Path:
    """Resolve an output path inside ``base_dir`` with extension normalization."""
    normalized_fmt = normalize_metadata_format(fmt)
    suffix = ".csv" if normalized_fmt == "csv" else ".json"

    base = base_dir.expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)

    sanitized = (raw_path or "").strip()
    if not sanitized:
        return base / f"{default_name}{suffix}"

    parts = [part for part in sanitized.replace("\\", "/").split("/") if part]
    if parts and parts[0] in STRIPPABLE_PATH_PREFIXES:
        parts = parts[1:]
    relative = "/".join(parts) if parts else f"{default_name}{suffix}"

    resolved = (base / relative).resolve()
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise InvalidMetadataPathError(invalid_path_message) from exc

    if resolved.suffix.lower() != suffix:
        resolved = resolved.with_suffix(suffix)

    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def save_dataframe_csv(dataframe: Any, destination: Path) -> None:
    """Persist a dataframe-like object to CSV without an index column."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(destination, index=False)


def write_metadata_csv(destination: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write metadata rows to CSV using the union of all keys as columns."""
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                columns.append(key)

    with destination.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_metadata_json(destination: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write metadata rows to JSON with a count + data envelope."""
    with destination.open("w", encoding="utf-8") as fp:
        json.dump({"count": len(rows), "data": list(rows)}, fp, indent=2)


def save_metadata_records(
    rows: Sequence[Mapping[str, Any]],
    destination: Path,
    *,
    fmt: str,
) -> None:
    """Persist metadata rows to ``destination`` using JSON or CSV serialization."""
    normalized_fmt = normalize_metadata_format(fmt)
    if normalized_fmt == "csv":
        write_metadata_csv(destination, rows)
        return
    write_metadata_json(destination, rows)
