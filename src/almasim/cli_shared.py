"""Shared CLI helpers for ALMASim commands."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer


def default_output_path(filename: str) -> Path:
    """Return the default examples output path for a filename."""
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "examples" / "output" / filename


def split_csv_values(values: Optional[List[str]]) -> Optional[List[str]]:
    """Split repeated/comma-separated option values while preserving order."""
    if not values:
        return None
    flattened = [item.strip() for value in values for item in value.split(",") if item.strip()]
    return flattened or None


def dedupe_keep_order(values: list[str]) -> list[str]:
    """Remove duplicates while preserving first-seen order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def validate_range(
    *,
    min_value: Optional[float],
    max_value: Optional[float],
    label: str,
) -> Optional[tuple[float, float]]:
    """Validate min/max pairs and return a typed range tuple."""
    if min_value is None and max_value is None:
        return None
    if min_value is None or max_value is None:
        typer.echo(f"Both min and max are required for {label}.", err=True)
        raise typer.Exit(code=2)
    if min_value > max_value:
        typer.echo(
            f"Invalid {label}: min ({min_value}) is greater than max ({max_value}).",
            err=True,
        )
        raise typer.Exit(code=2)
    return (min_value, max_value)


def validate_date_range(
    *,
    start_date: Optional[str],
    end_date: Optional[str],
) -> Optional[tuple[str, str]]:
    """Validate date start/end pairing and return date tuple."""
    if start_date is None and end_date is None:
        return None
    if start_date is None or end_date is None:
        typer.echo(
            "Both --observation-date-start and --observation-date-end are required.",
            err=True,
        )
        raise typer.Exit(code=2)
    return (start_date, end_date)
