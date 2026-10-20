"""Shared CLI helpers for ALMASim commands."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, NoReturn, Optional

import typer

from ..services.utils import dedupe_keep_order  # noqa: F401  (re-exported)


def default_output_path(filename: str) -> Path:
    """Return the default user-scoped output path for a filename."""
    return Path.home() / "almasim_outputs" / filename


def split_csv_values(values: Optional[List[str]]) -> Optional[List[str]]:
    """Split repeated/comma-separated option values while preserving order."""
    if not values:
        return None
    flattened = [item.strip() for value in values for item in value.split(",") if item.strip()]
    return flattened or None


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


def usage_error(message: str) -> NoReturn:
    """Print ``message`` to stderr and exit 2: the command line itself is wrong."""
    typer.echo(message, err=True)
    raise typer.Exit(code=2)


def fail(message: str, code: int = 1) -> NoReturn:
    """Print ``message`` to stderr and exit ``code`` (1): the command could not do its work."""
    typer.echo(message, err=True)
    raise typer.Exit(code=code)


class _CurrentStderrHandler(logging.StreamHandler):
    """A handler writing to whatever ``sys.stderr`` is when a record is emitted."""

    @property  # type: ignore[override]
    def stream(self):
        return sys.stderr

    @stream.setter
    def stream(self, value) -> None:
        pass


class _LevelPrefixFormatter(logging.Formatter):
    """INFO records as plain lines (as the commands echo them), others as ``LEVEL: message``."""

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        return message if record.levelno == logging.INFO else f"{record.levelname}: {message}"


def configure_logging(level: int = logging.INFO) -> None:
    """Show the library's log records (``almasim.*``) on stderr, once (MADIV-28 T2.10).

    Messages a command echoes itself are logged at DEBUG by
    ``services.utils.report``, so nothing is shown twice.
    """
    log = logging.getLogger("almasim")
    log.setLevel(level)
    if not any(isinstance(handler, _CurrentStderrHandler) for handler in log.handlers):
        handler = _CurrentStderrHandler()
        handler.setFormatter(_LevelPrefixFormatter())
        log.addHandler(handler)
