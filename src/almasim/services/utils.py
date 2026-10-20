"""Utility functions for services."""

from __future__ import annotations

import logging
from typing import Callable, Optional

LogFn = Optional[Callable[[str], None]]


def log_message(logger: LogFn, message: str, remote: bool = False) -> None:
    """Log a message using the logger callback or print."""
    if remote:
        print(message)
    elif logger is not None:
        logger(message)


def report(log: logging.Logger, logger_fn: LogFn, message: str, level: int = logging.INFO) -> None:
    """Log ``message`` and pass it to ``logger_fn`` (the CLI passes ``typer.echo``).

    Warnings reach ``logger_fn`` prefixed with ``WARNING:``.
    """
    log.log(level, message)
    if logger_fn is not None:
        logger_fn(f"WARNING: {message}" if level >= logging.WARNING else message)


class ProgressEmitterAdapter:
    """Adapter to convert callback functions to progress emitters."""

    def __init__(self, callback: Callable[[float], None]):
        self._callback = callback

    def emit(self, value):
        """Emit progress value."""
        self._callback(value)


def as_progress_emitter(callback):
    """Convert a callback to a progress emitter if needed."""
    if callback is None:
        return None
    if hasattr(callback, "emit"):
        return callback
    return ProgressEmitterAdapter(callback)
