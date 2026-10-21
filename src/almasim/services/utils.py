"""Utility functions for services."""

from __future__ import annotations

import logging
from collections.abc import Callable

LogFn = Callable[[str], None] | None

_simulation_log = logging.getLogger("almasim.simulation")


def log_message(logger: LogFn, message: str, remote: bool = False) -> None:
    """Pass ``message`` to the caller's ``logger`` callback, or log it.

    ``remote`` runs (on a worker, where no callback reaches the caller) always
    log. They used to print, the one library print besides streamed child
    output (MADIV-28 T2.10).
    """
    if logger is not None and not remote:
        logger(message)
    else:
        _simulation_log.info(message)


def report(log: logging.Logger, logger_fn: LogFn, message: str, level: int = logging.INFO) -> None:
    """One message to the caller's sink (the CLI passes ``typer.echo``), or to ``log``.

    Never both: with a sink the record is only logged at DEBUG, so a CLI that
    also shows log records prints each line once. Warnings reach the sink
    prefixed with ``WARNING:``.
    """
    if logger_fn is None:
        log.log(level, message)
        return
    log.debug(message)
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
