"""Utility functions for services."""

from __future__ import annotations

from typing import Callable, Optional

LogFn = Optional[Callable[[str], None]]


def log_message(logger: LogFn, message: str, remote: bool = False) -> None:
    """Log a message using the logger callback or print."""
    if remote:
        print(message)
    elif logger is not None:
        logger(message)


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
