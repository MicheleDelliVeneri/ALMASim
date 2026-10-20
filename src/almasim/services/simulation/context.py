"""The callbacks one simulation run reports through (MADIV-28 T5.7)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from ..utils import log_message
from .params import LogFn, StopFn


@dataclass
class RunContext:
    """Log, status and cancellation for one run, built from the caller's callbacks."""

    logger: LogFn = None
    status_callback: Optional[Callable[[str], None]] = None
    stop_requested: StopFn = False
    remote: bool = False

    def log(self, message: str) -> None:
        log_message(self.logger, message, remote=self.remote)

    def status(self, message: str) -> None:
        if self.status_callback is not None:
            self.status_callback(message)

    def stopped(self) -> bool:
        if callable(self.stop_requested):
            return bool(self.stop_requested())
        return bool(self.stop_requested)

    def check_cancelled(self) -> None:
        """Raise when the caller asked the run to stop."""
        if self.stopped():
            raise RuntimeError("Simulation cancelled")
