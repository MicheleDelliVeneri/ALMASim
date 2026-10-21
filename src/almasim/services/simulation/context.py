"""The callbacks one simulation run reports through (MADIV-28 T5.7)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ..utils import log_message
from .params import LogFn, StopFn


class SimulationCancelled(RuntimeError):
    """Raised by :meth:`RunContext.check_cancelled` when the caller asked the run to stop.

    A ``RuntimeError`` as before, so code that caught that keeps working; the
    caller tells it from a failure by its type (MADIV-28 W0.3).
    """


@dataclass
class RunContext:
    """Log, status, cancellation and random numbers for one run (MADIV-28 T5.1).

    ``rng`` draws the sky: the spectral lines, the source shape and the
    serendipitous sources (``run_rng(params.seed, "sky")``).
    """

    logger: LogFn = None
    status_callback: Callable[[str], None] | None = None
    stop_requested: StopFn = False
    remote: bool = False
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

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
            raise SimulationCancelled("Simulation cancelled")
