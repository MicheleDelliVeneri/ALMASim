"""The runner an imaging task's steps share (MADIV-28 T4.2)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

from almasim.services.imaging.markers import write_imaging_failure_marker
from almasim.services.imaging.wsclean.runner import _run_with_retries


@dataclass
class TaskRun:
    """Where one imaging task writes, and how its WSClean commands run and fail.

    :meth:`run` retries WSClean deaths by signal up to ``retries`` times and
    keeps the last lines of output in ``tail``; :meth:`fail` writes the
    task's failure marker and returns the error to raise.
    """

    out: Path
    log_path: Path
    ms_path: str
    spw: int
    env: dict[str, str] = field(default_factory=dict)
    timeout: float | None = None
    retries: int = 0
    tail: deque[str] = field(default_factory=lambda: deque(maxlen=60))

    def __post_init__(self) -> None:
        self.retries = max(int(self.retries), 0)

    def run(self, cmd: list[str], log: IO[str]) -> int:
        """Run one WSClean command, writing its output to ``log``; the attempts it took."""
        return _run_with_retries(
            cmd, self.out, log, self.env, self.timeout, self.tail, self.fail, self.retries
        )

    def fail(self, error: str, returncode: int | None = None) -> RuntimeError:
        write_imaging_failure_marker(
            self.out,
            ms_path=self.ms_path,
            spw=self.spw,
            error=error,
            log_path=self.log_path,
            returncode=returncode,
        )
        return RuntimeError(f"{error}\nFull log: {self.log_path}")
