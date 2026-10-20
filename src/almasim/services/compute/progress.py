"""Progress, heartbeat and failure helpers shared by the CLI stages and the simulator.

The archive stages (unpack, calibrate) and the imaging runs report a failed
task the same way (``TaskFailure``, ``report_failures``), print the same
heartbeat from each running task's log (``heartbeat_lines``) and inspect
backend futures the same way (``future_status``, ``future_done``). The
simulator's channel and sky-model computations report a percentage with
``watch_futures``.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

# Backend future states that mean the task did not succeed.
FAILED_FUTURE_STATES = {"error", "failed", "cancelled", "lost"}


@dataclass(frozen=True)
class TaskFailure:
    """One task (a UID's stage, an imaging task) that failed and was skipped."""

    label: str
    error: str
    log_path: Optional[str] = None


def error_headline(error: str, limit: int = 240) -> str:
    """First informative line of an error message, trimmed for console output."""
    for line in str(error).splitlines():
        stripped = line.strip()
        if stripped:
            return stripped if len(stripped) <= limit else stripped[: limit - 1] + "…"
    return str(error)[:limit]


def tail_line(
    path: Path, *, modified_since: Optional[float] = None, max_bytes: int = 4096
) -> Optional[str]:
    """The last non-empty line of ``path``, reading only its last ``max_bytes``.

    A file last modified before ``modified_since`` (a ``time.time()`` stamp) is
    ignored, so a log left by an earlier run is not taken for a running task.
    """
    try:
        if modified_since is not None and path.stat().st_mtime < modified_since:
            return None
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            chunk = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    lines = [line.strip() for line in chunk.splitlines() if line.strip()]
    return lines[-1] if lines else None


def heartbeat_lines(
    tasks: Iterable[tuple[str, Optional[Path]]], *, started_at: float
) -> tuple[list[str], int]:
    """``[label] last log line`` for each task whose log was written since ``started_at``.

    Returns the lines and how many tasks they cover (the ones running now).
    """
    lines = []
    for label, log_path in tasks:
        last = tail_line(log_path, modified_since=started_at - 1.0) if log_path else None
        if last is not None:
            lines.append(f"  [{label}] {last}")
    return lines, len(lines)


def report_failures(
    stage_label: str,
    failures: Sequence[TaskFailure],
    *,
    unit: str = "task(s)",
    echo: Callable[[str], None] = lambda text: print(text, file=sys.stderr),
) -> None:
    """Print one summary line per failed task, with its log when there is one."""
    if not failures:
        return
    echo(f"{stage_label}: {len(failures)} {unit} failed and were skipped:")
    for failure in failures:
        echo(f"  {failure.label}: {error_headline(failure.error)}")
        if failure.log_path:
            echo(f"    log: {failure.log_path}")


def future_status(future: Any) -> str:
    """A backend future's state in lower case ("" when it has none)."""
    status = getattr(future, "status", None)
    if callable(status):
        try:
            status = status()
        except TypeError:
            status = None
    return str(status).lower() if status is not None else ""


def future_done(future: Any) -> bool:
    """Whether a backend future has finished, successfully or not."""
    done_attr = getattr(future, "done", None)
    if callable(done_attr):
        try:
            return bool(done_attr())
        except TypeError:
            return False
    if isinstance(done_attr, bool):
        return done_attr
    return future_status(future) in {"finished", "done", "error", "failed", "cancelled"}


def watch_futures(
    futures: Sequence[Any],
    on_percent: Callable[[int], Any],
    *,
    poll_interval: float = 1.0,
) -> None:
    """Report the percentage of ``futures`` done every ``poll_interval`` s until all are.

    Plain values (an eager backend's results) count as done.
    """
    total = len(futures)
    if total == 0:
        return
    while True:
        done = sum(future_done(f) if hasattr(f, "done") else True for f in futures)
        on_percent(int(done / total * 100))
        if done >= total:
            return
        time.sleep(poll_interval)
