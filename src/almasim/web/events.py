"""Live job events: one ticker, many listeners (web console plan, S2.3).

A background thread wakes every ``interval`` seconds (or at once when an
action pokes it), moves this user's pipelines on, starts queued server
jobs, and compares every listed run's state with the last look. Changes
become numbered events; ``GET /api/v2/events`` streams them as server-sent
events. A browser that reconnects sends the last number it saw
(``Last-Event-ID``) and gets what it missed, or a fresh snapshot when that
has scrolled out of the buffer.
"""

from __future__ import annotations

import asyncio
import getpass
import json
import logging
import threading
from collections import deque
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from almasim.jobs.profile import Profile
from almasim.jobs.service import advance_pipelines, list_runs, start_queued_server_jobs

from .api.runs import run_json

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2", tags=["events"])

BUFFER = 500


class EventHub:
    def __init__(self, profile: Profile, interval: float = 15.0, limit: int = 200) -> None:
        self.profile = profile
        self.interval = interval
        self.limit = limit
        self.seq = 0
        self.events: deque[tuple[int, dict[str, Any]]] = deque(maxlen=BUFFER)
        self.states: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="almasim-events", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=10)

    def poke(self) -> None:
        """Look again now (after a submit or cancel), not at the next tick."""
        self._wake.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception:
                logger.exception("Updating the job states failed")
            self._wake.wait(self.interval)
            self._wake.clear()

    def tick(self) -> list[dict[str, Any]]:
        """One look: move work on, then record what changed; the new events."""
        advance_pipelines(self.profile)
        start_queued_server_jobs(self.profile)
        current = {
            r.job.id: run_json(r) for r in list_runs(self.profile, owner=None, limit=self.limit)
        }
        new: list[dict[str, Any]] = []
        with self._lock:
            for run_id, run in current.items():
                before = self.states.get(run_id)
                if before is None or (before["state"], before["reason"]) != (
                    run["state"],
                    run["reason"],
                ):
                    self.seq += 1
                    event = {"type": "run", "run": run}
                    self.events.append((self.seq, event))
                    new.append(event)
            self.states = current
        return new

    def snapshot(self) -> tuple[int, list[dict[str, Any]]]:
        with self._lock:
            return self.seq, list(self.states.values())

    def since(self, seq: int) -> list[tuple[int, dict[str, Any]]] | None:
        """Events after ``seq``, or None when some have left the buffer."""
        with self._lock:
            if self.events and seq < self.events[0][0] - 1:
                return None
            return [(n, e) for n, e in self.events if n > seq]


def _sse(seq: int | None, event: str, data: Any) -> str:
    head = f"id: {seq}\n" if seq is not None else ""
    return f"{head}event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


async def stream(hub: EventHub, last_id: int | None, poll: float = 1.0) -> AsyncIterator[str]:
    """The event stream for one listener."""
    missed = hub.since(last_id) if last_id is not None else None
    if missed is None:
        seq, runs = hub.snapshot()
        yield _sse(seq, "snapshot", {"runs": runs, "user": getpass.getuser()})
    else:
        seq = last_id or 0
        for n, event in missed:
            yield _sse(n, event["type"], event)
            seq = n
    while True:
        await asyncio.sleep(poll)
        batch = hub.since(seq)
        if batch is None:
            seq, runs = hub.snapshot()
            yield _sse(seq, "snapshot", {"runs": runs, "user": getpass.getuser()})
            continue
        if not batch:
            yield ": keep-alive\n\n"
            continue
        for n, event in batch:
            yield _sse(n, event["type"], event)
            seq = n


@router.get("/events")
async def events(request: Request) -> StreamingResponse:
    """Server-sent events: a snapshot of the runs, then each change."""
    raw = request.headers.get("last-event-id")
    last_id = int(raw) if raw and raw.isdigit() else None
    return StreamingResponse(
        stream(request.app.state.events, last_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
