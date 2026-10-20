"""Shared progress, heartbeat and failure helpers (MADIV-28 T2.9)."""

from __future__ import annotations

import os
import time
from types import SimpleNamespace

import pytest

from almasim.services.compute.progress import (
    TaskFailure,
    error_headline,
    future_done,
    future_status,
    heartbeat_lines,
    report_failures,
    tail_line,
    watch_futures,
)


@pytest.mark.unit
def test_error_headline_takes_the_first_line_and_trims():
    assert error_headline("\n  first line \nsecond") == "first line"
    assert error_headline("x" * 300, limit=10) == "x" * 9 + "…"
    assert error_headline("") == ""


@pytest.mark.unit
def test_heartbeat_lines_cover_only_logs_written_since_the_start(tmp_path):
    fresh, stale = tmp_path / "a.log", tmp_path / "b.log"
    fresh.write_text("step 1\nstep 2\n")
    stale.write_text("old run\n")
    started = time.time()
    os.utime(stale, (started - 3600, started - 3600))
    lines, running = heartbeat_lines(
        [("A", fresh), ("B", stale), ("C", None), ("D", tmp_path / "missing.log")],
        started_at=started,
    )
    assert lines == ["  [A] step 2"]
    assert running == 1
    assert tail_line(fresh, max_bytes=3) == "2"


@pytest.mark.unit
def test_report_failures_prints_one_line_per_task_with_its_log():
    printed: list[str] = []
    report_failures(
        "Calibrate",
        [TaskFailure("uid://A", "Traceback\nboom", "/logs/a.log"), TaskFailure("uid://B", "x")],
        unit="UID(s)",
        echo=printed.append,
    )
    assert printed == [
        "Calibrate: 2 UID(s) failed and were skipped:",
        "  uid://A: Traceback",
        "    log: /logs/a.log",
        "  uid://B: x",
    ]
    report_failures("Image", [], echo=printed.append)
    assert len(printed) == 4


@pytest.mark.unit
def test_future_state_helpers_accept_every_backend_shape():
    assert future_status(SimpleNamespace(status="Finished")) == "finished"
    assert future_status(SimpleNamespace(status=lambda: "ERROR")) == "error"
    assert future_status(object()) == ""
    assert future_done(SimpleNamespace(done=lambda: True))
    assert future_done(SimpleNamespace(done=False)) is False
    assert future_done(SimpleNamespace(status="cancelled"))
    assert not future_done(SimpleNamespace(status="pending"))


@pytest.mark.unit
def test_watch_futures_reports_percentages_until_everything_is_done():
    state = {"polls": 0}

    def done():
        return state["polls"] >= 2

    slow = SimpleNamespace(done=done)
    percents: list[int] = []

    def on_percent(value):
        percents.append(value)
        state["polls"] += 1

    watch_futures([slow, 42], on_percent, poll_interval=0)
    assert percents == [50, 50, 100]
    watch_futures([], percents.append)
    assert percents[-1] == 100
