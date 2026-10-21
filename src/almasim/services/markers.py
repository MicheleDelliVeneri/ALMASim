"""Completion and failure markers for pipeline stages (MADIV-28 T2.6).

Each unpack, calibrate and image output has a JSON ``.done`` marker written
when the stage finishes and a ``.failed`` marker that records why it did not.
The stages decide where the markers live and what they record; this module
writes them the same way for all three.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


@dataclass(frozen=True)
class StageMarkers:
    """The ``.done`` and ``.failed`` marker paths of one stage output."""

    done: Path
    failed: Path

    def is_done(self) -> bool:
        return self.done.is_file()

    def has_failed(self) -> bool:
        return self.failed.is_file()

    def write_done(self, **fields: Any) -> Path:
        """Record success with ``fields`` and ``completed_at``; clears a failure.

        An output that succeeds now must not keep looking failed to the next run.
        """
        self.failed.unlink(missing_ok=True)
        return _write_json(self.done, {**fields, "completed_at": _now()})

    def write_failed(
        self,
        error: str,
        *,
        stage: str,
        log_path: str | os.PathLike[str] | None = None,
        **fields: Any,
    ) -> Path:
        """Record why the stage failed. ``error`` is kept verbatim for triage."""
        payload = {**fields, "stage": stage, "error": error, "failed_at": _now()}
        if log_path is not None:
            payload["log"] = str(log_path)
        self.failed.parent.mkdir(parents=True, exist_ok=True)
        return _write_json(self.failed, payload)
