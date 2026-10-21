"""Done/failed markers and live logs of imaging tasks.

Moved out of archive_imaging.py (MADIV-28 module split).
"""

from __future__ import annotations

import os
import shlex
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from almasim.services.imaging.products import (
    IMAGE_FILENAME,
)
from almasim.services.markers import StageMarkers


def imaging_marker_path(output_dir: str | os.PathLike[str]) -> Path:
    """``<root>/<ms>/SPW-3`` -> ``<root>/<ms>/SPW-3.done``."""
    out = Path(output_dir)
    return out.with_name(out.name + ".done")


def imaging_failure_marker_path(output_dir: str | os.PathLike[str]) -> Path:
    out = Path(output_dir)
    return out.with_name(out.name + ".failed")


def _imaging_markers(output_dir: str | os.PathLike[str]) -> StageMarkers:
    return StageMarkers(imaging_marker_path(output_dir), imaging_failure_marker_path(output_dir))


def imaging_log_path(output_dir: str | os.PathLike[str]) -> Path:
    out = Path(output_dir)
    return out.with_name(out.name + ".log")


def is_imaging_complete(
    output_dir: str | os.PathLike[str], required: Sequence[str] = (IMAGE_FILENAME,)
) -> bool:
    """True when the task has its marker and every image in ``required``."""
    out = Path(output_dir)
    return imaging_marker_path(out).is_file() and all((out / name).is_file() for name in required)


def write_imaging_marker(
    output_dir: str | os.PathLike[str],
    *,
    ms_path: str,
    spw: int,
    command: Sequence[str] | None = None,
    seconds: float | None = None,
    commands: Sequence[Sequence[str]] | None = None,
    images: Sequence[str] | None = None,
) -> Path:
    out = Path(output_dir)
    payload: dict[str, Any] = {
        "ms": ms_path,
        "spw": int(spw),
        "output": str(out),
        "image": str(out / IMAGE_FILENAME),
    }
    if images:
        payload["image"] = str(out / images[0])
        payload["images"] = [str(out / name) for name in images]
    if command is not None:
        payload["command"] = shlex.join(str(part) for part in command)
    if commands is not None:
        payload["commands"] = [shlex.join(str(part) for part in c) for c in commands]
    if seconds is not None:
        payload["seconds"] = round(float(seconds), 1)
    return _imaging_markers(out).write_done(**payload)


def write_imaging_failure_marker(
    output_dir: str | os.PathLike[str],
    *,
    ms_path: str,
    spw: int,
    error: str,
    log_path: str | os.PathLike[str] | None = None,
    returncode: int | None = None,
) -> Path:
    """Record why a task failed. ``error`` is kept verbatim for triage."""
    out = Path(output_dir)
    markers = _imaging_markers(out)
    # Unlike unpack and calibrate, a failed image task also drops its old .done.
    markers.done.unlink(missing_ok=True)
    fields: dict[str, Any] = {"ms": ms_path, "spw": int(spw), "output": str(out)}
    if returncode is not None:
        fields["returncode"] = int(returncode)
    return markers.write_failed(error, stage="image", log_path=log_path, **fields)
