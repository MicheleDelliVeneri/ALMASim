"""Split the calibrated science windows into ``<uid>.ms.split.cal`` and mark it (MADIV-28 T4.8)."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Callable

from ...casa import tools as casa_tools
from ...markers import StageMarkers
from ...utils import LogFn, report

logger = logging.getLogger(__name__)


def science_spws(ms_path: str | os.PathLike[str]) -> str:
    """Return target science SPWs with more than four channels."""
    msmetadata = casa_tools().msmetadata

    msmd = msmetadata()
    msmd.open(str(ms_path))
    try:
        target_spws = msmd.spwsforintent("OBSERVE_TARGET*")
        selected = [str(spw) for spw in target_spws if msmd.nchan(int(spw)) > 4]
    finally:
        msmd.close()

    if not selected:
        raise RuntimeError(f"No science SPWs found in {ms_path}")
    return ",".join(selected)


def calibrated_output_path(output_root: str | os.PathLike[str], uid: str) -> Path:
    """Return the calibrated science MS path for ``uid``."""
    return Path(output_root).expanduser().resolve() / f"{uid}.ms.split.cal"


def calibration_marker_path(output_root: str | os.PathLike[str], uid: str) -> Path:
    """Return the completion marker written next to a finished ``<uid>.ms.split.cal``.

    The marker is what distinguishes a finished calibration from a partial
    output left behind by a crash. An empty file is accepted, so an output
    produced before markers existed can be trusted by simply creating it.
    """
    return calibrated_output_path(output_root, uid).with_name(f"{uid}.ms.split.cal.done")


def calibration_failure_marker_path(output_root: str | os.PathLike[str], uid: str) -> Path:
    """Return the failure marker written when ``uid`` could not be calibrated.

    Counterpart of :func:`calibration_marker_path`. Without it the filesystem
    cannot distinguish a UID that was never attempted from one that failed, so
    a rerun silently retries deterministic failures (a missing calibration
    delivery, say) that will fail again the same way.
    """
    return calibrated_output_path(output_root, uid).with_name(f"{uid}.ms.split.cal.failed")


def is_calibration_complete(output_root: str | os.PathLike[str], uid: str) -> bool:
    """Return True when ``uid`` has both a calibrated MS directory and its marker."""
    return (
        calibrated_output_path(output_root, uid).is_dir()
        and calibration_marker_path(output_root, uid).is_file()
    )


def has_calibration_failed(output_root: str | os.PathLike[str], uid: str) -> bool:
    """Return True when ``uid`` carries a failure marker from an earlier run."""
    return calibration_failure_marker_path(output_root, uid).is_file()


def _calibration_markers(output_root: str | os.PathLike[str], uid: str) -> StageMarkers:
    return StageMarkers(
        calibration_marker_path(output_root, uid),
        calibration_failure_marker_path(output_root, uid),
    )


def write_calibration_marker(output_root: str | os.PathLike[str], uid: str) -> Path:
    return _calibration_markers(output_root, uid).write_done(
        uid=uid, output=str(calibrated_output_path(output_root, uid))
    )


def write_calibration_failure_marker(
    output_root: str | os.PathLike[str],
    uid: str,
    error: str,
    *,
    stage: str = "calibrate",
    log_path: str | os.PathLike[str] | None = None,
) -> Path:
    """Record why ``uid`` failed, next to where its output would have gone.

    ``error`` is stored verbatim so a rerun can tell a deterministic failure
    (no delivered calibration tables) from a transient one (a truncated read)
    without re-reading the driver log, which is per-job and easily lost.
    """
    return _calibration_markers(output_root, uid).write_failed(
        error, stage=stage, log_path=log_path, uid=uid
    )


def split_calibrated_science_ms(
    mstransform: Callable[..., object],
    calibrated_working_ms: Path,
    output_root: str | os.PathLike[str],
    reindex: bool = False,
    overwrite: bool = False,
    logger_fn: LogFn = None,
) -> Path:
    """Split science SPWs into ``<uid>.ms.split.cal`` and write its completion marker."""
    output_path = Path(output_root).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    output_ms = output_path / f"{calibrated_working_ms.name}.split.cal"
    uid = calibrated_working_ms.name.removesuffix(".ms")
    marker = calibration_marker_path(output_path, uid)
    if output_ms.exists():
        if not overwrite:
            raise RuntimeError(f"Calibrated output MS already exists: {output_ms}")
        shutil.rmtree(output_ms)
    marker.unlink(missing_ok=True)

    spws = science_spws(calibrated_working_ms)
    report(
        logger,
        logger_fn,
        f"Splitting calibrated science SPWs for {calibrated_working_ms.name}: {spws}",
    )
    mstransform(
        vis=str(calibrated_working_ms),
        outputvis=str(output_ms),
        spw=spws,
        reindex=reindex,
    )

    if not output_ms.is_dir():
        raise RuntimeError(f"Expected calibrated MS was not created: {output_ms}")
    write_calibration_marker(output_path, uid)
    report(logger, logger_fn, f"Created calibrated MeasurementSet: {output_ms}")
    return output_ms
