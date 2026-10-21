"""Apply the delivered calibration to a working copy of a raw MS (MADIV-28 T4.8).

The PI restore script uses CASA pipeline globals such as ``hifa_restoredata``.
Standalone ``casatasks`` does not expose that pipeline task, so the delivered
calibration-application file is replayed with ``casatasks.applycal``.
"""

from __future__ import annotations

import logging
import shutil
import tarfile
from collections.abc import Callable
from pathlib import Path

from ...casa import tools as casa_tools
from ...extraction import member_is_safe
from ...utils import LogFn, report
from .discovery import _parse_applycal_calls

logger = logging.getLogger(__name__)


def _safe_extract_tar(tar_path: Path, destination: Path) -> list[Path]:
    """Extract a tarball without allowing absolute or escaping paths."""
    extracted: list[Path] = []
    with tarfile.open(tar_path, "r:*") as archive:
        members = archive.getmembers()
        for member in members:
            if not member_is_safe(member, destination):
                logger.warning("Skipping unsafe tar member: %s", member.name)
                continue
            archive.extract(member, destination, filter="data")
            if not member.isdir():
                extracted.append((destination / member.name).resolve())
    return extracted


def _normalize_intent(intent: str, ms_path: Path) -> str:
    """Translate pipeline shorthand intents to intents present in an imported ASDM MS."""
    if not intent:
        return intent

    msmetadata = casa_tools().msmetadata

    msmd = msmetadata()
    msmd.open(str(ms_path))
    try:
        available = set(msmd.intents())
    finally:
        msmd.close()

    if intent in available:
        return intent

    mapping = {
        "TARGET": "OBSERVE_TARGET#ON_SOURCE",
        "PHASE": "CALIBRATE_PHASE#ON_SOURCE",
        "BANDPASS": "CALIBRATE_BANDPASS#ON_SOURCE",
        "AMPLITUDE": "CALIBRATE_FLUX#ON_SOURCE",
        "FLUX": "CALIBRATE_FLUX#ON_SOURCE",
    }
    translated = []
    for token in [part.strip() for part in intent.split(",") if part.strip()]:
        translated.append(mapping.get(token, token))

    normalized = ",".join(translated)
    if all(token in available for token in translated):
        return normalized

    logger.warning(
        "Dropping applycal intent selection %r for %s; available intents are %s",
        intent,
        ms_path,
        sorted(available),
    )
    return ""


def _prepare_working_directory(
    raw_ms: Path,
    calibration_dir: Path,
    working_dir: Path,
    overwrite: bool = False,
    logger_fn: LogFn = None,
) -> Path:
    """Copy raw MS and unpack delivered calibration tables into ``working_dir``."""
    working_dir.mkdir(parents=True, exist_ok=True)
    working_ms = working_dir / raw_ms.name
    if working_ms.exists():
        if not overwrite:
            raise RuntimeError(f"Working MS already exists: {working_ms}")
        shutil.rmtree(working_ms)

    report(logger, logger_fn, f"Copying raw MS {raw_ms} -> {working_ms}")
    shutil.copytree(raw_ms, working_ms, symlinks=True)

    for tar_path in sorted(calibration_dir.glob("*caltables.tgz")):
        report(logger, logger_fn, f"Extracting calibration tables: {tar_path}")
        _safe_extract_tar(tar_path, working_dir)

    return working_ms


PIPELINE_FLAG_VERSION = "Pipeline_Final"


def restore_pipeline_flags(
    flagmanager: Callable[..., object],
    working_ms: Path,
    calibration_dir: Path,
    working_dir: Path,
    *,
    version: str = PIPELINE_FLAG_VERSION,
    required: bool = True,
    logger_fn: LogFn = None,
) -> bool:
    """Restore the pipeline's final flag state onto ``working_ms`` before ``applycal``.

    ALMA delivers ``<uid>.ms.flagversions.tgz`` next to the calapply file; it
    holds the ``Pipeline_Final`` flag version, everything the pipeline flagged
    (bad Tsys solutions, antennas, scans, edge channels, shadowing …).
    ``hifa_restoredata`` restores it before applying the tables, and so must
    we: applied to unflagged data, those calibration solutions produce
    visibilities with amplitudes ~10³ off and WEIGHT ~10⁷ (10⁶ × normal)
    that then dominate any image. Found on 40/40 sampled Cycle-11 EBs on
    2026-09-25 after the whole cycle had been calibrated without this step.

    The tarball is extracted next to the working MS (``flagmanager`` looks
    for ``<vis>.flagversions``), the version is checked against
    ``FLAG_VERSION_LIST`` and restored with ``merge='replace'``. A missing
    tarball or version is an error unless ``required`` is False, in which
    case it is logged and the data stay as imported. Returns whether flags
    were restored.
    """
    tar_path = calibration_dir / f"{working_ms.name}.flagversions.tgz"
    if not tar_path.is_file():
        message = (
            f"No delivered flag versions for {working_ms.name} ({tar_path} is missing): "
            f"the pipeline's flags would stay unset and the calibration tables would be "
            f"applied to data the pipeline had flagged"
        )
        if required:
            raise RuntimeError(message)
        report(logger, logger_fn, message, logging.WARNING)
        return False
    report(logger, logger_fn, f"Extracting flag versions: {tar_path}")
    _safe_extract_tar(tar_path, working_dir)
    flag_dir = working_dir / f"{working_ms.name}.flagversions"
    if not flag_dir.is_dir():
        raise RuntimeError(f"{tar_path} did not contain {flag_dir.name}")
    listing = flag_dir / "FLAG_VERSION_LIST"
    versions = (
        [line.split(":", 1)[0].strip() for line in listing.read_text().splitlines() if line.strip()]
        if listing.is_file()
        else []
    )
    if version not in versions:
        raise RuntimeError(
            f"Flag version {version!r} is not in {listing} (available: {versions or 'none'})"
        )
    report(logger, logger_fn, f"Restoring flag version {version!r} onto {working_ms.name}")
    flagmanager(vis=str(working_ms), mode="restore", versionname=version, merge="replace")
    return True


def apply_delivered_calibration(
    applycal: Callable[..., object],
    clearcal: Callable[..., object] | None,
    raw_ms: Path,
    calibration_dir: Path,
    working_dir: Path,
    overwrite: bool = False,
    logger_fn: LogFn = None,
    *,
    flagmanager: Callable[..., object] | None = None,
    require_pipeline_flags: bool = True,
) -> Path:
    """Run delivered ``applycal`` commands against one raw MS.

    The pipeline's ``Pipeline_Final`` flags are restored first (see
    :func:`restore_pipeline_flags`); ``flagmanager`` is the CASA task to do it
    with and is mandatory unless ``require_pipeline_flags`` is False.
    """
    if flagmanager is None and require_pipeline_flags:
        raise RuntimeError(
            "apply_delivered_calibration needs casatasks.flagmanager to restore the "
            "pipeline's flags before applycal (pass require_pipeline_flags=False to skip)"
        )
    working_ms = _prepare_working_directory(
        raw_ms, calibration_dir, working_dir, overwrite=overwrite, logger_fn=logger_fn
    )
    calapply_path = calibration_dir / f"{working_ms.name}.calapply.txt"
    if not calapply_path.is_file():
        raise RuntimeError(f"Cannot find calibration apply file: {calapply_path}")
    if clearcal is not None:
        clearcal(vis=str(working_ms), addmodel=False)
    if flagmanager is not None:
        restore_pipeline_flags(
            flagmanager,
            working_ms,
            calibration_dir,
            working_dir,
            required=require_pipeline_flags,
            logger_fn=logger_fn,
        )

    calls = _parse_applycal_calls(calapply_path)
    report(logger, logger_fn, f"Applying {len(calls)} calibration command(s) to {working_ms.name}")
    for index, kwargs in enumerate(calls, start=1):
        kwargs["vis"] = str(working_ms)
        if "gaintable" in kwargs:
            kwargs["gaintable"] = [
                str(working_dir / tbl) if not Path(tbl).is_absolute() else tbl
                for tbl in kwargs["gaintable"]
            ]
        if "callib" in kwargs and kwargs["callib"] and not Path(kwargs["callib"]).is_absolute():
            kwargs["callib"] = str(working_dir / kwargs["callib"])
        kwargs["intent"] = _normalize_intent(str(kwargs.get("intent", "")), working_ms)
        # Flag backup files on shared storage are not needed for this disposable working copy
        # and can fail under heavy concurrent Slurm I/O.
        kwargs["flagbackup"] = False
        report(
            logger,
            logger_fn,
            f"  applycal {index}/{len(calls)} for {working_ms.name}: "
            f"field={kwargs.get('field', '')!r} spw={kwargs.get('spw', '')!r} "
            f"intent={kwargs.get('intent', '')!r} gaintables={len(kwargs.get('gaintable', []))}",
        )
        applycal(**kwargs)

    return working_ms
