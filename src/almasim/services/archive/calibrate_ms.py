"""Create calibrated ALMA MeasurementSets from raw MS and delivered calibration products.

The PI restore script uses CASA pipeline globals such as ``hifa_restoredata``.
Standalone ``casatasks`` does not expose that pipeline task, so this service
uses the delivered calibration-application file with ``casatasks.applycal`` and
then writes the science-only calibrated MS with ``casatasks.mstransform``.
"""

from __future__ import annotations

import ast
import logging
import os
import shutil
import tarfile
from pathlib import Path
from typing import Any, Callable

from .unpack_ms import (
    LogFn,
    _emit,
    configure_casa_environment,
    ensure_casa_runtime_data,
    find_existing_casa_data,
)


logger = logging.getLogger(__name__)


def _safe_extract_tar(tar_path: Path, destination: Path) -> list[Path]:
    """Extract a tarball without allowing absolute or escaping paths."""
    extracted: list[Path] = []
    with tarfile.open(tar_path, "r:*") as archive:
        members = archive.getmembers()
        for member in members:
            member_path = Path(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                logger.warning("Skipping unsafe tar member: %s", member.name)
                continue
            resolved = (destination / member.name).resolve()
            if not str(resolved).startswith(str(destination.resolve())):
                logger.warning("Skipping escaping tar member: %s", member.name)
                continue
            archive.extract(member, destination, filter="data")
            if not member.isdir():
                extracted.append(resolved)
    return extracted


def find_calibration_directory(input_root: str | os.PathLike[str], asdm_uid: str | None = None) -> Path:
    """Find the calibration directory matching the requested ALMA delivery."""
    input_path = Path(input_root).expanduser().resolve()
    candidates = [path for path in input_path.rglob("calibration") if path.is_dir()]
    if asdm_uid is not None:
        candidates = [path for path in candidates if (path / f"{asdm_uid}.ms.calapply.txt").is_file()]
    if not candidates:
        raise RuntimeError(f"No calibration directory found under {input_path}")
    if len(candidates) > 1 and asdm_uid is None:
        raise RuntimeError(
            "Found more than one calibration directory. Pass asdm_uid to select a specific execution block."
        )
    return sorted(candidates)[0]


def find_raw_ms_directories(
    raw_ms_root: str | os.PathLike[str],
    asdm_uid: str | None = None,
) -> list[Path]:
    """Find raw imported MS directories below ``raw_ms_root``."""
    root = Path(raw_ms_root).expanduser().resolve()
    if not root.is_dir():
        raise RuntimeError(f"Raw MS root does not exist or is not a directory: {root}")

    raw_mss = []
    for candidate in root.rglob("*.ms"):
        if not candidate.is_dir():
            continue
        if asdm_uid is not None and candidate.name != f"{asdm_uid}.ms":
            continue
        raw_mss.append(candidate)

    if not raw_mss:
        if asdm_uid is None:
            raise RuntimeError(f"No raw *.ms directories found under {root}")
        raise RuntimeError(f"No raw MS named {asdm_uid}.ms found under {root}")

    return sorted(raw_mss)


def _parse_applycal_calls(calapply_path: Path) -> list[dict[str, Any]]:
    """Parse delivered ``applycal(...)`` lines into keyword dictionaries."""
    calls: list[dict[str, Any]] = []
    tree = ast.parse(calapply_path.read_text(encoding="utf-8"), filename=str(calapply_path))
    for node in tree.body:
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if not isinstance(call.func, ast.Name) or call.func.id != "applycal":
            continue
        kwargs = {keyword.arg: ast.literal_eval(keyword.value) for keyword in call.keywords if keyword.arg}
        calls.append(kwargs)

    if not calls:
        raise RuntimeError(f"No applycal calls found in {calapply_path}")
    return calls


def _normalize_intent(intent: str, ms_path: Path) -> str:
    """Translate pipeline shorthand intents to intents present in an imported ASDM MS."""
    if not intent:
        return intent

    from casatools import msmetadata

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
) -> Path:
    """Copy raw MS and unpack delivered calibration tables into ``working_dir``."""
    working_dir.mkdir(parents=True, exist_ok=True)
    working_ms = working_dir / raw_ms.name
    if working_ms.exists():
        if not overwrite:
            raise RuntimeError(f"Working MS already exists: {working_ms}")
        shutil.rmtree(working_ms)

    shutil.copytree(raw_ms, working_ms, symlinks=True)

    for tar_path in sorted(calibration_dir.glob("*caltables.tgz")):
        logger.info("Extracting calibration tables: %s", tar_path)
        _safe_extract_tar(tar_path, working_dir)

    return working_ms


def apply_delivered_calibration(
    applycal: Callable[..., object],
    raw_ms: Path,
    calibration_dir: Path,
    working_dir: Path,
    overwrite: bool = False,
    logger_fn: LogFn = None,
) -> Path:
    """Run delivered ``applycal`` commands against one raw MS."""
    working_ms = _prepare_working_directory(raw_ms, calibration_dir, working_dir, overwrite=overwrite)
    calapply_path = calibration_dir / f"{working_ms.name}.calapply.txt"
    if not calapply_path.is_file():
        raise RuntimeError(f"Cannot find calibration apply file: {calapply_path}")

    calls = _parse_applycal_calls(calapply_path)
    _emit(logger_fn, f"Applying {len(calls)} calibration command(s) to {working_ms.name}")
    current_dir = Path.cwd()
    try:
        os.chdir(working_dir)
        for kwargs in calls:
            kwargs["vis"] = working_ms.name
            kwargs["intent"] = _normalize_intent(str(kwargs.get("intent", "")), working_ms)
            applycal(**kwargs)
    finally:
        os.chdir(current_dir)

    return working_ms


def science_spws(ms_path: str | os.PathLike[str]) -> str:
    """Return target science SPWs with more than four channels."""
    from casatools import msmetadata

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


def split_calibrated_science_ms(
    mstransform: Callable[..., object],
    calibrated_working_ms: Path,
    output_root: str | os.PathLike[str],
    reindex: bool = False,
    overwrite: bool = False,
    logger_fn: LogFn = None,
) -> Path:
    """Split science SPWs into ``<uid>.ms.split.cal``."""
    output_path = Path(output_root).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    output_ms = output_path / f"{calibrated_working_ms.name}.split.cal"
    if output_ms.exists():
        if not overwrite:
            raise RuntimeError(f"Calibrated output MS already exists: {output_ms}")
        shutil.rmtree(output_ms)

    spws = science_spws(calibrated_working_ms)
    logger.info("Splitting science SPWs for %s: %s", calibrated_working_ms, spws)
    _emit(logger_fn, f"Splitting calibrated science SPWs for {calibrated_working_ms.name}: {spws}")
    mstransform(
        vis=str(calibrated_working_ms),
        outputvis=str(output_ms),
        spw=spws,
        reindex=reindex,
    )

    if not output_ms.is_dir():
        raise RuntimeError(f"Expected calibrated MS was not created: {output_ms}")
    _emit(logger_fn, f"Created calibrated MeasurementSet: {output_ms}")
    return output_ms


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _remove_tree_after_success(path: str | os.PathLike[str], protected: list[Path], logger_fn: LogFn = None) -> None:
    target = Path(path).expanduser().resolve()
    if not target.exists():
        return
    for protected_path in protected:
        protected_path = protected_path.resolve()
        if target == protected_path or _is_relative_to(protected_path, target):
            logger.warning("Skipping cleanup of %s because it contains protected output %s", target, protected_path)
            _emit(logger_fn, f"Skipping cleanup of {target}; it contains protected output {protected_path}")
            return
    shutil.rmtree(target)
    _emit(logger_fn, f"Removed intermediate data: {target}")


def create_calibrated_measurement_sets(
    input_root: str | os.PathLike[str],
    raw_ms_root: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    asdm_uid: str | None = None,
    casa_data_root: str | os.PathLike[str] | None = None,
    skip_casa_data_update: bool = False,
    overwrite: bool = False,
    clean_intermediate: bool = False,
    original_data_root: str | os.PathLike[str] | None = None,
    logger_fn: LogFn = None,
) -> list[Path]:
    """Create calibrated science MS products from raw MS and ALMA calibration products."""
    input_path = Path(input_root).expanduser().resolve()
    output_path = Path(output_root).expanduser().resolve()
    raw_mss = find_raw_ms_directories(raw_ms_root, asdm_uid)

    casa_data = find_existing_casa_data(input_path, output_path, casa_data_root)
    configure_casa_environment(output_path, casa_data)
    ensure_casa_runtime_data(casa_data, skip_update=skip_casa_data_update, logger_fn=logger_fn)

    from casatasks import applycal, mstransform

    calibrated_mss = []
    for raw_ms in raw_mss:
        uid = raw_ms.name.removesuffix(".ms")
        calibration_dir = find_calibration_directory(input_path, uid)
        working_ms = apply_delivered_calibration(
            applycal,
            raw_ms,
            calibration_dir,
            output_path / "working" / f"{uid}.calibration",
            overwrite=overwrite,
            logger_fn=logger_fn,
        )
        calibrated_mss.append(
            split_calibrated_science_ms(
                mstransform,
                working_ms,
                output_path,
                overwrite=overwrite,
                logger_fn=logger_fn,
            )
        )

    if clean_intermediate:
        protected = [*calibrated_mss, output_path]
        _remove_tree_after_success(raw_ms_root, protected, logger_fn=logger_fn)
        _remove_tree_after_success(output_path / "working", protected, logger_fn=logger_fn)
        if original_data_root is not None:
            _remove_tree_after_success(original_data_root, protected, logger_fn=logger_fn)

    return calibrated_mss


def restore_calibrated_measurement_sets(
    input_root: str | os.PathLike[str],
    raw_ms_root: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    **kwargs: Any,
) -> list[Path]:
    """Compatibility wrapper for the PI-script restore concept.

    The standalone ``casatasks`` package used by ALMASim does not expose
    ``hifa_restoredata``. This wrapper therefore creates the calibrated MS by
    applying the delivered calibration tables and splitting science SPWs.
    """
    return create_calibrated_measurement_sets(input_root, raw_ms_root, output_root, **kwargs)
