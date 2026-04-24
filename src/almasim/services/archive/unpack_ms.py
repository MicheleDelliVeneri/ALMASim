"""Import ALMA ASDM directories into raw MeasurementSets.

This module intentionally performs only the raw ASDM-to-MS import via
``casatasks.importasdm``. It does not calibrate, split, image, or restore
pipeline products.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable


logger = logging.getLogger(__name__)


LogFn = Callable[[str], None] | None


def _emit(logger_fn: LogFn, message: str) -> None:
    if logger_fn is not None:
        logger_fn(message)


def has_casa_runtime_data(path: str | os.PathLike[str]) -> bool:
    """Return true when ``path`` looks like populated CASA runtime data."""
    data_path = Path(path)
    return (data_path / "readme.txt").is_file() and (data_path / "geodetic").is_dir()


def find_existing_casa_data(
    input_root: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    casa_data_root: str | os.PathLike[str] | None = None,
) -> Path:
    """Choose a CASA runtime data directory for standalone ``casatasks`` use."""
    if casa_data_root is not None:
        return Path(casa_data_root).expanduser().resolve()

    output_casa_data = Path(output_root).expanduser().resolve() / ".casa-data"
    if has_casa_runtime_data(output_casa_data):
        return output_casa_data

    input_path = Path(input_root).expanduser().resolve()
    for candidate in input_path.rglob(".casa-data"):
        if candidate.is_dir() and has_casa_runtime_data(candidate):
            return candidate

    return output_casa_data


def configure_casa_environment(
    output_root: str | os.PathLike[str],
    casa_data: str | os.PathLike[str],
) -> Path:
    """Create and point CASA at a local site config and Matplotlib cache."""
    output_path = Path(output_root).expanduser().resolve()
    casa_data_path = Path(casa_data).expanduser().resolve()
    mpl_config = output_path / ".matplotlib"
    casa_config_dir = output_path / ".casa-config"
    casa_site_config = casa_config_dir / "casasiteconfig.py"

    casa_data_path.mkdir(parents=True, exist_ok=True)
    mpl_config.mkdir(parents=True, exist_ok=True)
    casa_config_dir.mkdir(parents=True, exist_ok=True)

    casa_site_config.write_text(
        "measurespath = {0!r}\n"
        "data_auto_update = False\n"
        "measures_auto_update = False\n".format(str(casa_data_path)),
        encoding="utf-8",
    )

    os.environ.setdefault("CASASITECONFIG", str(casa_site_config))
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    return casa_data_path


def ensure_casa_runtime_data(
    casa_data: Path,
    skip_update: bool = False,
    logger_fn: LogFn = None,
) -> None:
    """Populate CASA runtime data when needed."""
    if has_casa_runtime_data(casa_data):
        logger.info("Using CASA runtime data: %s", casa_data)
        _emit(logger_fn, f"Using CASA runtime data: {casa_data}")
        return

    if skip_update:
        raise RuntimeError(
            "CASA runtime data is missing at {0}. Re-run without "
            "--skip-casa-data-update or pass --casa-data-root pointing to a "
            "populated data directory.".format(casa_data)
        )

    logger.info("Populating CASA runtime data: %s", casa_data)
    _emit(logger_fn, f"Populating CASA runtime data: {casa_data}")
    from casaconfig import update_all

    update_all(path=str(casa_data), logger=logger)


def find_asdm_directories(
    input_root: str | os.PathLike[str],
    asdm_uid: str | None = None,
) -> list[Path]:
    """Find ASDM directories below ``input_root``."""
    input_path = Path(input_root).expanduser().resolve()
    if not input_path.is_dir():
        raise RuntimeError(f"Input root does not exist or is not a directory: {input_path}")

    asdm_dirs = []
    for candidate in input_path.rglob("*.asdm.sdm"):
        if not candidate.is_dir():
            continue
        if asdm_uid is not None and candidate.name != f"{asdm_uid}.asdm.sdm":
            continue
        asdm_dirs.append(candidate)

    if not asdm_dirs:
        if asdm_uid is None:
            raise RuntimeError(f"No *.asdm.sdm directories found under {input_path}")
        raise RuntimeError(f"No ASDM named {asdm_uid}.asdm.sdm found under {input_path}")

    return sorted(asdm_dirs)


def asdm_name(asdm_path: str | os.PathLike[str]) -> str:
    """Return the ASDM UID without the ``.asdm.sdm`` suffix."""
    return Path(asdm_path).name.replace(".asdm.sdm", "")


def create_measurement_set(
    importasdm: Callable[..., object],
    raw_asdm: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    overwrite: bool = False,
    logger_fn: LogFn = None,
) -> Path:
    """Create one raw MeasurementSet from one ASDM directory."""
    raw_asdm_path = Path(raw_asdm).expanduser().resolve()
    asdm_uid = asdm_name(raw_asdm_path)
    working_dir = Path(output_root).expanduser().resolve() / "working"

    if not raw_asdm_path.is_dir():
        raise RuntimeError(f"Cannot find raw ASDM directory: {raw_asdm_path}")

    working_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Working directory: %s", working_dir)

    output_ms = working_dir / f"{asdm_uid}.ms"
    if output_ms.exists() and not overwrite:
        raise RuntimeError(f"MeasurementSet already exists: {output_ms}")

    logger.info("Creating MeasurementSet from %s", raw_asdm_path)
    _emit(logger_fn, f"Creating raw MeasurementSet from {raw_asdm_path}")
    current_dir = Path.cwd()
    try:
        os.chdir(working_dir)
        importasdm(
            asdm=str(raw_asdm_path),
            vis=str(output_ms),
            overwrite=overwrite,
        )
    finally:
        os.chdir(current_dir)

    if not output_ms.is_dir():
        raise RuntimeError(f"Expected MeasurementSet was not created: {output_ms}")

    logger.info("Created MeasurementSet: %s", output_ms)
    _emit(logger_fn, f"Created raw MeasurementSet: {output_ms}")
    return output_ms


def create_measurement_sets(
    input_root: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    asdm_uid: str | None = None,
    casa_data_root: str | os.PathLike[str] | None = None,
    skip_casa_data_update: bool = False,
    overwrite: bool = False,
    logger_fn: LogFn = None,
) -> list[Path]:
    """Create raw MeasurementSets for all matching ASDMs below ``input_root``."""
    asdm_dirs = find_asdm_directories(input_root, asdm_uid)
    casa_data = find_existing_casa_data(input_root, output_root, casa_data_root)
    configure_casa_environment(output_root, casa_data)
    ensure_casa_runtime_data(casa_data, skip_update=skip_casa_data_update, logger_fn=logger_fn)

    from casatasks import importasdm

    logger.info("Found %d ASDM director%s", len(asdm_dirs), "y" if len(asdm_dirs) == 1 else "ies")
    _emit(logger_fn, f"Found {len(asdm_dirs)} ASDM director{'y' if len(asdm_dirs) == 1 else 'ies'}")
    return [
        create_measurement_set(
            importasdm,
            raw_asdm,
            output_root,
            overwrite=overwrite,
            logger_fn=logger_fn,
        )
        for raw_asdm in asdm_dirs
    ]
