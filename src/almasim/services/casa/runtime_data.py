"""Where CASA's runtime (measures) data lives, and populating it (MADIV-28 T2.5)."""

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


def find_input_casa_data(input_root: str | os.PathLike[str]) -> Path | None:
    """The first populated ``.casa-data`` directory below ``input_root``, if any."""
    input_path = Path(input_root).expanduser().resolve()
    for candidate in input_path.rglob(".casa-data"):
        if candidate.is_dir() and has_casa_runtime_data(candidate):
            return candidate
    return None


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

    found = find_input_casa_data(input_root)
    return found if found is not None else output_casa_data


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

    # casaconfig expects a CASA logsink object, not a Python logger — pass None
    # so it falls back to its own stdout/stderr output.
    update_all(path=str(casa_data), logger=None)
