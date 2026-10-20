"""Check that a MeasurementSet finished writing (MADIV-28 T4.8)."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from ..casa import tools as casa_tools

logger = logging.getLogger(__name__)


def measurement_set_row_count(ms_path: str | os.PathLike[str]) -> int:
    """Return the number of rows in the MAIN table of ``ms_path``.

    A killed ``importasdm`` leaves a directory that looks complete: the schema
    and tens of GB of storage-manager files are on disk, but the row count was
    never committed, so the table opens cleanly and reports zero rows. Checking
    the directory exists is therefore not enough to call an import finished.
    """
    table = casa_tools().table

    tb = table()
    try:
        if not tb.open(str(ms_path)):
            raise RuntimeError(f"Could not open MeasurementSet: {ms_path}")
        return int(tb.nrows())
    finally:
        try:
            tb.close()
        except Exception:  # pragma: no cover - close failures are not actionable
            pass


REQUIRED_SUBTABLES = ("ANTENNA", "SPECTRAL_WINDOW", "FIELD", "DATA_DESCRIPTION", "POLARIZATION")


def empty_required_subtables(ms_path: str | os.PathLike[str]) -> list[str]:
    """Return the names of ``REQUIRED_SUBTABLES`` that are missing or have no rows.

    importasdm writes the MAIN table before it commits the subtables, so an
    import that dies late leaves every visibility row in place and the
    subtables' storage files on disk, but their row counts at zero. The MAIN
    row count then matches the reference and only the subtables show the
    import is unfinished; calibration aborts on such an MS with ``minMax -
    Array has no elements`` (uid___A002_X10f3768_Xb604, Cycle-11 recalibration,
    2026-10-07).
    """
    table = casa_tools().table

    empty = []
    for name in REQUIRED_SUBTABLES:
        tb = table()
        try:
            rows = int(tb.nrows()) if tb.open(str(Path(ms_path) / name)) else 0
        except Exception:
            rows = 0
        finally:
            try:
                tb.close()
            except Exception:  # pragma: no cover - close failures are not actionable
                pass
        if rows <= 0:
            empty.append(name)
    return empty


def verify_measurement_set(ms_path: str | os.PathLike[str]) -> int:
    """Raise unless ``ms_path`` has MAIN rows and rows in every required subtable."""
    path = Path(ms_path)
    if not path.is_dir():
        raise RuntimeError(f"Expected MeasurementSet was not created: {path}")
    rows = measurement_set_row_count(path)
    if rows <= 0:
        raise RuntimeError(
            f"MeasurementSet has no rows, the import did not finish: {path}. "
            "Re-import it from its ASDM; the directory on disk is not usable."
        )
    empty = empty_required_subtables(path)
    if empty:
        raise RuntimeError(
            f"MeasurementSet subtables have no rows ({', '.join(empty)}), the import did "
            f"not finish: {path}. Re-import it from its ASDM; the directory on disk is "
            "not usable."
        )
    return rows
