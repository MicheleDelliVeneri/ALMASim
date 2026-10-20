"""Compatibility re-exports: the ASDM import moved to ``archive.asdm``/``archive.unpack``,
the MeasurementSet checks to ``services.ms.verify`` and the CASA set-up to
``services.casa`` (MADIV-28 T4.8). Removed in T8.4.
"""

from __future__ import annotations

from ..casa.environment import _EXIT_CLEANUP, configure_casa_environment
from ..casa.runtime_data import (
    ensure_casa_runtime_data,
    find_existing_casa_data,
    has_casa_runtime_data,
)
from ..ms.verify import (
    REQUIRED_SUBTABLES,
    empty_required_subtables,
    measurement_set_row_count,
    verify_measurement_set,
)
from ..utils import LogFn
from .asdm import asdm_name, find_asdm_directories, verify_asdm_directory
from .unpack import (
    casa_log_path,
    create_measurement_set,
    create_measurement_sets,
    is_unpack_complete,
    raw_ms_failure_marker_path,
    raw_ms_marker_path,
    raw_ms_path,
    write_raw_ms_failure_marker,
    write_raw_ms_marker,
)

__all__ = [
    "LogFn",
    "REQUIRED_SUBTABLES",
    "_EXIT_CLEANUP",
    "asdm_name",
    "casa_log_path",
    "configure_casa_environment",
    "create_measurement_set",
    "create_measurement_sets",
    "empty_required_subtables",
    "ensure_casa_runtime_data",
    "find_asdm_directories",
    "find_existing_casa_data",
    "has_casa_runtime_data",
    "is_unpack_complete",
    "measurement_set_row_count",
    "raw_ms_failure_marker_path",
    "raw_ms_marker_path",
    "raw_ms_path",
    "verify_asdm_directory",
    "verify_measurement_set",
    "write_raw_ms_failure_marker",
    "write_raw_ms_marker",
]
