"""Compatibility re-exports: calibration moved to ``archive.calibration`` (MADIV-28 T4.8).

Removed in T8.4.
"""

from __future__ import annotations

from .calibration import (
    PIPELINE_FLAG_VERSION,
    apply_delivered_calibration,
    calibrated_output_path,
    calibration_failure_marker_path,
    calibration_marker_path,
    cleanup_intermediate_calibration_data,
    create_calibrated_measurement_sets,
    find_calibration_directory,
    find_raw_ms_directories,
    has_calibration_failed,
    is_calibration_complete,
    restore_pipeline_flags,
    science_spws,
    split_calibrated_science_ms,
    write_calibration_failure_marker,
    write_calibration_marker,
)

__all__ = [
    "PIPELINE_FLAG_VERSION",
    "apply_delivered_calibration",
    "calibrated_output_path",
    "calibration_failure_marker_path",
    "calibration_marker_path",
    "cleanup_intermediate_calibration_data",
    "create_calibrated_measurement_sets",
    "find_calibration_directory",
    "find_raw_ms_directories",
    "has_calibration_failed",
    "is_calibration_complete",
    "restore_pipeline_flags",
    "science_spws",
    "split_calibrated_science_ms",
    "write_calibration_failure_marker",
    "write_calibration_marker",
]
