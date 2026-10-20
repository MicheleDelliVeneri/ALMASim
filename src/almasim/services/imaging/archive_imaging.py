"""Per-task bookkeeping for imaging archive MeasurementSets with WSClean.

One imaging task is one (MeasurementSet, spectral window) pair and produces
``<output_root>/<ms stem>/SPW-<n>/wsclean-*.fits``. Exactly as for the unpack
and calibrate stages, a task leaves a marker next to its output directory:

* ``SPW-<n>.done``   -- the task finished and ``wsclean-image.fits`` exists;
* ``SPW-<n>.failed`` -- the task failed, with the error and the log path;
* ``SPW-<n>.log``    -- everything WSClean printed, written live on the shared
  filesystem so a run on Slurm can be followed with ``tail -f``.

Directory existence alone means nothing: a killed WSClean leaves a partial
output tree that looks finished. Reruns skip tasks with a ``.done`` marker and
retry everything else.
"""

from __future__ import annotations

from almasim.services.imaging.continuum import (  # noqa: F401  (moved; re-exported)
    _channel_ranges,
    _fit_continuum,
    _line_free_from_ranges,
    fit_uv_continuum,
    line_free_channels,
    subtract_continuum,
)
from almasim.services.imaging.cube import (  # noqa: F401  (moved; re-exported)
    _CHANNEL_FILE,
    _line_mask_pass,
    _zero_non_finite_models,
    line_mask_from_channels,
    stack_channel_images,
)
from almasim.services.imaging.markers import (  # noqa: F401  (moved; re-exported)
    _imaging_markers,
    imaging_failure_marker_path,
    imaging_log_path,
    imaging_marker_path,
    is_imaging_complete,
    write_imaging_failure_marker,
    write_imaging_marker,
)
from almasim.services.imaging.ms.chunks import (  # noqa: F401  (moved; re-exported)
    _CHUNK_SAMPLES,
    _default_rows_per_chunk,
    _has_spectrum_weight,
    _parallel_hands,
    _read_rows,
    _source_spectrum,
    rows_for_chunk,
)
from almasim.services.imaging.ms.single_window import (  # noqa: F401  (moved; re-exported)
    _LOCK_FILE_ATTEMPTS,
    _LOCK_FILE_ERROR,
    _LOCK_FILE_RETRY_SECONDS,
    DEFAULT_MAX_WEIGHT,
    _extract_single_window_ms_once,
    extract_single_window_ms,
    extract_single_window_ms_subprocess,
)
from almasim.services.imaging.products import (  # noqa: F401  (moved; re-exported)
    CONTINUUM_COLUMN,
    CONTSUB_FILENAME,
    CUBE_IMAGE_FILENAME,
    CUBE_NAME,
    CUBE_PB_FILENAME,
    CUBE_PBCOR_FILENAME,
    CUBE_SELFCAL_CORRECTION_FILENAME,
    CUBE_TOTAL_RESIDUAL_FILENAME,
    IMAGE_FILENAME,
    LINE_COLUMN,
    LINE_MASK_FILENAME,
    PB_FILENAME,
    PBCOR_FILENAME,
    SELFCAL_COLUMN,
    SELFCAL_CONTINUUM_COLUMN,
    SELFCAL_CORRECTION_FILENAME,
    SELFCAL_GAINS_DIRNAME,
    SELFCAL_LINE_COLUMN,
    SINGLE_WINDOW_PLACEHOLDER,
    TOTAL_RESIDUAL_FILENAME,
)
from almasim.services.imaging.selfcal.casa_ops import (  # noqa: F401  (moved; re-exported)
    _configure_casa,
    adopt_corrected_data,
    casa_selfcal_apply,
    casa_selfcal_solve,
    export_gain_table,
    selfcal_prepare,
)
from almasim.services.imaging.selfcal.loop import (  # noqa: F401  (moved; re-exported)
    _call_in_child,
    _keep_gain_tables,
    _selfcal_noise_products,
    image_peak,
    image_snr,
    parse_selfcal_rounds,
    self_calibrate,
)
from almasim.services.imaging.selfcal.noise_products import (  # noqa: F401  (moved; re-exported)
    _single_image,
    add_images,
    selfcal_correction_columns,
)
from almasim.services.imaging.task import (  # noqa: F401  (moved; re-exported)
    _run_and_mark,
    _subtract_uv_continuum_step,
    pb_correct_task,
    pb_note,
    run_wsclean_task,
)
from almasim.services.imaging.uv_contsub import (  # noqa: F401  (moved; re-exported)
    drop_corrected_data,
    subtract_uv_continuum,
    subtract_uv_continuum_subprocess,
)
from almasim.services.imaging.wsclean.command import (  # noqa: F401  (moved; re-exported)
    _dirty_command,
    _selfcal_command,
    _with_data_column,
    continuum_restart_commands,
)
from almasim.services.imaging.wsclean.runner import (  # noqa: F401  (moved; re-exported)
    _error_line,
    _remove_reorder_files,
    _run_once,
    _run_with_retries,
)
from almasim.services.runtime.env import (  # noqa: F401  (moved; re-exported)
    _SYSTEM_LIBRARY_DIRS,
    casa_environment,
    wsclean_environment,
)

__all__ = [
    "CONTINUUM_COLUMN",
    "CUBE_SELFCAL_CORRECTION_FILENAME",
    "CUBE_TOTAL_RESIDUAL_FILENAME",
    "SELFCAL_CORRECTION_FILENAME",
    "SELFCAL_GAINS_DIRNAME",
    "TOTAL_RESIDUAL_FILENAME",
    "add_images",
    "export_gain_table",
    "selfcal_correction_columns",
    "CONTSUB_FILENAME",
    "CUBE_IMAGE_FILENAME",
    "CUBE_NAME",
    "DEFAULT_MAX_WEIGHT",
    "IMAGE_FILENAME",
    "LINE_COLUMN",
    "LINE_MASK_FILENAME",
    "SINGLE_WINDOW_PLACEHOLDER",
    "extract_single_window_ms",
    "extract_single_window_ms_subprocess",
    "imaging_failure_marker_path",
    "imaging_log_path",
    "imaging_marker_path",
    "line_mask_from_channels",
    "is_imaging_complete",
    "run_wsclean_task",
    "stack_channel_images",
    "fit_uv_continuum",
    "line_free_channels",
    "continuum_restart_commands",
    "subtract_continuum",
    "subtract_uv_continuum",
    "subtract_uv_continuum_subprocess",
    "self_calibrate",
    "casa_selfcal_apply",
    "casa_selfcal_solve",
    "drop_corrected_data",
    "adopt_corrected_data",
    "parse_selfcal_rounds",
    "selfcal_prepare",
    "rows_for_chunk",
    "write_imaging_failure_marker",
    "write_imaging_marker",
    "wsclean_environment",
]
