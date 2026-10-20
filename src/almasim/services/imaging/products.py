"""Product file names and MeasurementSet column names written by archive imaging.

Moved out of archive_imaging.py (MADIV-28 module split).
"""

from __future__ import annotations

# WSClean is invoked with ``-name <output_dir>/wsclean`` and a single SPW, so
# the restored image is always this file.
IMAGE_FILENAME = "wsclean-image.fits"


# A cube run uses ``-name <output_dir>/wsclean-cube``; its per-channel images
# are stacked into ``wsclean-cube-<product>.fits`` (see stack_channel_images).
CUBE_NAME = "wsclean-cube"


CUBE_IMAGE_FILENAME = f"{CUBE_NAME}-image.fits"


CONTSUB_FILENAME = f"{CUBE_NAME}-image.contsub.fits"


# The moment-0 mask the cube's second pass cleans in (``line_mask_from_channels``).
LINE_MASK_FILENAME = f"{CUBE_NAME}-linemask.fits"


# The primary beam and the primary-beam corrected image, continuum and cube
# (``primary_beam.primary_beam_correct``).
PB_FILENAME = "wsclean-pb.fits"


PBCOR_FILENAME = "wsclean-image.pbcor.fits"


CUBE_PB_FILENAME = f"{CUBE_NAME}-pb.fits"


CUBE_PBCOR_FILENAME = f"{CUBE_NAME}-image.pbcor.fits"


# Columns ``subtract_uv_continuum`` adds to the per-task single-window MS: the
# continuum fitted per visibility over the line-free channels, and the data
# minus it (the lines). The continuum image is made from the first, the cube
# from the second.
CONTINUUM_COLUMN = "CONTINUUM_DATA"


LINE_COLUMN = "LINE_DATA"


# What self-calibration removed (DATA - CORRECTED_DATA), whole or split into
# continuum and lines like the data (``selfcal_correction_columns``), and the
# images made from it: the dirty image of the correction and the total
# residual, the kept residual plus the correction (``_selfcal_noise_products``).
SELFCAL_COLUMN = "SELFCAL_DATA"


SELFCAL_CONTINUUM_COLUMN = "SELFCAL_CONTINUUM_DATA"


SELFCAL_LINE_COLUMN = "SELFCAL_LINE_DATA"


SELFCAL_CORRECTION_FILENAME = "wsclean-selfcal-correction.fits"


TOTAL_RESIDUAL_FILENAME = "wsclean-residual-total.fits"


CUBE_SELFCAL_CORRECTION_FILENAME = f"{CUBE_NAME}-selfcal-correction.fits"


CUBE_TOTAL_RESIDUAL_FILENAME = f"{CUBE_NAME}-residual-total.fits"


SELFCAL_GAINS_DIRNAME = "selfcal_gains"


# Placeholder in a task command for the per-task single-window MS that the
# wrapper extracts on the worker (see ``extract_single_window_ms``).
SINGLE_WINDOW_PLACEHOLDER = "__SINGLE_WINDOW_MS__"


def continuum_product(kind: str) -> str:
    """File name of a continuum WSClean product ("image", "residual", "model", ...)."""
    return f"wsclean-{kind}.fits"


def cube_product(kind: str) -> str:
    """File name of a cube WSClean product ("image", "residual", "model", ...)."""
    return f"{CUBE_NAME}-{kind}.fits"
