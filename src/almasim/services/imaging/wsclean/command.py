"""Build and rewrite WSClean command lines.

Moved out of archive_imaging.py (MADIV-28 module split).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence


def _selfcal_command(cmd: list[str], out: Path, name: str) -> list[str]:
    """The continuum command imaging into ``out/name`` and writing MODEL_DATA."""
    result = list(cmd)
    result[result.index("-name") + 1] = str(out / name)
    return [
        "-update-model-required" if part == "-no-update-model-required" else part for part in result
    ]


def _dirty_command(cmd: Sequence[str], out: Path, name: str, column: str) -> list[str]:
    """``cmd`` turned into a dirty image of ``column`` into ``out/name``: same grid,
    weighting and channels, no cleaning, no PSF, the model column left alone."""
    parts = [str(part) for part in cmd]
    with_value = {"-auto-mask", "-auto-threshold", "-fits-mask", "-data-column", "-name", "-niter"}
    alone = {"-continue", "-make-psf", "-update-model-required", "-no-update-model-required"}
    kept: list[str] = []
    index = 1
    while index < len(parts) - 1:
        if parts[index] in with_value:
            index += 2
            continue
        if parts[index] not in alone:
            kept.append(parts[index])
        index += 1
    return [
        parts[0],
        "-name",
        str(out / name),
        *kept,
        "-niter",
        "0",
        "-no-update-model-required",
        "-data-column",
        column,
        parts[-1],
    ]


def _with_data_column(cmd: list[str], column: str) -> list[str]:
    """``cmd`` reading ``column``: ``-data-column`` goes before the MS, its last argument."""
    return [*cmd[:-1], "-data-column", column, cmd[-1]]


def continuum_restart_commands(cmd: Sequence[str]) -> list[list[str]]:
    """The continuum ``cmd`` as two passes: one writing MODEL_DATA, then ``-continue``.

    WSClean's auto-mask is built in the first major cycle, at ``-auto-mask``
    times the noise it measures then; a bright source's sidelobes inflate that
    noise, so the mask misses its fainter parts and later cycles, which clean
    only inside it, never reach them. uid___A002_X10ded83_X6af4 SPW 16: noise
    91 mJy at mask creation against 26 mJy at the end, a 167 mJy feature 0.7
    beam off the peak left uncleaned, 16 % of the flux within a beam in the
    residual. The second pass starts from the true residual, measures the
    noise again and builds a new mask: residual 3 %, model flux +20 %
    (2026-10-18). The model visibilities go into the task's own copy of the MS.
    """
    first = [
        "-update-model-required" if part == "-no-update-model-required" else str(part)
        for part in cmd
    ]
    if "-update-model-required" not in first:
        first.insert(len(first) - 1, "-update-model-required")
    return [first, [*first[:-1], "-continue", first[-1]]]
