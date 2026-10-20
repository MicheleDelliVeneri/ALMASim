"""FITS writing shared by the imaging products (MADIV-28 T2.8)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

from astropy.io import fits


def write_fits_atomic(
    target: str | os.PathLike[str], hdus: Sequence[fits.hdu.base.ExtensionHDU]
) -> Path:
    """Write ``hdus`` (primary first) to ``target`` through ``<target>.partial``.

    The rename is atomic, so a reader, or a rerun after a crash, never finds a
    half-written product under its final name.
    """
    target = Path(target)
    partial = target.with_name(target.name + ".partial")
    fits.HDUList(list(hdus)).writeto(partial, overwrite=True)
    partial.replace(target)
    return target
