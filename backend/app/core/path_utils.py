"""Path validation utilities for API routers."""

import os
from pathlib import Path

from fastapi import HTTPException, status


def resolve_safe_path(
    raw: str,
    base_dir: Path,
    *,
    detail: str = "Invalid path",
) -> Path:
    """Resolve *raw* relative to *base_dir* and verify it stays within the base.

    Implements the pattern CodeQL explicitly recommends for py/path-injection:
    normalise with ``os.path.realpath(os.path.join(base, user_input))`` then
    verify the result starts with the safe base prefix.

    Raises HTTP 400 if the resolved path escapes *base_dir* or contains a
    null byte.  Works correctly for both relative and absolute *raw* inputs
    (absolute inputs must still resolve to a path inside *base_dir*).
    """
    if "\x00" in raw:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)
    base = os.path.realpath(str(base_dir))
    full = os.path.realpath(os.path.join(base, raw))
    if full != base and not full.startswith(base + os.sep):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)
    return Path(full)
