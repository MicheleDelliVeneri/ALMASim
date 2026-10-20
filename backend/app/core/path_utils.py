"""Path validation utilities for API routers."""

from pathlib import Path

from fastapi import HTTPException, status

from almasim.services.paths import UnsafePathError, safe_join


def resolve_safe_path(
    raw: str,
    base_dir: Path,
    *,
    detail: str = "Invalid path",
) -> Path:
    """Resolve *raw* relative to *base_dir* and verify it stays within the base.

    The check is :func:`almasim.services.paths.safe_join` (``realpath`` of the
    joined path, then a prefix check on the base, as CodeQL recommends for
    py/path-injection). Raises HTTP 400 with *detail* if the path escapes
    *base_dir* or contains a null byte.
    """
    try:
        return safe_join(base_dir, raw)
    except UnsafePathError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail) from exc
