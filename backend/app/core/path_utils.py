"""Path validation utilities for API routers."""

import os
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


def require_allowed_path(raw: str, *, field: str, relative_to: Path | None = None) -> Path:
    """``raw`` resolved, if it lies under one of the configured write roots.

    Relative paths are taken from ``relative_to`` (default: the first root,
    HOST_DIR). Raises HTTP 400 naming ``field`` and the allowed roots otherwise.
    """
    from app.core.config import settings

    roots = settings.write_roots()
    base = relative_to if relative_to is not None else roots[0]
    expanded = os.path.expanduser(raw)
    candidate = expanded if os.path.isabs(expanded) else os.path.join(base, expanded)
    for root in roots:
        try:
            return safe_join(root, candidate)
        except UnsafePathError:
            continue
    allowed = ", ".join(str(r) for r in roots)
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"{field} must be inside one of the allowed directories ({allowed})",
    )


def require_allowed_scheduler(address: str) -> str:
    """``address`` if it is the configured Dask scheduler or an allowed one, else HTTP 400."""
    from app.core.config import settings

    allowed = {settings.DASK_SCHEDULER, *settings.ALLOWED_SCHEDULERS}
    if address not in allowed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="scheduler address is not in ALLOWED_SCHEDULERS",
        )
    return address
