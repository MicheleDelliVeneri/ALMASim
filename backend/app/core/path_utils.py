"""Path validation utilities for API routers."""

from pathlib import Path

from fastapi import HTTPException, status


def validate_user_path(
    raw: str,
    *,
    allow_absolute: bool = False,
    detail: str = "Invalid path",
) -> None:
    """Raise HTTP 400 if *raw* contains path-traversal or null-byte sequences.

    Call this before any ``Path()`` or filesystem operation that uses
    user-supplied input so that CodeQL taint analysis sees the barrier guard
    ahead of the sink.

    Parameters
    ----------
    raw:
        The raw string received from the user (query param, path param, body).
    allow_absolute:
        When *False* (default) absolute paths are also rejected.
    detail:
        The error detail included in the HTTP response.
    """
    if "\x00" in raw or ".." in raw:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)
    if not allow_absolute and Path(raw).is_absolute():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)
