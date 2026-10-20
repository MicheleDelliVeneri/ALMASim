"""Keep user-supplied paths inside a base directory."""

from __future__ import annotations

import os
from pathlib import Path


class UnsafePathError(ValueError):
    """Raised when a path would resolve outside its base directory."""


def safe_join(base: str | os.PathLike[str], raw: str | os.PathLike[str]) -> Path:
    """Resolve ``raw`` against ``base`` and require the result to stay inside it.

    ``raw`` may be relative (taken from ``base``) or absolute; ``~`` is expanded
    in both. Symlinks are resolved before the check, so a link pointing out of
    ``base`` is refused too. Null bytes are refused.
    """
    raw_text = os.fspath(raw)
    if "\x00" in raw_text:
        raise UnsafePathError("Path contains a null byte")
    root = os.path.realpath(os.path.expanduser(os.fspath(base)))
    full = os.path.realpath(os.path.join(root, os.path.expanduser(raw_text)))
    if full != root and not full.startswith(root.rstrip(os.sep) + os.sep):
        raise UnsafePathError(f"{raw_text!r} is outside {root}")
    return Path(full)
