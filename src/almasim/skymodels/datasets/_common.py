"""Helpers the dataset downloaders share."""

from __future__ import annotations

import locale
from functools import lru_cache
from pathlib import Path


def _ensure_directory(path: Path) -> Path:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)
    return path


@lru_cache(maxsize=1)
def _load_kaggle_api():
    """Load Kaggle API (lazy import)."""
    from kaggle import api as kaggle_api  # local import to avoid side effects

    return kaggle_api


def _run_with_c_locale(func):
    """Run function with C locale."""
    saved = locale.setlocale(locale.LC_ALL)
    try:
        locale.setlocale(locale.LC_ALL, "C")
        return func()
    finally:
        locale.setlocale(locale.LC_ALL, saved)
