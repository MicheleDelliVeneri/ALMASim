"""Hubble dataset download utilities."""

from __future__ import annotations

import locale
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover - typing only
    from kaggle import api as _kaggle_api  # noqa: F401

DEFAULT_HUBBLE_DATASET = "redwankarimsony/top-100-hubble-telescope-images"


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


def download_hubble_top100(
    destination: Optional[Path | str] = None,
    token: Optional[Path | str] = None,
) -> Path:
    """Download the Hubble Top-100 dataset via Kaggle.

    ``token`` optionally points to a ``kaggle.json`` credentials file (or the
    directory containing it); otherwise the Kaggle defaults are used.
    """
    from .galaxy_zoo import _download_dataset, configure_kaggle_token

    configure_kaggle_token(token)
    base_path = Path(destination or Path.cwd() / "hubble" / "top100").expanduser().resolve()
    _ensure_directory(base_path)
    _download_dataset(DEFAULT_HUBBLE_DATASET, base_path)
    return base_path
