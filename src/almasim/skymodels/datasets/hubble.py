"""Hubble dataset download utilities."""
from __future__ import annotations

import locale
from functools import lru_cache
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from kaggle import api as _kaggle_api

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


def download_hubble_top100(destination: Optional[Path | str] = None) -> Path:
    """Download the Hubble Top-100 dataset via Kaggle."""
    base_path = Path(destination or Path.cwd() / "hubble" / "top100").expanduser().resolve()
    _ensure_directory(base_path)

    def _download():
        api = _load_kaggle_api()
        api.authenticate()
        api.dataset_download_files(
            DEFAULT_HUBBLE_DATASET,
            path=str(base_path),
            unzip=True,
        )

    _run_with_c_locale(_download)
    return base_path


