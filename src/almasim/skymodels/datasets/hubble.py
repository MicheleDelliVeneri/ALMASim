"""Hubble dataset download utilities."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ._common import _ensure_directory

if TYPE_CHECKING:  # pragma: no cover - typing only
    from kaggle import api as _kaggle_api  # noqa: F401

DEFAULT_HUBBLE_DATASET = "redwankarimsony/top-100-hubble-telescope-images"


def download_hubble_top100(
    destination: Path | str | None = None,
    token: Path | str | None = None,
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
