"""Galaxy Zoo dataset download utilities."""

from __future__ import annotations

import json
import locale
import os
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover - typing only
    from kaggle import api as _kaggle_api  # noqa: F401

DEFAULT_GALAXY_ZOO_DATASET = "jaimetrickz/galaxy-zoo-2-images"


def configure_kaggle_token(token: Optional[Path | str]) -> None:
    """Configure Kaggle credentials from an explicit token/credentials path.

    Handles both authentication schemes:

    * **Modern bearer token** (``KGAT_...``): a file containing a single token
      string, or the token string itself. Exposed via ``KAGGLE_API_TOKEN`` so the
      ``kagglehub`` stack authenticates with ``Authorization: Bearer <token>``.
    * **Legacy username/key**: a ``kaggle.json`` file (or a directory containing
      one). Exposed via ``KAGGLE_USERNAME``/``KAGGLE_KEY`` (or ``KAGGLE_CONFIG_DIR``)
      for HTTP Basic auth.

    Must be called before the Kaggle client is imported/authenticated.
    """
    if token is None:
        return
    p = Path(str(token)).expanduser()
    if p.is_dir():
        os.environ["KAGGLE_CONFIG_DIR"] = str(p)
        return
    if p.is_file():
        text = p.read_text().strip()
        # Legacy kaggle.json -> username/key
        try:
            creds = json.loads(text)
        except Exception:
            creds = None
        if isinstance(creds, dict) and creds.get("username") and creds.get("key"):
            os.environ["KAGGLE_USERNAME"] = str(creds["username"])
            os.environ["KAGGLE_KEY"] = str(creds["key"])
            return
        # Otherwise treat it as a modern bearer token file (e.g. KGAT_...).
        # kagglehub/kagglesdk accept either the token value or a path to it.
        os.environ["KAGGLE_API_TOKEN"] = str(p)
        return
    # Not a path on disk: treat the value itself as a bearer token.
    os.environ["KAGGLE_API_TOKEN"] = str(token)


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


def download_galaxy_zoo(
    destination: Optional[Path | str] = None,
    token: Optional[Path | str] = None,
) -> Path:
    """Download the Galaxy Zoo 2 dataset via the Kaggle API.

    ``token`` optionally points to a ``kaggle.json`` credentials file (or the
    directory containing it); otherwise the Kaggle defaults are used
    (``~/.kaggle/kaggle.json`` or ``KAGGLE_USERNAME``/``KAGGLE_KEY``).
    """
    configure_kaggle_token(token)
    base_path = Path(destination or Path.cwd() / "galaxy_zoo").expanduser().resolve()
    _ensure_directory(base_path)
    _download_dataset(DEFAULT_GALAXY_ZOO_DATASET, base_path)
    return base_path


def _download_dataset(handle: str, base_path: Path) -> None:
    """Download a Kaggle dataset into ``base_path``.

    Prefers ``kagglehub`` (modern bearer-token auth, reads ``KAGGLE_API_TOKEN`` /
    ``~/.kaggle/access_token``), and falls back to the legacy ``kaggle`` package
    (username/key Basic auth) only if ``kagglehub`` is unavailable.
    """
    try:
        import kagglehub  # modern client, supports KGAT_ bearer tokens

        _run_with_c_locale(
            lambda: kagglehub.dataset_download(handle, output_dir=str(base_path))
        )
        return
    except ImportError:
        pass

    def _legacy():
        api = _load_kaggle_api()
        api.authenticate()
        api.dataset_download_files(handle, path=str(base_path), unzip=True)

    _run_with_c_locale(_legacy)
