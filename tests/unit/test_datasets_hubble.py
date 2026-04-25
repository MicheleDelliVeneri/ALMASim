"""Unit tests for almasim.skymodels.datasets.hubble."""

from __future__ import annotations

import locale
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from almasim.skymodels.datasets.hubble import (
    DEFAULT_HUBBLE_DATASET,
    _ensure_directory,
    _load_kaggle_api,
    _run_with_c_locale,
    download_hubble_top100,
)

# ===========================================================================
# _ensure_directory
# ===========================================================================


@pytest.mark.unit
def test_ensure_directory_creates_nested_path(tmp_path):
    """_ensure_directory creates the full directory tree."""
    target = tmp_path / "hubble" / "top100" / "images"
    result = _ensure_directory(target)
    assert result.is_dir()
    assert result == target


@pytest.mark.unit
def test_ensure_directory_idempotent(tmp_path):
    """_ensure_directory is idempotent for existing directories."""
    target = tmp_path / "existing"
    target.mkdir()
    result = _ensure_directory(target)
    assert result == target


# ===========================================================================
# _run_with_c_locale
# ===========================================================================


@pytest.mark.unit
def test_run_with_c_locale_executes_function():
    """_run_with_c_locale calls and returns the result of the wrapped function."""

    def my_func():
        return "result"

    assert _run_with_c_locale(my_func) == "result"


@pytest.mark.unit
def test_run_with_c_locale_restores_locale_on_success():
    """_run_with_c_locale restores the original locale after success."""
    original = locale.setlocale(locale.LC_ALL)
    _run_with_c_locale(lambda: None)
    assert locale.setlocale(locale.LC_ALL) == original


@pytest.mark.unit
def test_run_with_c_locale_restores_locale_on_exception():
    """_run_with_c_locale restores locale even when the function raises."""
    original = locale.setlocale(locale.LC_ALL)
    with pytest.raises(RuntimeError):
        _run_with_c_locale(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert locale.setlocale(locale.LC_ALL) == original


# ===========================================================================
# _load_kaggle_api
# ===========================================================================


@pytest.mark.unit
def test_load_kaggle_api_returns_api_object():
    """_load_kaggle_api returns the kaggle.api object."""
    mock_api = MagicMock()
    mock_kaggle = MagicMock()
    mock_kaggle.api = mock_api

    _load_kaggle_api.cache_clear()
    try:
        with patch.dict(sys.modules, {"kaggle": mock_kaggle}):
            result = _load_kaggle_api()
        assert result is mock_api
    finally:
        _load_kaggle_api.cache_clear()


@pytest.mark.unit
def test_load_kaggle_api_is_cached():
    """_load_kaggle_api caches the API object after first call."""
    mock_api = MagicMock()
    mock_kaggle = MagicMock()
    mock_kaggle.api = mock_api

    _load_kaggle_api.cache_clear()
    try:
        with patch.dict(sys.modules, {"kaggle": mock_kaggle}):
            r1 = _load_kaggle_api()
            r2 = _load_kaggle_api()
        assert r1 is r2
    finally:
        _load_kaggle_api.cache_clear()


# ===========================================================================
# download_hubble_top100
# ===========================================================================


@pytest.mark.unit
def test_download_hubble_top100_creates_destination(tmp_path):
    """download_hubble_top100 creates the destination directory."""
    dest = tmp_path / "hubble"
    mock_api = MagicMock()

    _load_kaggle_api.cache_clear()
    with patch(
        "almasim.skymodels.datasets.hubble._load_kaggle_api",
        return_value=mock_api,
    ):
        result = download_hubble_top100(dest)

    assert dest.is_dir()
    assert result == dest


@pytest.mark.unit
def test_download_hubble_top100_calls_authenticate(tmp_path):
    """download_hubble_top100 calls api.authenticate()."""
    dest = tmp_path / "hubble"
    mock_api = MagicMock()

    with patch(
        "almasim.skymodels.datasets.hubble._load_kaggle_api",
        return_value=mock_api,
    ):
        download_hubble_top100(dest)

    mock_api.authenticate.assert_called_once()


@pytest.mark.unit
def test_download_hubble_top100_calls_dataset_download_files(tmp_path):
    """download_hubble_top100 calls api.dataset_download_files correctly."""
    dest = tmp_path / "hubble"
    mock_api = MagicMock()

    with patch(
        "almasim.skymodels.datasets.hubble._load_kaggle_api",
        return_value=mock_api,
    ):
        download_hubble_top100(dest)

    mock_api.dataset_download_files.assert_called_once_with(
        DEFAULT_HUBBLE_DATASET,
        path=str(dest),
        unzip=True,
    )


@pytest.mark.unit
def test_download_hubble_top100_default_destination():
    """download_hubble_top100 with None uses cwd/hubble/top100."""
    mock_api = MagicMock()

    with patch(
        "almasim.skymodels.datasets.hubble._load_kaggle_api",
        return_value=mock_api,
    ):
        with patch("pathlib.Path.cwd", return_value=Path("/tmp/test_cwd")):
            result = download_hubble_top100(None)

    assert result == Path("/tmp/test_cwd/hubble/top100").resolve()


@pytest.mark.unit
def test_download_hubble_top100_returns_path_object(tmp_path):
    """download_hubble_top100 returns a Path."""
    dest = tmp_path / "hubble"
    mock_api = MagicMock()

    with patch(
        "almasim.skymodels.datasets.hubble._load_kaggle_api",
        return_value=mock_api,
    ):
        result = download_hubble_top100(dest)

    assert isinstance(result, Path)


@pytest.mark.unit
def test_download_hubble_top100_uses_c_locale(tmp_path):
    """download_hubble_top100 wraps download in C locale."""
    dest = tmp_path / "hubble"
    mock_api = MagicMock()

    with patch(
        "almasim.skymodels.datasets.hubble._load_kaggle_api",
        return_value=mock_api,
    ):
        with patch(
            "almasim.skymodels.datasets.hubble._run_with_c_locale",
            side_effect=lambda fn: fn(),
        ) as mock_locale_runner:
            download_hubble_top100(dest)

    mock_locale_runner.assert_called_once()


@pytest.mark.unit
def test_default_hubble_dataset_constant():
    """DEFAULT_HUBBLE_DATASET has the expected value."""
    assert DEFAULT_HUBBLE_DATASET == "redwankarimsony/top-100-hubble-telescope-images"
