"""Unit tests for almasim.skymodels.datasets.galaxy_zoo."""

from __future__ import annotations

import locale
import sys
from functools import lru_cache
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from almasim.skymodels.datasets.galaxy_zoo import (
    DEFAULT_GALAXY_ZOO_DATASET,
    _ensure_directory,
    _load_kaggle_api,
    _run_with_c_locale,
    download_galaxy_zoo,
)


# ===========================================================================
# _ensure_directory
# ===========================================================================


@pytest.mark.unit
def test_ensure_directory_creates_nested_path(tmp_path):
    """_ensure_directory creates nested directories."""
    target = tmp_path / "a" / "b" / "c"
    result = _ensure_directory(target)
    assert result.is_dir()
    assert result == target


@pytest.mark.unit
def test_ensure_directory_idempotent(tmp_path):
    """_ensure_directory is idempotent."""
    target = tmp_path / "existing"
    target.mkdir()
    result = _ensure_directory(target)
    assert result == target


# ===========================================================================
# _run_with_c_locale
# ===========================================================================


@pytest.mark.unit
def test_run_with_c_locale_calls_function():
    """_run_with_c_locale calls the wrapped function and returns its result."""
    called = {"n": 0}

    def my_func():
        called["n"] += 1
        return 42

    result = _run_with_c_locale(my_func)
    assert called["n"] == 1
    assert result == 42


@pytest.mark.unit
def test_run_with_c_locale_restores_locale_on_success():
    """_run_with_c_locale restores the original locale after the function succeeds."""
    original = locale.setlocale(locale.LC_ALL)

    def noop():
        pass

    _run_with_c_locale(noop)
    restored = locale.setlocale(locale.LC_ALL)
    assert restored == original


@pytest.mark.unit
def test_run_with_c_locale_restores_locale_on_exception():
    """_run_with_c_locale restores locale even if the wrapped function raises."""
    original = locale.setlocale(locale.LC_ALL)

    def bad_func():
        raise ValueError("oops")

    with pytest.raises(ValueError, match="oops"):
        _run_with_c_locale(bad_func)

    assert locale.setlocale(locale.LC_ALL) == original


# ===========================================================================
# _load_kaggle_api
# ===========================================================================


@pytest.mark.unit
def test_load_kaggle_api_returns_api_object():
    """_load_kaggle_api returns the kaggle API object."""
    mock_api = MagicMock()
    mock_kaggle = MagicMock()
    mock_kaggle.api = mock_api

    # Clear the lru_cache before testing
    _load_kaggle_api.cache_clear()
    try:
        with patch.dict(sys.modules, {"kaggle": mock_kaggle}):
            result = _load_kaggle_api()
        assert result is mock_api
    finally:
        _load_kaggle_api.cache_clear()


@pytest.mark.unit
def test_load_kaggle_api_caches_result():
    """_load_kaggle_api is cached (lru_cache)."""
    mock_api = MagicMock()
    mock_kaggle = MagicMock()
    mock_kaggle.api = mock_api

    _load_kaggle_api.cache_clear()
    try:
        with patch.dict(sys.modules, {"kaggle": mock_kaggle}):
            result1 = _load_kaggle_api()
            result2 = _load_kaggle_api()
        # Both calls should return the same object (cached)
        assert result1 is result2
    finally:
        _load_kaggle_api.cache_clear()


# ===========================================================================
# download_galaxy_zoo
# ===========================================================================


@pytest.mark.unit
def test_download_galaxy_zoo_creates_destination(tmp_path):
    """download_galaxy_zoo creates the destination directory."""
    dest = tmp_path / "gz"
    mock_api = MagicMock()

    _load_kaggle_api.cache_clear()
    with patch(
        "almasim.skymodels.datasets.galaxy_zoo._load_kaggle_api",
        return_value=mock_api,
    ):
        result = download_galaxy_zoo(dest)

    assert dest.is_dir()
    assert result == dest


@pytest.mark.unit
def test_download_galaxy_zoo_calls_authenticate(tmp_path):
    """download_galaxy_zoo calls api.authenticate()."""
    dest = tmp_path / "gz"
    mock_api = MagicMock()

    with patch(
        "almasim.skymodels.datasets.galaxy_zoo._load_kaggle_api",
        return_value=mock_api,
    ):
        download_galaxy_zoo(dest)

    mock_api.authenticate.assert_called_once()


@pytest.mark.unit
def test_download_galaxy_zoo_calls_dataset_download_files(tmp_path):
    """download_galaxy_zoo calls api.dataset_download_files with correct args."""
    dest = tmp_path / "gz"
    mock_api = MagicMock()

    with patch(
        "almasim.skymodels.datasets.galaxy_zoo._load_kaggle_api",
        return_value=mock_api,
    ):
        download_galaxy_zoo(dest)

    mock_api.dataset_download_files.assert_called_once_with(
        DEFAULT_GALAXY_ZOO_DATASET,
        path=str(dest),
        unzip=True,
    )


@pytest.mark.unit
def test_download_galaxy_zoo_default_destination(tmp_path):
    """download_galaxy_zoo with None destination uses cwd/galaxy_zoo."""
    mock_api = MagicMock()

    with patch(
        "almasim.skymodels.datasets.galaxy_zoo._load_kaggle_api",
        return_value=mock_api,
    ):
        with patch("pathlib.Path.cwd", return_value=tmp_path):
            result = download_galaxy_zoo(None)

    expected = (tmp_path / "galaxy_zoo").resolve()
    assert result == expected


@pytest.mark.unit
def test_download_galaxy_zoo_uses_c_locale(tmp_path):
    """download_galaxy_zoo wraps the download in C locale."""
    dest = tmp_path / "gz"
    mock_api = MagicMock()
    locale_calls = []

    original_run = _run_with_c_locale.__module__

    with patch(
        "almasim.skymodels.datasets.galaxy_zoo._load_kaggle_api",
        return_value=mock_api,
    ):
        with patch(
            "almasim.skymodels.datasets.galaxy_zoo._run_with_c_locale",
            side_effect=lambda fn: fn(),
        ) as mock_locale_runner:
            download_galaxy_zoo(dest)

    mock_locale_runner.assert_called_once()


@pytest.mark.unit
def test_download_galaxy_zoo_returns_path_object(tmp_path):
    """download_galaxy_zoo returns a Path object."""
    dest = tmp_path / "gz"
    mock_api = MagicMock()

    with patch(
        "almasim.skymodels.datasets.galaxy_zoo._load_kaggle_api",
        return_value=mock_api,
    ):
        result = download_galaxy_zoo(dest)

    assert isinstance(result, Path)


@pytest.mark.unit
def test_default_galaxy_zoo_dataset_constant():
    """DEFAULT_GALAXY_ZOO_DATASET has the expected value."""
    assert DEFAULT_GALAXY_ZOO_DATASET == "jaimetrickz/galaxy-zoo-2-images"
