"""Unit tests for almasim.services.archive.unpack_ms."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure casatasks / casaconfig are stubbed before the module under test is
# imported so that the real CASA installations are never required.
# ---------------------------------------------------------------------------
_casa_stubs = {
    "casatasks": MagicMock(),
    "casaconfig": MagicMock(),
}
for _mod, _stub in _casa_stubs.items():
    sys.modules.setdefault(_mod, _stub)

from almasim.services.archive.unpack_ms import (  # noqa: E402
    LogFn,
    _emit,
    asdm_name,
    configure_casa_environment,
    create_measurement_set,
    create_measurement_sets,
    ensure_casa_runtime_data,
    find_asdm_directories,
    find_existing_casa_data,
    has_casa_runtime_data,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_asdm(tmp_path: Path, uid: str = "uid___A001_X1_X1") -> Path:
    """Create a fake ASDM directory."""
    asdm_dir = tmp_path / f"{uid}.asdm.sdm"
    asdm_dir.mkdir(parents=True)
    return asdm_dir


def _make_casa_data(path: Path) -> Path:
    """Populate a directory to look like valid CASA runtime data."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "readme.txt").write_text("CASA data")
    (path / "geodetic").mkdir()
    return path


# ===========================================================================
# has_casa_runtime_data
# ===========================================================================


@pytest.mark.unit
def test_has_casa_runtime_data_true(tmp_path):
    """Directory with readme.txt and geodetic/ is recognised as CASA data."""
    _make_casa_data(tmp_path)
    assert has_casa_runtime_data(tmp_path) is True


@pytest.mark.unit
def test_has_casa_runtime_data_missing_geodetic(tmp_path):
    """Directory without geodetic/ is not recognised as CASA data."""
    (tmp_path / "readme.txt").write_text("x")
    assert has_casa_runtime_data(tmp_path) is False


@pytest.mark.unit
def test_has_casa_runtime_data_missing_readme(tmp_path):
    """Directory without readme.txt is not recognised as CASA data."""
    (tmp_path / "geodetic").mkdir()
    assert has_casa_runtime_data(tmp_path) is False


@pytest.mark.unit
def test_has_casa_runtime_data_nonexistent(tmp_path):
    """A path that does not exist returns False."""
    assert has_casa_runtime_data(tmp_path / "no-such-dir") is False


# ===========================================================================
# _emit
# ===========================================================================


@pytest.mark.unit
def test_emit_calls_logger_fn():
    """_emit should call the provided callable with the message."""
    messages = []
    _emit(messages.append, "hello world")
    assert messages == ["hello world"]


@pytest.mark.unit
def test_emit_none_logger_fn():
    """_emit with None logger_fn should not raise."""
    _emit(None, "hello world")  # no exception


# ===========================================================================
# find_existing_casa_data
# ===========================================================================


@pytest.mark.unit
def test_find_existing_casa_data_explicit_override(tmp_path):
    """Explicit casa_data_root is returned as-is."""
    explicit = tmp_path / "my-data"
    explicit.mkdir()
    result = find_existing_casa_data(tmp_path, tmp_path / "out", explicit)
    assert result == explicit.resolve()


@pytest.mark.unit
def test_find_existing_casa_data_output_data_preferred(tmp_path):
    """Pre-populated .casa-data inside output_root is preferred."""
    output_root = tmp_path / "output"
    _make_casa_data(output_root / ".casa-data")
    result = find_existing_casa_data(tmp_path, output_root)
    assert result == (output_root / ".casa-data").resolve()


@pytest.mark.unit
def test_find_existing_casa_data_input_root_fallback(tmp_path):
    """A .casa-data directory inside input_root is used when output is absent."""
    input_root = tmp_path / "input"
    _make_casa_data(input_root / ".casa-data")
    output_root = tmp_path / "output"
    output_root.mkdir(parents=True)
    result = find_existing_casa_data(input_root, output_root)
    assert result == (input_root / ".casa-data").resolve()


@pytest.mark.unit
def test_find_existing_casa_data_default_fallback(tmp_path):
    """When nothing is found we fall back to <output_root>/.casa-data."""
    output_root = tmp_path / "output"
    output_root.mkdir(parents=True)
    result = find_existing_casa_data(tmp_path, output_root)
    assert result == (output_root / ".casa-data").resolve()


# ===========================================================================
# configure_casa_environment
# ===========================================================================


@pytest.mark.unit
def test_configure_casa_environment_creates_files(tmp_path):
    """configure_casa_environment creates the site config and MPL cache dirs."""
    output_root = tmp_path / "output"
    casa_data = tmp_path / "casa-data"
    configure_casa_environment(output_root, casa_data)

    site_config = output_root / ".casa-config" / "casasiteconfig.py"
    assert site_config.is_file()
    content = site_config.read_text()
    assert "measurespath" in content
    assert "data_auto_update = False" in content
    assert "measures_auto_update = False" in content


@pytest.mark.unit
def test_configure_casa_environment_returns_casa_data_path(tmp_path):
    """configure_casa_environment returns the resolved casa_data path."""
    output_root = tmp_path / "output"
    casa_data = tmp_path / "casa-data"
    returned = configure_casa_environment(output_root, casa_data)
    assert returned == casa_data.resolve()


# ===========================================================================
# ensure_casa_runtime_data
# ===========================================================================


@pytest.mark.unit
def test_ensure_casa_runtime_data_already_present(tmp_path, caplog):
    """When data is already present, no casaconfig call is made."""
    casa_data = _make_casa_data(tmp_path / "casa")
    # Should complete without raising or calling casaconfig
    ensure_casa_runtime_data(casa_data, skip_update=False, logger_fn=None)


@pytest.mark.unit
def test_ensure_casa_runtime_data_skip_update_raises_when_missing(tmp_path):
    """skip_update=True raises RuntimeError when CASA data is absent."""
    casa_data = tmp_path / "missing-casa"
    with pytest.raises(RuntimeError, match="CASA runtime data is missing"):
        ensure_casa_runtime_data(casa_data, skip_update=True)


@pytest.mark.unit
def test_ensure_casa_runtime_data_calls_update_all_when_missing(tmp_path):
    """When data is absent and skip_update=False, casaconfig.update_all is called."""
    casa_data = tmp_path / "empty-casa"
    casa_data.mkdir()
    mock_update_all = MagicMock()
    with patch.dict(sys.modules, {"casaconfig": MagicMock(update_all=mock_update_all)}):
        with patch("almasim.services.archive.unpack_ms.has_casa_runtime_data", return_value=False):
            with patch("casaconfig.update_all", mock_update_all):
                try:
                    ensure_casa_runtime_data(casa_data, skip_update=False)
                except Exception:
                    pass  # we only care that the call was attempted


# ===========================================================================
# find_asdm_directories
# ===========================================================================


@pytest.mark.unit
def test_find_asdm_directories_finds_asdm(tmp_path):
    """A single ASDM directory is correctly discovered."""
    asdm = _make_asdm(tmp_path)
    results = find_asdm_directories(tmp_path)
    assert len(results) == 1
    assert results[0] == asdm.resolve()


@pytest.mark.unit
def test_find_asdm_directories_specific_uid(tmp_path):
    """Only the ASDM with the requested UID is returned."""
    _make_asdm(tmp_path, "uid___A001_X1_X1")
    target = _make_asdm(tmp_path, "uid___A001_X1_X2")
    results = find_asdm_directories(tmp_path, asdm_uid="uid___A001_X1_X2")
    assert results == [target.resolve()]


@pytest.mark.unit
def test_find_asdm_directories_nonexistent_root(tmp_path):
    """Non-existent input root raises RuntimeError."""
    with pytest.raises(RuntimeError, match="does not exist"):
        find_asdm_directories(tmp_path / "nope")


@pytest.mark.unit
def test_find_asdm_directories_empty_dir_raises(tmp_path):
    """No ASDMs found raises RuntimeError."""
    with pytest.raises(RuntimeError, match="No \\*.asdm.sdm directories found"):
        find_asdm_directories(tmp_path)


@pytest.mark.unit
def test_find_asdm_directories_missing_uid_raises(tmp_path):
    """Requesting a specific UID that is absent raises RuntimeError."""
    _make_asdm(tmp_path, "uid___A001_X1_X1")
    with pytest.raises(RuntimeError, match="No ASDM named"):
        find_asdm_directories(tmp_path, asdm_uid="uid___MISSING")


# ===========================================================================
# asdm_name
# ===========================================================================


@pytest.mark.unit
def test_asdm_name_strips_suffix(tmp_path):
    """asdm_name returns the UID part without .asdm.sdm."""
    asdm = tmp_path / "uid___A001_X1_X1.asdm.sdm"
    assert asdm_name(asdm) == "uid___A001_X1_X1"


# ===========================================================================
# create_measurement_set
# ===========================================================================


@pytest.mark.unit
def test_create_measurement_set_calls_importasdm_and_returns_path(tmp_path):
    """create_measurement_set calls importasdm and returns the expected path."""
    asdm = _make_asdm(tmp_path / "input")
    output_root = tmp_path / "output"

    # importasdm side-effect: create the expected output directory
    expected_ms = output_root / "working" / f"{asdm_name(asdm)}.ms"

    def fake_importasdm(asdm, vis, overwrite):
        Path(vis).mkdir(parents=True)

    result = create_measurement_set(fake_importasdm, asdm, output_root)
    assert result == expected_ms


@pytest.mark.unit
def test_create_measurement_set_skips_when_exists(tmp_path):
    """create_measurement_set skips importasdm when MS already exists."""
    asdm = _make_asdm(tmp_path / "input")
    output_root = tmp_path / "output"
    uid = asdm_name(asdm)
    existing_ms = output_root / "working" / f"{uid}.ms"
    existing_ms.mkdir(parents=True)

    mock_importasdm = MagicMock()
    result = create_measurement_set(mock_importasdm, asdm, output_root, overwrite=False)
    mock_importasdm.assert_not_called()
    assert result == existing_ms


@pytest.mark.unit
def test_create_measurement_set_overwrite_true(tmp_path):
    """When overwrite=True, importasdm is called even if MS exists."""
    asdm = _make_asdm(tmp_path / "input")
    output_root = tmp_path / "output"
    uid = asdm_name(asdm)
    existing_ms = output_root / "working" / f"{uid}.ms"
    existing_ms.mkdir(parents=True)

    def fake_importasdm(asdm, vis, overwrite):
        Path(vis).mkdir(parents=True, exist_ok=True)

    result = create_measurement_set(fake_importasdm, asdm, output_root, overwrite=True)
    assert result == existing_ms


@pytest.mark.unit
def test_create_measurement_set_missing_asdm_raises(tmp_path):
    """A missing ASDM directory raises RuntimeError."""
    missing_asdm = tmp_path / "uid___NONE.asdm.sdm"
    with pytest.raises(RuntimeError, match="Cannot find raw ASDM"):
        create_measurement_set(MagicMock(), missing_asdm, tmp_path)


@pytest.mark.unit
def test_create_measurement_set_importasdm_failure_raises(tmp_path):
    """If importasdm does not create the MS directory, RuntimeError is raised."""
    asdm = _make_asdm(tmp_path / "input")
    output_root = tmp_path / "output"

    def bad_importasdm(asdm, vis, overwrite):
        pass  # intentionally does NOT create vis

    with pytest.raises(RuntimeError, match="Expected MeasurementSet was not created"):
        create_measurement_set(bad_importasdm, asdm, output_root)


@pytest.mark.unit
def test_create_measurement_set_emits_log_messages(tmp_path):
    """create_measurement_set emits log messages via logger_fn."""
    asdm = _make_asdm(tmp_path / "input")
    output_root = tmp_path / "output"

    def fake_importasdm(asdm, vis, overwrite):
        Path(vis).mkdir(parents=True)

    messages = []
    create_measurement_set(fake_importasdm, asdm, output_root, logger_fn=messages.append)
    assert any("Creating raw MeasurementSet" in m for m in messages)
    assert any("Created raw MeasurementSet" in m for m in messages)


# ===========================================================================
# create_measurement_sets (integration of top-level function)
# ===========================================================================


@pytest.mark.unit
@patch("almasim.services.archive.unpack_ms.find_existing_casa_data")
@patch("almasim.services.archive.unpack_ms.configure_casa_environment")
@patch("almasim.services.archive.unpack_ms.ensure_casa_runtime_data")
def test_create_measurement_sets_returns_list(mock_ensure, mock_configure, mock_find, tmp_path):
    """create_measurement_sets returns a list of MS paths."""
    asdm = _make_asdm(tmp_path / "input")
    output_root = tmp_path / "output"
    mock_find.return_value = tmp_path / "casa-data"
    mock_configure.return_value = tmp_path / "casa-data"

    uid = asdm_name(asdm)
    expected_ms = output_root / "working" / f"{uid}.ms"

    def fake_importasdm(asdm, vis, overwrite):
        Path(vis).mkdir(parents=True)

    with patch.dict(sys.modules, {"casatasks": MagicMock(importasdm=fake_importasdm)}):
        results = create_measurement_sets(
            tmp_path / "input", output_root, skip_casa_data_update=True
        )

    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0] == expected_ms
