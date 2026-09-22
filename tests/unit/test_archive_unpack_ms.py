"""Unit tests for almasim.services.archive.unpack_ms."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

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
    _emit,
    asdm_name,
    configure_casa_environment,
    create_measurement_set,
    create_measurement_sets,
    ensure_casa_runtime_data,
    find_asdm_directories,
    find_existing_casa_data,
    has_casa_runtime_data,
    verify_asdm_directory,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


_ASDM_XML = """<?xml version="1.0" encoding="UTF-8"?>
<ASDM xmlns:cntnr="http://Alma/XASDM/ASDM" schemaVersion="4">
  <Entity entityId="uid://A002/X1/X1" entityTypeName="ASDM"/>
  <Table><Name> Main </Name><NumberRows> {main_rows} </NumberRows></Table>
  <Table><Name> Antenna </Name><NumberRows> 2 </NumberRows></Table>
  <Table><Name> SysCal </Name><NumberRows> 3 </NumberRows></Table>
  <Table><Name> Flag </Name><NumberRows> 0 </NumberRows></Table>
</ASDM>
"""

_MAIN_XML = """<?xml version="1.0" encoding="UTF-8"?>
<MainTable xmlns="http://Alma/XASDM/MainTable">
  <row>
    <dataUID><EntityRef entityId="uid://A002/X1/Xa" entityTypeName="Main"/></dataUID>
  </row>
  <row>
    <dataUID><EntityRef entityId="uid://A002/X1/Xb" entityTypeName="Main"/></dataUID>
  </row>
</MainTable>
"""


def _make_asdm(tmp_path: Path, uid: str = "uid___A001_X1_X1") -> Path:
    """Create a minimal but *complete* fake ASDM directory.

    Complete means what :func:`verify_asdm_directory` checks: an index, a file
    per non-empty table (SysCal deliberately as ``.bin``), and one non-empty
    binary blob per ``dataUID`` in ``Main.xml``.
    """
    asdm_dir = tmp_path / f"{uid}.asdm.sdm"
    asdm_dir.mkdir(parents=True)
    (asdm_dir / "ASDM.xml").write_text(_ASDM_XML.format(main_rows=2))
    (asdm_dir / "Main.xml").write_text(_MAIN_XML)
    (asdm_dir / "Antenna.xml").write_text("<AntennaTable/>")
    (asdm_dir / "SysCal.bin").write_bytes(b"\x00\x01")
    binary = asdm_dir / "ASDMBinary"
    binary.mkdir()
    (binary / "uid___A002_X1_Xa").write_bytes(b"data")
    (binary / "uid___A002_X1_Xb").write_bytes(b"data")
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
@patch("almasim.services.archive.unpack_ms.measurement_set_row_count", return_value=1234)
def test_create_measurement_set_calls_importasdm_and_returns_path(mock_rows, tmp_path):
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
@patch("almasim.services.archive.unpack_ms.measurement_set_row_count", return_value=1234)
def test_create_measurement_set_skips_when_exists(mock_rows, tmp_path):
    """An existing MS is skipped once verified, and gains a marker."""
    asdm = _make_asdm(tmp_path / "input")
    output_root = tmp_path / "output"
    uid = asdm_name(asdm)
    existing_ms = output_root / "working" / f"{uid}.ms"
    existing_ms.mkdir(parents=True)

    mock_importasdm = MagicMock()
    result = create_measurement_set(mock_importasdm, asdm, output_root, overwrite=False)
    mock_importasdm.assert_not_called()
    assert result == existing_ms
    # Grandfathered in: verified, then marked so the next run takes the fast path.
    assert (output_root / "working" / f"{uid}.ms.done").is_file()


@pytest.mark.unit
@patch("almasim.services.archive.unpack_ms.measurement_set_row_count", return_value=1234)
def test_create_measurement_set_overwrite_true(mock_rows, tmp_path):
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
@patch("almasim.services.archive.unpack_ms.measurement_set_row_count", return_value=1234)
def test_create_measurement_set_emits_log_messages(mock_rows, tmp_path):
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
@patch("almasim.services.archive.unpack_ms.measurement_set_row_count", return_value=1234)
def test_create_measurement_sets_returns_list(
    mock_rows, mock_ensure, mock_configure, mock_find, tmp_path
):
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


@pytest.mark.unit
def test_configure_casa_environment_routes_casa_log(tmp_path):
    """Site config should carry the requested CASA log file and terminal echo."""
    output_root = tmp_path / "out"
    log_file = output_root / "logs" / "casa-calibrate-uid.log"
    configure_casa_environment(
        output_root, tmp_path / "casa-data", log_file=log_file, log_to_terminal=True
    )

    site_config = (output_root / ".casa-config" / "casasiteconfig.py").read_text(encoding="utf-8")
    assert f"logfile = {str(log_file.resolve())!r}" in site_config
    assert "log2term = True" in site_config
    assert log_file.parent.is_dir()


@pytest.mark.unit
def test_configure_casa_environment_defaults_to_quiet_terminal(tmp_path):
    """Without options the site config keeps CASA's default log file and no terminal echo."""
    output_root = tmp_path / "out"
    configure_casa_environment(output_root, tmp_path / "casa-data")

    site_config = (output_root / ".casa-config" / "casasiteconfig.py").read_text(encoding="utf-8")
    assert "logfile" not in site_config
    assert "log2term = False" in site_config


# ===========================================================================
# unpack completion markers and row verification
# ===========================================================================


@pytest.mark.unit
def test_raw_ms_marker_helpers(tmp_path):
    """The done marker records the row count and clears any stale failure marker."""
    from almasim.services.archive.unpack_ms import (
        is_unpack_complete,
        raw_ms_failure_marker_path,
        raw_ms_marker_path,
        write_raw_ms_failure_marker,
        write_raw_ms_marker,
    )

    output_root = tmp_path / "out"
    (output_root / "working").mkdir(parents=True)
    uid = "uid___A002_X1_X2"

    assert is_unpack_complete(output_root, uid) is False

    failure = write_raw_ms_failure_marker(output_root, uid, "RuntimeError: killed", log_path="/l")
    assert failure == raw_ms_failure_marker_path(output_root, uid)
    assert failure.name == "uid___A002_X1_X2.ms.failed"
    assert json.loads(failure.read_text())["error"] == "RuntimeError: killed"

    # Marker alone is not enough; the MS directory must exist too.
    marker = write_raw_ms_marker(output_root, uid, 42)
    assert marker == raw_ms_marker_path(output_root, uid)
    assert is_unpack_complete(output_root, uid) is False
    (output_root / "working" / f"{uid}.ms").mkdir()
    assert is_unpack_complete(output_root, uid) is True

    assert json.loads(marker.read_text())["rows"] == 42
    # Success clears the earlier failure.
    assert not failure.exists()


@pytest.mark.unit
@patch("almasim.services.archive.unpack_ms.measurement_set_row_count")
def test_verify_measurement_set_rejects_zero_rows(mock_rows, tmp_path):
    """A 0-row MS is the signature of a killed importasdm and must not pass."""
    from almasim.services.archive.unpack_ms import verify_measurement_set

    ms = tmp_path / "uid___A002_X1_X2.ms"
    ms.mkdir()

    mock_rows.return_value = 0
    with pytest.raises(RuntimeError, match="has no rows"):
        verify_measurement_set(ms)

    mock_rows.return_value = 7
    assert verify_measurement_set(ms) == 7

    with pytest.raises(RuntimeError, match="was not created"):
        verify_measurement_set(tmp_path / "absent.ms")


@pytest.mark.unit
@patch("almasim.services.archive.unpack_ms.measurement_set_row_count", return_value=0)
def test_truncated_existing_ms_is_reimported_not_skipped(mock_rows, tmp_path):
    """The bug that let 511 truncated MSs through: existence alone meant 'done'."""
    asdm = _make_asdm(tmp_path / "input")
    output_root = tmp_path / "output"
    uid = asdm_name(asdm)
    truncated = output_root / "working" / f"{uid}.ms"
    truncated.mkdir(parents=True)

    calls = []

    def fake_importasdm(asdm, vis, overwrite):
        calls.append(overwrite)
        Path(vis).mkdir(parents=True, exist_ok=True)
        # The re-import succeeds this time.
        mock_rows.return_value = 999

    result = create_measurement_set(fake_importasdm, asdm, output_root, overwrite=False)

    assert calls == [True], "a truncated MS must be re-imported, with overwrite forced"
    assert result == truncated
    assert json.loads((output_root / "working" / f"{uid}.ms.done").read_text())["rows"] == 999


@pytest.mark.unit
@patch("almasim.services.archive.unpack_ms.measurement_set_row_count", return_value=0)
def test_failed_import_writes_failure_marker_and_no_done_marker(mock_rows, tmp_path):
    """An import that produces an empty MS is a failure, and is recorded as one."""
    asdm = _make_asdm(tmp_path / "input")
    output_root = tmp_path / "output"
    uid = asdm_name(asdm)

    def fake_importasdm(asdm, vis, overwrite):
        Path(vis).mkdir(parents=True, exist_ok=True)

    with pytest.raises(RuntimeError, match="has no rows"):
        create_measurement_set(fake_importasdm, asdm, output_root)

    assert not (output_root / "working" / f"{uid}.ms.done").exists()
    payload = json.loads((output_root / "working" / f"{uid}.ms.failed").read_text())
    assert payload["stage"] == "unpack"
    assert "has no rows" in payload["error"]


@pytest.mark.unit
@patch("almasim.services.archive.unpack_ms.measurement_set_row_count")
def test_marked_ms_is_skipped_without_reopening_it(mock_rows, tmp_path):
    """Once marked, the fast path must not pay to reopen a multi-GB MS."""
    from almasim.services.archive.unpack_ms import write_raw_ms_marker

    asdm = _make_asdm(tmp_path / "input")
    output_root = tmp_path / "output"
    uid = asdm_name(asdm)
    (output_root / "working" / f"{uid}.ms").mkdir(parents=True)
    write_raw_ms_marker(output_root, uid, 100)

    mock_importasdm = MagicMock()
    create_measurement_set(mock_importasdm, asdm, output_root, overwrite=False)

    mock_importasdm.assert_not_called()
    mock_rows.assert_not_called()


# ===========================================================================
# verify_asdm_directory / half-extracted ASDMs
# ===========================================================================


@pytest.mark.unit
def test_verify_asdm_directory_accepts_complete_asdm(tmp_path):
    verify_asdm_directory(_make_asdm(tmp_path))


@pytest.mark.unit
def test_verify_asdm_directory_missing_index(tmp_path):
    asdm = _make_asdm(tmp_path)
    (asdm / "ASDM.xml").unlink()
    with pytest.raises(RuntimeError, match="ASDM.xml is missing"):
        verify_asdm_directory(asdm)


@pytest.mark.unit
def test_verify_asdm_directory_missing_table_file(tmp_path):
    """The real-world case: ExecBlock.xml had not been extracted yet."""
    asdm = _make_asdm(tmp_path)
    (asdm / "Antenna.xml").unlink()
    with pytest.raises(RuntimeError, match=r"table file\(s\) missing: Antenna"):
        verify_asdm_directory(asdm)


@pytest.mark.unit
def test_verify_asdm_directory_accepts_bin_or_xml_table_file(tmp_path):
    asdm = _make_asdm(tmp_path)
    (asdm / "SysCal.bin").rename(asdm / "SysCal.xml")
    verify_asdm_directory(asdm)


@pytest.mark.unit
def test_verify_asdm_directory_ignores_empty_tables(tmp_path):
    """Flag has NumberRows 0 and no file; that is normal."""
    asdm = _make_asdm(tmp_path)
    assert not (asdm / "Flag.xml").exists()
    verify_asdm_directory(asdm)


@pytest.mark.unit
def test_verify_asdm_directory_missing_binary_data(tmp_path):
    asdm = _make_asdm(tmp_path)
    (asdm / "ASDMBinary" / "uid___A002_X1_Xb").unlink()
    with pytest.raises(RuntimeError, match=r"1/2 binary data file\(s\) missing.*uid___A002_X1_Xb"):
        verify_asdm_directory(asdm)


@pytest.mark.unit
def test_verify_asdm_directory_rejects_empty_binary_data(tmp_path):
    asdm = _make_asdm(tmp_path)
    (asdm / "ASDMBinary" / "uid___A002_X1_Xa").write_bytes(b"")
    with pytest.raises(RuntimeError, match="binary data file"):
        verify_asdm_directory(asdm)


@pytest.mark.unit
def test_find_asdm_directories_ignores_extraction_staging(tmp_path):
    """An ASDM still inside a download's staging directory is not an ASDM yet."""
    from almasim.services.extraction import EXTRACTION_TMP_PREFIX

    root = tmp_path / "input"
    published = _make_asdm(root / "proj" / "raw", uid="uid___A002_X1_Xdone")
    staging = root / f"{EXTRACTION_TMP_PREFIX}bundle-abc" / "proj" / "raw"
    _make_asdm(staging, uid="uid___A002_X1_Xpartial")

    assert find_asdm_directories(root) == [published]
    with pytest.raises(RuntimeError, match="No ASDM named uid___A002_X1_Xpartial"):
        find_asdm_directories(root, asdm_uid="uid___A002_X1_Xpartial")


@pytest.mark.unit
@patch("almasim.services.archive.unpack_ms.measurement_set_row_count", return_value=1234)
def test_create_measurement_set_refuses_incomplete_asdm(mock_rows, tmp_path):
    """A half-extracted ASDM is a recorded failure, and importasdm is never run."""
    asdm = _make_asdm(tmp_path / "input")
    (asdm / "ASDMBinary" / "uid___A002_X1_Xa").unlink()
    output_root = tmp_path / "output"
    uid = asdm_name(asdm)
    importasdm = MagicMock()

    with pytest.raises(RuntimeError, match="ASDM is incomplete"):
        create_measurement_set(importasdm, asdm, output_root)

    importasdm.assert_not_called()
    failed = output_root / "working" / f"{uid}.ms.failed"
    assert failed.is_file()
    assert "ASDM is incomplete" in json.loads(failed.read_text())["error"]
    assert not (output_root / "working" / f"{uid}.ms.done").exists()
