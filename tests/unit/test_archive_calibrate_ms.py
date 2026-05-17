"""Unit tests for almasim.services.archive.calibrate_ms.

CASA dependencies (casatools, casatasks) are mocked at the function level
using patch.dict(sys.modules) so that the module-level sys.modules state is
never permanently polluted. This prevents cross-test interference with
tests that require the *real* casatools (e.g. test_measurement_set_native_ms).
"""

from __future__ import annotations

import sys
import tarfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Import only the pure-Python helpers that don't require CASA at import time.
# Functions that call `from casatools import ...` inside their body are tested
# with explicit patch.dict(sys.modules) in each test.
from almasim.services.archive.calibrate_ms import (
    _find_input_casa_data,
    _is_relative_to,
    _parse_applycal_calls,
    _prepare_casa_environment,
    _prepare_working_directory,
    _remove_tree_after_success,
    _safe_extract_tar,
    apply_delivered_calibration,
    create_calibrated_measurement_sets,
    find_calibration_directory,
    find_raw_ms_directories,
    restore_calibrated_measurement_sets,
    science_spws,
    split_calibrated_science_ms,
)

# ===========================================================================
# helpers
# ===========================================================================


def _make_calibration_dir(tmp_path: Path, uid: str = "uid___A001_X1_X1") -> Path:
    """Create a fake calibration directory with a .calapply.txt file."""
    cal_dir = tmp_path / "calibration"
    cal_dir.mkdir(parents=True, exist_ok=True)
    (cal_dir / f"{uid}.ms.calapply.txt").write_text(
        f'applycal(vis="{uid}.ms", gaintable=["{uid}.ms.gcal"], intent="TARGET")\n'
    )
    return cal_dir


def _make_raw_ms(tmp_path: Path, uid: str = "uid___A001_X1_X1") -> Path:
    """Create a fake raw MS directory."""
    ms_dir = tmp_path / f"{uid}.ms"
    ms_dir.mkdir(parents=True, exist_ok=True)
    return ms_dir


def _mock_casatools():
    """Return a MagicMock that looks like casatools with msmetadata."""
    mock_msmd = MagicMock()
    mock_msmd.spwsforintent.return_value = [0, 1, 2]
    mock_msmd.nchan.side_effect = lambda spw: 128 if spw in [0, 2] else 2
    mock_msmd.intents.return_value = ["OBSERVE_TARGET#ON_SOURCE"]
    mock_casatools = MagicMock()
    mock_casatools.msmetadata.return_value = mock_msmd
    return mock_casatools, mock_msmd


# ===========================================================================
# _safe_extract_tar
# ===========================================================================


@pytest.mark.unit
def test_safe_extract_tar_normal_file(tmp_path):
    """Normal tar member is extracted and returned in list."""
    src_file = tmp_path / "data.txt"
    src_file.write_text("content")
    tar_path = tmp_path / "archive.tgz"
    dest = tmp_path / "dest"
    dest.mkdir()
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(src_file, arcname="data.txt")
    extracted = _safe_extract_tar(tar_path, dest)
    assert len(extracted) == 1
    assert (dest / "data.txt").is_file()


@pytest.mark.unit
def test_safe_extract_tar_skips_parent_traversal(tmp_path):
    """Tar members with .. in path are skipped."""
    src_file = tmp_path / "x.txt"
    src_file.write_text("x")
    tar_path = tmp_path / "traversal.tgz"
    dest = tmp_path / "dest"
    dest.mkdir()
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(src_file, arcname="x.txt")

    with tarfile.open(tar_path, "r:gz") as tf:
        members = tf.getmembers()

    members[0].name = "../outside.txt"
    bad_tar = tmp_path / "bad_traversal.tgz"
    with tarfile.open(bad_tar, "w:gz") as tf2:
        for m in members:
            tf2.addfile(m, src_file.open("rb"))

    _safe_extract_tar(bad_tar, dest)
    assert not (tmp_path / "outside.txt").exists()


@pytest.mark.unit
def test_safe_extract_tar_multiple_files(tmp_path):
    """Multiple safe files are all extracted."""
    dest = tmp_path / "dest"
    dest.mkdir()
    tar_path = tmp_path / "multi.tgz"
    file1 = tmp_path / "a.txt"
    file2 = tmp_path / "b.txt"
    file1.write_text("a")
    file2.write_text("b")
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(file1, arcname="a.txt")
        tf.add(file2, arcname="b.txt")
    extracted = _safe_extract_tar(tar_path, dest)
    assert len(extracted) == 2


# ===========================================================================
# find_calibration_directory
# ===========================================================================


@pytest.mark.unit
def test_find_calibration_directory_finds_dir(tmp_path):
    """find_calibration_directory returns the calibration directory."""
    uid = "uid___A001_X1_X1"
    cal_dir = _make_calibration_dir(tmp_path, uid)
    result = find_calibration_directory(tmp_path, asdm_uid=uid)
    assert result == cal_dir.resolve()


@pytest.mark.unit
def test_find_calibration_directory_no_match_raises(tmp_path):
    """No matching calibration directory raises RuntimeError."""
    with pytest.raises(RuntimeError, match="No calibration directory found"):
        find_calibration_directory(tmp_path, asdm_uid="uid___MISSING")


@pytest.mark.unit
def test_find_calibration_directory_no_uid_single_ok(tmp_path):
    """A single calibration dir without asdm_uid is returned."""
    _make_calibration_dir(tmp_path, "uid___A001_X1_X1")
    result = find_calibration_directory(tmp_path)
    assert result.name == "calibration"


@pytest.mark.unit
def test_find_calibration_directory_no_uid_multiple_raises(tmp_path):
    """Multiple calibration dirs without asdm_uid raises RuntimeError."""
    (tmp_path / "A" / "calibration").mkdir(parents=True)
    (tmp_path / "B" / "calibration").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="Found more than one"):
        find_calibration_directory(tmp_path)


# ===========================================================================
# find_raw_ms_directories
# ===========================================================================


@pytest.mark.unit
def test_find_raw_ms_directories_finds_ms(tmp_path):
    """find_raw_ms_directories discovers *.ms directories."""
    ms = _make_raw_ms(tmp_path)
    results = find_raw_ms_directories(tmp_path)
    assert ms.resolve() in results


@pytest.mark.unit
def test_find_raw_ms_directories_specific_uid(tmp_path):
    """find_raw_ms_directories filters by UID."""
    _make_raw_ms(tmp_path, "uid___A001_X1_X1")
    target = _make_raw_ms(tmp_path, "uid___A001_X1_X2")
    results = find_raw_ms_directories(tmp_path, asdm_uid="uid___A001_X1_X2")
    assert results == [target.resolve()]


@pytest.mark.unit
def test_find_raw_ms_directories_missing_root_raises(tmp_path):
    """Non-existent root raises RuntimeError."""
    with pytest.raises(RuntimeError, match="does not exist or is not a directory"):
        find_raw_ms_directories(tmp_path / "nope")


@pytest.mark.unit
def test_find_raw_ms_directories_empty_raises(tmp_path):
    """No MS directories found raises RuntimeError."""
    with pytest.raises(RuntimeError, match="No raw \\*.ms directories found"):
        find_raw_ms_directories(tmp_path)


@pytest.mark.unit
def test_find_raw_ms_directories_missing_uid_raises(tmp_path):
    """Specific UID not found raises RuntimeError."""
    _make_raw_ms(tmp_path, "uid___A001_X1_X1")
    with pytest.raises(RuntimeError, match="No raw MS named"):
        find_raw_ms_directories(tmp_path, asdm_uid="uid___MISSING")


# ===========================================================================
# _parse_applycal_calls
# ===========================================================================


@pytest.mark.unit
def test_parse_applycal_calls_basic(tmp_path):
    """_parse_applycal_calls parses a simple applycal call."""
    calapply = tmp_path / "test.ms.calapply.txt"
    calapply.write_text('applycal(vis="test.ms", gaintable=["test.gcal"], intent="TARGET")\n')
    calls = _parse_applycal_calls(calapply)
    assert len(calls) == 1
    assert calls[0]["vis"] == "test.ms"
    assert calls[0]["gaintable"] == ["test.gcal"]
    assert calls[0]["intent"] == "TARGET"


@pytest.mark.unit
def test_parse_applycal_calls_multiple(tmp_path):
    """_parse_applycal_calls handles multiple applycal blocks."""
    calapply = tmp_path / "multi.calapply.txt"
    calapply.write_text(
        'applycal(vis="a.ms", gaintable=["a.gcal"])\napplycal(vis="b.ms", gaintable=["b.gcal"])\n'
    )
    calls = _parse_applycal_calls(calapply)
    assert len(calls) == 2


@pytest.mark.unit
def test_parse_applycal_calls_no_applycal_raises(tmp_path):
    """No applycal() calls in file raises RuntimeError."""
    calapply = tmp_path / "empty.calapply.txt"
    calapply.write_text("# This file has no applycal calls\nx = 1\n")
    with pytest.raises(RuntimeError, match="No applycal calls found"):
        _parse_applycal_calls(calapply)


@pytest.mark.unit
def test_parse_applycal_calls_ignores_non_applycal(tmp_path):
    """Non-applycal function calls in file are ignored."""
    calapply = tmp_path / "mixed.calapply.txt"
    calapply.write_text('flagcmd(vis="test.ms")\napplycal(vis="test.ms", gaintable=["g.gcal"])\n')
    calls = _parse_applycal_calls(calapply)
    assert len(calls) == 1


@pytest.mark.unit
def test_find_input_casa_data_returns_populated_runtime_dir(tmp_path):
    """_find_input_casa_data should return the first populated .casa-data directory."""
    input_root = tmp_path / "input"
    candidate = input_root / "nested" / ".casa-data"
    candidate.mkdir(parents=True)
    (candidate / "readme.txt").write_text("ok")
    (candidate / "geodetic").mkdir()

    result = _find_input_casa_data(input_root)

    assert result == candidate.resolve()


@pytest.mark.unit
@patch("almasim.services.archive.calibrate_ms.tempfile.mkdtemp")
def test_prepare_casa_environment_honors_explicit_casa_data_root(mock_mkdtemp, tmp_path):
    """Explicit casa_data_root should be used directly with an isolated workspace."""
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    explicit = tmp_path / "explicit-casa"
    input_root.mkdir()
    output_root.mkdir()

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    mock_mkdtemp.return_value = str(workspace)

    casa_data, workspace_root = _prepare_casa_environment(input_root, output_root, explicit)

    assert casa_data == explicit.resolve()
    assert workspace_root == workspace


@pytest.mark.unit
@patch("almasim.services.archive.calibrate_ms.tempfile.mkdtemp")
@patch("almasim.services.archive.calibrate_ms.find_existing_casa_data")
def test_prepare_casa_environment_falls_back_to_input_runtime_data(
    mock_find_existing,
    mock_mkdtemp,
    tmp_path,
):
    """When output/runtime data is missing, calibration should reuse populated input data."""
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    input_root.mkdir()
    output_root.mkdir()

    input_casa = input_root / "delivery" / ".casa-data"
    input_casa.mkdir(parents=True)
    (input_casa / "readme.txt").write_text("ok")
    (input_casa / "geodetic").mkdir()

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    mock_mkdtemp.return_value = str(workspace)
    mock_find_existing.return_value = output_root / ".casa-data"

    casa_data, workspace_root = _prepare_casa_environment(input_root, output_root, None)

    assert casa_data == input_casa.resolve()
    assert workspace_root == workspace


# ===========================================================================
# _is_relative_to
# ===========================================================================


@pytest.mark.unit
def test_is_relative_to_true():
    """Child is relative to parent."""
    assert _is_relative_to(Path("/a/b/c"), Path("/a/b")) is True


@pytest.mark.unit
def test_is_relative_to_false():
    """Unrelated path is not relative."""
    assert _is_relative_to(Path("/x/y"), Path("/a/b")) is False


@pytest.mark.unit
def test_is_relative_to_same_path():
    """A path is relative to itself."""
    assert _is_relative_to(Path("/a/b"), Path("/a/b")) is True


# ===========================================================================
# _remove_tree_after_success
# ===========================================================================


@pytest.mark.unit
def test_remove_tree_after_success_removes(tmp_path):
    """_remove_tree_after_success removes the target when safe."""
    target = tmp_path / "intermediate"
    target.mkdir()
    (target / "file.txt").write_text("junk")
    protected = [tmp_path / "output.ms"]
    _remove_tree_after_success(target, protected)
    assert not target.exists()


@pytest.mark.unit
def test_remove_tree_after_success_skips_protected(tmp_path):
    """Protected output inside target prevents deletion."""
    target = tmp_path / "working"
    target.mkdir()
    protected_path = target / "output.ms"
    protected_path.mkdir()
    _remove_tree_after_success(target, [protected_path])
    assert target.exists()


@pytest.mark.unit
def test_remove_tree_after_success_nonexistent_is_noop(tmp_path):
    """Non-existent target is silently ignored."""
    _remove_tree_after_success(tmp_path / "nope", [])


@pytest.mark.unit
def test_remove_tree_after_success_emits_log(tmp_path):
    """Log message is emitted after removal."""
    target = tmp_path / "to_remove"
    target.mkdir()
    messages = []
    _remove_tree_after_success(target, [], logger_fn=messages.append)
    assert any("Removed intermediate data" in m for m in messages)


# ===========================================================================
# _prepare_working_directory
# ===========================================================================


@pytest.mark.unit
def test_prepare_working_directory_copies_ms(tmp_path):
    """_prepare_working_directory copies the raw MS into working_dir."""
    raw_ms = _make_raw_ms(tmp_path / "raw")
    cal_dir = _make_calibration_dir(tmp_path)
    working_dir = tmp_path / "working"
    result = _prepare_working_directory(raw_ms, cal_dir, working_dir)
    assert result.is_dir()
    assert result.name == raw_ms.name


@pytest.mark.unit
def test_prepare_working_directory_raises_when_exists_no_overwrite(tmp_path):
    """RuntimeError when working MS exists and overwrite=False."""
    raw_ms = _make_raw_ms(tmp_path / "raw")
    cal_dir = _make_calibration_dir(tmp_path)
    working_dir = tmp_path / "working"
    working_dir.mkdir()
    (working_dir / raw_ms.name).mkdir()
    with pytest.raises(RuntimeError, match="Working MS already exists"):
        _prepare_working_directory(raw_ms, cal_dir, working_dir, overwrite=False)


@pytest.mark.unit
def test_prepare_working_directory_overwrite_removes_old(tmp_path):
    """overwrite=True replaces existing working MS."""
    raw_ms = _make_raw_ms(tmp_path / "raw")
    cal_dir = _make_calibration_dir(tmp_path)
    working_dir = tmp_path / "working"
    working_dir.mkdir()
    (working_dir / raw_ms.name).mkdir()
    result = _prepare_working_directory(raw_ms, cal_dir, working_dir, overwrite=True)
    assert result.is_dir()


# ===========================================================================
# apply_delivered_calibration
# ===========================================================================


@pytest.mark.unit
def test_apply_delivered_calibration_calls_applycal(tmp_path):
    """apply_delivered_calibration calls the applycal callable for each block."""
    uid = "uid___A001_X1_X1"
    raw_ms = _make_raw_ms(tmp_path / "raw", uid)
    cal_dir = _make_calibration_dir(tmp_path / "cal", uid)
    working_dir = tmp_path / "working"

    mock_applycal = MagicMock()
    mock_clearcal = MagicMock()

    with patch(
        "almasim.services.archive.calibrate_ms._normalize_intent",
        side_effect=lambda intent, ms_path: intent,
    ):
        result = apply_delivered_calibration(
            mock_applycal, mock_clearcal, raw_ms, cal_dir, working_dir
        )

    mock_clearcal.assert_called_once()
    mock_applycal.assert_called_once()
    assert mock_applycal.call_args.kwargs["flagbackup"] is False
    assert result.name == f"{uid}.ms"


@pytest.mark.unit
def test_apply_delivered_calibration_missing_calapply_raises(tmp_path):
    """Missing .calapply.txt raises RuntimeError."""
    uid = "uid___A001_X1_X1"
    raw_ms = _make_raw_ms(tmp_path / "raw", uid)
    cal_dir = tmp_path / "calibration"
    cal_dir.mkdir()
    working_dir = tmp_path / "working"
    with pytest.raises(RuntimeError, match="Cannot find calibration apply file"):
        apply_delivered_calibration(MagicMock(), MagicMock(), raw_ms, cal_dir, working_dir)


@pytest.mark.unit
def test_apply_delivered_calibration_emits_log(tmp_path):
    """apply_delivered_calibration emits log messages."""
    uid = "uid___A001_X1_X1"
    raw_ms = _make_raw_ms(tmp_path / "raw", uid)
    cal_dir = _make_calibration_dir(tmp_path / "cal", uid)
    working_dir = tmp_path / "working"
    messages = []
    mock_applycal = MagicMock()
    mock_clearcal = MagicMock()

    with patch(
        "almasim.services.archive.calibrate_ms._normalize_intent",
        side_effect=lambda intent, ms_path: intent,
    ):
        apply_delivered_calibration(
            mock_applycal,
            mock_clearcal,
            raw_ms,
            cal_dir,
            working_dir,
            logger_fn=messages.append,
        )

    assert any("Applying" in m for m in messages)


@pytest.mark.unit
def test_apply_delivered_calibration_resolves_relative_callib(tmp_path):
    """Relative callib paths in delivered applycal should be resolved under working_dir."""
    uid = "uid___A001_X1_X1"
    raw_ms = _make_raw_ms(tmp_path / "raw", uid)
    cal_dir = tmp_path / "calibration"
    cal_dir.mkdir(parents=True)
    (cal_dir / f"{uid}.ms.calapply.txt").write_text(
        'applycal(vis="x.ms", gaintable=["g.gcal"], callib="cal/chain.callib")\n'
    )
    working_dir = tmp_path / "working"

    mock_applycal = MagicMock()
    with patch(
        "almasim.services.archive.calibrate_ms._normalize_intent",
        side_effect=lambda intent, ms_path: intent,
    ):
        apply_delivered_calibration(
            mock_applycal,
            MagicMock(),
            raw_ms,
            cal_dir,
            working_dir,
        )

    kwargs = mock_applycal.call_args.kwargs
    assert kwargs["callib"] == str(working_dir / "cal/chain.callib")


# ===========================================================================
# science_spws — uses casatools internally
# ===========================================================================


@pytest.mark.unit
def test_science_spws_returns_comma_separated_string(tmp_path):
    """science_spws returns SPW IDs as a comma-separated string."""
    ms_path = tmp_path / "test.ms"
    ms_path.mkdir()

    mock_casatools, mock_msmd = _mock_casatools()

    with patch.dict(sys.modules, {"casatools": mock_casatools}):
        result = science_spws(ms_path)

    assert "0" in result
    assert "2" in result
    assert "1" not in result  # nchan == 2 <= 4


@pytest.mark.unit
def test_science_spws_no_science_spws_raises(tmp_path):
    """RuntimeError when no science SPWs with >4 channels exist."""
    ms_path = tmp_path / "test.ms"
    ms_path.mkdir()

    mock_msmd = MagicMock()
    mock_msmd.spwsforintent.return_value = [0, 1]
    mock_msmd.nchan.return_value = 1  # all <= 4
    mock_casatools = MagicMock()
    mock_casatools.msmetadata.return_value = mock_msmd

    with patch.dict(sys.modules, {"casatools": mock_casatools}):
        with pytest.raises(RuntimeError, match="No science SPWs found"):
            science_spws(ms_path)


# ===========================================================================
# split_calibrated_science_ms
# ===========================================================================


@pytest.mark.unit
def test_split_calibrated_science_ms_calls_mstransform_and_returns_path(tmp_path):
    """split_calibrated_science_ms calls mstransform and returns the output path."""
    uid = "uid___A001_X1_X1"
    calibrated_ms = tmp_path / "working" / f"{uid}.ms"
    calibrated_ms.mkdir(parents=True)
    output_root = tmp_path / "output"

    def fake_mstransform(vis, outputvis, spw, reindex):
        Path(outputvis).mkdir(parents=True)

    with patch("almasim.services.archive.calibrate_ms.science_spws", return_value="0,1,2"):
        result = split_calibrated_science_ms(fake_mstransform, calibrated_ms, output_root)

    expected = output_root / f"{uid}.ms.split.cal"
    assert result == expected


@pytest.mark.unit
def test_split_calibrated_science_ms_raises_if_output_not_created(tmp_path):
    """RuntimeError if mstransform does not create the output MS."""
    uid = "uid___A001_X1_X1"
    calibrated_ms = tmp_path / "working" / f"{uid}.ms"
    calibrated_ms.mkdir(parents=True)
    output_root = tmp_path / "output"

    def bad_mstransform(vis, outputvis, spw, reindex):
        pass  # intentionally does NOT create outputvis

    with patch("almasim.services.archive.calibrate_ms.science_spws", return_value="0,1"):
        with pytest.raises(RuntimeError, match="Expected calibrated MS was not created"):
            split_calibrated_science_ms(bad_mstransform, calibrated_ms, output_root)


@pytest.mark.unit
def test_split_calibrated_science_ms_raises_when_exists_no_overwrite(tmp_path):
    """RuntimeError when output MS already exists and overwrite=False."""
    uid = "uid___A001_X1_X1"
    calibrated_ms = tmp_path / "working" / f"{uid}.ms"
    calibrated_ms.mkdir(parents=True)
    output_root = tmp_path / "output"
    output_root.mkdir()
    (output_root / f"{uid}.ms.split.cal").mkdir()

    with patch("almasim.services.archive.calibrate_ms.science_spws", return_value="0"):
        with pytest.raises(RuntimeError, match="Calibrated output MS already exists"):
            split_calibrated_science_ms(MagicMock(), calibrated_ms, output_root, overwrite=False)


@pytest.mark.unit
def test_split_calibrated_science_ms_overwrite_removes_old(tmp_path):
    """overwrite=True replaces an existing output MS."""
    uid = "uid___A001_X1_X1"
    calibrated_ms = tmp_path / "working" / f"{uid}.ms"
    calibrated_ms.mkdir(parents=True)
    output_root = tmp_path / "output"
    output_root.mkdir()
    existing_out = output_root / f"{uid}.ms.split.cal"
    existing_out.mkdir()

    def fake_mstransform(vis, outputvis, spw, reindex):
        Path(outputvis).mkdir(parents=True, exist_ok=True)

    with patch("almasim.services.archive.calibrate_ms.science_spws", return_value="0"):
        result = split_calibrated_science_ms(
            fake_mstransform, calibrated_ms, output_root, overwrite=True
        )
    assert result == existing_out


# ===========================================================================
# restore_calibrated_measurement_sets (smoke test)
# ===========================================================================


@pytest.mark.unit
@patch("almasim.services.archive.calibrate_ms.create_calibrated_measurement_sets")
def test_restore_calibrated_measurement_sets_delegates(mock_create, tmp_path):
    """restore_calibrated_measurement_sets calls create_calibrated_measurement_sets."""
    mock_create.return_value = [tmp_path / "result.ms"]
    results = restore_calibrated_measurement_sets(
        tmp_path / "input", tmp_path / "raw", tmp_path / "output"
    )
    mock_create.assert_called_once()
    assert results == [tmp_path / "result.ms"]


@pytest.mark.unit
@patch("almasim.services.archive.calibrate_ms.tempfile.mkdtemp")
@patch("almasim.services.archive.calibrate_ms.configure_casa_environment")
@patch("almasim.services.archive.calibrate_ms.ensure_casa_runtime_data")
@patch("almasim.services.archive.calibrate_ms.find_calibration_directory")
@patch("almasim.services.archive.calibrate_ms.find_raw_ms_directories")
@patch("almasim.services.archive.calibrate_ms.split_calibrated_science_ms")
@patch("almasim.services.archive.calibrate_ms.apply_delivered_calibration")
def test_create_calibrated_measurement_sets_uses_temp_workspace_when_no_casa_root(
    mock_apply,
    mock_split,
    mock_find_raw,
    mock_find_cal,
    mock_ensure_casa,
    mock_configure_casa,
    mock_mkdtemp,
    tmp_path,
):
    """Calibration should avoid writing CASA scratch under the shared output tree by default."""
    input_root = tmp_path / "input"
    raw_ms_root = tmp_path / "raw"
    output_root = tmp_path / "output"
    input_root.mkdir()
    raw_ms_root.mkdir()
    output_root.mkdir()

    raw_ms = raw_ms_root / "uid___A001_X1_X1.ms"
    raw_ms.mkdir()
    cal_dir = input_root / "calibration"
    cal_dir.mkdir()
    (cal_dir / f"{raw_ms.name}.calapply.txt").write_text('applycal(vis="x")\n')

    mock_find_raw.return_value = [raw_ms]
    mock_find_cal.return_value = cal_dir
    mock_apply.return_value = raw_ms
    mock_split.return_value = output_root / f"{raw_ms.name}.split.cal"

    scratch_root = tmp_path / "scratch"
    scratch_root.mkdir()
    mock_mkdtemp.return_value = str(scratch_root)

    fake_casatasks = MagicMock()
    fake_casatasks.applycal = mock_apply
    fake_casatasks.clearcal = MagicMock()
    fake_casatasks.mstransform = MagicMock()

    with patch.dict(sys.modules, {"casatasks": fake_casatasks}):
        results = create_calibrated_measurement_sets(
            input_root,
            raw_ms_root,
            output_root,
            skip_casa_data_update=True,
        )

    mock_configure_casa.assert_called_once()
    assert mock_configure_casa.call_args.kwargs["workspace_root"] == scratch_root
    assert results == [output_root / f"{raw_ms.name}.split.cal"]


@pytest.mark.unit
@patch("almasim.services.archive.calibrate_ms.tempfile.mkdtemp")
@patch("almasim.services.archive.calibrate_ms.configure_casa_environment")
@patch("almasim.services.archive.calibrate_ms.ensure_casa_runtime_data")
@patch("almasim.services.archive.calibrate_ms.find_calibration_directory")
@patch("almasim.services.archive.calibrate_ms.find_raw_ms_directories")
@patch("almasim.services.archive.calibrate_ms.split_calibrated_science_ms")
@patch("almasim.services.archive.calibrate_ms.apply_delivered_calibration")
def test_create_calibrated_measurement_sets_reuses_preflighted_output_casa_data(
    mock_apply,
    mock_split,
    mock_find_raw,
    mock_find_cal,
    mock_ensure_casa,
    mock_configure_casa,
    mock_mkdtemp,
    tmp_path,
):
    """Slurm preflighted CASA data under the output root should be reused, not recreated."""
    input_root = tmp_path / "input"
    raw_ms_root = tmp_path / "raw"
    output_root = tmp_path / "output"
    input_root.mkdir()
    raw_ms_root.mkdir()
    output_root.mkdir()
    (output_root / ".casa-data").mkdir()
    (output_root / ".casa-data" / "readme.txt").write_text("data")
    (output_root / ".casa-data" / "geodetic").mkdir()

    raw_ms = raw_ms_root / "uid___A001_X1_X1.ms"
    raw_ms.mkdir()
    cal_dir = input_root / "calibration"
    cal_dir.mkdir()
    (cal_dir / f"{raw_ms.name}.calapply.txt").write_text('applycal(vis="x")\n')

    mock_find_raw.return_value = [raw_ms]
    mock_find_cal.return_value = cal_dir
    mock_apply.return_value = raw_ms
    mock_split.return_value = output_root / f"{raw_ms.name}.split.cal"

    fake_casatasks = MagicMock()
    fake_casatasks.applycal = mock_apply
    fake_casatasks.clearcal = MagicMock()
    fake_casatasks.mstransform = MagicMock()

    scratch_root = tmp_path / "scratch"
    mock_mkdtemp.return_value = str(scratch_root)

    with patch.dict(sys.modules, {"casatasks": fake_casatasks}):
        results = create_calibrated_measurement_sets(
            input_root,
            raw_ms_root,
            output_root,
            skip_casa_data_update=True,
        )

    mock_mkdtemp.assert_called_once_with(prefix="almasim-casa-")
    mock_configure_casa.assert_called_once()
    assert mock_configure_casa.call_args.kwargs["workspace_root"] == scratch_root
    assert mock_ensure_casa.call_args.args[0] == output_root / ".casa-data"
    assert results == [output_root / f"{raw_ms.name}.split.cal"]


@pytest.mark.unit
@patch("almasim.services.archive.calibrate_ms.shutil.rmtree")
@patch("almasim.services.archive.calibrate_ms.tempfile.mkdtemp")
@patch("almasim.services.archive.calibrate_ms.configure_casa_environment")
@patch("almasim.services.archive.calibrate_ms.ensure_casa_runtime_data")
@patch("almasim.services.archive.calibrate_ms.find_raw_ms_directories")
def test_create_calibrated_measurement_sets_cleans_workspace_on_failure(
    mock_find_raw,
    mock_ensure_casa,
    mock_configure_casa,
    mock_mkdtemp,
    mock_rmtree,
    tmp_path,
):
    """Temporary workspace should be removed in finally even when setup fails."""
    input_root = tmp_path / "input"
    raw_ms_root = tmp_path / "raw"
    output_root = tmp_path / "output"
    input_root.mkdir()
    raw_ms_root.mkdir()
    output_root.mkdir()

    mock_find_raw.return_value = []
    scratch_root = tmp_path / "scratch"
    scratch_root.mkdir()
    mock_mkdtemp.return_value = str(scratch_root)
    mock_ensure_casa.side_effect = RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        create_calibrated_measurement_sets(
            input_root,
            raw_ms_root,
            output_root,
            skip_casa_data_update=False,
        )

    mock_configure_casa.assert_called_once()
    mock_rmtree.assert_any_call(scratch_root, ignore_errors=True)


@pytest.mark.unit
@patch("almasim.services.archive.calibrate_ms.tempfile.mkdtemp")
@patch("almasim.services.archive.calibrate_ms._remove_tree_after_success")
@patch("almasim.services.archive.calibrate_ms.configure_casa_environment")
@patch("almasim.services.archive.calibrate_ms.ensure_casa_runtime_data")
@patch("almasim.services.archive.calibrate_ms.find_calibration_directory")
@patch("almasim.services.archive.calibrate_ms.find_raw_ms_directories")
@patch("almasim.services.archive.calibrate_ms.split_calibrated_science_ms")
@patch("almasim.services.archive.calibrate_ms.apply_delivered_calibration")
def test_create_calibrated_measurement_sets_cleans_intermediate_paths(
    mock_apply,
    mock_split,
    mock_find_raw,
    mock_find_cal,
    mock_ensure_casa,
    mock_configure_casa,
    mock_remove_tree,
    mock_mkdtemp,
    tmp_path,
):
    """Successful clean_intermediate mode should request cleanup of raw/working/original roots."""
    input_root = tmp_path / "input"
    raw_ms_root = tmp_path / "raw"
    output_root = tmp_path / "output"
    original_data_root = tmp_path / "original"
    input_root.mkdir()
    raw_ms_root.mkdir()
    output_root.mkdir()
    original_data_root.mkdir()

    raw_ms = raw_ms_root / "uid___A001_X1_X1.ms"
    raw_ms.mkdir()
    cal_dir = input_root / "calibration"
    cal_dir.mkdir()
    (cal_dir / f"{raw_ms.name}.calapply.txt").write_text('applycal(vis="x")\n')

    mock_find_raw.return_value = [raw_ms]
    mock_find_cal.return_value = cal_dir
    mock_apply.return_value = raw_ms
    split_output = output_root / f"{raw_ms.name}.split.cal"
    mock_split.return_value = split_output

    scratch_root = tmp_path / "scratch"
    scratch_root.mkdir()
    mock_mkdtemp.return_value = str(scratch_root)

    fake_casatasks = MagicMock()
    fake_casatasks.applycal = mock_apply
    fake_casatasks.clearcal = MagicMock()
    fake_casatasks.mstransform = MagicMock()

    with patch.dict(sys.modules, {"casatasks": fake_casatasks}):
        results = create_calibrated_measurement_sets(
            input_root,
            raw_ms_root,
            output_root,
            clean_intermediate=True,
            original_data_root=original_data_root,
            skip_casa_data_update=True,
        )

    assert results == [split_output]
    cleanup_targets = [call.args[0] for call in mock_remove_tree.call_args_list]
    assert raw_ms_root in cleanup_targets
    assert output_root / "working" in cleanup_targets
    assert original_data_root in cleanup_targets
