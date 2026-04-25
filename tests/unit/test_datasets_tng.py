"""Unit tests for almasim.skymodels.datasets.tng."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Stub pysftp and paramiko before importing the module under test.
# ---------------------------------------------------------------------------
_mock_pysftp = MagicMock()
_mock_paramiko = MagicMock()
sys.modules.setdefault("pysftp", _mock_pysftp)
sys.modules.setdefault("paramiko", _mock_paramiko)

from almasim.skymodels.datasets.tng import (  # noqa: E402
    TNG_SIMULATION_URL,
    RemoteMachine,
    _ensure_directory,
    _require_pysftp,
    _wget_tng,
    download_tng_structure,
)

# ===========================================================================
# _ensure_directory
# ===========================================================================


@pytest.mark.unit
def test_ensure_directory_creates_path(tmp_path):
    """_ensure_directory creates the directory and returns it."""
    target = tmp_path / "a" / "b" / "c"
    result = _ensure_directory(target)
    assert result.is_dir()
    assert result == target


@pytest.mark.unit
def test_ensure_directory_existing_is_noop(tmp_path):
    """_ensure_directory is idempotent when directory already exists."""
    target = tmp_path / "existing"
    target.mkdir()
    result = _ensure_directory(target)
    assert result == target


# ===========================================================================
# _require_pysftp
# ===========================================================================


@pytest.mark.unit
def test_require_pysftp_returns_module_when_available():
    """_require_pysftp returns the pysftp module when available."""
    import almasim.skymodels.datasets.tng as tng_mod

    original_pysftp = tng_mod.pysftp
    tng_mod.pysftp = _mock_pysftp
    try:
        result = _require_pysftp()
        assert result is _mock_pysftp
    finally:
        tng_mod.pysftp = original_pysftp


@pytest.mark.unit
def test_require_pysftp_raises_when_unavailable():
    """_require_pysftp raises RuntimeError when pysftp is None."""
    import almasim.skymodels.datasets.tng as tng_mod

    original = tng_mod.pysftp
    tng_mod.pysftp = None
    try:
        with pytest.raises(RuntimeError, match="pysftp"):
            _require_pysftp()
    finally:
        tng_mod.pysftp = original


# ===========================================================================
# _wget_tng
# ===========================================================================


@pytest.mark.unit
def test_wget_tng_calls_subprocess_run(tmp_path):
    """_wget_tng runs wget with the correct arguments."""
    dest = tmp_path / "simulation.hdf5"
    with patch("almasim.skymodels.datasets.tng.subprocess.run") as mock_run:
        _wget_tng("MY_API_KEY", dest)
    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert "wget" in cmd
    assert "--header=API-Key:MY_API_KEY" in cmd
    assert str(dest) in cmd
    assert TNG_SIMULATION_URL in cmd


@pytest.mark.unit
def test_wget_tng_uses_check_true(tmp_path):
    """_wget_tng passes check=True to subprocess.run."""
    with patch("almasim.skymodels.datasets.tng.subprocess.run") as mock_run:
        _wget_tng("key", tmp_path / "out.hdf5")
    assert mock_run.call_args[1].get("check") is True


# ===========================================================================
# download_tng_structure — local path
# ===========================================================================


@pytest.mark.unit
def test_download_tng_structure_local_creates_dirs(tmp_path):
    """download_tng_structure creates the expected directory structure."""
    with patch("almasim.skymodels.datasets.tng._wget_tng") as mock_wget:
        download_tng_structure("MY_KEY", tmp_path)

    assert (tmp_path / "TNG100-1").is_dir()
    assert (tmp_path / "TNG100-1" / "output").is_dir()
    assert (tmp_path / "TNG100-1" / "postprocessing" / "offsets").is_dir()
    mock_wget.assert_called_once()


@pytest.mark.unit
def test_download_tng_structure_local_returns_structure_path(tmp_path):
    """download_tng_structure returns the path to simulation.hdf5."""
    with patch("almasim.skymodels.datasets.tng._wget_tng"):
        result = download_tng_structure("KEY", tmp_path)

    assert result == tmp_path / "TNG100-1" / "simulation.hdf5"


@pytest.mark.unit
def test_download_tng_structure_local_skips_wget_if_exists(tmp_path):
    """download_tng_structure skips wget when simulation.hdf5 already exists."""
    structure_path = tmp_path / "TNG100-1" / "simulation.hdf5"
    structure_path.parent.mkdir(parents=True)
    structure_path.write_text("fake")

    with patch("almasim.skymodels.datasets.tng._wget_tng") as mock_wget:
        result = download_tng_structure("KEY", tmp_path)

    mock_wget.assert_not_called()
    assert result == structure_path


# ===========================================================================
# download_tng_structure — remote path
# ===========================================================================


@pytest.mark.unit
def test_download_tng_structure_remote_creates_dirs_via_sftp(tmp_path):
    """Remote path creates directories on remote host via SFTP."""
    remote = RemoteMachine(
        host="remote.host",
        username="user",
        ssh_key=tmp_path / "id_rsa",
        ssh_key_passphrase=None,
    )
    (tmp_path / "id_rsa").write_text("fake key")

    mock_sftp_instance = MagicMock()
    mock_sftp_instance.exists.return_value = False
    mock_sftp_instance.__enter__ = lambda s: mock_sftp_instance
    mock_sftp_instance.__exit__ = MagicMock(return_value=False)

    mock_pysftp_mod = MagicMock()
    mock_pysftp_mod.Connection.return_value = mock_sftp_instance

    mock_ssh = MagicMock()
    import almasim.skymodels.datasets.tng as tng_mod

    original_pysftp = tng_mod.pysftp
    try:
        tng_mod.pysftp = mock_pysftp_mod
        with patch("almasim.skymodels.datasets.tng.paramiko") as mock_paramiko:
            mock_paramiko.RSAKey.from_private_key_file.return_value = MagicMock()
            mock_paramiko.SSHClient.return_value = mock_ssh
            mock_paramiko.AutoAddPolicy = MagicMock()
            download_tng_structure("MY_KEY", tmp_path / "remote_dest", remote=remote)
    finally:
        tng_mod.pysftp = original_pysftp

    # SFTP Connection should have been used
    assert mock_pysftp_mod.Connection.called


@pytest.mark.unit
def test_download_tng_structure_remote_skips_ssh_if_file_exists(tmp_path):
    """Remote download skips SSH wget call if file already exists on remote."""
    remote = RemoteMachine(
        host="remote.host",
        username="user",
        ssh_key=tmp_path / "id_rsa",
        ssh_key_passphrase=None,
    )
    (tmp_path / "id_rsa").write_text("fake key")

    mock_sftp_instance = MagicMock()
    # All remote paths exist including the file — so no SSH wget needed
    mock_sftp_instance.exists.return_value = True
    mock_sftp_instance.__enter__ = lambda s: mock_sftp_instance
    mock_sftp_instance.__exit__ = MagicMock(return_value=False)

    mock_pysftp_mod = MagicMock()
    mock_pysftp_mod.Connection.return_value = mock_sftp_instance

    import almasim.skymodels.datasets.tng as tng_mod

    original_pysftp = tng_mod.pysftp
    try:
        tng_mod.pysftp = mock_pysftp_mod
        with patch("almasim.skymodels.datasets.tng.paramiko") as mock_paramiko:
            mock_paramiko.RSAKey.from_private_key_file.return_value = MagicMock()
            download_tng_structure("KEY", tmp_path / "dest", remote=remote)
    finally:
        tng_mod.pysftp = original_pysftp

    # SSHClient should NOT have been instantiated — file already exists
    mock_paramiko.SSHClient.assert_not_called()


# ===========================================================================
# RemoteMachine dataclass
# ===========================================================================


@pytest.mark.unit
def test_remote_machine_dataclass(tmp_path):
    """RemoteMachine stores all fields."""
    key = tmp_path / "id_rsa"
    rm = RemoteMachine(host="h", username="u", ssh_key=key, ssh_key_passphrase="pw")
    assert rm.host == "h"
    assert rm.username == "u"
    assert rm.ssh_key == key
    assert rm.ssh_key_passphrase == "pw"


@pytest.mark.unit
def test_remote_machine_passphrase_optional(tmp_path):
    """RemoteMachine passphrase defaults to None."""
    rm = RemoteMachine(host="h", username="u", ssh_key=tmp_path / "key")
    assert rm.ssh_key_passphrase is None
