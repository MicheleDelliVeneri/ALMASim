"""TNG dataset download utilities."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import paramiko

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

try:  # pragma: no cover - exercised indirectly
    import pysftp
except Exception as exc:  # pragma: no cover - import-time guard
    pysftp = None
    _PYSFTP_IMPORT_ERROR = exc
else:
    _PYSFTP_IMPORT_ERROR = None

TNG_SIMULATION_URL = "http://www.tng-project.org/api/TNG100-1/files/simulation.hdf5"


@dataclass
class RemoteMachine:
    """Configuration for remote machine access."""

    host: str
    username: str
    ssh_key: Path
    ssh_key_passphrase: Optional[str] = None


def _ensure_directory(path: Path) -> Path:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def _require_pysftp() -> "pysftp":  # type: ignore[name-defined]
    """Require pysftp module, raising error if not available."""
    if pysftp is None:
        raise RuntimeError(
            "pysftp (and its paramiko dependency) could not be imported. "
            "Remote dataset downloads currently require pysftp; install "
            "compatible versions (e.g. paramiko<4) or update the dependency. "
            f"Original import error: {_PYSFTP_IMPORT_ERROR}"
        )
    return pysftp


def _wget_tng(api_key: str, destination_file: Path) -> None:
    """Download TNG structure file using wget."""
    cmd = [
        "wget",
        "-nv",
        "--content-disposition",
        f"--header=API-Key:{api_key}",
        "-O",
        str(destination_file),
        TNG_SIMULATION_URL,
    ]
    subprocess.run(cmd, check=True)


def download_tng_structure(
    api_key: str,
    destination: Path | str,
    remote: Optional[RemoteMachine] = None,
) -> Path:
    """Download the Illustris TNG structure file locally or to a remote host."""
    destination = Path(destination).expanduser().resolve()
    if remote is None:
        structure_path = destination / "TNG100-1" / "simulation.hdf5"
        _ensure_directory(structure_path.parent)
        _ensure_directory(destination / "TNG100-1" / "output")
        _ensure_directory(destination / "TNG100-1" / "postprocessing" / "offsets")
        if not structure_path.exists():
            _wget_tng(api_key, structure_path)
        return structure_path

    # Remote download path via SFTP/SSH
    ssh_key = paramiko.RSAKey.from_private_key_file(
        str(remote.ssh_key),
        password=remote.ssh_key_passphrase,
    )
    remote_base = Path(destination)
    pysftp_mod = _require_pysftp()
    with pysftp_mod.Connection(
        remote.host,
        username=remote.username,
        private_key=str(remote.ssh_key),
        private_key_pass=remote.ssh_key_passphrase,
    ) as sftp:
        for rel in [
            Path("TNG100-1"),
            Path("TNG100-1") / "output",
            Path("TNG100-1") / "postprocessing",
            Path("TNG100-1") / "postprocessing" / "offsets",
        ]:
            remote_dir = os.path.join(str(remote_base), str(rel))
            if not sftp.exists(remote_dir):
                sftp.mkdir(remote_dir)
        remote_file = os.path.join(str(remote_base), "TNG100-1", "simulation.hdf5")
        if not sftp.exists(remote_file):
            ssh = paramiko.SSHClient()
            ssh.load_system_host_keys()
            ssh.connect(remote.host, username=remote.username, pkey=ssh_key)
            cmd = (
                f"wget -nv --content-disposition --header=API-Key:{api_key} "
                f"-O {remote_file} {TNG_SIMULATION_URL}"
            )
            ssh.exec_command(cmd)
            ssh.close()
        return Path(remote_file)
