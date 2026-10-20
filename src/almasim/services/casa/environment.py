"""Point CASA at its runtime data, a per-process site config and a log (MADIV-28 T2.5)."""

from __future__ import annotations

import atexit
import os
import socket
from pathlib import Path


def _site_config_name() -> str:
    """Site config file name unique to this process, also across NFS clients."""
    host = socket.gethostname().split(".", 1)[0] or "localhost"
    return f"casasiteconfig-{host}-{os.getpid()}.py"


def _write_atomically(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` so that no reader can ever observe it partially written."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


_EXIT_CLEANUP: set[Path] = set()


def _remove_at_exit(path: Path) -> None:
    """Delete ``path`` when the interpreter exits (registered once per path)."""
    if path in _EXIT_CLEANUP:
        return
    _EXIT_CLEANUP.add(path)
    atexit.register(path.unlink, missing_ok=True)


def configure_casa_environment(
    output_root: str | os.PathLike[str],
    casa_data: str | os.PathLike[str],
    workspace_root: str | os.PathLike[str] | None = None,
    log_file: str | os.PathLike[str] | None = None,
    log_to_terminal: bool = False,
) -> Path:
    """Create and point CASA at a local site config and Matplotlib cache.

    ``log_file`` redirects the CASA logger away from ``casa-<timestamp>.log`` in
    the current directory; ``log_to_terminal`` additionally echoes every CASA log
    message to the console so long-running tasks show progress.

    The site config is **per process**: it is written to
    ``.casa-config/casasiteconfig-<host>-<pid>.py``, atomically (temp file +
    rename), exported through ``CASASITECONFIG`` unconditionally, and removed
    at interpreter exit. It used to be one shared ``casasiteconfig.py`` that
    every worker truncated and rewrote; with eight children importing casatools
    concurrently, one would read the file while another was rewriting it, see
    an empty config, fall back to ``~/.casa/data`` and die with ``measures data
    is not available`` before doing any work. The content is also per UID (the
    ``logfile`` line), so a shared file was wrong even without the race.
    """
    output_path = Path(output_root).expanduser().resolve()
    casa_data_path = Path(casa_data).expanduser().resolve()
    workspace_path = (
        Path(workspace_root).expanduser().resolve() if workspace_root is not None else output_path
    )

    workspace_path.mkdir(parents=True, exist_ok=True)

    mpl_config = workspace_path / ".matplotlib"
    casa_config_dir = workspace_path / ".casa-config"
    casa_site_config = casa_config_dir / _site_config_name()

    casa_data_path.mkdir(parents=True, exist_ok=True)
    mpl_config.mkdir(parents=True, exist_ok=True)
    casa_config_dir.mkdir(parents=True, exist_ok=True)

    config_lines = [
        f"measurespath = {str(casa_data_path)!r}",
        "data_auto_update = False",
        "measures_auto_update = False",
    ]
    if log_file is not None:
        log_path = Path(log_file).expanduser().resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        config_lines.append(f"logfile = {str(log_path)!r}")
    config_lines.append(f"log2term = {bool(log_to_terminal)!r}")
    _write_atomically(casa_site_config, "\n".join(config_lines) + "\n")
    _remove_at_exit(casa_site_config)

    # Not setdefault: casatools reads this variable at import time and the
    # config we just wrote is the only one that describes *this* process.
    os.environ["CASASITECONFIG"] = str(casa_site_config)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    return casa_data_path
