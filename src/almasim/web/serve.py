"""Start, find and stop the console server (web console plan, S2.1).

``~/.almasim/serve.json`` (mode 0600) records the running server: its pid,
where it listens and its token, so ``almasim serve --status`` can show the
link again and ``--stop`` can end it.
"""

from __future__ import annotations

import json
import os
import secrets
import signal
import socket
import time
from dataclasses import asdict, dataclass
from pathlib import Path

STATE_DIR = Path("~/.almasim")
STATE_NAME = "serve.json"
SOCKET_NAME = "serve.sock"
LOCAL_PORT = 8765  # the port the ssh line forwards on the laptop
# For a console in a container (docker-compose.yml): a token fixed in advance,
# so the web app's dev server can be given it, and the address to listen on
# with --port, since 127.0.0.1 inside a container is unreachable from outside.
TOKEN_ENV = "ALMASIM_TOKEN"
HOST_ENV = "ALMASIM_SERVE_HOST"
MIN_TOKEN = 16


@dataclass(frozen=True)
class ServerInfo:
    pid: int
    token: str
    host: str  # this machine, for the ssh line
    socket: str | None = None
    port: int | None = None
    bind: str = "127.0.0.1"  # the address it listens on, with a port

    def url(self, local_port: int = LOCAL_PORT) -> str:
        port = local_port if self.socket else self.port
        return f"http://localhost:{port}/?token={self.token}"

    def ssh_line(self, local_port: int = LOCAL_PORT) -> str:
        user = os.environ.get("USER", "you")
        target = self.socket if self.socket else f"localhost:{self.port}"
        port = local_port if self.socket else self.port
        return f"ssh -N -L {port}:{target} {user}@{self.host}"


def state_dir() -> Path:
    path = STATE_DIR.expanduser()
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.chmod(0o700)  # holds the socket and the token: this user only
    return path


def new_token() -> str:
    return secrets.token_urlsafe(32)


def write_info(info: ServerInfo) -> Path:
    path = state_dir() / STATE_NAME
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(asdict(info), f)
    path.chmod(0o600)
    return path


def read_info() -> ServerInfo | None:
    """The recorded server, if its process still runs."""
    path = STATE_DIR.expanduser() / STATE_NAME
    try:
        info = ServerInfo(**json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError):
        return None
    try:
        os.kill(info.pid, 0)
    except ProcessLookupError:
        return None
    except PermissionError:  # another user's pid: not our server
        return None
    return info


def clear_info() -> None:
    (STATE_DIR.expanduser() / STATE_NAME).unlink(missing_ok=True)


def stop(timeout: float = 10.0) -> ServerInfo | None:
    """Stop the recorded server, wait for it to exit, and remove what it leaves."""
    info = read_info()
    if info is not None:
        os.kill(info.pid, signal.SIGTERM)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                os.kill(info.pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        if info.socket:
            Path(info.socket).unlink(missing_ok=True)
    clear_info()
    return info


def this_host() -> str:
    return socket.getfqdn()
