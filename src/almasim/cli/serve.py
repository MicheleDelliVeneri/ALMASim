"""``almasim serve``: the web console on this machine (web console plan, S2.1)."""

from __future__ import annotations

import os
from pathlib import Path

import typer

from ._shared import fail

# sockaddr_un holds 104 bytes on macOS, 108 on Linux, the terminator included.
SOCKET_PATH_MAX = 103


def _print_access(info) -> None:
    typer.echo("ALMASim web console")
    if info.socket:
        typer.echo(f"  socket:  {info.socket} (only you can open it)")
        typer.echo("On your laptop, open the tunnel:")
        typer.echo(f"  {info.ssh_line()}")
        typer.echo("then browse to:")
    else:
        typer.echo(f"  listening on 127.0.0.1:{info.port}")
        typer.echo("From your laptop, tunnel the port, then browse to the link:")
        typer.echo(f"  {info.ssh_line()}")
    typer.echo(f"  {info.url()}")


def serve(port: int | None, status: bool, stop_server: bool) -> None:
    try:
        import uvicorn

        from almasim.web.app import create_app
    except ImportError:
        fail("The web console needs the web extra: pip install 'almasim[web]'")
    from almasim.web import serve as state

    if status:
        info = state.read_info()
        if info is None:
            fail("No console is running (start one with: almasim serve)")
        _print_access(info)
        return
    if stop_server:
        info = state.stop()
        typer.echo("Stopped the console." if info else "No console was running.")
        return

    running = state.read_info()
    if running is not None:
        typer.echo("A console is already running:", err=True)
        _print_access(running)
        raise typer.Exit(code=1)

    token = state.new_token()
    folder = state.state_dir()
    sock = None if port else folder / state.SOCKET_NAME
    if sock is not None:
        if len(str(sock).encode()) > SOCKET_PATH_MAX:
            fail(f"{sock} is too long for a socket path; use --port instead")
        sock.unlink(missing_ok=True)  # left over from a server that did not exit cleanly
    info = state.ServerInfo(
        pid=os.getpid(),
        token=token,
        host=state.this_host(),
        socket=str(sock) if sock else None,
        port=port,
    )
    state.write_info(info)
    _print_access(info)
    os.umask(0o077)  # the socket file: this user only
    try:
        app = create_app(token)
        if port is None:
            uvicorn.run(app, uds=str(sock), log_level="warning")
        else:
            uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
    finally:
        state.clear_info()
        if sock is not None:
            Path(sock).unlink(missing_ok=True)
