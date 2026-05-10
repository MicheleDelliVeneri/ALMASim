"""WSClean passthrough command for ALMASim CLI."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path

import typer


def _bundled_wsclean_candidates() -> list[Path]:
    repo_root = Path(__file__).resolve().parents[2]
    return [
        repo_root / "third_party" / "wsclean" / "build" / "wsclean",
        repo_root / "bin" / "wsclean",
    ]


def clean_command(
    ctx: typer.Context,
    wsclean_bin: str = typer.Option(
        "wsclean",
        "--wsclean-bin",
        help="Path or executable name for WSClean binary.",
    ),
    cwd: Path | None = typer.Option(
        None,
        "--cwd",
        help="Optional working directory for WSClean execution.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print the WSClean command without running it.",
    ),
) -> None:
    """Run WSClean and forward all extra arguments unchanged."""
    forwarded_args = list(ctx.args)

    if not forwarded_args:
        typer.echo(
            "No WSClean arguments provided. Use 'almasim clean -- --help' to see WSClean options.",
            err=True,
        )
        raise typer.Exit(code=2)

    requested_executable = os.environ.get("ALMASIM_WSCLEAN_BIN", wsclean_bin)
    executable = requested_executable
    if not os.path.isabs(executable):
        resolved = shutil.which(executable)
        if resolved:
            executable = resolved

    if shutil.which(executable) is None and not Path(executable).exists():
        for candidate in _bundled_wsclean_candidates():
            if candidate.is_file() and os.access(candidate, os.X_OK):
                executable = str(candidate)
                break

    if shutil.which(executable) is None and not Path(executable).exists():
        typer.echo(
            f"WSClean executable not found: {requested_executable}. "
            "Install WSClean, set ALMASIM_WSCLEAN_BIN, or pass --wsclean-bin /path/to/wsclean.",
            err=True,
        )
        raise typer.Exit(code=127)

    command = [executable, *forwarded_args]
    typer.echo("Running: " + " ".join(shlex.quote(part) for part in command))

    if dry_run:
        return

    run_cwd = cwd.expanduser().resolve() if cwd is not None else None
    try:
        completed = subprocess.run(
            command,
            cwd=run_cwd,
            check=False,
        )
    except OSError as exc:
        typer.echo(f"Failed to start WSClean: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if completed.returncode != 0:
        raise typer.Exit(code=completed.returncode)
