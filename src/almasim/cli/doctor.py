"""``almasim doctor``: check the job profile on this machine (web console plan, J1.3)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import typer

from ._shared import fail


def doctor(slurm: bool, site_profile: Path | None, user_profile: Path | None) -> None:
    from almasim.jobs.doctor import format_report, passed, run_checks, srun_command
    from almasim.jobs.profile import ProfileError, load_profile

    try:
        profile = load_profile(site_profile, user_profile)
    except ProfileError as exc:
        fail(str(exc))

    if slurm:
        argv = ["doctor"]
        if site_profile is not None:
            argv += ["--site-profile", str(site_profile)]
        if user_profile is not None:
            argv += ["--profile", str(user_profile)]
        cmd = srun_command(profile, argv)
        typer.echo(f"Running on a compute node: {' '.join(cmd)}", err=True)
        raise typer.Exit(code=subprocess.call(cmd))

    checks = run_checks(profile)
    typer.echo(format_report(checks))
    if not passed(checks):
        raise typer.Exit(code=1)
