"""``almasim update-shared``: update the team's shared install from a git ref."""

from __future__ import annotations

import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import typer

from ._shared import fail


def update_shared(
    ref: str, prefix: Path | None, repo: str | None, template: str | None, force: bool
) -> None:
    from almasim.jobs import shared_install as si
    from almasim.jobs.profile import ProfileError, load_profile
    from almasim.jobs.service import list_runs, short_id, unfinished

    prefix = prefix.resolve() if prefix is not None else si.find_prefix()
    if prefix is None:
        fail("this almasim is not from a shared install: pass --prefix")
    if not (prefix / "almasim.sh").is_file():
        fail(f"{prefix} is not a shared install (no almasim.sh)")
    try:
        source = si.resolve_source(si.read_source(prefix), repo, template)
        profile = load_profile(prefix / "site.toml")
    except (si.SharedInstallError, ProfileError) as exc:
        fail(str(exc))
    missing = [tool for tool in si.TOOLS if shutil.which(tool) is None]
    if missing:
        fail(f"the update job needs {', '.join(missing)} on the PATH")

    # A job loads almasim from the environment while it runs: files replaced
    # under it can break it.
    busy = unfinished(list_runs(profile))
    if busy and not force:
        for run in busy:
            typer.echo(f"  {short_id(run.job.id)}  {run.job.owner}  {run.job.command}", err=True)
        fail(f"{len(busy)} run(s) under {profile.runs_root} are unfinished: wait, or pass --force")

    os.umask(0o002)  # the group shares the install, and its updates folder
    work = prefix / "updates" / datetime.now().strftime("%Y%m%d-%H%M%S")
    work.mkdir(parents=True)
    script = work / "update.sh"
    script.write_text(si.job_script(prefix, work, source, ref), encoding="utf-8")
    log = work / "update.log"
    typer.echo(f"Updating {prefix} to {ref} of {source.repo}, as a Slurm job; log: {log}")
    result = subprocess.run(si.sbatch_command(profile, script, log), capture_output=True, text=True)
    if log.is_file():
        typer.echo("\n".join(log.read_text(encoding="utf-8").splitlines()[-25:]))
    if result.returncode != 0:
        typer.echo(result.stderr.strip(), err=True)
        fail(f"the update failed (job {result.stdout.strip() or '?'}): see {log}")
    typer.echo(
        "Updated. Running consoles keep the old version until restarted "
        "(almasim serve --stop, then almasim serve)."
    )
