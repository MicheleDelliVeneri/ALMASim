"""`almasim submit` and `almasim jobs`: runs as jobs from the terminal (web console plan, J1.7).

The web console uses the same runs: whatever is submitted here shows up
there, and the other way round.
"""

from __future__ import annotations

import shlex
import sys
import time
from datetime import datetime
from pathlib import Path

import typer

from ._shared import fail

jobs_app = typer.Typer(
    help="List, follow and control runs submitted with `almasim submit` or the web console.",
    no_args_is_help=True,
)

ProfileOpt = typer.Option(
    None, "--profile", help="User profile (default: $ALMASIM_PROFILE or ~/.config/almasim)."
)


def _profile(user_profile: Path | None):
    from almasim.jobs.profile import ProfileError, load_profile

    try:
        return load_profile(user=user_profile)
    except ProfileError as exc:
        fail(str(exc))


def submit(
    command: list[str],
    *,
    partition: str | None,
    account: str | None,
    cpus: int | None,
    memory: str | None,
    time_limit: str | None,
    title: str,
    wait: bool,
    user_profile: Path | None,
) -> None:
    from almasim.jobs import ResourcesError
    from almasim.jobs.executors.slurm import SlurmError
    from almasim.jobs.policy import PolicyError
    from almasim.jobs.service import short_id, submit_command
    from almasim.jobs.service import wait as wait_for

    profile = _profile(user_profile)
    resources = {
        "partition": partition,
        "account": account,
        "cpus": cpus,
        "memory": memory,
        "time": time_limit,
    }
    try:
        run = submit_command(command, profile, resources=resources, title=title)
    except (PolicyError, ResourcesError, SlurmError) as exc:
        fail(str(exc))
    where = f"Slurm job {run.job.slurm_job_id}" if run.job.slurm_job_id else "the server node"
    typer.echo(f"Run {short_id(run.job.id)} on {where}: almasim {shlex.join(run.job.argv)}")
    typer.echo(f"  folder: {run.run_dir}")
    if wait:
        final = wait_for(run.run_dir)
        typer.echo(f"  {final.state.value}{f': {final.reason}' if final.reason else ''}")
        if final.state.value != "succeeded":
            raise typer.Exit(code=1)


def _age(created: float) -> str:
    return datetime.fromtimestamp(created).strftime("%Y-%m-%d %H:%M")


@jobs_app.command("list")
def list_command(
    everyone: bool = typer.Option(False, "--all", help="Include the team's runs, not only yours."),
    limit: int = typer.Option(20, "--limit", min=1, help="How many runs to show, newest first."),
    user_profile: Path | None = ProfileOpt,
) -> None:
    """List runs, newest first."""
    import getpass

    from almasim.jobs.service import list_runs, short_id

    profile = _profile(user_profile)
    runs = list_runs(profile, owner=None if everyone else getpass.getuser(), limit=limit)
    if not runs:
        typer.echo(f"No runs under {profile.runs_root}")
        return
    rows = [("RUN", "STATE", "COMMAND", "OWNER", "WHERE", "CREATED")]
    for r in runs:
        where = f"slurm {r.job.slurm_job_id}" if r.job.slurm_job_id else r.job.executor
        rows.append(
            (
                short_id(r.job.id),
                r.state.value,
                r.job.command,
                r.job.owner,
                where,
                _age(r.job.created_at),
            )
        )
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    for row in rows:
        typer.echo("  ".join(cell.ljust(w) for cell, w in zip(row, widths, strict=True)).rstrip())


@jobs_app.command("show")
def show_command(
    run_id: str = typer.Argument(..., help="Run id, or its first or last characters."),
    user_profile: Path | None = ProfileOpt,
) -> None:
    """Show one run: its command, resources, state and files."""
    from almasim.jobs.executors.slurm import poll
    from almasim.jobs.service import RunError, find_run, view

    profile = _profile(user_profile)
    try:
        run_dir = find_run(profile, run_id)
    except RunError as exc:
        fail(str(exc))
    job_id = view(run_dir).job.slurm_job_id
    run = view(run_dir, poll([job_id]).get(job_id) if job_id else None)
    job, status = run.job, run.status
    r = job.resources
    lines = [
        f"run       {job.id}",
        f"state     {run.state.value}{f' ({run.reason})' if run.reason else ''}",
        f"command   almasim {shlex.join(job.argv)}",
        f"owner     {job.owner}",
        f"executor  {job.executor}{f' (Slurm job {job.slurm_job_id})' if job.slurm_job_id else ''}",
        f"resources {r.cpus} CPU, {r.memory}, {r.time}"
        + (f", partition {r.partition}" if r.partition else ""),
        f"created   {_age(job.created_at)}",
    ]
    if status.host:
        lines.append(f"host      {status.host}")
    if status.exit_code is not None:
        lines.append(
            f"exit code {status.exit_code}{f' ({status.signal})' if status.signal else ''}"
        )
    lines.append(f"folder    {run_dir}")
    typer.echo("\n".join(lines))


@jobs_app.command("logs")
def logs_command(
    run_id: str = typer.Argument(..., help="Run id, or its first or last characters."),
    stderr: bool = typer.Option(False, "--stderr", help="Show stderr.log instead of stdout.log."),
    follow: bool = typer.Option(False, "--follow", "-f", help="Keep printing until the run ends."),
    user_profile: Path | None = ProfileOpt,
) -> None:
    """Print a run's output."""
    from almasim.jobs.service import RunError, find_run, view
    from almasim.jobs.wrapper import STDERR_NAME, STDOUT_NAME

    profile = _profile(user_profile)
    try:
        run_dir = find_run(profile, run_id)
    except RunError as exc:
        fail(str(exc))
    path = run_dir / (STDERR_NAME if stderr else STDOUT_NAME)
    offset = 0
    while True:
        if path.exists():
            with path.open("rb") as f:
                f.seek(offset)
                chunk = f.read()
            offset += len(chunk)
            sys.stdout.write(chunk.decode("utf-8", errors="replace"))
            sys.stdout.flush()
        if not follow or view(run_dir).state.finished:
            break
        time.sleep(2)


@jobs_app.command("cancel")
def cancel_command(
    run_id: str = typer.Argument(..., help="Run id, or its first or last characters."),
    user_profile: Path | None = ProfileOpt,
) -> None:
    """Stop a run (yours only)."""
    from almasim.jobs.executors.slurm import SlurmError
    from almasim.jobs.service import RunError, cancel_run, find_run, short_id

    profile = _profile(user_profile)
    try:
        run = cancel_run(find_run(profile, run_id))
    except (RunError, SlurmError) as exc:
        fail(str(exc))
    typer.echo(f"Cancelling run {short_id(run.job.id)}")


@jobs_app.command("resubmit")
def resubmit_command(
    run_id: str = typer.Argument(..., help="Run id, or its first or last characters."),
    user_profile: Path | None = ProfileOpt,
) -> None:
    """Run the same command again, with the same resources, as a new run."""
    from almasim.jobs.executors.slurm import SlurmError
    from almasim.jobs.service import RunError, find_run, resubmit, short_id

    profile = _profile(user_profile)
    try:
        run = resubmit(find_run(profile, run_id), profile)
    except (RunError, SlurmError) as exc:
        fail(str(exc))
    typer.echo(f"Run {short_id(run.job.id)} resubmitted: almasim {shlex.join(run.job.argv)}")
