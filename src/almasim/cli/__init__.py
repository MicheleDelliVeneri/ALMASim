"""ALMASim CLI entrypoint and lazy command registration.

The command groups live in this package (``image``, ``products``, ...); the
old ``almasim.cli_<group>`` aliases were removed in MADIV-28 T8.4.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

import click
import typer

app = typer.Typer(
    help="ALMASim command-line interface.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def _setup() -> None:
    """ALMASim command-line interface."""
    from ._shared import configure_logging

    configure_logging()


def _invoke_click_command(command: click.Command, *, args: list[str], prog_name: str) -> None:
    """Invoke a click command while preserving Typer-style exit behavior."""
    try:
        result = command.main(
            args=args,
            prog_name=prog_name,
            standalone_mode=False,
        )
    except click.exceptions.Exit as exc:
        raise typer.Exit(code=exc.exit_code) from exc

    if isinstance(result, int) and result != 0:
        raise typer.Exit(code=result)


def _forward_typer_subapp(
    module_name: str,
    app_name: str,
    *,
    ctx: typer.Context,
    prog_name: str,
) -> None:
    """Lazy-load a Typer sub-app and forward CLI args to it."""
    from ._options import deprecated_flag_warnings

    module = import_module(module_name, package=__package__)
    subapp = getattr(module, app_name)
    command = typer.main.get_command(subapp)
    args = list(ctx.args) or ["--help"]
    for warning in deprecated_flag_warnings(args):
        typer.echo(warning, err=True)

    # Typer may emit a plain Command for single-command apps; in that case,
    # users still type an explicit subcommand token (e.g. `simulation run ...`).
    if not isinstance(command, click.MultiCommand) and args and args[0] == command.name:
        args = args[1:]

    _invoke_click_command(command, args=args, prog_name=prog_name)


def _forward_clean_command(*, ctx: typer.Context) -> None:
    """Lazy-load and invoke the WSClean passthrough command."""
    module = import_module(".clean", package=__package__)
    clean_app = typer.Typer(add_completion=False)
    clean_app.command(
        "clean",
        context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    )(module.clean_command)
    command = typer.main.get_command(clean_app)
    _invoke_click_command(command, args=list(ctx.args) or ["--help"], prog_name="almasim clean")


# (command, module, Typer app, help): each group's module is imported only
# when the group runs, so ``almasim --help`` stays fast.
_GROUPS = (
    ("metadata", ".metadata", "metadata_app", "Metadata query commands."),
    ("products", ".products", "products_app", "Data product resolution and download commands."),
    ("simulation", ".simulation", "simulation_app", "Simulation workflow commands."),
    ("visibilities", ".visibilities", "visibilities_app", "Visibility processing commands."),
    ("predict", ".predict", "predict_app", "Model prediction commands."),
    ("image", ".image", "image_app", "Data product batch imaging."),
)

_FORWARDED = {
    "allow_extra_args": True,
    "ignore_unknown_options": True,
    "help_option_names": [],
}


def _register_group(name: str, module_name: str, app_name: str, help_text: str) -> None:
    def forward(ctx: typer.Context) -> None:
        _forward_typer_subapp(module_name, app_name, ctx=ctx, prog_name=f"almasim {name}")

    forward.__name__ = f"{name}_command"
    forward.__doc__ = f"Run {name} subcommands."
    app.command(name, help=help_text, context_settings=_FORWARDED)(forward)


for _group in _GROUPS:
    _register_group(*_group)


@app.command(
    "clean",
    help="Run WSClean and forward all extra arguments unchanged.",
    context_settings=_FORWARDED,
)
def clean_command_proxy(ctx: typer.Context) -> None:
    """Run clean passthrough command."""
    _forward_clean_command(ctx=ctx)


@app.command("doctor", help="Check that the job profile works on this machine.")
def doctor_command(
    slurm: bool = typer.Option(
        False, "--slurm", help="Run the checks on a compute node (with srun), as a job sees them."
    ),
    site_profile: Path | None = typer.Option(
        None, "--site-profile", help="Site profile (default: $ALMASIM_SITE_PROFILE)."
    ),
    profile: Path | None = typer.Option(
        None,
        "--profile",
        help="User profile (default: $ALMASIM_PROFILE, else ~/.config/almasim/almasim.toml).",
    ),
) -> None:
    """Check the runs root, Python, CASA, WSClean and Slurm the profile names."""
    from .doctor import doctor

    doctor(slurm, site_profile, profile)


def main() -> None:
    """Run the ALMASim CLI application."""
    app()
