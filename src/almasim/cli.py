"""ALMASim CLI entrypoint and lazy command registration."""

from __future__ import annotations

from importlib import import_module

import click
import typer

app = typer.Typer(
    help="ALMASim command-line interface.",
    no_args_is_help=True,
    add_completion=False,
)


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
    module = import_module(module_name, package=__package__)
    subapp = getattr(module, app_name)
    command = typer.main.get_command(subapp)
    args = list(ctx.args) or ["--help"]

    # Typer may emit a plain Command for single-command apps; in that case,
    # users still type an explicit subcommand token (e.g. `simulation run ...`).
    if not isinstance(command, click.MultiCommand) and args and args[0] == command.name:
        args = args[1:]

    _invoke_click_command(command, args=args, prog_name=prog_name)


def _forward_clean_command(*, ctx: typer.Context) -> None:
    """Lazy-load and invoke the WSClean passthrough command."""
    module = import_module(".cli_clean", package=__package__)
    clean_app = typer.Typer(add_completion=False)
    clean_app.command(
        "clean",
        context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    )(getattr(module, "clean_command"))
    command = typer.main.get_command(clean_app)
    _invoke_click_command(command, args=list(ctx.args) or ["--help"], prog_name="almasim clean")


@app.command(
    "metadata",
    help="Metadata query commands.",
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
        "help_option_names": [],
    },
)
def metadata_command(ctx: typer.Context) -> None:
    """Run metadata subcommands."""
    _forward_typer_subapp(
        ".cli_metadata",
        "metadata_app",
        ctx=ctx,
        prog_name="almasim metadata",
    )


@app.command(
    "products",
    help="Data product resolution and download commands.",
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
        "help_option_names": [],
    },
)
def products_command(ctx: typer.Context) -> None:
    """Run products subcommands."""
    _forward_typer_subapp(
        ".cli_products",
        "products_app",
        ctx=ctx,
        prog_name="almasim products",
    )


@app.command(
    "simulation",
    help="Simulation workflow commands.",
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
        "help_option_names": [],
    },
)
def simulation_command(ctx: typer.Context) -> None:
    """Run simulation subcommands."""
    _forward_typer_subapp(
        ".cli_simulation",
        "simulation_app",
        ctx=ctx,
        prog_name="almasim simulation",
    )


@app.command(
    "visibilities",
    help="Visibility processing commands.",
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
        "help_option_names": [],
    },
)
def visibilities_command(ctx: typer.Context) -> None:
    """Run visibilities subcommands."""
    _forward_typer_subapp(
        ".cli_visibilities",
        "visibilities_app",
        ctx=ctx,
        prog_name="almasim visibilities",
    )


@app.command(
    "predict",
    help="Model prediction commands.",
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
        "help_option_names": [],
    },
)
def predict_command(ctx: typer.Context) -> None:
    """Run predict subcommands."""
    _forward_typer_subapp(
        ".cli_predict",
        "predict_app",
        ctx=ctx,
        prog_name="almasim predict",
    )


@app.command(
    "image",
    help="Data product batch imaging.",
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
        "help_option_names": [],
    },
)
def image_command(ctx: typer.Context) -> None:
    """Run image subcommands."""
    _forward_typer_subapp(
        ".cli_image",
        "image_app",
        ctx=ctx,
        prog_name="almasim image",
    )


@app.command(
    "clean",
    help="Run WSClean and forward all extra arguments unchanged.",
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
        "help_option_names": [],
    },
)
def clean_command_proxy(ctx: typer.Context) -> None:
    """Run clean passthrough command."""
    _forward_clean_command(ctx=ctx)


def main() -> None:
    """Run the ALMASim CLI application."""
    app()


if __name__ == "__main__":
    main()
