"""ALMASim CLI entrypoint and command registration."""

from __future__ import annotations

import typer

from .cli_clean import clean_command
from .cli_image import image_app
from .cli_metadata import metadata_app
from .cli_products import products_app
from .cli_simulation import simulation_app

app = typer.Typer(
    help="ALMASim command-line interface.",
    no_args_is_help=True,
    add_completion=False,
)
app.add_typer(metadata_app, name="metadata")
app.add_typer(products_app, name="products")
app.add_typer(simulation_app, name="simulation")
app.add_typer(image_app, name="image")
app.command(
    "clean",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)(clean_command)


def main() -> None:
    """Run the ALMASim CLI application."""
    app()


if __name__ == "__main__":
    main()
