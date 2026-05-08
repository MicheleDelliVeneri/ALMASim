"""ALMASim CLI entrypoint and command registration."""

from __future__ import annotations

import typer

from .cli_metadata import metadata_app
from .cli_products import products_app

app = typer.Typer(
    help="ALMASim command-line interface.",
    no_args_is_help=True,
    add_completion=False,
)
app.add_typer(metadata_app, name="metadata")
app.add_typer(products_app, name="products")


def main() -> None:
    """Run the ALMASim CLI application."""
    app()


if __name__ == "__main__":
    main()
