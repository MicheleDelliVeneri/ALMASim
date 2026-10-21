"""The `visibilities` extract/unpack/calibrate commands are the `products` ones (MADIV-28 T3.4).

One implementation each, so the two names cannot drift apart again (T1.14
found the aliases refusing `--slurm-workers 0`, which the products
commands accept as one worker per UID).
"""

from __future__ import annotations

import pytest
import typer

from almasim.cli import image as cli_image
from almasim.cli import products as cli_products
from almasim.cli import visibilities as cli_visibilities

pytestmark = pytest.mark.unit

ALIASES = {
    "extract": cli_products.products_extract,
    "unpack": cli_products.products_unpack,
    "calibrate": cli_products.products_calibrate,
    "spw-overview": cli_image.derive_parameters,
}


def test_visibilities_commands_are_the_single_implementations():
    registered = {
        cmd.name: cmd.callback for cmd in cli_visibilities.visibilities_app.registered_commands
    }
    for name, function in ALIASES.items():
        assert registered[name] is function, name


@pytest.mark.parametrize("command", ["unpack", "calibrate"])
def test_zero_workers_is_accepted(tmp_path, command):
    group = typer.main.get_group(cli_visibilities.visibilities_app)
    ctx = group.commands[command].make_context(
        command, ["--input-root", str(tmp_path), "--slurm-workers", "0"]
    )
    assert ctx.params["slurm_workers"] == 0
