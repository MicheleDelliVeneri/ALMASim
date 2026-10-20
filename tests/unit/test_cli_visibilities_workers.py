"""`visibilities unpack/calibrate --slurm-workers 0` is accepted (MADIV-28 T1.14).

The `products` commands they forward to treat 0 as one worker per UID; the
aliases used to refuse it.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from almasim import cli_products, cli_visibilities

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("command", "target"), [("unpack", "products_unpack"), ("calibrate", "products_calibrate")]
)
def test_zero_workers_is_forwarded(monkeypatch, tmp_path, command, target):
    captured = {}
    monkeypatch.setattr(cli_products, target, lambda **kwargs: captured.update(kwargs))
    result = CliRunner().invoke(
        cli_visibilities.visibilities_app,
        [command, "--input-root", str(tmp_path), "--slurm-workers", "0"],
    )
    assert result.exit_code == 0, result.output
    assert captured["slurm_workers"] == 0
