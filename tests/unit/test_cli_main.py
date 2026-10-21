"""The almasim entry point, the clean passthrough and the cli_* aliases (MADIV-28 T8.2)."""

from __future__ import annotations

from types import SimpleNamespace

import click
import pytest
import typer
from typer.testing import CliRunner

from almasim import cli, cli_clean

runner = CliRunner()


def test_clean_passthrough_runs_wsclean(monkeypatch):
    """Top-level clean command should forward unknown options to WSClean."""
    captured: dict[str, object] = {}

    monkeypatch.setattr(cli_clean.shutil, "which", lambda executable: f"/usr/bin/{executable}")

    def _fake_run(command, cwd=None, check=False):
        captured["command"] = command
        captured["cwd"] = cwd
        captured["check"] = check
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli_clean.subprocess, "run", _fake_run)

    result = runner.invoke(
        cli.app,
        [
            "clean",
            "--wsclean-bin",
            "wsclean",
            "--",
            "-name",
            "img",
            "-niter",
            "1000",
            "input.ms",
        ],
    )

    assert result.exit_code == 0
    assert captured["command"] == [
        "/usr/bin/wsclean",
        "-name",
        "img",
        "-niter",
        "1000",
        "input.ms",
    ]


def test_invoke_click_command_raises_typer_exit_for_nonzero_int_result():
    """Click command return codes should map to Typer exit codes."""

    class _FakeClickCommand:
        def main(self, **kwargs):
            del kwargs
            return 9

    with pytest.raises(typer.Exit) as exc:
        cli._invoke_click_command(_FakeClickCommand(), args=[], prog_name="almasim")

    assert exc.value.exit_code == 9


def test_invoke_click_command_maps_click_exit_to_typer_exit():
    """Click Exit exceptions should be mapped to Typer Exit with same code."""

    class _FakeClickCommand:
        def main(self, **kwargs):
            del kwargs
            raise click.exceptions.Exit(3)

    with pytest.raises(typer.Exit) as exc:
        cli._invoke_click_command(_FakeClickCommand(), args=[], prog_name="almasim")

    assert exc.value.exit_code == 3


def test_cli_modules_are_the_package_modules():
    """``almasim.cli_<group>`` names the module in ``almasim.cli`` (MADIV-28 T3.1)."""
    import importlib

    for old, new in [
        ("cli_clean", "clean"),
        ("cli_image", "image"),
        ("cli_metadata", "metadata"),
        ("cli_predict", "predict"),
        ("cli_products", "products"),
        ("cli_shared", "_shared"),
        ("cli_simulation", "simulation"),
        ("cli_visibilities", "visibilities"),
    ]:
        alias = importlib.import_module(f"almasim.{old}")
        assert alias is importlib.import_module(f"almasim.cli.{new}"), old


def test_simulation_main_dir_defaults_to_the_package_data():
    """The default ``--main-dir`` is ``src/almasim``, where the data directories live."""
    import inspect

    from almasim.cli import simulation

    default = inspect.signature(simulation.simulation_run).parameters["main_dir"].default.default
    assert (default / "cli").is_dir() and default.name == "almasim"


def test_python_m_almasim_cli_runs_the_app(monkeypatch, capsys):
    """``python -m almasim.cli --help`` prints the command groups."""
    import runpy
    import sys

    monkeypatch.setattr(sys, "argv", ["almasim", "--help"])
    with pytest.raises(SystemExit) as info:
        runpy.run_module("almasim.cli", run_name="__main__")
    assert info.value.code == 0
    assert "visibilities" in capsys.readouterr().out
