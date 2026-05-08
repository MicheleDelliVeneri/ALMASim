"""Unit tests for ALMASim simulation CLI commands."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
from typer.testing import CliRunner

from almasim import cli, cli_simulation

runner = CliRunner()


class _DummyBackend:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return None


def test_simulation_csv_mode_requires_metadata_path():
    """CSV metadata mode must require --metadata-path."""
    result = runner.invoke(
        cli.app,
        [
            "simulation",
            "run",
            "--metadata-mode",
            "csv",
        ],
    )

    assert result.exit_code == 2
    assert "--metadata-path is required" in result.output


def test_simulation_invalid_backend():
    """Invalid backend values should fail fast."""
    result = runner.invoke(
        cli.app,
        [
            "simulation",
            "run",
            "--backend",
            "dask",
        ],
    )

    assert result.exit_code == 2
    assert "--backend must be one of" in result.output


def test_simulation_run_query_happy_path(monkeypatch):
    """Simulation run should execute staged workflow and print completion output."""
    metadata = pd.DataFrame({"member_ous_uid": ["uid://A"]})

    monkeypatch.setattr(cli_simulation, "query_metadata_by_science", lambda **kwargs: metadata)
    monkeypatch.setattr(cli_simulation.astro, "get_line_info", lambda main_dir: (100.0, None))
    monkeypatch.setattr(
        cli_simulation,
        "sample_given_redshift",
        lambda metadata, n, rest_frequency, extended, zmax: metadata,
    )

    monkeypatch.setattr(
        cli_simulation.SimulationParams,
        "from_metadata_row",
        lambda *args, **kwargs: SimpleNamespace(ml_dataset_path="/tmp/mock_ml.h5"),
    )
    monkeypatch.setattr(cli_simulation, "create_backend", lambda *args, **kwargs: _DummyBackend())
    monkeypatch.setattr(
        cli_simulation,
        "generate_clean_cube",
        lambda *args, **kwargs: SimpleNamespace(model_cube=SimpleNamespace(shape=(1, 2, 3))),
    )
    monkeypatch.setattr(
        cli_simulation,
        "simulate_observation",
        lambda *args, **kwargs: {
            "dirty_cube": SimpleNamespace(shape=(1, 2, 3)),
            "uv_mask_cube": SimpleNamespace(shape=(1, 2, 3)),
        },
    )
    monkeypatch.setattr(
        cli_simulation,
        "export_results",
        lambda *args, **kwargs: {
            "ml_dataset_path": "/tmp/mock_ml.h5",
            "dirty_cube": "dummy",
        },
    )

    result = runner.invoke(
        cli.app,
        [
            "simulation",
            "run",
            "--science-keyword",
            "Galaxies",
        ],
    )

    assert result.exit_code == 0
    assert "Run complete" in result.output
    assert "ML shard written to: /tmp/mock_ml.h5" in result.output
