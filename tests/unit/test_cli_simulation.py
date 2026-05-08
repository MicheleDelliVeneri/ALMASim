"""Unit tests for ALMASim simulation CLI commands."""

from __future__ import annotations

import re
from types import SimpleNamespace

import pandas as pd
from typer.testing import CliRunner

from almasim import cli, cli_simulation

runner = CliRunner()


_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


class _DummyBackend:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return None


def _write_metadata_csv(tmp_path, rows: list[dict]) -> str:
    metadata_path = tmp_path / "metadata.csv"
    pd.DataFrame(rows).to_csv(metadata_path, index=False)
    return str(metadata_path)


def test_simulation_requires_metadata_path():
    """Simulation runs must require --metadata-path."""
    result = runner.invoke(
        cli.app,
        [
            "simulation",
            "run",
        ],
    )

    assert result.exit_code == 2
    cleaned = _ANSI_ESCAPE_RE.sub("", result.output)
    assert "Missing option" in cleaned
    assert "--metadata-path" in cleaned


def test_simulation_invalid_backend(tmp_path):
    """Invalid backend values should fail fast."""
    metadata_path = _write_metadata_csv(tmp_path, [{"member_ous_uid": "uid://A"}])

    result = runner.invoke(
        cli.app,
        [
            "simulation",
            "run",
            "--metadata-path",
            metadata_path,
            "--backend",
            "dask",
        ],
    )

    assert result.exit_code == 2
    assert "--backend must be one of" in result.output


def test_simulation_run_csv_happy_path(monkeypatch, tmp_path):
    """Simulation run should execute staged workflow and print completion output."""
    metadata_path = _write_metadata_csv(tmp_path, [{"member_ous_uid": "uid://A"}])

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
            "--metadata-path",
            metadata_path,
        ],
    )

    assert result.exit_code == 0
    assert "All done. Simulated 1 run(s) across 1 row(s)." in result.output
    assert "ML shard written to: /tmp/mock_ml.h5" in result.output


def test_resolve_input_path_for_row_single_file(tmp_path):
    """A single input file should be reused for every metadata row."""
    input_file = tmp_path / "shared_model.fits"
    input_file.write_text("dummy", encoding="utf-8")
    row = pd.Series({"member_ous_uid": "uid://A", "group_ous_uid": "uid://G"})

    resolved = cli_simulation._resolve_input_path_for_row(input_file, row)

    assert resolved == input_file.resolve()


def test_resolve_input_path_for_row_directory_match(tmp_path):
    """Input directory should match files using member_ous_uid/group_ous_uid."""
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    matched = input_dir / "skymodel_uidA001X1X1_uidA001X1.fits"
    matched.write_text("dummy", encoding="utf-8")
    (input_dir / "other_model.fits").write_text("dummy", encoding="utf-8")
    row = pd.Series(
        {
            "member_ous_uid": "uid://A001/X1/X1",
            "group_ous_uid": "uid://A001/X1",
        }
    )

    resolved = cli_simulation._resolve_input_path_for_row(input_dir, row)

    assert resolved == matched.resolve()
