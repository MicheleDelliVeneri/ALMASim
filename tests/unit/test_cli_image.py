"""Unit tests for ALMASim image CLI commands."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
import typer
from typer.testing import CliRunner

from almasim import cli, cli_image

runner = CliRunner()


class _FakeTable:
    def __init__(self, columns: dict[str, np.ndarray]):
        self._columns = columns

    def getcol(self, name: str) -> np.ndarray:
        return self._columns[name]


@pytest.mark.unit
def test_import_casacore_tables_returns_table_symbol(monkeypatch):
    """import_casacore_tables should return casacore.tables.table on success."""
    fake_table_symbol = object()
    casacore_module = ModuleType("casacore")
    tables_module = ModuleType("casacore.tables")
    tables_module.table = fake_table_symbol
    casacore_module.tables = tables_module

    monkeypatch.setitem(sys.modules, "casacore", casacore_module)
    monkeypatch.setitem(sys.modules, "casacore.tables", tables_module)

    table_symbol = cli_image.import_casacore_tables()

    assert table_symbol is fake_table_symbol


@pytest.mark.unit
def test_import_casacore_tables_raises_when_casacore_missing(monkeypatch):
    """import_casacore_tables should raise a friendly message if casacore is unavailable."""
    casacore_module = ModuleType("casacore")
    monkeypatch.setitem(sys.modules, "casacore", casacore_module)
    monkeypatch.delitem(sys.modules, "casacore.tables", raising=False)

    with pytest.raises(typer.Exit):
        cli_image.import_casacore_tables()


@pytest.mark.unit
def test_compute_imaging_parameters_builds_expected_dataframe(monkeypatch):
    """compute_imaging_parameters should populate expected imaging columns."""

    spectral_window = _FakeTable(
        {
            "REF_FREQUENCY": np.array([100.0e9, 200.0e9]),
        }
    )
    antenna = _FakeTable(
        {
            "DISH_DIAMETER": np.array([12.0, 10.0]),
            "POSITION": np.array(
                [
                    [0.0, 0.0, 0.0],
                    [3.0, 4.0, 0.0],
                    [0.0, 0.0, 12.0],
                ]
            ),
        }
    )

    def _fake_casacore_table(table_name: str, ack: bool = False):
        del ack
        if table_name.endswith("::SPECTRAL_WINDOW"):
            return spectral_window
        if table_name.endswith("::ANTENNA"):
            return antenna

    monkeypatch.setattr(cli_image, "import_casacore_tables", lambda: _fake_casacore_table)

    output = cli_image.compute_imaging_parameters(Path("test_dataset.cal"))

    expected_frequencies = np.array([100.0e9, 200.0e9])
    speed_of_light = 299_792_458.0
    radians_to_arcsec = 180.0 * 3600.0 / np.pi
    expected_max_baseline_size = 13.0
    expected_wavelengths = speed_of_light / expected_frequencies
    expected_fov_per_frequency = (
        1.22 * expected_wavelengths / np.min(antenna.getcol("DISH_DIAMETER")) * radians_to_arcsec
    )
    expected_synthetized_beam_size = (
        expected_wavelengths / expected_max_baseline_size * radians_to_arcsec
    )

    assert list(output["filename"]) == [str(Path("test_dataset.cal").resolve())] * 2
    np.testing.assert_array_equal(output["spectral_window_id"].to_numpy(), np.array([0, 1]))
    np.testing.assert_array_equal(output["reference_frequency"].to_numpy(), expected_frequencies)
    np.testing.assert_allclose(
        output["max_baseline_size"].to_numpy(),
        np.array([expected_max_baseline_size, expected_max_baseline_size]),
    )
    np.testing.assert_allclose(
        output["fov_per_frequency"].to_numpy(),
        expected_fov_per_frequency,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        output["synthetized_beam_size"].to_numpy(),
        expected_synthetized_beam_size,
        rtol=1e-12,
    )


@pytest.mark.unit
def test_imaging_parameter_to_command_arg_returns_expected_tokens():
    """Command arg helper should return split tokens ready for subprocess usage."""
    params = {
        "spectral_window_id": 3,
        "fov_per_frequency": 8.0,
        "synthetized_beam_size": 2.0,
    }

    cmd_args = cli_image.imaging_parameter_to_command_arg(
        params,
        fov_fraction=1.5,
        beam_sampling=2,
    )

    assert cmd_args[:7] == ["-scale", "1.0asec", "-size", "16", "16", "-spws", "3"]
    assert "-update-model-required" in cmd_args


@pytest.mark.unit
def test_ms_overview_command_prints_dataframe(monkeypatch):
    """ms-overview should print the computed dataframe."""
    df = pd.DataFrame(
        {
            "filename": ["a.cal"],
            "spectral_window_id": [0],
            "reference_frequency": [100.0],
            "fov_per_frequency": [10.0],
            "max_baseline_size": [50.0],
            "synthetized_beam_size": [1.0],
        }
    )
    monkeypatch.setattr(cli_image, "compute_imaging_parameters", lambda _: df)

    result = runner.invoke(cli.app, ["image", "ms-overview", "a.cal"])

    assert result.exit_code == 0
    assert "filename" in result.output
    assert "synthetized_beam_size" in result.output


@pytest.mark.unit
def test_ms_overview_snake_case_command_is_rejected():
    """Only hyphenated command naming should be supported."""
    result = runner.invoke(cli.app, ["image", "ms_overview", "a.cal"])

    assert result.exit_code != 0
    assert "No such command" in result.output


@pytest.mark.unit
def test_compute_parameters_exits_when_no_ms_found(tmp_path):
    """compute-parameters should fail with exit code 1 if no .cal datasets exist."""
    out_csv = tmp_path / "imaging_parameters.csv"

    result = runner.invoke(
        cli.app,
        ["image", "compute-parameters", str(tmp_path), str(out_csv)],
    )

    assert result.exit_code == 1
    assert "Cannot find any MS" in result.output


@pytest.mark.unit
def test_compute_parameters_writes_csv_for_all_datasets(monkeypatch, tmp_path):
    """compute-parameters should aggregate rows across all matching datasets."""
    (tmp_path / "first.cal").mkdir()
    (tmp_path / "second.cal").mkdir()
    out_csv = tmp_path / "imaging_parameters.csv"

    monkeypatch.setattr(cli_image, "tqdm", lambda iterable: iterable)

    def _fake_compute(input_ms: Path) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "filename": [str(input_ms.resolve())],
                "spectral_window_id": [0],
                "reference_frequency": [100.0],
                "fov_per_frequency": [10.0],
                "max_baseline_size": [50.0],
                "synthetized_beam_size": [1.0],
            }
        )

    monkeypatch.setattr(cli_image, "compute_imaging_parameters", _fake_compute)

    result = runner.invoke(
        cli.app,
        ["image", "compute-parameters", str(tmp_path), str(out_csv)],
    )

    assert result.exit_code == 0
    saved = pd.read_csv(out_csv)
    assert len(saved) == 2
    assert sorted(Path(f).name for f in saved["filename"].tolist()) == ["first.cal", "second.cal"]


@pytest.mark.unit
def test_batch_image_submits_sbatch_with_wrap(monkeypatch, tmp_path):
    """batch-image should call sbatch once per dataframe row with wrapped wsclean command."""
    imaging_csv = tmp_path / "imaging_parameters.csv"
    output_dir = tmp_path / "images"
    output_dir.mkdir()

    pd.DataFrame(
        {
            "filename": ["uid___A001_X1_X1.cal"],
            "spectral_window_id": [2],
            "reference_frequency": [100.0e9],
            "fov_per_frequency": [8.0],
            "max_baseline_size": [100.0],
            "synthetized_beam_size": [2.0],
        }
    ).to_csv(imaging_csv, index=False)

    calls: list[tuple[list[str], bool]] = []

    def _fake_subprocess_run(cmd: list[str], check: bool = False):
        calls.append((cmd, check))

    monkeypatch.setattr(cli_image, "tqdm", lambda iterable, total=None: iterable)
    monkeypatch.setattr(cli_image.subprocess, "run", _fake_subprocess_run)

    result = runner.invoke(
        cli.app,
        [
            "image",
            "batch-image",
            str(imaging_csv),
            str(output_dir),
            "--num-cores",
            "8",
            "--max-cores-per-node",
            "64",
        ],
    )

    assert result.exit_code == 0
    assert len(calls) == 1
    sbatch_cmd, check_flag = calls[0]
    assert check_flag is True
    assert sbatch_cmd[0] == "sbatch"
    assert "--wrap" in sbatch_cmd
    assert "-o" in sbatch_cmd
    assert "-e" in sbatch_cmd
    assert "-c" in sbatch_cmd
    wrap_idx = sbatch_cmd.index("--wrap")
    assert "wsclean" in sbatch_cmd[wrap_idx + 1]
    assert "-name" in sbatch_cmd[wrap_idx + 1]


@pytest.mark.unit
def test_batch_image_enforces_positive_max_cores_per_node(tmp_path):
    """batch-image should reject invalid max cores value at CLI parsing time."""
    imaging_csv = tmp_path / "imaging_parameters.csv"
    output_dir = tmp_path / "images"
    output_dir.mkdir()

    pd.DataFrame(
        {
            "filename": ["uid___A001_X1_X1.cal"],
            "spectral_window_id": [2],
            "reference_frequency": [100.0e9],
            "fov_per_frequency": [8.0],
            "max_baseline_size": [100.0],
            "synthetized_beam_size": [2.0],
        }
    ).to_csv(imaging_csv, index=False)

    result = runner.invoke(
        cli.app,
        [
            "image",
            "batch-image",
            str(imaging_csv),
            str(output_dir),
            "--max-cores-per-node",
            "0",
        ],
    )

    assert result.exit_code == 2
