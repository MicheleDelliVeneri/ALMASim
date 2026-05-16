"""Unit tests for ALMASim image CLI commands."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any

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
        1.12 * expected_wavelengths / np.min(antenna.getcol("DISH_DIAMETER")) * radians_to_arcsec
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
def test_batch_image_submits_commands_via_slurm_cluster(monkeypatch, tmp_path):
    """batch-image should dispatch wsclean commands via the SLURM cluster helper."""
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

    captured: dict[str, Any] = {}

    def _fake_run_with_slurm_cluster(
        commands,
        *,
        cores_per_task,
        node_cores,
        queue,
        project,
        walltime,
        memory,
        n_jobs,
        scheduler_host,
        scheduler_interface,
        task_timeout,
    ):
        captured["commands"] = commands
        captured["cores_per_task"] = cores_per_task
        captured["node_cores"] = node_cores
        captured["queue"] = queue
        captured["project"] = project
        captured["walltime"] = walltime
        captured["memory"] = memory
        captured["n_jobs"] = n_jobs
        captured["scheduler_host"] = scheduler_host
        captured["scheduler_interface"] = scheduler_interface
        captured["task_timeout"] = task_timeout

    monkeypatch.setattr(cli_image, "tqdm", lambda iterable, total=None: iterable)
    monkeypatch.setattr(cli_image, "_run_commands_with_slurm_cluster", _fake_run_with_slurm_cluster)

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
    assert captured["cores_per_task"] == 8
    assert captured["node_cores"] == 64
    assert captured["queue"] == "normal"
    assert captured["project"] is None
    assert captured["walltime"] == "02:00:00"
    assert captured["memory"] == "16GB"
    assert captured["n_jobs"] == 1

    commands = captured["commands"]
    assert len(commands) == 1
    _, cmd_tokens = commands[0]
    assert cmd_tokens[0] == "wsclean"
    assert "-name" in cmd_tokens


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


@pytest.mark.unit
def test_predict_batch_submits_jobs_via_slurm_cluster(monkeypatch, tmp_path):
    """predict-batch should submit predict-single commands via SLURM cluster helper."""
    imaging_csv = tmp_path / "imaging_parameters.csv"
    output_dir = tmp_path / "images"
    output_dir.mkdir()

    first_ms = tmp_path / "uid___A001_X1_X1.cal"
    second_ms = tmp_path / "uid___A001_X2_X2.cal"
    pd.DataFrame(
        {
            "filename": [str(first_ms), str(second_ms)],
            "spectral_window_id": [1, 3],
            "reference_frequency": [100.0e9, 101.0e9],
            "fov_per_frequency": [8.0, 8.2],
            "max_baseline_size": [100.0, 100.0],
            "synthetized_beam_size": [2.0, 2.1],
        }
    ).to_csv(imaging_csv, index=False)

    first_model = output_dir / first_ms.stem / "SPW-1" / "wsclean-model.fits"
    second_model = output_dir / second_ms.stem / "SPW-3" / "wsclean-model.fits"
    first_model.parent.mkdir(parents=True)
    second_model.parent.mkdir(parents=True)
    first_model.touch()
    second_model.touch()

    captured: dict[str, Any] = {}

    def _fake_run_with_slurm_cluster(
        commands,
        *,
        cores_per_task,
        node_cores,
        queue,
        project,
        walltime,
        memory,
        n_jobs,
        scheduler_host,
        scheduler_interface,
        task_timeout,
    ):
        captured["commands"] = commands
        captured["cores_per_task"] = cores_per_task
        captured["node_cores"] = node_cores
        captured["queue"] = queue
        captured["project"] = project
        captured["walltime"] = walltime
        captured["memory"] = memory
        captured["n_jobs"] = n_jobs
        captured["scheduler_host"] = scheduler_host
        captured["scheduler_interface"] = scheduler_interface
        captured["task_timeout"] = task_timeout

    monkeypatch.setattr(cli_image, "_run_commands_with_slurm_cluster", _fake_run_with_slurm_cluster)
    monkeypatch.setattr(cli_image, "tqdm", lambda iterable, total=None: iterable)

    result = runner.invoke(
        cli.app,
        ["image", "predict-batch", str(imaging_csv), str(output_dir), "--use-slurm"],
    )

    assert result.exit_code == 0
    assert captured["cores_per_task"] == 1
    assert captured["node_cores"] == 95
    assert captured["queue"] == "normal"
    assert captured["project"] is None
    assert captured["walltime"] == "02:00:00"
    assert captured["memory"] == "16GB"
    assert captured["n_jobs"] == 1

    commands = captured["commands"]
    assert len(commands) == 2
    first_cmd = commands[0][1]
    second_cmd = commands[1][1]
    assert first_cmd[:3] == ["almasim", "image", "predict-single"]
    assert second_cmd[:3] == ["almasim", "image", "predict-single"]
    assert str(first_model) in first_cmd
    assert str(second_model) in second_cmd
    assert str(first_model.parent / f"{first_ms.name}.predicted") in first_cmd
    assert str(second_model.parent / f"{second_ms.name}.predicted") in second_cmd


@pytest.mark.unit
def test_predict_batch_skips_when_model_is_missing(monkeypatch, tmp_path):
    """predict-batch should log and skip rows whose model FITS file is missing."""
    imaging_csv = tmp_path / "imaging_parameters.csv"
    output_dir = tmp_path / "images"
    output_dir.mkdir()

    ms_path = tmp_path / "uid___A001_X9_X9.cal"
    pd.DataFrame(
        {
            "filename": [str(ms_path)],
            "spectral_window_id": [0],
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
        ["image", "predict-batch", str(imaging_csv), str(output_dir)],
    )

    assert result.exit_code == 0
    assert "[debug] missing model FITS, skipping row" in result.output
    assert len(calls) == 0
