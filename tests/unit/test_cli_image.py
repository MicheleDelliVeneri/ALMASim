"""Unit tests for ALMASim image CLI commands."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
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

    fake_observation = [
        {
            "TIME_RANGE": [57000 * 86400.0, 57001 * 86400.0],
        }
    ]

    def _fake_casacore_table(table_name: str, ack: bool = False):
        del ack
        if table_name.endswith("::SPECTRAL_WINDOW"):
            return spectral_window
        if table_name.endswith("::ANTENNA"):
            return antenna
        if table_name.endswith("::OBSERVATION"):
            return fake_observation

    monkeypatch.setattr(cli_image, "import_casacore_tables", lambda: _fake_casacore_table)
    monkeypatch.setattr(cli_image, "science_selection", lambda _ms: ({0: 10, 1: 20}, [0, 2]))

    output = cli_image.compute_imaging_parameters(Path("test_dataset.cal"))
    assert list(output["n_visibility_rows"]) == [10, 20]
    assert list(output["target_field_ids"]) == ["0,2", "0,2"]

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

    def _fake_compute(input_ms: Path, science_only: bool = True) -> pd.DataFrame:
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
def test_compute_parameters_defaults_to_cwd_and_default_output(monkeypatch, tmp_path):
    """compute-parameters should run without args using cwd and default output filename."""
    monkeypatch.setattr(cli_image, "tqdm", lambda iterable: iterable)

    def _fake_compute(input_ms: Path, science_only: bool = True) -> pd.DataFrame:
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

    with runner.isolated_filesystem(temp_dir=str(tmp_path)):
        Path("sample.cal").mkdir()
        result = runner.invoke(cli.app, ["image", "compute-parameters"])

    assert result.exit_code == 0
    output_candidates = list(tmp_path.rglob("imaging_parameters.csv"))
    assert output_candidates


@pytest.mark.unit
def test_run_commands_with_slurm_cluster_no_commands_is_noop(monkeypatch):
    """Slurm command runner should return immediately when no commands are provided."""
    called = {"create_backend": False}

    def _fake_create_backend(*args, **kwargs):
        del args, kwargs
        called["create_backend"] = True
        raise AssertionError("create_backend should not be called for an empty command list")

    monkeypatch.setattr("almasim.services.compute.create_backend", _fake_create_backend)

    cli_image._run_commands_with_slurm_cluster(
        commands=[],
        cores_per_task=1,
        node_cores=2,
        queue="normal",
        project=None,
        walltime="00:10:00",
        memory="2GB",
        n_jobs=1,
        scheduler_host=None,
        scheduler_interface=None,
        task_timeout=None,
    )

    assert called["create_backend"] is False


@pytest.mark.unit
def test_predict_from_model_exits_when_output_exists(tmp_path):
    """predict_from_model should fail fast when the output MS already exists."""
    input_ms = tmp_path / "input.ms"
    output_ms = tmp_path / "output.ms"
    model = tmp_path / "model.fits"
    input_ms.mkdir()
    output_ms.mkdir()
    model.write_text("dummy", encoding="utf-8")

    with pytest.raises(typer.Exit) as exc:
        cli_image.predict_from_model(input_ms=input_ms, model=model, output_ms=output_ms)

    assert exc.value.exit_code == 1


@pytest.mark.unit
def test_run_commands_with_slurm_cluster_forwards_scheduler_interface(monkeypatch):
    """Slurm command runner should pass scheduler_interface to backend creation."""
    captured: dict[str, object] = {}

    class _Future:
        def result(self):
            return SimpleNamespace(returncode=0, stderr="")

    class _Backend:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return None

        def submit_subcommand(self, **kwargs):
            captured["submit_kwargs"] = kwargs
            return _Future()

    def _fake_create_backend(*args, **kwargs):
        captured["create_args"] = args
        captured["create_kwargs"] = kwargs
        return _Backend()

    monkeypatch.setattr("almasim.services.compute.create_backend", _fake_create_backend)
    monkeypatch.setattr(cli_image, "tqdm", lambda iterable, **kwargs: iterable)

    cli_image._run_commands_with_slurm_cluster(
        commands=[("job-1", ["echo", "ok"])],
        cores_per_task=2,
        node_cores=8,
        queue="normal",
        project=None,
        walltime="00:10:00",
        memory="4GB",
        n_jobs=1,
        scheduler_host="headnode",
        scheduler_interface="ib0",
        task_timeout=10.0,
    )

    assert captured["create_kwargs"]["scheduler_interface"] == "ib0"


@pytest.mark.unit
def test_batch_image_submits_commands_via_slurm_cluster(monkeypatch, tmp_path):
    """batch-image should build one WSClean task per row and hand them to the runner."""
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
            "target_field_ids": ["3,4"],
        }
    ).to_csv(imaging_csv, index=False)

    captured: dict[str, Any] = {}

    def _fake_run_imaging_tasks(tasks, **kwargs):
        captured["tasks"] = tasks
        captured.update(kwargs)
        return []

    monkeypatch.setattr(cli_image, "run_imaging_tasks", _fake_run_imaging_tasks)

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
            "--wsclean-bin",
            "/opt/wsclean/bin/wsclean",
            "--no-single-window-ms",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["single_window"] is False
    assert captured["cores_per_task"] == 8
    assert captured["node_cores"] == 64
    assert captured["queue"] == "normal"
    assert captured["project"] is None
    assert captured["walltime"] == "02:00:00"
    assert captured["memory"] == "16GB"
    assert captured["n_jobs"] == 1
    assert captured["backend_kind"] == "slurm"
    assert captured["continue_on_error"] is True

    tasks = captured["tasks"]
    assert len(tasks) == 1
    task = tasks[0]
    assert task.spw == 2
    assert task.output_dir == output_dir / "uid___A001_X1_X1" / "SPW-2"
    cmd = task.command
    assert cmd[0] == "/opt/wsclean/bin/wsclean"
    assert cmd[cmd.index("-j") + 1] == "8"
    assert cmd[cmd.index("-field") + 1] == "3,4"
    assert cmd[cmd.index("-spws") + 1] == "2"
    assert cmd[-1] == "uid___A001_X1_X1.cal"
    assert cmd[cmd.index("-auto-threshold") + 1] == "3.0", "3 sigma stopping point by default"
    assert "-auto-mask" not in cmd
    assert cmd[cmd.index("-mem") + 1] == "12.5", "-mem is a percentage: 8 of 64 cores"
    assert cmd[cmd.index("-temp-dir") + 1] == str(task.output_dir)
    assert "-no-update-model-required" in cmd and "-update-model-required" not in cmd


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


# ---------------------------------------------------------------------------
# Imaging markers, the worker-side WSClean wrapper and the resilient runner.

from almasim.services.imaging import archive_imaging as ai  # noqa: E402


def _params_csv(tmp_path: Path, rows: int = 2) -> Path:
    csv_path = tmp_path / "params.csv"
    pd.DataFrame(
        {
            "filename": ["/data/uid___A001_X1_X1.ms.split.cal"] * rows,
            "spectral_window_id": list(range(rows)),
            "reference_frequency": [100.0e9] * rows,
            "fov_per_frequency": [8.0] * rows,
            "max_baseline_size": [100.0] * rows,
            "synthetized_beam_size": [2.0] * rows,
        }
    ).to_csv(csv_path, index=False)
    return csv_path


class _FakeWsclean:
    """Stand-in for ``subprocess.Popen`` running WSClean.

    ``returncode`` drives the outcome; on success the restored image is written
    unless ``write_image`` is False, mimicking a run that exited 0 without output.
    """

    def __init__(self, returncode: int = 0, lines=None, write_image: bool = True):
        self.returncode = returncode
        self.lines = lines or ["WSClean version 3.7", "Cleaning up temporary files..."]
        self.write_image = write_image
        self.calls: list[dict] = []

    def __call__(self, cmd, **kwargs):
        self.calls.append({"cmd": list(cmd), **kwargs})
        outdir = Path(cmd[cmd.index("-name") + 1]).parent
        if self.returncode == 0 and self.write_image:
            (outdir / ai.IMAGE_FILENAME).write_bytes(b"SIMPLE")
        lines = self.lines
        rc = self.returncode

        class _Proc:
            stdout = iter(line + "\n" for line in lines)

            @staticmethod
            def wait():
                return rc

            @staticmethod
            def kill():
                pass

        return _Proc()


def _wsclean_cmd(outdir: Path) -> list[str]:
    return ["wsclean", "-name", str(outdir / "wsclean"), "-spws", "1", "in.ms"]


@pytest.mark.unit
def test_imaging_marker_paths_sit_next_to_the_spw_directory(tmp_path):
    outdir = tmp_path / "ms" / "SPW-3"
    assert ai.imaging_marker_path(outdir) == tmp_path / "ms" / "SPW-3.done"
    assert ai.imaging_failure_marker_path(outdir) == tmp_path / "ms" / "SPW-3.failed"
    assert ai.imaging_log_path(outdir) == tmp_path / "ms" / "SPW-3.log"
    assert not ai.is_imaging_complete(outdir)
    outdir.mkdir(parents=True)
    ai.write_imaging_marker(outdir, ms_path="in.ms", spw=3)
    assert not ai.is_imaging_complete(outdir), "a marker without the image is not complete"
    (outdir / ai.IMAGE_FILENAME).write_bytes(b"")
    assert ai.is_imaging_complete(outdir)


@pytest.mark.unit
def test_wsclean_environment_pins_openblas_to_one_thread():
    env = ai.wsclean_environment(6, base={"LD_LIBRARY_PATH": "/spack/lib", "PATH": "/bin"})
    assert env["OPENBLAS_NUM_THREADS"] == "1"
    assert env["OMP_NUM_THREADS"] == "6"
    assert env["LD_LIBRARY_PATH"].startswith("/lib64:")
    assert env["LD_LIBRARY_PATH"].endswith(":/spack/lib")
    assert env["PATH"] == "/bin"


@pytest.mark.unit
def test_run_wsclean_task_success_writes_done_marker_and_log(tmp_path, monkeypatch):
    fake = _FakeWsclean(returncode=0)
    monkeypatch.setattr(ai.subprocess, "Popen", fake)
    outdir = tmp_path / "ms" / "SPW-1"

    result = ai.run_wsclean_task(
        command=_wsclean_cmd(outdir), output_dir=str(outdir), ms_path="in.ms", spw=1, threads=4
    )

    assert ai.is_imaging_complete(outdir)
    assert result["image"] == str(outdir / ai.IMAGE_FILENAME)
    payload = json.loads(ai.imaging_marker_path(outdir).read_text())
    assert payload["spw"] == 1 and payload["ms"] == "in.ms" and "wsclean" in payload["command"]
    log_text = ai.imaging_log_path(outdir).read_text()
    assert log_text.splitlines()[0].startswith("# wsclean -name")
    assert "Cleaning up temporary files" in log_text
    assert fake.calls[0]["env"]["OPENBLAS_NUM_THREADS"] == "1"
    assert fake.calls[0]["env"]["OMP_NUM_THREADS"] == "4"
    assert not ai.imaging_failure_marker_path(outdir).exists()


@pytest.mark.unit
def test_run_wsclean_task_hard_abort_writes_failure_marker(tmp_path, monkeypatch):
    lines = [
        "Gridding...",
        "terminate called after throwing an instance of 'std::bad_alloc'",
        "  what():  std::bad_alloc",
    ]
    monkeypatch.setattr(ai.subprocess, "Popen", _FakeWsclean(returncode=-6, lines=lines))
    outdir = tmp_path / "ms" / "SPW-1"

    with pytest.raises(RuntimeError, match="return code -6"):
        ai.run_wsclean_task(
            command=_wsclean_cmd(outdir), output_dir=str(outdir), ms_path="in.ms", spw=1, threads=2
        )

    marker = ai.imaging_failure_marker_path(outdir)
    assert marker.is_file()
    payload = json.loads(marker.read_text())
    assert payload["returncode"] == -6
    assert "what():  std::bad_alloc" in payload["error"]
    assert Path(payload["log"]).is_file()
    assert payload["stage"] == "image"
    assert not ai.imaging_marker_path(outdir).exists()


@pytest.mark.unit
def test_run_wsclean_task_without_image_is_a_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(ai.subprocess, "Popen", _FakeWsclean(returncode=0, write_image=False))
    outdir = tmp_path / "ms" / "SPW-2"

    with pytest.raises(RuntimeError, match="wrote no wsclean-image.fits"):
        ai.run_wsclean_task(
            command=_wsclean_cmd(outdir), output_dir=str(outdir), ms_path="in.ms", spw=2, threads=2
        )
    assert ai.imaging_failure_marker_path(outdir).is_file()


@pytest.mark.unit
def test_run_wsclean_task_missing_binary_is_a_failure(tmp_path, monkeypatch):
    def _no_binary(*args, **kwargs):
        raise FileNotFoundError("wsclean")

    monkeypatch.setattr(ai.subprocess, "Popen", _no_binary)
    outdir = tmp_path / "ms" / "SPW-2"
    with pytest.raises(RuntimeError, match="Could not start wsclean"):
        ai.run_wsclean_task(
            command=_wsclean_cmd(outdir), output_dir=str(outdir), ms_path="in.ms", spw=2, threads=2
        )
    payload = json.loads(ai.imaging_failure_marker_path(outdir).read_text())
    assert "Could not start wsclean" in payload["error"]


@pytest.mark.unit
def test_run_wsclean_task_replaces_stale_markers(tmp_path, monkeypatch):
    outdir = tmp_path / "ms" / "SPW-1"
    outdir.mkdir(parents=True)
    ai.write_imaging_failure_marker(outdir, ms_path="in.ms", spw=1, error="old")
    monkeypatch.setattr(ai.subprocess, "Popen", _FakeWsclean(returncode=0))

    ai.run_wsclean_task(
        command=_wsclean_cmd(outdir), output_dir=str(outdir), ms_path="in.ms", spw=1, threads=1
    )
    assert ai.is_imaging_complete(outdir)
    assert not ai.imaging_failure_marker_path(outdir).exists()


@pytest.mark.unit
def test_build_imaging_tasks_skips_done_and_grandfathers_images(tmp_path):
    csv_path = _params_csv(tmp_path, rows=3)
    parameters = pd.read_csv(csv_path)
    out = tmp_path / "images"
    done_dir = out / "uid___A001_X1_X1.ms.split" / "SPW-0"
    done_dir.mkdir(parents=True)
    (done_dir / ai.IMAGE_FILENAME).write_bytes(b"")
    ai.write_imaging_marker(done_dir, ms_path="x", spw=0)
    legacy_dir = out / "uid___A001_X1_X1.ms.split" / "SPW-1"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / ai.IMAGE_FILENAME).write_bytes(b"")

    common = dict(fov_fraction=1.5, beam_sampling=8, num_cores=10, max_cores_per_node=95)

    tasks, skipped = cli_image.build_imaging_tasks(parameters, out, **common)
    assert skipped == 1
    assert [t.spw for t in tasks] == [1, 2], "an image without a marker is retried by default"

    tasks, skipped = cli_image.build_imaging_tasks(
        parameters, out, trust_existing_images=True, **common
    )
    assert skipped == 2
    assert [t.spw for t in tasks] == [2]
    assert ai.imaging_marker_path(legacy_dir).is_file(), "grandfathered image gets its marker"

    tasks, skipped = cli_image.build_imaging_tasks(
        parameters, out, overwrite_outputs=True, **common
    )
    assert skipped == 0 and [t.spw for t in tasks] == [0, 1, 2]
    assert "-field" not in tasks[0].command, "no target_field_ids column: no field selection"


class _FakeFuture:
    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error
        self.cancelled = False

    def done(self):
        return True

    def exception(self):
        return self._error

    def result(self):
        if self._error is not None:
            raise self._error
        return self._result

    def cancel(self):
        self.cancelled = True


class _FakeBackend:
    def __init__(self, outcomes):
        self.outcomes = outcomes
        self.submitted: list[dict] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def submit_callable(self, func, *, cores, **kwargs):
        self.submitted.append({"func": func, "cores": cores, **kwargs})
        outcome = self.outcomes[len(self.submitted) - 1]
        if isinstance(outcome, Exception):
            return _FakeFuture(error=outcome)
        return _FakeFuture(result=outcome)


def _tasks(tmp_path: Path, n: int) -> list[cli_image.ImagingTask]:
    out = tmp_path / "images" / "ms"
    return [
        cli_image.ImagingTask(
            label=f"ms_{i}",
            ms_path=Path("in.ms"),
            spw=i,
            output_dir=out / f"SPW-{i}",
            command=["wsclean", "-name", str(out / f"SPW-{i}" / "wsclean"), "in.ms"],
        )
        for i in range(n)
    ]


def _run(tasks, backend, monkeypatch, **overrides):
    monkeypatch.setattr("almasim.services.compute.create_backend", lambda *a, **k: backend)
    monkeypatch.setattr(cli_image, "tqdm", _QuietTqdm)
    kwargs = dict(
        backend_kind="slurm",
        cores_per_task=4,
        node_cores=32,
        queue="q",
        project=None,
        walltime="01:00:00",
        memory="8GB",
        n_jobs=2,
        scheduler_host=None,
        scheduler_interface=None,
        task_timeout=10.0,
        heartbeat_interval=1e9,
    )
    kwargs.update(overrides)
    return cli_image.run_imaging_tasks(tasks, **kwargs)


class _QuietTqdm:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def update(self, n=1):
        pass

    def write(self, text):
        pass

    def set_postfix_str(self, text):
        pass


@pytest.mark.unit
def test_run_imaging_tasks_continues_past_failures_and_marks_lost_tasks(tmp_path, monkeypatch):
    tasks = _tasks(tmp_path, 3)
    backend = _FakeBackend(
        [{"ok": 0}, RuntimeError("WSClean exited with return code 1"), {"ok": 2}]
    )
    failures: list[cli_image.ImagingFailure] = []

    results = _run(tasks, backend, monkeypatch, failures=failures)

    assert results == [{"ok": 0}, {"ok": 2}]
    assert [f.label for f in failures] == ["ms_1"]
    assert "return code 1" in failures[0].error
    # the worker never wrote a marker for the lost task: the driver does it
    payload = json.loads(ai.imaging_failure_marker_path(tasks[1].output_dir).read_text())
    assert payload["spw"] == 1 and "return code 1" in payload["error"]
    assert not ai.imaging_failure_marker_path(tasks[0].output_dir).exists()
    # resources and task kwargs both reach the backend
    assert backend.submitted[0]["cores"] == 4
    assert backend.submitted[0]["threads"] == 4
    assert backend.submitted[0]["func"] is ai.run_wsclean_task
    assert backend.submitted[0]["timeout"] == 10.0


@pytest.mark.unit
def test_run_imaging_tasks_fail_fast_stops_and_cancels(tmp_path, monkeypatch):
    tasks = _tasks(tmp_path, 2)
    backend = _FakeBackend([RuntimeError("boom"), {"ok": 1}])
    failures: list[cli_image.ImagingFailure] = []

    with pytest.raises(typer.Exit) as exc:
        _run(tasks, backend, monkeypatch, failures=failures, continue_on_error=False)
    assert exc.value.exit_code == 1
    assert [f.label for f in failures] == ["ms_0"]


@pytest.mark.unit
def test_run_imaging_tasks_sync_backend_runs_in_process(tmp_path, monkeypatch):
    tasks = _tasks(tmp_path, 2)
    calls = []

    def _fake_task(**kwargs):
        calls.append(kwargs)
        if kwargs["spw"] == 0:
            raise RuntimeError("no image")
        return {"spw": kwargs["spw"]}

    monkeypatch.setattr(ai, "run_wsclean_task", _fake_task)
    failures: list[cli_image.ImagingFailure] = []
    results = _run(tasks, None, monkeypatch, backend_kind="sync", failures=failures)

    assert results == [{"spw": 1}]
    assert [f.label for f in failures] == ["ms_0"]
    assert calls[0]["threads"] == 4
    assert ai.imaging_failure_marker_path(tasks[0].output_dir).is_file()


@pytest.mark.unit
def test_image_from_ms_reports_failures_and_exits_one(monkeypatch, tmp_path):
    csv_path = _params_csv(tmp_path, rows=2)

    def _fake_run(tasks, *, failures, **kwargs):
        failures.append(
            cli_image.ImagingFailure(tasks[0].label, "WSClean exited with return code 1", "/l")
        )
        return [{"ok": 1}]

    monkeypatch.setattr(cli_image, "run_imaging_tasks", _fake_run)
    result = runner.invoke(
        cli.app,
        [
            "image",
            "image-from-ms",
            str(csv_path),
            str(tmp_path / "out"),
            "--postprocess-backend",
            "sync",
        ],
    )
    assert result.exit_code == 1
    assert "Imaged 1/2 task(s), 1 failed." in result.output
    assert "1 task(s) failed and were skipped" in result.output
    assert "log: /l" in result.output


@pytest.mark.unit
def test_image_from_ms_nothing_to_do_when_all_done(monkeypatch, tmp_path):
    csv_path = _params_csv(tmp_path, rows=1)
    out = tmp_path / "out"
    done_dir = out / "uid___A001_X1_X1.ms.split" / "SPW-0"
    done_dir.mkdir(parents=True)
    (done_dir / ai.IMAGE_FILENAME).write_bytes(b"")
    ai.write_imaging_marker(done_dir, ms_path="x", spw=0)
    monkeypatch.setattr(cli_image, "run_imaging_tasks", lambda *a, **k: pytest.fail("no run"))

    result = runner.invoke(cli.app, ["image", "image-from-ms", str(csv_path), str(out)])
    assert result.exit_code == 0, result.output
    assert "Skipped 1 already-imaged SPW(s)." in result.output
    assert "nothing to do" in result.output


@pytest.mark.unit
def test_image_from_ms_rejects_unknown_backend(tmp_path):
    csv_path = _params_csv(tmp_path, rows=1)
    result = runner.invoke(
        cli.app,
        ["image", "image-from-ms", str(csv_path), str(tmp_path), "--postprocess-backend", "k8s"],
    )
    assert result.exit_code == 2


@pytest.mark.unit
def test_compute_parameters_records_unreadable_ms_and_continues(monkeypatch, tmp_path):
    folder = tmp_path / "cal"
    folder.mkdir()
    for name in ("a.ms.split.cal", "b.ms.split.cal", "c.ms.split.cal"):
        (folder / name).mkdir()
        if name != "c.ms.split.cal":
            (folder / f"{name}.done").write_text("{}")
    monkeypatch.setattr(cli_image, "tqdm", lambda iterable: iterable)

    def _fake_compute(ms: Path, science_only: bool = True):
        if ms.name.startswith("b"):
            raise RuntimeError("Table does not exist")
        return pd.DataFrame({"filename": [str(ms)], "spectral_window_id": [0]})

    monkeypatch.setattr(cli_image, "compute_imaging_parameters", _fake_compute)
    out_csv = tmp_path / "params.csv"

    result = runner.invoke(cli.app, ["image", "compute-parameters", str(folder), str(out_csv)])

    assert result.exit_code == 1
    assert "Skipping 1 MS(s) without a .done calibration marker." in result.output
    assert "FAILED b.ms.split.cal" in result.output
    written = pd.read_csv(out_csv)
    assert list(written["filename"]) == [str(folder / "a.ms.split.cal")]
    failed = (tmp_path / "params.csv.failed.tsv").read_text()
    assert "b.ms.split.cal\tRuntimeError: Table does not exist" in failed


@pytest.mark.unit
def test_science_selection_reports_used_spws_and_target_fields(monkeypatch):
    class _Tab:
        def __init__(self, cols, nrows=None):
            self.cols = cols
            self._n = nrows

        def getcol(self, name):
            return self.cols[name]

        def nrows(self):
            return self._n if self._n is not None else len(next(iter(self.cols.values())))

    tables = {
        "::DATA_DESCRIPTION": _Tab({"SPECTRAL_WINDOW_ID": np.array([0, 5, 7])}),
        "::STATE": _Tab({"OBS_MODE": ["CALIBRATE_PHASE#ON_SOURCE", "OBSERVE_TARGET#ON_SOURCE"]}),
        "": _Tab(
            {
                "DATA_DESC_ID": np.array([1, 1, 2, 2, 2, 1]),
                "STATE_ID": np.array([0, 1, 1, 0, 1, 1]),
                "FIELD_ID": np.array([0, 3, 3, 0, 4, 3]),
            }
        ),
    }

    def _fake_table(name, ack=False):
        for suffix, tab in tables.items():
            if suffix and name.endswith(suffix):
                return tab
        return tables[""]

    monkeypatch.setattr(cli_image, "import_casacore_tables", lambda: _fake_table)
    rows_per_spw, target_fields = cli_image.science_selection(Path("x.ms"))
    assert rows_per_spw == {5: 3, 7: 3}, "SPW 0 has no rows and is left out"
    assert target_fields == [3, 4]


@pytest.mark.unit
def test_imaging_parameter_to_command_arg_threshold_flags():
    row = pd.Series(
        {"spectral_window_id": 1, "fov_per_frequency": 10.0, "synthetized_beam_size": 1.0}
    )
    plain = cli_image.imaging_parameter_to_command_arg(row, 1.0, 4.0)
    assert "-auto-threshold" not in plain and "-auto-mask" not in plain
    flagged = cli_image.imaging_parameter_to_command_arg(
        row, 1.0, 4.0, auto_threshold=2.5, auto_mask=5.0
    )
    assert flagged[flagged.index("-auto-threshold") + 1] == "2.5"
    assert flagged[flagged.index("-auto-mask") + 1] == "5.0"
    assert cli_image.imaging_parameter_to_command_arg(row, 1.0, 4.0, auto_threshold=0) == plain


@pytest.mark.unit
def test_build_imaging_tasks_abs_mem_and_scratch_placeholder(tmp_path):
    parameters = pd.read_csv(_params_csv(tmp_path, rows=1))
    tasks, _ = cli_image.build_imaging_tasks(
        parameters,
        tmp_path / "out",
        fov_fraction=1.5,
        beam_sampling=8,
        num_cores=10,
        max_cores_per_node=96,
        task_memory_gb=40,
        scratch_dir="/tmp",
    )
    cmd = tasks[0].command
    assert cmd[cmd.index("-abs-mem") + 1] == "40" and "-mem" not in cmd
    assert cmd[cmd.index("-temp-dir") + 1] == "__SCRATCH__"


@pytest.mark.unit
def test_run_wsclean_task_uses_and_removes_scratch_dir(tmp_path, monkeypatch):
    fake = _FakeWsclean(returncode=0)
    monkeypatch.setattr(ai.subprocess, "Popen", fake)
    outdir = tmp_path / "ms" / "SPW-1"
    scratch_root = tmp_path / "scratch"
    cmd = ["wsclean", "-name", str(outdir / "wsclean"), "-temp-dir", "__SCRATCH__", "in.ms"]

    ai.run_wsclean_task(
        command=cmd,
        output_dir=str(outdir),
        ms_path="in.ms",
        spw=1,
        threads=2,
        scratch_root=str(scratch_root),
    )
    used = fake.calls[0]["cmd"]
    temp = Path(used[used.index("-temp-dir") + 1])
    assert temp.parent == scratch_root and temp.name.startswith("SPW-1-")
    assert not temp.exists(), "per-task scratch is removed after the run"
    assert ai.is_imaging_complete(outdir)


@pytest.mark.unit
def test_run_wsclean_task_removes_scratch_dir_on_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(ai.subprocess, "Popen", _FakeWsclean(returncode=-11))
    outdir = tmp_path / "ms" / "SPW-1"
    scratch_root = tmp_path / "scratch"
    cmd = ["wsclean", "-name", str(outdir / "wsclean"), "-temp-dir", "__SCRATCH__", "in.ms"]
    with pytest.raises(RuntimeError):
        ai.run_wsclean_task(
            command=cmd,
            output_dir=str(outdir),
            ms_path="in.ms",
            spw=1,
            threads=2,
            scratch_root=str(scratch_root),
        )
    assert list(scratch_root.iterdir()) == []
    assert ai.imaging_failure_marker_path(outdir).is_file()


class _FlakyWsclean:
    """Crashes with a signal ``crashes`` times, then behaves like ``final``."""

    def __init__(self, crashes: int, final: int = 0):
        self.remaining = crashes
        self.final = final
        self.calls = 0
        self.envs: list[dict] = []

    def __call__(self, cmd, **kwargs):
        self.calls += 1
        self.envs.append(kwargs)
        outdir = Path(cmd[cmd.index("-name") + 1]).parent
        if self.remaining > 0:
            self.remaining -= 1
            (outdir / "x-part0000-I.tmp").write_bytes(b"junk")
            return _FakeWsclean(returncode=-11, lines=["Reordering: 0%....10%"])(cmd, **kwargs)
        return _FakeWsclean(returncode=self.final)(cmd, **kwargs)


@pytest.mark.unit
def test_run_wsclean_task_retries_signal_crashes(tmp_path, monkeypatch):
    flaky = _FlakyWsclean(crashes=2)
    monkeypatch.setattr(ai.subprocess, "Popen", flaky)
    outdir = tmp_path / "ms" / "SPW-1"

    result = ai.run_wsclean_task(
        command=_wsclean_cmd(outdir),
        output_dir=str(outdir),
        ms_path="in.ms",
        spw=1,
        threads=2,
        retries=3,
    )

    assert flaky.calls == 3 and result["attempts"] == 3
    assert ai.is_imaging_complete(outdir)
    pads = [call["env"]["ALMASIM_WSCLEAN_PAD"] for call in flaky.envs]
    assert pads == ["", "x", "xx"], "each attempt shifts the environment layout"
    log = ai.imaging_log_path(outdir).read_text()
    assert "# attempt 1 died with signal 11; retrying (3 left)" in log
    assert "# attempt 2 died with signal 11; retrying (2 left)" in log
    assert not list(outdir.glob("*.tmp")), "reorder leftovers are removed between attempts"


@pytest.mark.unit
def test_run_wsclean_task_gives_up_after_retries(tmp_path, monkeypatch):
    flaky = _FlakyWsclean(crashes=10)
    monkeypatch.setattr(ai.subprocess, "Popen", flaky)
    outdir = tmp_path / "ms" / "SPW-1"
    with pytest.raises(RuntimeError, match="after 3 attempts"):
        ai.run_wsclean_task(
            command=_wsclean_cmd(outdir),
            output_dir=str(outdir),
            ms_path="in.ms",
            spw=1,
            threads=2,
            retries=2,
        )
    assert flaky.calls == 3
    payload = json.loads(ai.imaging_failure_marker_path(outdir).read_text())
    assert payload["returncode"] == -11 and "after 3 attempts" in payload["error"]


@pytest.mark.unit
def test_run_wsclean_task_does_not_retry_error_exits(tmp_path, monkeypatch):
    fake = _FakeWsclean(returncode=255, lines=["+ >>> Error opening meta file"])
    monkeypatch.setattr(ai.subprocess, "Popen", fake)
    outdir = tmp_path / "ms" / "SPW-1"
    with pytest.raises(RuntimeError, match="return code 255"):
        ai.run_wsclean_task(
            command=_wsclean_cmd(outdir),
            output_dir=str(outdir),
            ms_path="in.ms",
            spw=1,
            threads=2,
            retries=3,
        )
    assert len(fake.calls) == 1


@pytest.mark.unit
def test_build_imaging_tasks_single_window_default(tmp_path):
    csv_path = tmp_path / "params.csv"
    pd.DataFrame(
        {
            "filename": ["/data/uid___A001_X1_X1.ms.split.cal"],
            "spectral_window_id": [5],
            "reference_frequency": [100.0e9],
            "fov_per_frequency": [8.0],
            "max_baseline_size": [100.0],
            "synthetized_beam_size": [2.0],
            "target_field_ids": ["2,3"],
        }
    ).to_csv(csv_path, index=False)
    tasks, _ = cli_image.build_imaging_tasks(
        pd.read_csv(csv_path),
        tmp_path / "out",
        fov_fraction=1.5,
        beam_sampling=8,
        num_cores=10,
        max_cores_per_node=96,
    )
    cmd = tasks[0].command
    assert cmd[-1] == ai.SINGLE_WINDOW_PLACEHOLDER
    assert "-no-reorder" in cmd and "-spws" not in cmd and "-field" not in cmd
    assert tasks[0].field_ids == (2, 3) and tasks[0].spw == 5


@pytest.mark.unit
def test_run_wsclean_task_single_window_extracts_and_cleans_up(tmp_path, monkeypatch):
    fake = _FakeWsclean(returncode=0)
    monkeypatch.setattr(ai.subprocess, "Popen", fake)
    calls = []

    def fake_extract(ms_path, spw, field_ids, out_path):
        calls.append((ms_path, spw, list(field_ids), Path(out_path)))
        Path(out_path).mkdir(parents=True)
        return {"rows": 10, "bytes": 1, "data_desc_ids": [spw]}

    monkeypatch.setattr(ai, "extract_single_window_ms", fake_extract)
    outdir = tmp_path / "ms" / "SPW-5"
    cmd = ["wsclean", "-name", str(outdir / "wsclean"), "-no-reorder", ai.SINGLE_WINDOW_PLACEHOLDER]

    ai.run_wsclean_task(
        command=cmd,
        output_dir=str(outdir),
        ms_path="/data/in.ms",
        spw=5,
        threads=2,
        single_window=True,
        field_ids=[2, 3],
    )
    assert calls == [("/data/in.ms", 5, [2, 3], outdir / "spw5.single.ms")]
    used = fake.calls[0]["cmd"]
    assert used[-1] == str(outdir / "spw5.single.ms")
    assert not (outdir / "spw5.single.ms").exists(), "the per-task MS is removed afterwards"
    assert ai.is_imaging_complete(outdir)


@pytest.mark.unit
def test_run_wsclean_task_single_window_extraction_failure_is_marked(tmp_path, monkeypatch):
    def failing_extract(*args, **kwargs):
        raise RuntimeError("No visibilities for spectral window 5")

    monkeypatch.setattr(ai, "extract_single_window_ms", failing_extract)
    monkeypatch.setattr(ai.subprocess, "Popen", lambda *a, **k: pytest.fail("WSClean must not run"))
    outdir = tmp_path / "ms" / "SPW-5"
    with pytest.raises(RuntimeError, match="Could not extract spectral window 5"):
        ai.run_wsclean_task(
            command=["wsclean", "-name", str(outdir / "wsclean"), ai.SINGLE_WINDOW_PLACEHOLDER],
            output_dir=str(outdir),
            ms_path="/data/in.ms",
            spw=5,
            threads=2,
            single_window=True,
        )
    payload = json.loads(ai.imaging_failure_marker_path(outdir).read_text())
    assert "No visibilities" in payload["error"]
