"""The predict command (MADIV-28 T8.2: split from test_cli.py)."""

from __future__ import annotations

from typer.testing import CliRunner

from almasim import cli
from almasim.cli import image as cli_image
from almasim.cli import predict as cli_predict

runner = CliRunner()


def test_predict_ms_from_image_single_ms_invokes_prediction(monkeypatch, tmp_path):
    """predict ms-from-image should handle a single MS without slurm."""
    ms_dir = tmp_path / "sample.ms"
    ms_dir.mkdir()
    out_dir = tmp_path / "models"
    out_dir.mkdir()

    captured: dict[str, object] = {}

    def _fake_predict_all_models_for_ms(input_ms, output_directory):
        captured["input_ms"] = input_ms
        captured["output_directory"] = output_directory
        return [output_directory / input_ms.stem / "SPW-0" / "sample.ms.predicted"]

    monkeypatch.setattr(cli_predict, "_predict_all_models_for_ms", _fake_predict_all_models_for_ms)

    result = runner.invoke(
        cli.app,
        ["predict", "ms-from-image", str(ms_dir), str(out_dir), "--no-use-slurm"],
    )

    assert result.exit_code == 0
    assert captured["input_ms"] == ms_dir
    assert captured["output_directory"] == out_dir


def test_predict_ms_from_image_slurm_dispatches_one_job_per_ms(monkeypatch, tmp_path):
    """predict ms-from-image should submit one Slurm command per MS in a folder."""
    ms_dir = tmp_path / "inputs"
    ms_dir.mkdir()
    (ms_dir / "a.ms").mkdir()
    (ms_dir / "b.ms").mkdir()
    out_dir = tmp_path / "models"
    out_dir.mkdir()

    captured: dict[str, object] = {}

    def _fake_run_with_slurm_cluster(*args, **kwargs):
        captured["commands"] = args[0]

    monkeypatch.setattr(cli_image, "_run_commands_with_slurm_cluster", _fake_run_with_slurm_cluster)

    result = runner.invoke(
        cli.app,
        ["predict", "ms-from-image", str(ms_dir), str(out_dir)],
    )

    assert result.exit_code == 0
    assert len(captured["commands"]) == 2
    assert captured["commands"][0][1][:3] == ["almasim", "predict", "ms-from-image"]
