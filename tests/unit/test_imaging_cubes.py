"""Spectral-cube imaging: channel choice, per-channel stacking, continuum subtraction."""

from __future__ import annotations

import json
import math
from pathlib import Path

import click
import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from typer.testing import CliRunner

from almasim import cli, cli_image
from almasim.services.imaging import archive_imaging as ai

runner = CliRunner()


def _row(**overrides) -> pd.Series:
    """A Hanning-smoothed FDM window: 3840 x 488 kHz channels, 977 kHz resolution."""
    values = {
        "filename": "/data/uid___A002_X1_X2.ms.split.cal",
        "spectral_window_id": 25,
        "reference_frequency": 230.0e9,
        "fov_per_frequency": 28.0,
        "synthetized_beam_size": 0.5,
        "target_field_ids": "2",
        "n_channels": 3840,
        "channel_width_hz": 488.28125e3,
        "spectral_resolution_hz": 976.5625e3,
        "total_bandwidth_hz": 1875.0e6,
    }
    values.update(overrides)
    return pd.Series(values)


@pytest.mark.unit
def test_cube_channels_nyquist_keeps_native_hanning_channels():
    options = cli_image.CubeOptions(max_cube_gb=math.inf)
    assert cli_image.cube_channels(_row(), 100, options) == 3840


@pytest.mark.unit
def test_cube_channels_spectral_sampling_one_halves_hanning_channels():
    options = cli_image.CubeOptions(spectral_sampling=1.0, max_cube_gb=math.inf)
    assert cli_image.cube_channels(_row(), 100, options) == 1920


@pytest.mark.unit
def test_cube_channels_averaged_window_is_never_upsampled():
    # Online-averaged window: resolution ~ width, so Nyquist asks for half
    # the width; the native channel is the floor.
    row = _row(n_channels=480, channel_width_hz=3906.25e3, spectral_resolution_hz=3904.3e3)
    options = cli_image.CubeOptions(max_cube_gb=math.inf)
    assert cli_image.cube_channels(row, 100, options) == 480


@pytest.mark.unit
def test_cube_channels_velocity_resolution_widens_channels():
    # 2 km/s at 230 GHz is 1.534 MHz: 3.14 native channels, so 1875 MHz /
    # 1.534 MHz = 1222 channels, rounded down to a divisor of 3840.
    options = cli_image.CubeOptions(velocity_resolution_km_s=2.0, max_cube_gb=math.inf)
    channels = cli_image.cube_channels(_row(), 100, options)
    assert channels == 960
    assert 3840 % channels == 0


@pytest.mark.unit
def test_cube_channels_size_cap_uses_fewer_channels_on_big_images():
    # 2000 x 2000 x 4 bytes = 16 MB per plane; 1 GB fits 62 planes, and the
    # largest divisor of 3840 not above 62 is 60.
    options = cli_image.CubeOptions(max_cube_gb=1.0)
    assert cli_image.cube_channels(_row(), 2000, options) == 60
    assert cli_image.cube_channels(_row(), 100, options) == 3840


@pytest.mark.unit
def test_cube_channels_needs_the_spectral_columns():
    row = _row().drop(["n_channels", "channel_width_hz"])
    with pytest.raises(ValueError, match="n_channels, channel_width_hz"):
        cli_image.cube_channels(row, 100, cli_image.CubeOptions())


@pytest.mark.unit
def test_command_args_add_channels_out_only_for_cubes():
    plain = cli_image.imaging_parameter_to_command_arg(_row(), 1.0, 4.0)
    assert "-channels-out" not in plain
    cube = cli_image.imaging_parameter_to_command_arg(
        _row(), 1.0, 4.0, cube=cli_image.CubeOptions(velocity_resolution_km_s=2.0)
    )
    assert cube[cube.index("-channels-out") + 1] == "960"
    one = cli_image.imaging_parameter_to_command_arg(
        _row(n_channels=1, channel_width_hz=1e9, spectral_resolution_hz=1e9),
        1.0,
        4.0,
        cube=cli_image.CubeOptions(),
    )
    assert "-channels-out" not in one, "a single-channel window stays one plane"


def _write_channel(path: Path, value: np.ndarray, freq: float, *, bmaj: float | None = None):
    header = fits.Header()
    header["CTYPE1"], header["CTYPE2"] = "RA---SIN", "DEC--SIN"
    header["CTYPE3"], header["CRVAL3"], header["CDELT3"] = "FREQ", freq, 1e6
    header["CTYPE4"], header["CRVAL4"] = "STOKES", 1.0
    if bmaj is not None:
        header["BMAJ"], header["BMIN"], header["BPA"] = bmaj, bmaj / 2, 30.0
    fits.PrimaryHDU(value.reshape(1, 1, *value.shape).astype(np.float32), header).writeto(
        path, overwrite=True
    )


@pytest.mark.unit
def test_stack_channel_images_builds_cubes_with_a_beam_table(tmp_path):
    for channel in range(3):
        plane = np.full((4, 5), float(channel))
        freq = 230e9 + channel * 1e6
        _write_channel(tmp_path / f"wsclean-cube-{channel:04d}-image.fits", plane, freq, bmaj=1e-4)
        _write_channel(tmp_path / f"wsclean-cube-{channel:04d}-psf.fits", plane, freq)
        _write_channel(tmp_path / f"wsclean-cube-{channel:04d}-model.fits", plane, freq)
    _write_channel(tmp_path / "wsclean-image.fits", np.zeros((4, 5)), 230e9)  # continuum

    written = ai.stack_channel_images(tmp_path, prefix="wsclean-cube", keep=("image", "psf"))

    assert written == {"image": 3, "psf": 3}
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "wsclean-cube-image.fits",
        "wsclean-cube-psf.fits",
        "wsclean-image.fits",
    ], "per-channel files and the unwanted model are gone; the continuum is untouched"
    with fits.open(tmp_path / "wsclean-cube-image.fits") as hdus:
        assert hdus[0].data.shape == (1, 3, 4, 5)
        assert [hdus[0].data[0, c, 0, 0] for c in range(3)] == [0.0, 1.0, 2.0]
        assert hdus[0].header["CRVAL3"] == 230e9 and hdus[0].header["CDELT3"] == 1e6
        assert hdus[0].header["CASAMBM"] is True
        assert hdus["BEAMS"].data["BMAJ"] == pytest.approx([0.36] * 3)
        assert list(hdus["BEAMS"].data["CHAN"]) == [0, 1, 2]
    with fits.open(tmp_path / "wsclean-cube-psf.fits") as hdus:
        assert len(hdus) == 1, "no beam keywords, no BEAMS table"


@pytest.mark.unit
def test_stack_channel_images_rejects_uneven_channels(tmp_path):
    for channel, freq in enumerate((230e9, 230.001e9, 230.005e9)):
        _write_channel(tmp_path / f"w-{channel:04d}-image.fits", np.zeros((2, 2)), freq)
    with pytest.raises(RuntimeError, match="not evenly spaced"):
        ai.stack_channel_images(tmp_path, prefix="w")


def _cube_file(path: Path, cube: np.ndarray) -> Path:
    header = fits.Header()
    header["CTYPE3"], header["CRVAL3"], header["CDELT3"] = "FREQ", 230e9, 1e6
    fits.PrimaryHDU(cube[None].astype(np.float32), header).writeto(path)
    return path


@pytest.mark.unit
@pytest.mark.parametrize("order", [0, 1])
def test_subtract_continuum_removes_the_continuum_and_keeps_the_line(tmp_path, order):
    rng = np.random.default_rng(1)
    n_channels = 120
    channel = np.arange(n_channels)[:, None, None]
    continuum = 5.0 + (0.02 * channel if order == 1 else 0.0)
    line = np.where((channel >= 50) & (channel < 60), 3.0, 0.0) * np.ones((1, 6, 7))
    cube = continuum + line + rng.normal(0, 0.05, (n_channels, 6, 7))
    cube[10] = 0.0  # a fully flagged channel
    source = _cube_file(tmp_path / "cube.fits", cube)

    fit = ai.subtract_continuum(source, tmp_path / "contsub.fits", order=order, rows_per_chunk=4)

    result = fits.getdata(tmp_path / "contsub.fits")[0]
    assert np.isnan(result[10]).all(), "a blank channel stays blank"
    off_line = np.delete(result, [10, *range(50, 60)], axis=0)
    assert abs(np.nanmean(off_line)) < 0.02
    assert np.nanmean(result[50:60]) == pytest.approx(3.0, abs=0.05)
    assert 0.05 < fit["line_fraction"] < 0.15, "about the 10 line channels of 119 are clipped"
    assert "continuum subtracted" in str(fits.getheader(tmp_path / "contsub.fits")["HISTORY"])


class _FakeCubeWsclean:
    """``subprocess.Popen`` stand-in writing what WSClean writes for each command."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def __call__(self, cmd, **kwargs):
        self.calls.append(list(cmd))
        name = Path(cmd[cmd.index("-name") + 1])
        if "-channels-out" in cmd:
            n = int(cmd[cmd.index("-channels-out") + 1])
            products = ["image", "residual", "psf", "model"] + ["dirty"] * ("-no-dirty" not in cmd)
            for c in range(n):
                plane = np.full((3, 3), 2.0 + 0.01 * c)
                for product in products:
                    path = name.with_name(f"{name.name}-{c:04d}-{product}.fits")
                    _write_channel(path, plane, 230e9 + c * 1e6, bmaj=1e-4)
            for product in products:
                mfs = name.with_name(f"{name.name}-MFS-{product}.fits")
                _write_channel(mfs, np.ones((3, 3)), 230e9)
        else:
            _write_channel(name.with_name(f"{name.name}-image.fits"), np.ones((3, 3)), 230e9)

        class _Proc:
            stdout = iter(["WSClean version 3.7\n"])

            @staticmethod
            def wait():
                return 0

            @staticmethod
            def kill():
                pass

        return _Proc()


def _tasks(tmp_path, **kwargs):
    parameters = pd.DataFrame([_row(n_channels=8, filename=str(tmp_path / "x.ms.split.cal"))])
    common = dict(fov_fraction=1.0, beam_sampling=4.0, num_cores=2, max_cores_per_node=4)
    kwargs.setdefault("single_window", False)
    return cli_image.build_imaging_tasks(parameters, tmp_path / "out", **common, **kwargs)


@pytest.mark.unit
def test_run_wsclean_task_makes_continuum_cube_and_contsub(tmp_path, monkeypatch):
    fake = _FakeCubeWsclean()
    monkeypatch.setattr(ai.subprocess, "Popen", fake)
    products = ("image", "residual", "psf")
    (task,), _ = _tasks(
        tmp_path, cube=cli_image.CubeOptions(), cube_products=products, single_window=True
    )

    result = ai.run_wsclean_task(
        command=task.command,
        cube_command=task.cube_command,
        output_dir=str(task.output_dir),
        ms_path=str(task.ms_path),
        spw=task.spw,
        threads=2,
        cube_products=products,
        contsub_order=0,
    )

    out = task.output_dir
    assert [Path(c[c.index("-name") + 1]).name for c in fake.calls] == ["wsclean", "wsclean-cube"]
    assert "-no-dirty" in fake.calls[1]
    assert "-no-reorder" in fake.calls[0], "the continuum keeps the contiguous reader"
    assert "-no-reorder" not in fake.calls[1], "the cube reorders"
    assert sorted(p.name for p in out.iterdir()) == [
        "wsclean-cube-image.contsub.fits",
        "wsclean-cube-image.fits",
        "wsclean-cube-psf.fits",
        "wsclean-cube-residual.fits",
        "wsclean-image.fits",
    ]
    assert result["channels"] == 8
    assert fits.getdata(out / "wsclean-cube-image.fits").shape == (1, 8, 3, 3)
    marker = json.loads(ai.imaging_marker_path(out).read_text())
    assert len(marker["commands"]) == 2 and len(marker["images"]) == 3
    assert ai.is_imaging_complete(out, (ai.IMAGE_FILENAME, ai.CUBE_IMAGE_FILENAME))
    assert "stacked channel images into cubes: image (8 channels)" in (
        ai.imaging_log_path(out).read_text()
    )


@pytest.mark.unit
def test_cube_only_run_adds_cubes_next_to_existing_continuum(tmp_path, monkeypatch):
    fake = _FakeCubeWsclean()
    monkeypatch.setattr(ai.subprocess, "Popen", fake)
    (continuum_task,), _ = _tasks(tmp_path)
    ai.run_wsclean_task(
        command=continuum_task.command,
        output_dir=str(continuum_task.output_dir),
        ms_path="x",
        spw=25,
        threads=2,
    )
    continuum_bytes = (continuum_task.output_dir / ai.IMAGE_FILENAME).read_bytes()
    assert _tasks(tmp_path) == ([], 1), "the continuum task is done"

    cube = cli_image.CubeOptions()
    (task,), skipped = _tasks(tmp_path, continuum=False, cube=cube)
    assert skipped == 0 and task.command == [], "only the cube is still missing"
    ai.run_wsclean_task(
        command=task.command,
        cube_command=task.cube_command,
        output_dir=str(task.output_dir),
        ms_path="x",
        spw=25,
        threads=2,
    )

    assert len(fake.calls) == 2
    assert (task.output_dir / ai.IMAGE_FILENAME).read_bytes() == continuum_bytes
    assert _tasks(tmp_path, continuum=False, cube=cube) == ([], 1)
    assert _tasks(tmp_path, cube=cube) == ([], 1), "continuum and cube both present"


@pytest.mark.unit
def test_build_imaging_tasks_needs_something_to_image(tmp_path):
    with pytest.raises(ValueError, match="Nothing to image"):
        _tasks(tmp_path, continuum=False)


@pytest.mark.unit
def test_image_from_ms_plan_only_fills_spectral_columns_and_reports(tmp_path, monkeypatch):
    csv = tmp_path / "params.csv"
    pd.DataFrame(
        [
            _row().drop(
                ["n_channels", "channel_width_hz", "spectral_resolution_hz", "total_bandwidth_hz"]
            )
        ]
    ).to_csv(csv, index=False)

    class _SpectralWindow:
        def nrows(self):
            return 26

        def getcell(self, name, row):
            return np.full(3840, -488.28125e3 if name == "CHAN_WIDTH" else 976.5625e3)

        def getcol(self, name):
            return np.full(26, 3840 if name == "NUM_CHAN" else 1875e6)

    opened = []
    monkeypatch.setattr(
        cli_image,
        "import_casacore_tables",
        lambda: lambda name, ack=False: opened.append(name) or _SpectralWindow(),
    )

    result = runner.invoke(
        cli.app,
        [
            "image",
            "image-from-ms",
            str(csv),
            str(tmp_path / "out"),
            "--cube",
            "--velocity-resolution",
            "2",
            "--plan-only",
        ],
    )

    assert result.exit_code == 0, result.output
    assert opened == ["/data/uid___A002_X1_X2.ms.split.cal::SPECTRAL_WINDOW"]
    assert "channels per cube min 960, median 960, max 960" in result.output
    assert "image, residual, psf, model, dirty" in result.output
    assert not (tmp_path / "out").exists(), "--plan-only images nothing"


@pytest.mark.unit
@pytest.mark.parametrize(("flags", "contsub_order"), [([], None), (["--contsub"], 1)])
def test_image_from_ms_cube_keeps_every_product_and_contsub_is_opt_in(
    tmp_path, monkeypatch, flags, contsub_order
):
    csv = tmp_path / "params.csv"
    pd.DataFrame([_row()]).to_csv(csv, index=False)
    seen = {}

    def _run(tasks, **kwargs):
        seen.update(kwargs, tasks=tasks)

    monkeypatch.setattr(cli_image, "run_imaging_tasks", _run)
    result = runner.invoke(
        cli.app,
        ["image", "image-from-ms", str(csv), str(tmp_path / "out"), "--cube", *flags],
    )

    assert result.exit_code == 0, result.output
    assert seen["cube_products"] == cli_image.CUBE_PRODUCTS
    assert seen["contsub_order"] == contsub_order
    (task,) = seen["tasks"]
    assert "-no-dirty" not in task.cube_command, "the dirty cube is made by default"


@pytest.mark.unit
def test_image_from_ms_rejects_contsub_without_cube(tmp_path):
    csv = tmp_path / "params.csv"
    pd.DataFrame([_row()]).to_csv(csv, index=False)
    result = runner.invoke(
        cli.app, ["image", "image-from-ms", str(csv), str(tmp_path / "out"), "--contsub"]
    )
    assert result.exit_code != 0
    # Rich styles the error box when it detects CI (GITHUB_ACTIONS/FORCE_COLOR).
    assert "need --cube" in click.unstyle(result.output)


@pytest.mark.unit
@pytest.mark.parametrize("flags", [["--auto-mask", "1"], ["--auto-threshold", "3"]])
def test_image_from_ms_rejects_a_mask_not_above_the_threshold(tmp_path, flags):
    csv = tmp_path / "params.csv"
    pd.DataFrame([_row()]).to_csv(csv, index=False)
    result = runner.invoke(
        cli.app, ["image", "image-from-ms", str(csv), str(tmp_path / "out"), *flags]
    )
    assert result.exit_code != 0
    assert "--auto-mask must be above --auto-threshold" in click.unstyle(result.output)


@pytest.mark.unit
def test_command_args_join_channels_only_on_multichannel_cubes():
    joined = cli_image.imaging_parameter_to_command_arg(
        _row(), 1.0, 4.0, cube=cli_image.CubeOptions(), join_channels=True
    )
    assert "-join-channels" in joined and "-channels-out" in joined
    assert "-join-channels" not in cli_image.imaging_parameter_to_command_arg(
        _row(), 1.0, 4.0, join_channels=True
    ), "a continuum image has nothing to join"
    assert "-join-channels" not in cli_image.imaging_parameter_to_command_arg(
        _row(), 1.0, 4.0, cube=cli_image.CubeOptions()
    )


@pytest.mark.unit
def test_image_from_ms_rejects_join_channels_without_cube(tmp_path):
    csv = tmp_path / "params.csv"
    pd.DataFrame([_row()]).to_csv(csv, index=False)
    result = runner.invoke(
        cli.app, ["image", "image-from-ms", str(csv), str(tmp_path / "out"), "--join-channels"]
    )
    assert result.exit_code != 0
    assert "need --cube" in click.unstyle(result.output)


@pytest.mark.unit
def test_continuum_restart_commands_write_the_model_then_continue():
    first, second = ai.continuum_restart_commands(
        ["wsclean", "-name", "w", "-no-update-model-required", "x.ms"]
    )
    assert first == ["wsclean", "-name", "w", "-update-model-required", "x.ms"]
    assert second == ["wsclean", "-name", "w", "-update-model-required", "-continue", "x.ms"]

    first, _ = ai.continuum_restart_commands(["wsclean", "-name", "w", "x.ms"])
    assert first[-2:] == ["-update-model-required", "x.ms"]


@pytest.mark.unit
def test_run_wsclean_task_cleans_the_continuum_in_two_passes(tmp_path, monkeypatch):
    fake = _FakeCubeWsclean()
    monkeypatch.setattr(ai.subprocess, "Popen", fake)
    monkeypatch.setattr(
        ai, "extract_single_window_ms_subprocess", lambda *a, **k: {"rows": 1, "bytes": 0}
    )
    (task,), _ = _tasks(tmp_path, cube=cli_image.CubeOptions(), single_window=True)

    ai.run_wsclean_task(
        command=task.command,
        cube_command=task.cube_command,
        output_dir=str(task.output_dir),
        ms_path=str(task.ms_path),
        spw=task.spw,
        threads=2,
        single_window=True,
        continuum_restart=True,
    )

    names = [Path(c[c.index("-name") + 1]).name for c in fake.calls]
    assert names == ["wsclean", "wsclean", "wsclean-cube"]
    first, second, cube = fake.calls
    assert "-update-model-required" in first and "-continue" not in first
    assert second[-2] == "-continue" and second[-1] == first[-1]
    assert "-continue" not in cube and "-no-update-model-required" in cube
    marker = json.loads(ai.imaging_marker_path(task.output_dir).read_text())
    assert len(marker["commands"]) == 3


@pytest.mark.unit
def test_continuum_restart_needs_the_single_window_copy():
    with pytest.raises(ValueError, match="needs single_window"):
        ai.run_wsclean_task(
            command=["wsclean", "-name", "w", "x.ms"],
            output_dir="x",
            ms_path="x",
            spw=1,
            threads=1,
            continuum_restart=True,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("flags", "expected"),
    [([], True), (["--no-continuum-restart"], False), (["--no-single-window-ms"], False)],
)
def test_image_from_ms_continuum_restart_default(tmp_path, monkeypatch, flags, expected):
    csv = tmp_path / "params.csv"
    pd.DataFrame([_row()]).to_csv(csv, index=False)
    seen = {}
    monkeypatch.setattr(cli_image, "run_imaging_tasks", lambda tasks, **k: seen.update(k))
    result = runner.invoke(
        cli.app, ["image", "image-from-ms", str(csv), str(tmp_path / "out"), *flags]
    )
    assert result.exit_code == 0, result.output
    assert seen["continuum_restart"] is expected


def _line_channels(out, n_channels=40, size=64, seed=8):
    """Channel images of a faint line: 1 sigma per channel, ~6 sigma summed."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    blob = np.exp(-((yy - 32) ** 2 + (xx - 20) ** 2) / (2 * 2.0**2))
    out.mkdir(parents=True, exist_ok=True)
    for c in range(n_channels):
        plane = rng.normal(0, 1, (size, size)) + 1.0 * blob
        path = out / f"wsclean-cube-{c:04d}-image.fits"
        _write_channel(path, plane, 230e9 + c * 1e6, bmaj=4 * 2.5e-4)  # 4 pixels
        with fits.open(path, mode="update") as hdus:
            hdus[0].header["CDELT1"], hdus[0].header["CDELT2"] = -2.5e-4, 2.5e-4


@pytest.mark.unit
def test_line_mask_from_channels_finds_a_line_no_channel_shows(tmp_path):
    _line_channels(tmp_path)

    pixels = ai.line_mask_from_channels(tmp_path, tmp_path / "mask.fits")

    mask = fits.getdata(tmp_path / "mask.fits").squeeze().astype(bool)
    assert pixels == mask.sum() and mask[32, 20], "the blob is in the mask"
    assert mask[32, 24] and not mask[32, 40], "grown by about one beam, no further"
    assert mask.mean() < 0.1


@pytest.mark.unit
def test_line_mask_from_channels_without_channels_or_signal(tmp_path):
    assert ai.line_mask_from_channels(tmp_path, tmp_path / "mask.fits") == 0
    for c in range(3):
        _write_channel(tmp_path / f"wsclean-cube-{c:04d}-image.fits", np.zeros((8, 8)), 230e9)
    assert ai.line_mask_from_channels(tmp_path, tmp_path / "mask.fits") == 0
    assert not (tmp_path / "mask.fits").exists()


@pytest.mark.unit
def test_line_mask_pass_continues_inside_the_mask(tmp_path, monkeypatch):
    cube = ["wsclean", "-name", "c", "-auto-mask", "3.0", "-auto-threshold", "1.0", "x.ms"]
    with (tmp_path / "log").open("w") as log:
        monkeypatch.setattr(ai, "line_mask_from_channels", lambda out, path: 120)
        second = ai._line_mask_pass(cube, tmp_path, log)
        monkeypatch.setattr(ai, "line_mask_from_channels", lambda out, path: 0)
        assert ai._line_mask_pass(cube, tmp_path, log) is None
    assert second == [
        "wsclean",
        "-name",
        "c",
        "-auto-threshold",
        "1.0",
        "-fits-mask",
        str(tmp_path / ai.LINE_MASK_FILENAME),
        "-continue",
        "x.ms",
    ]
    log = (tmp_path / "log").read_text()
    assert "120 pixels; second pass" in log and "nothing above 3 sigma" in log


@pytest.mark.unit
def test_run_wsclean_task_cleans_the_cube_again_inside_the_line_mask(tmp_path, monkeypatch):
    fake = _FakeCubeWsclean()
    monkeypatch.setattr(ai.subprocess, "Popen", fake)
    monkeypatch.setattr(
        ai, "extract_single_window_ms_subprocess", lambda *a, **k: {"rows": 1, "bytes": 0}
    )
    monkeypatch.setattr(ai, "line_mask_from_channels", lambda out, path: 50)
    (task,), _ = _tasks(tmp_path, cube=cli_image.CubeOptions(), single_window=True)

    ai.run_wsclean_task(
        command=task.command,
        cube_command=task.cube_command,
        output_dir=str(task.output_dir),
        ms_path=str(task.ms_path),
        spw=task.spw,
        threads=2,
        single_window=True,
        cube_line_mask=True,
    )

    continuum, cube, second = fake.calls
    assert "-no-update-model-required" in continuum
    assert "-update-model-required" in cube and "-continue" not in cube
    assert second[-2] == "-continue" and "-fits-mask" in second and "-auto-mask" not in second
    marker = json.loads(ai.imaging_marker_path(task.output_dir).read_text())
    assert len(marker["commands"]) == 3
    assert fits.getdata(task.output_dir / "wsclean-cube-image.fits").shape[1] == 8


@pytest.mark.unit
def test_cube_line_mask_needs_the_single_window_copy():
    with pytest.raises(ValueError, match="cube_line_mask writes MODEL_DATA"):
        ai.run_wsclean_task(
            command=[],
            cube_command=["wsclean", "-name", "c", "x.ms"],
            output_dir="x",
            ms_path="x",
            spw=1,
            threads=1,
            cube_line_mask=True,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        (["--cube"], True),
        (["--cube", "--no-cube-line-mask"], False),
        (["--cube", "--no-single-window-ms"], False),
        ([], False),
    ],
)
def test_image_from_ms_cube_line_mask_default(tmp_path, monkeypatch, flags, expected):
    csv = tmp_path / "params.csv"
    pd.DataFrame([_row()]).to_csv(csv, index=False)
    seen = {}
    monkeypatch.setattr(cli_image, "run_imaging_tasks", lambda tasks, **k: seen.update(k))
    result = runner.invoke(
        cli.app, ["image", "image-from-ms", str(csv), str(tmp_path / "out"), *flags]
    )
    assert result.exit_code == 0, result.output
    assert seen["cube_line_mask"] is expected


@pytest.mark.unit
def test_line_mask_from_channels_without_a_pixel_scale_is_not_grown(tmp_path):
    rng = np.random.default_rng(3)
    for c in range(20):
        plane = rng.normal(0, 1, (32, 32))
        plane[16, 16] += 3.0
        _write_channel(tmp_path / f"wsclean-cube-{c:04d}-image.fits", plane, 230e9, bmaj=1e-3)
    pixels = ai.line_mask_from_channels(tmp_path, tmp_path / "mask.fits")
    mask = fits.getdata(tmp_path / "mask.fits").squeeze().astype(bool)
    assert mask[16, 16] and pixels <= 5, "the source pixel, plus at most a stray noise peak"
