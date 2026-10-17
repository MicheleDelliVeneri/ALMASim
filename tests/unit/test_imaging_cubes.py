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
    fits.PrimaryHDU(value.reshape(1, 1, *value.shape).astype(np.float32), header).writeto(path)


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


def _spectrum_with_line(n=400, line=slice(180, 200), noise=0.01, seed=3):
    rng = np.random.default_rng(seed)
    x = np.linspace(-1, 1, n)
    spectrum = 1.0 + 0.1 * x + rng.normal(0, noise, n)
    spectrum[line] += 0.5
    return spectrum


@pytest.mark.unit
def test_line_free_channels_drops_the_line_and_its_wings():
    spectrum = _spectrum_with_line()
    usable = np.ones(spectrum.size, bool)
    usable[:5] = False  # flagged band edge

    line_free = ai.line_free_channels(spectrum, usable)

    assert not line_free[:5].any(), "unusable channels are never line-free"
    assert not line_free[176:204].any(), "the line and its wings"
    off_line = np.r_[5:150, 230:400]
    assert line_free[off_line].mean() > 0.8, "noise spikes may cost a few channels, no more"


@pytest.mark.unit
def test_line_free_channels_finds_faint_broad_wings_by_smoothing():
    rng = np.random.default_rng(2)
    n = 2048
    spectrum = 1.0 + rng.normal(0, 0.1, n)
    spectrum[1000:1100] += 0.06  # 0.6 sigma per channel, 6 sigma over the 100 channels
    usable = np.ones(n, bool)

    line_free = ai.line_free_channels(spectrum, usable)
    assert line_free[1000:1100].mean() < 0.2, "the wing is found on the smoothed spectrum"
    single = ai.line_free_channels(spectrum, usable, widths=(1,))
    assert single[1000:1100].mean() > 0.8, "channel by channel it is not"


@pytest.mark.unit
@pytest.mark.parametrize("order", [0, 1])
def test_fit_uv_continuum_fits_each_row_over_the_line_free_channels(order):
    rng = np.random.default_rng(5)
    n_rows, n_channels, n_corr = 6, 64, 2
    x = np.linspace(-1, 1, n_channels)[None, :, None]
    offset = rng.normal(0, 1, (n_rows, 1, n_corr)) + 1j * rng.normal(0, 1, (n_rows, 1, n_corr))
    slope = (0.3 + 0.2j) * (order == 1)
    continuum = offset + slope * x
    data = (continuum + 0j).astype(np.complex64)
    data[:, 30:34] += 5.0  # the line
    data[:, 50] = 1e6  # a flagged spike
    flag = np.zeros(data.shape, bool)
    flag[:, 50] = True
    flag[0] = True  # a fully flagged row
    weight = np.ones(data.shape)
    line_free = np.ones(n_channels, bool)
    line_free[28:36] = False

    fitted = ai.fit_uv_continuum(data, weight, flag, line_free, order)

    assert fitted.shape == data.shape and fitted.dtype == data.dtype
    assert np.all(fitted[0] == 0), "no usable channel: no continuum"
    np.testing.assert_allclose(fitted[1:], continuum[1:].repeat(1, axis=1), atol=1e-4)
    assert np.allclose((data - fitted)[1:, 30:34], 5.0, atol=1e-4), "the line is kept"


@pytest.mark.unit
def test_fit_uv_continuum_rejects_other_orders():
    with pytest.raises(ValueError, match="order"):
        ai.fit_uv_continuum(
            np.zeros((1, 4, 1)), np.ones((1, 4, 1)), np.zeros((1, 4, 1), bool), np.ones(4, bool), 2
        )


@pytest.mark.unit
def test_channel_ranges_lists_runs_casa_style():
    mask = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1], bool)
    assert ai._channel_ranges(mask) == "1~2;5~5;7~9"
    assert ai._channel_ranges(np.zeros(4, bool)) == ""


def _uv_task(tmp_path, monkeypatch, fit):
    fake = _FakeCubeWsclean()
    monkeypatch.setattr(ai.subprocess, "Popen", fake)
    monkeypatch.setattr(
        ai, "extract_single_window_ms_subprocess", lambda *a, **k: {"rows": 10, "bytes": 0}
    )
    calls = []
    monkeypatch.setattr(
        ai, "subtract_uv_continuum_subprocess", lambda ms, **k: calls.append((ms, k)) or fit
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
        uv_contsub_order=1,
    )
    return fake, calls, ai.imaging_log_path(task.output_dir).read_text()


@pytest.mark.unit
def test_run_wsclean_task_images_continuum_and_lines_from_their_columns(tmp_path, monkeypatch):
    fit = {
        "source_column": "DATA",
        "order": 1,
        "seconds": 1.0,
        "usable_channels": 8,
        "line_free_channels": 6,
        "line_ranges": "3~4",
    }
    fake, calls, log = _uv_task(tmp_path, monkeypatch, fit)

    ((ms, kwargs),) = calls
    assert Path(ms).name == "spw25.single.ms" and kwargs["order"] == 1
    continuum, cube = fake.calls
    assert continuum[continuum.index("-data-column") + 1] == ai.CONTINUUM_COLUMN
    assert cube[cube.index("-data-column") + 1] == ai.LINE_COLUMN
    assert continuum[-1] == cube[-1] and continuum[-1].endswith("spw25.single.ms")
    assert "6 of 8 channels line-free, lines in channels 3~4" in log


@pytest.mark.unit
def test_run_wsclean_task_keeps_the_data_when_too_few_channels_are_line_free(tmp_path, monkeypatch):
    fake, _, log = _uv_task(
        tmp_path, monkeypatch, {"skipped": "only 1 of 8 usable channels are line-free"}
    )
    assert all("-data-column" not in c for c in fake.calls)
    assert "continuum NOT subtracted from the visibilities: only 1 of 8" in log


@pytest.mark.unit
def test_run_wsclean_task_uv_contsub_needs_a_single_window_cube():
    with pytest.raises(ValueError, match="single_window and a cube_command"):
        ai.run_wsclean_task(
            command=["wsclean"],
            output_dir="x",
            ms_path="x",
            spw=1,
            threads=1,
            uv_contsub_order=1,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("flags", "message"),
    [
        (["--uv-contsub"], "need --cube"),
        (["--cube", "--uv-contsub", "--contsub"], "drop --contsub"),
        (["--cube", "--uv-contsub", "--no-single-window-ms"], "needs --single-window-ms"),
    ],
)
def test_image_from_ms_checks_uv_contsub(tmp_path, flags, message):
    csv = tmp_path / "params.csv"
    pd.DataFrame([_row()]).to_csv(csv, index=False)
    result = runner.invoke(
        cli.app, ["image", "image-from-ms", str(csv), str(tmp_path / "out"), *flags]
    )
    assert result.exit_code != 0
    assert message in click.unstyle(result.output)


@pytest.mark.unit
def test_image_from_ms_passes_uv_contsub_order(tmp_path, monkeypatch):
    csv = tmp_path / "params.csv"
    pd.DataFrame([_row()]).to_csv(csv, index=False)
    seen = {}
    monkeypatch.setattr(cli_image, "run_imaging_tasks", lambda tasks, **k: seen.update(k))
    result = runner.invoke(
        cli.app,
        [
            "image",
            "image-from-ms",
            str(csv),
            str(tmp_path / "out"),
            "--cube",
            "--uv-contsub",
            "--contsub-order",
            "0",
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen["uv_contsub_order"] == 0 and seen["contsub_order"] is None
