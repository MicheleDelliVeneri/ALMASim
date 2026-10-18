"""Cleaning-quality statistics and moment maps (almasim image clean-quality)."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from typer.testing import CliRunner

from almasim import cli
from almasim.services.imaging import archive_imaging as ai
from almasim.services.imaging import quality as q

runner = CliRunner()
SIZE, CELL_DEG, BEAM_PIX = 64, 1 / 3600, 4.0
SOURCE = (32, 30)


def _header(n_channels: int | None = None) -> fits.Header:
    header = fits.Header()
    header["CTYPE1"], header["CDELT1"], header["CRPIX1"] = "RA---SIN", -CELL_DEG, 33
    header["CTYPE2"], header["CDELT2"], header["CRPIX2"] = "DEC--SIN", CELL_DEG, 33
    header["BMAJ"], header["BMIN"], header["BPA"] = BEAM_PIX * CELL_DEG, BEAM_PIX * CELL_DEG, 0.0
    if n_channels:
        header["CTYPE3"], header["CRVAL3"], header["CDELT3"], header["CRPIX3"] = (
            "FREQ",
            230e9,
            1e6,
            1.0,
        )
    return header


def _blob(amplitude: float) -> np.ndarray:
    yy, xx = np.mgrid[:SIZE, :SIZE]
    sigma = BEAM_PIX / 2.355
    return amplitude * np.exp(-((yy - SOURCE[0]) ** 2 + (xx - SOURCE[1]) ** 2) / (2 * sigma**2))


def _write(path: Path, data: np.ndarray, header: fits.Header) -> None:
    fits.PrimaryHDU(data.astype(np.float32), header=header).writeto(path, overwrite=True)


def _task(root: Path, *, log_lines: bool = True, line_mask: bool = False, cube: bool = True):
    """A task: continuum 1 Jy with 5 % uncleaned; a 40-channel cube, line in channels 10-19."""
    rng = np.random.default_rng(3)
    task = root / "uid___A002_X1.ms.split" / "SPW-1"
    task.mkdir(parents=True)
    noise = rng.normal(0, 0.01, (SIZE, SIZE))
    residual = noise + _blob(0.05)
    model = np.zeros((SIZE, SIZE))
    model[SOURCE] = 0.95
    header = _header()
    _write(task / "wsclean-image.fits", (_blob(0.95) + residual)[None, None], header)
    _write(task / "wsclean-residual.fits", residual[None, None], header)
    _write(task / "wsclean-model.fits", model[None, None], header)
    log = [
        f"# /bin/wsclean -name {task}/wsclean -j 1 x.ms",
        "Estimated standard deviation of background noise: 90 mJy",
        "Estimated standard deviation of background noise: 26 mJy",
        f"# /bin/wsclean -name {task}/wsclean -j 1 -continue x.ms",
        "Estimated standard deviation of background noise: 25 mJy",
    ]
    if log_lines:
        log.append("# continuum subtracted from DATA (order 1, 1 s): lines in channels 10~19")
    (task.parent / "SPW-1.log").write_text("\n".join(log) + "\n")
    if cube:
        n = 40
        image = rng.normal(0, 0.01, (n, SIZE, SIZE))
        residual_cube = image.copy()
        cube_model = np.zeros((n, SIZE, SIZE))
        for c in range(10, 20):
            image[c] += _blob(0.5)
            residual_cube[c] += _blob(0.05)
            cube_model[c][SOURCE] = 0.45
        image[39] = residual_cube[39] = np.nan  # a fully flagged channel
        ch = _header(n)
        _write(task / "wsclean-cube-image.fits", image[None], ch)
        _write(task / "wsclean-cube-residual.fits", residual_cube[None], ch)
        _write(task / "wsclean-cube-model.fits", cube_model[None], ch)
        if line_mask:
            yy, xx = np.mgrid[:SIZE, :SIZE]
            mask = np.hypot(yy - SOURCE[0], xx - SOURCE[1]) <= 6
            _write(task / ai.LINE_MASK_FILENAME, mask[None, None].astype(float), _header())
    return task


@pytest.mark.unit
def test_continuum_quality_measures_the_flux_left_at_the_source(tmp_path):
    task = _task(tmp_path)
    stats = q.continuum_quality(task)

    assert stats["peak_pixel"] == list(SOURCE)
    assert stats["noise_jy_beam"] == pytest.approx(0.01, rel=0.15)
    assert stats["snr"] == pytest.approx(100, rel=0.15)
    assert 0.03 < stats["residual_share_1beam"] < 0.08, "the 5 % left uncleaned"
    assert stats["residual_core_sigma"] > 3 and stats["residual_peak_sigma"] > 4
    assert set(stats["residual_rings_sigma"]) == {"0-0.5", "0.5-1", "1-1.5", "1.5-2", "2-3", "3-5"}
    assert stats["residual_symmetric_sigma"] > stats["residual_antisymmetric_sigma"]
    assert stats["model_flux_jy"] == pytest.approx(0.95) and stats["model_components"] == 1
    assert stats["noise_at_mask_creation_ratio"] == pytest.approx(90 / 26)
    assert stats["beam_arcsec"] == [4.0, 4.0]


@pytest.mark.unit
def test_continuum_quality_needs_all_three_images(tmp_path):
    task = _task(tmp_path, cube=False)
    (task / "wsclean-model.fits").unlink()
    assert q.continuum_quality(task) is None


@pytest.mark.unit
def test_cube_quality_with_logged_line_channels_and_the_task_mask(tmp_path):
    task = _task(tmp_path, line_mask=True)
    stats = q.cube_quality(task)

    assert stats["channels"] == 40 and stats["usable_channels"] == 39
    assert stats["line_channels"] == 10 and stats["line_channels_from"] == "log"
    assert stats["line_mask_from"] == "task" and stats["line_mask_pixels"] > 50
    assert stats["channel_noise_mjy"] == pytest.approx(10, rel=0.2)
    assert 0.05 < stats["line_residual_share"] < 0.2, "10 % of the line left in the residual"
    assert stats["line_residual_peak_sigma"] > 3
    assert abs(stats["linefree_residual_sigma"]) < 3, "no continuum in the line-free channels"
    assert stats["model_flux_jy"] == pytest.approx(4.5, rel=1e-5)
    assert stats["moments"] == [
        "cube-mom0.fits",
        "cube-mom1.fits",
        "cube-mom8.fits",
        "cube-residual-line.fits",
        "cube-residual-linefree.fits",
    ]
    mom0 = fits.getdata(task / "quality" / "cube-mom0.fits")
    mom8 = fits.getdata(task / "quality" / "cube-mom8.fits")
    assert np.unravel_index(np.nanargmax(mom0), mom0.shape) == SOURCE
    assert mom8[SOURCE] == pytest.approx(0.5, abs=0.05)
    assert fits.getheader(task / "quality" / "cube-mom0.fits")["BUNIT"] == "Jy/beam.km/s"
    mom1 = fits.getdata(task / "quality" / "cube-mom1.fits")
    assert np.isfinite(mom1[SOURCE]) and np.isnan(mom1[0, 0])


@pytest.mark.unit
def test_cube_quality_finds_the_line_on_the_spectrum_and_builds_a_mask(tmp_path):
    task = _task(tmp_path, log_lines=False)
    stats = q.cube_quality(task, write_moments=False)

    assert stats["line_channels_from"] == "source spectrum"
    assert 10 <= stats["line_channels"] <= 20, "the line, widened by the finder's margins"
    assert stats["line_mask_from"] == "moment-0" and stats["line_mask_pixels"] > 0
    assert "moments" not in stats and not (task / "quality").exists()


@pytest.mark.unit
def test_cube_quality_without_cubes(tmp_path):
    assert q.cube_quality(_task(tmp_path, cube=False)) is None


@pytest.mark.unit
def test_logged_line_channels_follow_averaged_cubes(tmp_path):
    task = tmp_path / "SPW-1"
    task.mkdir()
    log = tmp_path / "SPW-1.log"
    log.write_text(
        "# continuum subtracted ...: 60 of 80 channels line-free, lines in channels 2~19\n"
    )
    line = q._logged_line_channels(task, 40)  # 80 native channels averaged 2 by 2
    assert np.flatnonzero(line).tolist() == list(range(1, 10)), "lines low in the band"
    log.write_text(
        "# continuum subtracted ...: 1700 of 1920 channels line-free, lines in channels 100~199\n"
    )
    line = q._logged_line_channels(task, 2048)  # native channels, 128 of them flagged
    assert np.flatnonzero(line).tolist() == list(range(100, 200))
    log.write_text("lines in channels 20~39;60~61\n")  # an older log without the counts
    assert np.flatnonzero(q._logged_line_channels(task, 40)).tolist() == list(range(10, 20)) + [30]

    log.write_text("lines in channels none\n")
    assert not q._logged_line_channels(task, 40).any()
    log.write_text("nothing about lines\n")
    assert q._logged_line_channels(task, 40) is None
    log.unlink()
    assert q._logged_line_channels(task, 40) is None


@pytest.mark.unit
def test_noise_history_reads_the_first_run_of_a_name(tmp_path):
    log = tmp_path / "x.log"
    log.write_text(
        "# /bin/wsclean -name /o/selfcal-0 -j 1 x.ms\n"
        "Estimated standard deviation of background noise: 1 Jy\n"
        "# /bin/wsclean -name /o/wsclean -j 1 x.ms\n"
        "Estimated standard deviation of background noise: 91 mJy\n"
        "Estimated standard deviation of background noise: -nan Jy\n"
        "Estimated standard deviation of background noise: oops mJy\n"
        "Estimated standard deviation of background noise: 26 mJy\n"
        "# /bin/wsclean -name /o/wsclean -j 1 -continue x.ms\n"
        "Estimated standard deviation of background noise: 25 mJy\n"
    )
    history = q.noise_history(log, "wsclean")
    assert history[0] == pytest.approx(0.091) and history[-1] == pytest.approx(0.026)
    assert math.isnan(history[1]) and len(history) == 3
    assert q.noise_history(log, "wsclean-cube") == []


@pytest.mark.unit
def test_helpers():
    assert math.isnan(q.robust_sigma(np.array([np.nan])))
    assert math.isnan(q._ratio(1.0, 0.0))
    assert math.isnan(q._beam_pixels(fits.Header()))
    assert not q._moment_mask(np.zeros((8, 8)), 2.0).any()
    with pytest.raises(TypeError):
        q._json_default(object())
    assert q._json_default(np.float32(1.5)) == 1.5


@pytest.mark.unit
def test_assess_task_and_summary(tmp_path):
    task = _task(tmp_path)
    result = q.assess_task(task)

    saved = json.loads((task / q.QUALITY_FILENAME).read_text())
    assert saved["continuum"]["snr"] == pytest.approx(result["continuum"]["snr"])
    assert "cube" in saved
    assert q.find_task_dirs(tmp_path) == [task]

    rows = q.summary_rows([result, {"task": str(tmp_path)}], tmp_path)
    assert rows[0]["task"] == "uid___A002_X1.ms.split/SPW-1" and rows[1]["task"] == "."
    assert rows[1]["cube.line_residual_share"] is None
    q.write_summary(rows, tmp_path / "s.csv")
    with (tmp_path / "s.csv").open() as handle:
        written = list(csv.DictReader(handle))
    assert written[0]["continuum.snr"] and len(written) == 2


@pytest.mark.unit
def test_clean_quality_command(tmp_path):
    task = _task(tmp_path)
    broken = tmp_path / "uid___A002_X2.ms.split" / "SPW-3"
    broken.mkdir(parents=True)
    (broken / "wsclean-image.fits").write_text("not a FITS file")
    (broken / "wsclean-residual.fits").write_text("not a FITS file")
    (broken / "wsclean-model.fits").write_text("not a FITS file")

    result = runner.invoke(cli.app, ["image", "clean-quality", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "uid___A002_X1.ms.split/SPW-1" in result.output and "cont.snr" in result.output
    assert "uid___A002_X2.ms.split/SPW-3" in result.output, "a broken task is reported"
    assert (tmp_path / q.SUMMARY_FILENAME).is_file()
    assert (task / "quality" / "cube-mom0.fits").is_file()

    custom = tmp_path / "custom.csv"
    result = runner.invoke(
        cli.app, ["image", "clean-quality", str(task), "--no-moments", "--summary", str(custom)]
    )
    assert result.exit_code == 0, result.output
    assert custom.is_file()


@pytest.mark.unit
def test_clean_quality_without_tasks(tmp_path):
    result = runner.invoke(cli.app, ["image", "clean-quality", str(tmp_path)])
    assert result.exit_code == 1 and "No imaged tasks" in result.output
