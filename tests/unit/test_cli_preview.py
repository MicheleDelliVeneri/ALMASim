"""`almasim preview`: maps and reconstructions as FITS for the viewer (U3.7)."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits
from typer.testing import CliRunner

from almasim import cli
from almasim.jobs.model import command_name
from almasim.jobs.policy import plan_job
from almasim.jobs.profile import Profile

pytestmark = pytest.mark.unit


def _cube(path, data):
    fits.PrimaryHDU(np.asarray(data, dtype=np.float32)).writeto(path)
    return path


def test_integrate_sums_the_channels_into_one_image(tmp_path):
    cube = np.zeros((4, 8, 8), dtype=np.float32)
    cube[1:3, 4, 4] = 2.0  # a line in channels 1 and 2
    source = _cube(tmp_path / "cube.fits", cube)
    out = tmp_path / "maps" / "mom0.fits"

    result = CliRunner().invoke(cli.app, ["preview", "integrate", str(source), str(out)])
    assert result.exit_code == 0, result.output
    with fits.open(out) as hdul:
        image, header = hdul[0].data, hdul[0].header
    assert image.shape == (8, 8)
    assert image[4, 4] == pytest.approx(4.0) and image.sum() == pytest.approx(4.0)
    assert header["METHOD"] == "sum" and header["SOURCE"] == "cube.fits"

    window = tmp_path / "one.fits"
    CliRunner().invoke(
        cli.app,
        ["preview", "integrate", str(source), str(window), "--channel-start", "2",
         "--channel-end", "2", "--method", "mean"],
    )  # fmt: skip
    assert fits.getdata(window)[4, 4] == pytest.approx(2.0)


def test_integrate_says_what_is_missing(tmp_path):
    result = CliRunner().invoke(
        cli.app, ["preview", "integrate", str(tmp_path / "nope.fits"), str(tmp_path / "o.fits")]
    )
    assert result.exit_code == 1 and "does not exist" in result.output


def test_wiener_writes_the_model_restored_and_residual(tmp_path):
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:16, :16]
    psf = np.exp(-((xx - 8) ** 2 + (yy - 8) ** 2) / 4.0)
    beam = np.stack([psf, psf])
    dirty = np.stack([np.roll(psf, (2, 1), (0, 1)), psf]) + rng.normal(0, 1e-3, (2, 16, 16))
    out = tmp_path / "wiener"
    result = CliRunner().invoke(
        cli.app,
        [
            "preview",
            "wiener",
            str(_cube(tmp_path / "dirty.fits", dirty)),
            str(_cube(tmp_path / "beam.fits", beam)),
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    for name in ("model", "restored", "residual"):
        assert fits.getdata(out / f"wiener-{name}.fits").shape == (2, 16, 16)


def test_the_console_can_run_them():
    argv = ["preview", "integrate", "/data/cube.fits", "{run_dir}/mom0.fits"]
    assert command_name(argv) == "preview integrate"
    assert plan_job(argv, Profile()).executor == "slurm"
    assert command_name(["preview", "wiener", "d", "b", "o"]) == "preview wiener"
