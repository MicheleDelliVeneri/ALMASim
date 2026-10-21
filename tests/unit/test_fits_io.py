"""FITS cube reading and writing (MADIV-28 T2.8)."""

import numpy as np
import pytest
from astropy.io import fits

from almasim.services.imaging.fits_io import (
    ChannelBeam,
    read_cube,
    read_image,
    write_cube,
)


def _header():
    header = fits.Header()
    header["CTYPE3"] = "FREQ"
    header["BUNIT"] = "Jy/beam"
    return header


def test_write_cube_without_beams_is_one_hdu(tmp_path):
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    path = write_cube(tmp_path / "cube.fits", data, _header())

    with fits.open(path) as hdus:
        assert len(hdus) == 1
        assert "CASAMBM" not in hdus[0].header
    cube = read_cube(path)
    np.testing.assert_array_equal(cube.data, data)
    assert cube.header["BUNIT"] == "Jy/beam"
    assert cube.beams is None
    assert not (tmp_path / "cube.fits.partial").exists()


def test_write_cube_round_trips_channel_beams(tmp_path):
    beams = [
        ChannelBeam(1e-4, 5e-5, 10.0),
        ChannelBeam(2e-4, 1e-4, 20.0),
        ChannelBeam(3e-4, 1.5e-4, 30.0),
    ]
    header = _header()
    path = write_cube(tmp_path / "cube.fits", np.zeros((3, 2, 2), np.float32), header, beams)

    assert "BMAJ" not in header  # the caller's header is not changed
    with fits.open(path) as hdus:
        primary = hdus[0].header
        assert primary["CASAMBM"] is True
        assert primary["BMAJ"] == pytest.approx(2e-4)
        assert hdus["BEAMS"].header["NCHAN"] == 3
        assert list(hdus["BEAMS"].data["CHAN"]) == [0, 1, 2]
    cube = read_cube(path)
    assert cube.beams is not None
    for read, written in zip(cube.beams, beams, strict=True):
        assert read.bmaj_deg == pytest.approx(written.bmaj_deg, rel=1e-6)
        assert read.bmin_deg == pytest.approx(written.bmin_deg, rel=1e-6)
        assert read.bpa_deg == pytest.approx(written.bpa_deg, rel=1e-6)


def test_channel_beam_from_header():
    header = fits.Header({"BMAJ": 1.0, "BMIN": 0.5, "BPA": 45.0})
    assert ChannelBeam.from_header(header) == ChannelBeam(1.0, 0.5, 45.0)


def test_read_cube_uses_the_first_image_extension_with_data(tmp_path):
    path = tmp_path / "ext.fits"
    data = np.ones((2, 2), np.float32)
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(data, name="SCI")]).writeto(path)

    cube = read_cube(path)
    np.testing.assert_array_equal(cube.data, data)
    assert cube.header["EXTNAME"] == "SCI"
    np.testing.assert_array_equal(read_image(path), data)


def test_readers_reject_files_without_image_data(tmp_path):
    path = tmp_path / "empty.fits"
    fits.PrimaryHDU().writeto(path)

    with pytest.raises(ValueError, match="does not contain image data"):
        read_cube(path)
    with pytest.raises(ValueError, match="does not contain image data"):
        read_image(path)
