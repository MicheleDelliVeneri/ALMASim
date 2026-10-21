"""Product previews: formats, channel windows and complex parts."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from almasim.services.products import preview

pytestmark = pytest.mark.unit


def test_classify_product_path_knows_npz_fits_and_msv2(tmp_path):
    np.savez(tmp_path / "cube.npz", clean_cube=np.ones((2, 3, 4), dtype=np.float32))
    fits.PrimaryHDU(np.ones((2, 3, 4), dtype=np.float32)).writeto(tmp_path / "cube.fits")
    (tmp_path / "example.ms").mkdir()
    (tmp_path / "notes.txt").write_text("x")

    assert preview.classify_product_path(tmp_path / "cube.npz") == "npz"
    assert preview.classify_product_path(tmp_path / "cube.fits") == "fits"
    assert preview.classify_product_path(tmp_path / "example.ms") == "msv2"
    assert preview.classify_product_path(tmp_path / "notes.txt") is None


def test_integrate_product_uses_an_inclusive_channel_window(tmp_path):
    cube = np.arange(2 * 4 * 5, dtype=np.float32).reshape(2, 4, 5)
    fits_path = tmp_path / "cube.fits"
    fits.PrimaryHDU(cube).writeto(fits_path)

    image, stats = preview.integrate_product(fits_path, "sum", channel_start=1, channel_end=1)

    assert stats["format"] == "fits"
    assert stats["integration_axis"] == 0
    assert (stats["channel_start"], stats["channel_end"]) == (1, 1)
    assert stats["window_shape"] == [1, 4, 5]
    np.testing.assert_array_equal(image, cube[1])


def test_complex_component_projection_uses_imaginary_part():
    cube = np.array([[[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]]], dtype=np.complex64)
    projected = preview.project_complex_component(cube, complex_component="imag")
    assert projected.tolist() == [[[2.0, 4.0], [6.0, 8.0]]]
