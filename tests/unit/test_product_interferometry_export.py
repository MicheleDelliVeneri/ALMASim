"""Unit tests for interferometric product exporters."""

import numpy as np
from astropy.io import fits

from almasim.services.products import write_interferometry_products


def _sample_cubes():
    real_cube = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    complex_cube = real_cube.astype(np.complex64) + 1j * (real_cube + 1)
    return real_cube, complex_cube


def test_write_interferometry_products_npz(tmp_path):
    real_cube, complex_cube = _sample_cubes()

    write_interferometry_products(
        tmp_path,
        idx=3,
        save_mode="npz",
        header=None,
        model_cube=real_cube,
        vis_cube=complex_cube,
        dirty_cube=real_cube,
        dirty_vis_cube=complex_cube,
        beam_cube=real_cube,
        totsampling_cube=real_cube,
        uv_mask_cube=real_cube,
        u_cube=real_cube,
        v_cube=real_cube,
    )

    assert (tmp_path / "clean-cube_3.npz").exists()
    assert (tmp_path / "dirty-cube_3.npz").exists()
    assert (tmp_path / "dirty-vis-cube_3.npz").exists()
    assert (tmp_path / "clean-vis-cube_3.npz").exists()
    assert (tmp_path / "beam-cube_3.npz").exists()
    assert (tmp_path / "totsampling-cube_3.npz").exists()
    assert (tmp_path / "uv-mask-cube_3.npz").exists()
    assert (tmp_path / "u-cube_3.npz").exists()
    assert (tmp_path / "v-cube_3.npz").exists()


def test_write_interferometry_products_h5(tmp_path):
    real_cube, complex_cube = _sample_cubes()

    write_interferometry_products(
        tmp_path,
        idx=1,
        save_mode="h5",
        header=None,
        model_cube=real_cube,
        vis_cube=complex_cube,
        dirty_cube=real_cube,
        dirty_vis_cube=complex_cube,
        beam_cube=real_cube,
        totsampling_cube=real_cube,
        uv_mask_cube=real_cube,
        u_cube=real_cube,
        v_cube=real_cube,
    )

    assert (tmp_path / "clean-cube_1.h5").exists()
    assert (tmp_path / "dirty-cube_1.h5").exists()
    assert (tmp_path / "dirty-vis-cube_1.h5").exists()
    assert (tmp_path / "clean-vis-cube_1.h5").exists()
    assert (tmp_path / "measurement-operator_1.h5").exists()


def test_write_interferometry_products_fits_preserves_names_and_header(tmp_path):
    real_cube, complex_cube = _sample_cubes()
    header = fits.Header()
    header["OBJECT"] = "UnitTest"

    write_interferometry_products(
        tmp_path,
        idx=2,
        save_mode="fits",
        header=header,
        model_cube=real_cube,
        vis_cube=complex_cube,
        dirty_cube=real_cube,
        dirty_vis_cube=complex_cube,
        beam_cube=real_cube,
        totsampling_cube=real_cube,
        uv_mask_cube=real_cube,
        u_cube=real_cube,
        v_cube=real_cube,
    )

    assert (tmp_path / "clean-cube_2.fits").exists()
    assert (tmp_path / "dirty-cube_2.fits").exists()
    assert (tmp_path / "dirty-vis-cube_real2.fits").exists()
    assert (tmp_path / "dirty-vis-cube_imag2.fits").exists()
    assert (tmp_path / "clean-vis-cube_real2.fits").exists()
    assert (tmp_path / "clean-vis-cube_imag2.fits").exists()

    saved_header = fits.getheader(tmp_path / "clean-cube_2.fits")
    assert saved_header["OBJECT"] == "UnitTest"
    assert "DATAMAX" in saved_header
    assert "DATAMIN" in saved_header
