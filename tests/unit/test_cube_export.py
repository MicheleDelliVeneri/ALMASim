"""Characterization tests for the cube exporters (MADIV-28 T8.2).

They pin the file names, formats and contents each save mode produces, so the
shared FITS I/O of T2.8 can replace the writers without changing products.
"""

from __future__ import annotations

import json
from pathlib import Path

import astropy.units as U
import h5py
import numpy as np
import pytest
from astropy.io import fits

from almasim.services.products import cube_export

pytestmark = pytest.mark.unit

_NAMES = ("clean-cube", "dirty-cube", "dirty-vis-cube", "clean-vis-cube", "beam-cube")
_OPERATOR = ("totsampling-cube", "uv-mask-cube", "u-cube", "v-cube")


def _products(rng):
    shape = (2, 4, 4)
    real = lambda: rng.normal(size=shape).astype(np.float32)  # noqa: E731
    return dict(
        model_cube=real(),
        dirty_cube=real(),
        vis_cube=real() + 1j * real(),
        dirty_vis_cube=real() + 1j * real(),
        beam_cube=real(),
        totsampling_cube=real(),
        uv_mask_cube=real(),
        u_cube=real(),
        v_cube=real(),
    )


def _header():
    header = fits.Header()
    header["BUNIT"] = "Jy/beam"
    header["CRVAL3"] = 1.0e11
    return header


def _write(tmp_path, mode, products):
    cube_export.write_interferometry_products(
        tmp_path, idx=3, save_mode=mode, header=_header(), **products
    )
    return sorted(p.name for p in Path(tmp_path).iterdir())


def test_npz_mode_writes_every_cube_as_npz(tmp_path):
    products = _products(np.random.default_rng(0))
    names = _write(tmp_path, "npz", products)
    assert names == sorted(f"{n}_3.npz" for n in (*_NAMES, *_OPERATOR))
    with np.load(tmp_path / "dirty-vis-cube_3.npz") as data:
        np.testing.assert_array_equal(data["arr_0"], products["dirty_vis_cube"])


def test_fits_mode_splits_complex_cubes_and_records_data_range(tmp_path):
    products = _products(np.random.default_rng(1))
    names = _write(tmp_path, "fits", products)
    expected = {
        "clean-cube_3.fits",
        "dirty-cube_3.fits",
        "dirty-vis-cube_real3.fits",
        "dirty-vis-cube_imag3.fits",
        "clean-vis-cube_real3.fits",
        "clean-vis-cube_imag3.fits",
        "beam-cube_3.fits",
        *(f"{n}_3.fits" for n in _OPERATOR),
    }
    assert set(names) == expected

    with fits.open(tmp_path / "dirty-cube_3.fits") as hdul:
        header, data = hdul[0].header, hdul[0].data
    np.testing.assert_array_equal(data, products["dirty_cube"])
    assert header["BUNIT"] == "Jy/beam"
    assert header["DATAMAX"] == pytest.approx(float(products["dirty_cube"].max()))
    assert header["DATAMIN"] == pytest.approx(float(products["dirty_cube"].min()))
    with fits.open(tmp_path / "clean-vis-cube_imag3.fits") as hdul:
        np.testing.assert_array_equal(hdul[0].data, products["vis_cube"].imag)
    with fits.open(tmp_path / "beam-cube_3.fits") as hdul:
        assert "BUNIT" not in hdul[0].header, "only the image cubes carry the sky header"


def test_both_mode_writes_npz_and_fits(tmp_path):
    names = _write(tmp_path, "both", _products(np.random.default_rng(2)))
    assert sum(n.endswith(".npz") for n in names) == 9
    assert sum(n.endswith(".fits") for n in names) == 11


def test_h5_mode_groups_the_measurement_operator(tmp_path):
    products = _products(np.random.default_rng(3))
    names = _write(tmp_path, "h5", products)
    assert names == sorted(
        f"{n}_3.h5"
        for n in (
            "clean-cube",
            "dirty-cube",
            "dirty-vis-cube",
            "clean-vis-cube",
            "measurement-operator",
        )
    )
    with h5py.File(tmp_path / "measurement-operator_3.h5") as handle:
        assert sorted(handle) == [
            "beam_cube",
            "totsampling_cube",
            "u_cube",
            "uv_mask_cube",
            "v_cube",
        ]
        np.testing.assert_array_equal(handle["u_cube"][()], products["u_cube"])


def test_unknown_mode_is_refused(tmp_path):
    with pytest.raises(ValueError, match="Unsupported save_mode"):
        _write(tmp_path, "zarr", _products(np.random.default_rng(4)))


@pytest.mark.parametrize(
    ("mode", "suffixes"),
    [("npz", [".npz"]), ("h5", [".h5"]), ("fits", [".fits"]), ("both", [".npz", ".fits"])],
)
def test_save_optional_cube_formats(tmp_path, mode, suffixes):
    cube = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    saved = cube_export.save_optional_cube(tmp_path, "noise-cube", 1, cube, mode, _header())
    assert [Path(p).suffix for p in saved] == suffixes
    assert all(Path(p).name.startswith("noise-cube_1") for p in saved)
    if mode == "h5":
        with h5py.File(saved[0]) as handle:
            assert list(handle) == ["noise_cube"]


def test_save_optional_cube_skips_a_missing_cube(tmp_path):
    assert cube_export.save_optional_cube(tmp_path, "x", 0, None, "both", _header()) == []
    assert not any(tmp_path.iterdir())


def test_ml_shard_holds_the_cubes_and_json_metadata(tmp_path):
    cube = np.ones((2, 3, 3), dtype=np.float32)
    path = cube_export.write_ml_dataset_shard(
        tmp_path / "shards" / "s0.h5",
        clean_cube=cube,
        dirty_cube=cube * 2,
        dirty_vis=cube * 3,
        uv_mask_cube=cube * 4,
        metadata={
            "band": np.int64(6),
            "path": tmp_path,
            "spws": np.array([1, 2]),
            "obj": object,
        },
    )
    with h5py.File(path) as handle:
        assert sorted(handle) == ["clean_cube", "dirty_cube", "dirty_vis", "uv_mask_cube"]
        assert handle["dirty_vis"][0, 0, 0] == 3
        metadata = json.loads(handle.attrs["metadata_json"])
    assert metadata["band"] == 6
    assert metadata["path"] == str(tmp_path)
    assert metadata["spws"] == [1, 2]
    assert metadata["obj"] == str(object)


@pytest.mark.xfail(
    strict=True,
    raises=NotImplementedError,
    reason="_json_safe tests np.ndarray before U.Quantity, and Quantity.tolist() refuses; "
    "the simulator pre-converts its metadata, so only direct API callers hit this. "
    "Fixed with the single JSON serialiser of T2.8.",
)
def test_ml_shard_metadata_may_hold_quantities(tmp_path):
    cube = np.ones((1, 1, 1), dtype=np.float32)
    cube_export.write_ml_dataset_shard(
        tmp_path / "q.h5",
        clean_cube=cube,
        dirty_cube=cube,
        dirty_vis=cube,
        uv_mask_cube=cube,
        metadata={"freq": 100.0 * U.GHz},
    )
