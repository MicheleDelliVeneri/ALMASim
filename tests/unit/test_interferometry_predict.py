"""One ducc0 prediction for the CLI and the simulator (MADIV-28 T3.7)."""

import numpy as np
import pytest

from almasim.services.interferometry import predict

pytestmark = pytest.mark.unit


def _uvw(n=20, seed=0):
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(-300, 300, (n, 2)), rng.uniform(-5, 5, n)])


def test_a_centred_point_has_flat_visibilities():
    pytest.importorskip("ducc0")
    plane = np.zeros((16, 16))
    plane[8, 8] = 2.0
    vis = predict.predict_visibilities(_uvw(), [100e9, 101e9], plane, 1e-6, 1e-6, nthreads=1)
    assert vis.shape == (20, 2)
    np.testing.assert_allclose(vis, 2.0, atol=1e-4)


def test_an_off_centre_point_lands_where_fits_has_it():
    """A FITS plane (RA to the left) predicts the visibilities WSClean images back
    at the same pixel: V = exp(+2 pi i (u l + v m)) in ALMA's (MS) uvw convention,
    l = -(x - n/2) cell (east), m = (y - n/2) cell (north). Before the
    correction it came back rotated by 90 degrees."""
    pytest.importorskip("ducc0")
    n, cell, freq = 64, 0.05 / 3600 * np.pi / 180, 230e9
    plane = np.zeros((n, n))
    y0, x0 = 20, 45
    plane[y0, x0] = 1.0
    uvw = _uvw(50)
    uvw[:, 2] = 0.0
    lam = 299_792_458.0 / freq
    ell, em = -(x0 - n // 2) * cell, (y0 - n // 2) * cell
    expected = np.exp(2j * np.pi * (uvw[:, 0] * ell + uvw[:, 1] * em) / lam)
    vis = predict.predict_visibilities(uvw, [freq], plane, cell, cell, nthreads=1)[:, 0]
    np.testing.assert_allclose(vis, expected, atol=1e-4)


def test_a_cube_predicts_each_channel_from_its_plane():
    pytest.importorskip("ducc0")
    planes = np.zeros((3, 16, 16))
    for c in range(3):
        planes[c, 8, 8] = c + 1.0
    vis = predict.predict_visibilities(
        _uvw(), [100e9, 101e9, 102e9], planes, 1e-6, 1e-6, nthreads=1
    )
    np.testing.assert_allclose(vis.real.mean(axis=0), [1.0, 2.0, 3.0], atol=1e-4)


def test_a_cube_needs_one_plane_per_channel():
    with pytest.raises(ValueError, match="2 model planes for 3 channels"):
        predict.predict_visibilities(_uvw(), [1e11, 1.1e11, 1.2e11], np.zeros((2, 8, 8)), 1, 1)


def test_too_many_w_planes_falls_back_to_2d_gridding():
    calls = []

    def fake(**kwargs):
        calls.append(kwargs["do_wgridding"])
        if kwargs["do_wgridding"]:
            raise RuntimeError("too many w planes")
        return "ok"

    assert predict._ducc0_with_wgridder_fallback(fake, do_wgridding=True) == "ok"
    assert calls == [True, False]
    with pytest.raises(RuntimeError, match="other"):
        predict._ducc0_with_wgridder_fallback(
            lambda **k: (_ for _ in ()).throw(RuntimeError("other")), do_wgridding=True
        )
