"""predict_from_model writes the model's visibilities to MODEL_DATA (MADIV-28 T1.4).

Built on a tiny real MeasurementSet, so it runs where python-casacore and
ducc0 are installed and is skipped elsewhere.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from almasim.cli import image as cli_image

tables = pytest.importorskip("casacore.tables")
wgridder = pytest.importorskip("ducc0.wgridder")

N_ROWS = 6
FREQS = np.array([100.0e9, 100.1e9, 100.2e9, 100.3e9])
CELL_RAD = np.deg2rad(0.5 / 3600)


def _make_ms(path, freqs=FREQS, corr_types=(9, 12)):
    """An MS with one spectral window, one polarization setup and DATA, no MODEL_DATA."""
    ms = tables.default_ms(str(path))
    ms.addcols(tables.maketabdesc(tables.makearrcoldesc("DATA", 0j, ndim=2, valuetype="complex")))
    ms.addrows(N_ROWS)
    rng = np.random.default_rng(1)
    ms.putcol("UVW", rng.uniform(-300, 300, size=(N_ROWS, 3)))
    ms.putcol("DATA_DESC_ID", np.zeros(N_ROWS, dtype=np.int32))
    for row in range(N_ROWS):
        ms.putcell("DATA", row, np.zeros((len(freqs), len(corr_types)), dtype=np.complex64))
    ms.close()
    spw = tables.table(f"{path}/SPECTRAL_WINDOW", readonly=False, ack=False)
    spw.addrows(1)
    spw.putcell("CHAN_FREQ", 0, freqs)
    spw.putcell("NUM_CHAN", 0, len(freqs))
    spw.close()
    pol = tables.table(f"{path}/POLARIZATION", readonly=False, ack=False)
    pol.addrows(1)
    pol.putcell("CORR_TYPE", 0, np.array(corr_types, dtype=np.int32))
    pol.putcell("NUM_CORR", 0, len(corr_types))
    pol.close()
    dd = tables.table(f"{path}/DATA_DESCRIPTION", readonly=False, ack=False)
    dd.addrows(1)
    dd.putcell("SPECTRAL_WINDOW_ID", 0, 0)
    dd.putcell("POLARIZATION_ID", 0, 0)
    dd.close()


def _write_model(path, planes):
    """FITS model with axes [stokes, freq, y, x] like WSClean's."""
    header = fits.Header()
    header["CDELT1"] = -np.rad2deg(CELL_RAD)
    header["CDELT2"] = np.rad2deg(CELL_RAD)
    fits.PrimaryHDU(planes[None].astype(np.float32), header).writeto(path)


def _expected(uvw, plane, freqs):
    return wgridder.dirty2vis(
        uvw=uvw,
        freq=freqs,
        dirty=plane.astype(np.float64),
        pixsize_x=CELL_RAD,
        pixsize_y=CELL_RAD,
        do_wgridding=True,
        # Near-exact reference; the prediction runs at epsilon 1e-6 (MADIV-28 T3.7).
        epsilon=1e-10,
        nthreads=1,
    )


def _model_data(path):
    ms = tables.table(str(path), ack=False)
    try:
        return ms.getcol("MODEL_DATA"), ms.getcol("UVW")
    finally:
        ms.close()


@pytest.fixture
def ms(tmp_path):
    _make_ms(tmp_path / "in.ms")
    return tmp_path / "in.ms"


def test_single_plane_model_predicts_every_channel_on_parallel_hands(ms, tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    plane = np.zeros((16, 16))
    plane[8, 8], plane[5, 10] = 1.0, 0.25
    _write_model(tmp_path / "model.fits", plane[None])

    cli_image.predict_from_model(ms, tmp_path / "model.fits", tmp_path / "out.ms")

    model_data, uvw = _model_data(tmp_path / "out.ms")
    assert model_data.shape == (N_ROWS, len(FREQS), 2), "(nrows, nchan, ncorr)"
    expected = _expected(uvw, plane, FREQS)
    # Stokes I on both parallel hands: XX = YY = I (I = (XX + YY) / 2).
    np.testing.assert_allclose(model_data[:, :, 0], expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(model_data[:, :, 1], expected, rtol=1e-5, atol=1e-6)


def test_cube_model_predicts_each_channel_from_its_own_plane(ms, tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    planes = np.zeros((len(FREQS), 16, 16))
    for c in range(len(FREQS)):
        planes[c, 4 + 2 * c, 8] = 1.0 + c
    _write_model(tmp_path / "cube.fits", planes)

    cli_image.predict_from_model(ms, tmp_path / "cube.fits", tmp_path / "out.ms")

    model_data, uvw = _model_data(tmp_path / "out.ms")
    for c in range(len(FREQS)):
        expected = _expected(uvw, planes[c], FREQS[c : c + 1])[:, 0]
        np.testing.assert_allclose(model_data[:, c, 0], expected, rtol=1e-5, atol=1e-6)


def test_cross_hands_stay_zero(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    _make_ms(tmp_path / "in.ms", corr_types=(9, 10, 11, 12))  # XX XY YX YY
    plane = np.zeros((16, 16))
    plane[8, 8] = 1.0
    _write_model(tmp_path / "model.fits", plane[None])

    cli_image.predict_from_model(tmp_path / "in.ms", tmp_path / "model.fits", tmp_path / "out.ms")

    model_data, _ = _model_data(tmp_path / "out.ms")
    assert np.all(model_data[:, :, 1:3] == 0)
    np.testing.assert_array_equal(model_data[:, :, 0], model_data[:, :, 3])


def test_model_with_wrong_number_of_planes_is_rejected(ms, tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    _write_model(tmp_path / "bad.fits", np.zeros((3, 16, 16)))
    with pytest.raises(ValueError, match="3 planes"):
        cli_image.predict_from_model(ms, tmp_path / "bad.fits", tmp_path / "out.ms")
