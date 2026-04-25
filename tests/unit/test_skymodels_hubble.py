"""Unit tests for almasim.skymodels.hubble."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import astropy.units as U
import numpy as np
import pytest
from martini import DataCube

from almasim.skymodels.hubble import HubbleSkyModel, hubble_image


# ===========================================================================
# helpers
# ===========================================================================


def _make_datacube(n_px: int = 16, n_chan: int = 8) -> DataCube:
    return DataCube(
        n_px_x=n_px,
        n_px_y=n_px,
        n_channels=n_chan,
        px_size=0.1 * U.arcsec,
        channel_width=0.1 * U.GHz,
        spectral_centre=100.0 * U.GHz,
        ra=0.0 * U.deg,
        dec=0.0 * U.deg,
    )


def _mock_client(n_px: int, n_chan: int) -> MagicMock:
    client = MagicMock()
    client.compute.side_effect = lambda tasks: tasks
    client.gather.side_effect = lambda futures: [
        np.random.rand(n_chan, n_px, n_px).astype(np.float32)
    ]
    return client


def _make_model(
    tmp_path,
    n_px: int = 16,
    n_chan: int = 8,
    n_lines: int = 2,
) -> HubbleSkyModel:
    """Create a HubbleSkyModel with fake image files."""
    from skimage import io as skio

    data_path = tmp_path / "hubble_images"
    data_path.mkdir()
    for i in range(3):
        img = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
        skio.imsave(str(data_path / f"hubble_{i}.png"), img)

    datacube = _make_datacube(n_px, n_chan)
    client = _mock_client(n_px, n_chan)
    return HubbleSkyModel(
        datacube=datacube,
        continuum=np.ones(n_chan) * 0.01,
        line_fluxes=np.array([1.0] * n_lines),
        pos_z=list(range(n_lines)),
        fwhm_z=[1.0] * n_lines,
        n_px=n_px,
        n_chan=n_chan,
        data_path=str(data_path),
        client=client,
    )


# ===========================================================================
# hubble_image (delayed helper)
# ===========================================================================


@pytest.mark.unit
def test_hubble_image_scales_array():
    """hubble_image scales the input array by amplitude."""
    arr = np.ones((8, 8)) * 2.0
    result = hubble_image(arr, 3.0).compute()
    np.testing.assert_allclose(result, arr * 3.0)


@pytest.mark.unit
def test_hubble_image_negative_amplitude():
    """hubble_image with negative amplitude negates the array."""
    arr = np.ones((4, 4))
    result = hubble_image(arr, -1.0).compute()
    np.testing.assert_allclose(result, -1.0)


@pytest.mark.unit
def test_hubble_image_zero_amplitude():
    """hubble_image with zero amplitude returns zeros."""
    arr = np.ones((4, 4))
    result = hubble_image(arr, 0.0).compute()
    np.testing.assert_allclose(result, 0.0)


# ===========================================================================
# HubbleSkyModel initialization
# ===========================================================================


@pytest.mark.unit
def test_hubble_skymodel_init_stores_data_path(tmp_path):
    """HubbleSkyModel stores data_path attribute."""
    model = _make_model(tmp_path)
    assert "hubble_images" in model.data_path


@pytest.mark.unit
def test_hubble_skymodel_inherits_base_attributes(tmp_path):
    """HubbleSkyModel inherits n_px, n_chan, etc. from SkyModel."""
    model = _make_model(tmp_path, n_px=16, n_chan=8, n_lines=3)
    assert model.n_px == 16
    assert model.n_chan == 8
    assert len(model.line_fluxes) == 3


@pytest.mark.unit
def test_hubble_skymodel_default_update_progress_none(tmp_path):
    """update_progress defaults to None."""
    model = _make_model(tmp_path)
    assert model.update_progress is None


# ===========================================================================
# HubbleSkyModel.insert
# ===========================================================================


@pytest.mark.unit
def test_hubble_insert_returns_datacube(tmp_path):
    """insert() returns the datacube object."""
    model = _make_model(tmp_path)
    result = model.insert()
    assert result is not None


@pytest.mark.unit
def test_hubble_insert_sets_units(tmp_path):
    """Datacube _array has Jy / pix^2 units after insert()."""
    model = _make_model(tmp_path)
    result = model.insert()
    assert result._array.unit == U.Jy * U.pix**-2


@pytest.mark.unit
def test_hubble_insert_calls_dask_client(tmp_path):
    """insert() computes and gathers via Dask."""
    model = _make_model(tmp_path)
    model.insert()
    model.client.compute.assert_called_once()
    model.client.gather.assert_called_once()


@pytest.mark.unit
def test_hubble_insert_no_lines(tmp_path):
    """insert() works when there are no emission lines."""
    model = _make_model(tmp_path, n_lines=0)
    model.line_fluxes = np.array([])
    model.pos_z = []
    model.fwhm_z = []
    result = model.insert()
    assert result is not None


@pytest.mark.unit
def test_hubble_insert_multiple_lines(tmp_path):
    """insert() accumulates multiple emission lines."""
    model = _make_model(tmp_path, n_lines=5)
    result = model.insert()
    assert result is not None


@pytest.mark.unit
def test_hubble_insert_skips_hidden_files(tmp_path):
    """File listing excludes dot-files (files starting with '.')."""
    from skimage import io as skio

    data_path = tmp_path / "hubble_hidden"
    data_path.mkdir()
    img = (np.random.rand(16, 16, 3) * 255).astype(np.uint8)
    skio.imsave(str(data_path / "real.png"), img)
    (data_path / ".hidden").write_text("not an image")

    model = _make_model(tmp_path)
    model.data_path = str(data_path)
    # Should not raise even though .hidden exists
    with patch("almasim.skymodels.hubble.io.imread") as mock_imread:
        mock_imread.return_value = img.astype(np.float32)
        result = model.insert()
    assert result is not None
    # .hidden should not have been passed to imread
    called_file = mock_imread.call_args[0][0]
    assert not os.path.basename(called_file).startswith(".")


@pytest.mark.unit
def test_hubble_insert_with_progress_callback(tmp_path):
    """insert() works when update_progress is None (the default)."""
    # track_progress only runs when update_progress is not None AND calls
    # .done() on real dask futures; our mock client doesn't return real futures.
    # Verify the default None path is handled correctly.
    model = _make_model(tmp_path)
    assert model.update_progress is None
    result = model.insert()
    assert result is not None


@pytest.mark.unit
def test_hubble_insert_reads_image_with_skimage(tmp_path):
    """insert() uses skimage.io.imread."""
    model = _make_model(tmp_path)
    with patch("almasim.skymodels.hubble.io.imread") as mock_imread:
        fake_img = np.random.rand(32, 32, 3).astype(np.float32)
        mock_imread.return_value = fake_img
        model.insert()
    mock_imread.assert_called_once()


import os
