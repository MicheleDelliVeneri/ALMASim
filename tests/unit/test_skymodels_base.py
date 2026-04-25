"""Unit tests for sky model base class."""

import numpy as np
from martini import DataCube
import astropy.units as U

from almasim.skymodels.base import SkyModel


class ConcreteSkyModel(SkyModel):
    """Concrete implementation for testing."""

    def insert(self):
        return self.datacube


def test_skymodel_initialization():
    """Test SkyModel base class initialization."""
    datacube = DataCube(
        n_px_x=64,
        n_px_y=64,
        n_channels=32,
        px_size=0.1 * U.arcsec,
        channel_width=0.1 * U.GHz,
        spectral_centre=100.0 * U.GHz,
        ra=0.0 * U.deg,
        dec=0.0 * U.deg,
    )
    continuum = np.ones(32)
    line_fluxes = np.ones(5)
    pos_z = [10, 11, 12, 13, 14]
    fwhm_z = [1.0, 1.0, 1.0, 1.0, 1.0]

    model = ConcreteSkyModel(
        datacube=datacube,
        continuum=continuum,
        line_fluxes=line_fluxes,
        pos_z=pos_z,
        fwhm_z=fwhm_z,
        n_px=64,
        n_chan=32,
    )

    assert model.datacube is datacube
    assert np.array_equal(model.continuum, continuum)
    assert np.array_equal(model.line_fluxes, line_fluxes)
    assert model.pos_z == pos_z
    assert model.fwhm_z == fwhm_z
    assert model.n_px == 64
    assert model.n_chan == 32
