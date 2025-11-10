"""Unit tests for pointlike sky model."""
import pytest
import numpy as np
from martini import DataCube
import astropy.units as U

from almasim.skymodels.pointlike import PointlikeSkyModel


def test_pointlike_sky_model_insert():
    """Test pointlike sky model insertion."""
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
    continuum = np.ones(32) * 0.1
    line_fluxes = np.array([1.0, 0.5, 0.3])
    pos_z = [10, 11, 12]
    fwhm_z = [2.0, 2.0, 2.0]
    
    model = PointlikeSkyModel(
        datacube=datacube,
        continuum=continuum,
        line_fluxes=line_fluxes,
        pos_x=32,
        pos_y=32,
        pos_z=pos_z,
        fwhm_z=fwhm_z,
        n_chan=32,
    )
    
    result = model.insert()
    assert result is not None
    assert hasattr(result, '_array')


