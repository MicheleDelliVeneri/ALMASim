"""Component tests for sky model integration."""
import pytest
import numpy as np
from martini import DataCube
import astropy.units as U

from almasim import skymodels


class InlineClient:
    """Minimal synchronous Dask-like client for component tests."""

    def compute(self, tasks):
        if isinstance(tasks, list):
            return [
                task.compute(scheduler="synchronous") if hasattr(task, "compute") else task
                for task in tasks
            ]
        return tasks.compute(scheduler="synchronous") if hasattr(tasks, "compute") else tasks

    def gather(self, futures):
        return futures if isinstance(futures, list) else [futures]


def test_pointlike_model_integration():
    """Test pointlike sky model end-to-end."""
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
    line_fluxes = np.array([1.0, 0.5])
    pos_z = [10, 11]
    fwhm_z = [2.0, 2.0]
    
    model = skymodels.PointlikeSkyModel(
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
    array = result._array.to_value(result._array.unit)
    assert array.shape == (32, 64, 64) or array.shape == (64, 64, 32)


def test_gaussian_model_integration():
    """Test Gaussian sky model end-to-end."""
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
    line_fluxes = np.array([1.0])
    pos_z = [10]
    fwhm_z = [2.0]
    
    model = skymodels.GaussianSkyModel(
        datacube=datacube,
        continuum=continuum,
        line_fluxes=line_fluxes,
        pos_x=32,
        pos_y=32,
        pos_z=pos_z,
        fwhm_x=5,
        fwhm_y=5,
        fwhm_z=fwhm_z,
        angle=45,
        n_px=64,
        n_chan=32,
        client=InlineClient(),
    )
    
    result = model.insert()
    assert result is not None
