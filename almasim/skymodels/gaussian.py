"""Gaussian sky model implementation."""
import numpy as np
import math
import astropy.units as U
from typing import Optional, Any
from dask import delayed
from dask.distributed import Client

from .base import SkyModel
from .utils import track_progress, gaussian


@delayed
def gaussian2d(amp, x, y, n_px, cen_x, cen_y, fwhm_x, fwhm_y, angle):
    """
    Generates a 2D Gaussian given the following input parameters:
    
    Parameters
    ----------
    x, y : np.ndarray
        Position arrays
    amp : float
        Amplitude
    cen_x, cen_y : float
        Center positions
    fwhm_x, fwhm_y : float
        FWHMs (full width at half maximum) along x and y axes
    angle : float
        Angle of rotation (in degrees)
    """
    angle_rad = math.radians(angle)
    # Rotate coordinates
    xp = (x - cen_x) * np.cos(angle_rad) - (y - cen_y) * np.sin(angle_rad) + cen_x
    yp = (x - cen_x) * np.sin(angle_rad) + (y - cen_y) * np.cos(angle_rad) + cen_y

    gaussian = np.exp(
        -(
            (xp - cen_x) ** 2 / (2 * (fwhm_x / 2.35482) ** 2)
            + (yp - cen_y) ** 2 / (2 * (fwhm_y / 2.35482) ** 2)
        )
    )
    norm = amp / np.sum(gaussian)

    result = norm * gaussian
    return result


class GaussianSkyModel(SkyModel):
    """2D Gaussian sky model."""
    
    def __init__(
        self,
        datacube: Any,
        continuum: np.ndarray,
        line_fluxes: np.ndarray,
        pos_x: int,
        pos_y: int,
        pos_z: list[int],
        fwhm_x: int,
        fwhm_y: int,
        fwhm_z: list[float],
        angle: int,
        n_px: int,
        n_chan: int,
        client: Client,
        update_progress: Optional[Any] = None,
    ):
        """
        Initialize Gaussian sky model.
        
        Parameters
        ----------
        datacube : Any
            DataCube object
        continuum : np.ndarray
            Continuum flux values per channel
        line_fluxes : np.ndarray
            Flux values for each emission line
        pos_x : int
            X position in pixels
        pos_y : int
            Y position in pixels
        pos_z : list[int]
            Channel positions for each line
        fwhm_x : int
            FWHM in x direction (pixels)
        fwhm_y : int
            FWHM in y direction (pixels)
        fwhm_z : list[float]
            FWHM in channels for each line
        angle : int
            Rotation angle in degrees
        n_px : int
            Number of pixels per axis
        n_chan : int
            Number of spectral channels
        client : Client
            Dask client for parallel processing
        update_progress : Optional[Any]
            Progress emitter callback
        """
        super().__init__(
            datacube=datacube,
            continuum=continuum,
            line_fluxes=line_fluxes,
            pos_z=pos_z,
            fwhm_z=fwhm_z,
            n_px=n_px,
            n_chan=n_chan,
            client=client,
            update_progress=update_progress,
        )
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.fwhm_x = fwhm_x
        self.fwhm_y = fwhm_y
        self.angle = angle
    
    def insert(self) -> Any:
        """Insert Gaussian source into datacube."""
        z_idxs = np.arange(0, self.n_chan)
        x, y = np.meshgrid(np.arange(self.n_px), np.arange(self.n_px))
        gs = np.zeros(self.n_chan)
        skymodel = []
        for i in range(len(self.line_fluxes)):
            gs += gaussian(z_idxs, self.line_fluxes[i], self.pos_z[i], self.fwhm_z[i])
        for z in range(0, self.n_chan):
            delayed_result = gaussian2d(
                gs[z] + self.continuum[z],
                x,
                y,
                self.n_px,
                self.pos_x,
                self.pos_y,
                self.fwhm_x,
                self.fwhm_y,
                self.angle,
            )
            skymodel.append(delayed_result)
        delayed_skymodel = delayed(np.stack)(skymodel, axis=0)
        futures = self.client.compute([delayed_skymodel])
        track_progress(self.update_progress, futures)
        skymodel = self.client.gather(futures)
        self.datacube._array = skymodel * U.Jy * U.pix**-2
        return self.datacube


