"""Pointlike sky model implementation."""
import numpy as np
import astropy.units as U
from typing import Optional, Any

from .base import SkyModel
from .utils import gaussian


class PointlikeSkyModel(SkyModel):
    """Point source sky model."""
    
    def __init__(
        self,
        datacube: Any,
        continuum: np.ndarray,
        line_fluxes: np.ndarray,
        pos_x: int,
        pos_y: int,
        pos_z: list[int],
        fwhm_z: list[float],
        n_chan: int,
        update_progress: Optional[Any] = None,
    ):
        """
        Initialize pointlike sky model.
        
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
        fwhm_z : list[float]
            FWHM in channels for each line
        n_chan : int
            Number of spectral channels
        update_progress : Optional[Any]
            Progress emitter callback
        """
        super().__init__(
            datacube=datacube,
            continuum=continuum,
            line_fluxes=line_fluxes,
            pos_z=pos_z,
            fwhm_z=fwhm_z,
            n_px=0,  # Not used for pointlike
            n_chan=n_chan,
            client=None,  # Not used for pointlike
            update_progress=update_progress,
        )
        self.pos_x = pos_x
        self.pos_y = pos_y
    
    def insert(self) -> Any:
        """Insert point source into datacube."""
        z_idxs = np.arange(0, self.n_chan)
        gs = np.zeros(self.n_chan)
        for i in range(len(self.line_fluxes)):
            gs += gaussian(z_idxs, self.line_fluxes[i], self.pos_z[i], self.fwhm_z[i])
            if self.update_progress is not None:
                self.update_progress.emit(i / len(self.line_fluxes) * 100)
        
        self.datacube._array[
            self.pos_x,
            self.pos_y,
            :,
        ] = (
            (self.continuum + gs) * U.Jy * U.pix**-2
        )
        return self.datacube


