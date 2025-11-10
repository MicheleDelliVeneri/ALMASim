"""Base class for sky model generation."""
from abc import ABC, abstractmethod
from typing import Optional, Any
import numpy as np
import astropy.units as U
from dask.distributed import Client

from .utils import gaussian


class SkyModel(ABC):
    """Base class for all sky model types."""
    
    def __init__(
        self,
        datacube: Any,
        continuum: np.ndarray,
        line_fluxes: np.ndarray,
        pos_z: list[int],
        fwhm_z: list[float],
        n_px: int,
        n_chan: int,
        client: Optional[Client] = None,
        update_progress: Optional[Any] = None,
    ):
        """
        Initialize base sky model.
        
        Parameters
        ----------
        datacube : Any
            DataCube object from martini
        continuum : np.ndarray
            Continuum flux values per channel
        line_fluxes : np.ndarray
            Flux values for each emission line
        pos_z : list[int]
            Channel positions for each line
        fwhm_z : list[float]
            FWHM in channels for each line
        n_px : int
            Number of pixels per axis
        n_chan : int
            Number of spectral channels
        client : Optional[Client]
            Dask client for parallel processing
        update_progress : Optional[Any]
            Progress emitter callback
        """
        self.datacube = datacube
        self.continuum = continuum
        self.line_fluxes = line_fluxes
        self.pos_z = pos_z
        self.fwhm_z = fwhm_z
        self.n_px = n_px
        self.n_chan = n_chan
        self.client = client
        self.update_progress = update_progress
    
    def _compute_spectral_profile(self) -> np.ndarray:
        """
        Compute the spectral profile from continuum and line fluxes.
        
        Returns
        -------
        np.ndarray
            Spectral profile as a function of channel
        """
        z_idxs = np.arange(0, self.n_chan)
        gs = np.zeros(self.n_chan)
        for i in range(len(self.line_fluxes)):
            gs += gaussian(z_idxs, self.line_fluxes[i], self.pos_z[i], self.fwhm_z[i])
        return gs
    
    @abstractmethod
    def insert(self) -> Any:
        """
        Insert the sky model into the datacube.
        
        Returns
        -------
        Any
            Modified datacube object
        """
        pass


