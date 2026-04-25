"""Hubble sky model implementation."""

import os
from typing import Any, Optional

import astropy.units as U
import numpy as np
from dask import delayed
from dask.distributed import Client
from skimage import io

from .base import SkyModel
from .utils import gaussian, interpolate_array, track_progress


@delayed
def hubble_image(hubble: np.ndarray, amp: float) -> np.ndarray:
    """Scale Hubble image by amplitude."""
    return hubble * amp


class HubbleSkyModel(SkyModel):
    """Hubble Top 100 sky model."""

    def __init__(
        self,
        datacube: Any,
        continuum: np.ndarray,
        line_fluxes: np.ndarray,
        pos_z: list[int],
        fwhm_z: list[float],
        n_px: int,
        n_chan: int,
        data_path: str,
        client: Client,
        update_progress: Optional[Any] = None,
    ):
        """
        Initialize Hubble sky model.

        Parameters
        ----------
        datacube : Any
            DataCube object
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
        data_path : str
            Path to Hubble image directory
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
        self.data_path = data_path

    def insert(self) -> Any:
        """Insert Hubble source into datacube."""
        files = np.array([file for file in os.listdir(self.data_path) if not file.startswith(".")])
        imfile = os.path.join(self.data_path, np.random.choice(files))
        img = io.imread(imfile).astype(np.float32)
        dims = np.shape(img)
        d3 = min(2, dims[2])
        avimg = np.average(img[:, :, :d3], axis=2)
        avimg -= np.min(avimg)
        avimg *= 1 / np.max(avimg)
        avimg = interpolate_array(avimg, self.n_px)
        avimg /= np.sum(avimg)
        z_idxs = np.arange(0, self.n_chan)
        gs = np.zeros(self.n_chan)
        skymodel = []
        for i in range(len(self.line_fluxes)):
            gs += gaussian(z_idxs, self.line_fluxes[i], self.pos_z[i], self.fwhm_z[i])
        for z in range(0, self.n_chan):
            delayed_result = hubble_image(avimg, gs[z] + self.continuum[z])
            skymodel.append(delayed_result)
        delayed_skymodel = delayed(np.stack)(skymodel, axis=0)
        futures = self.client.compute([delayed_skymodel])
        track_progress(self.update_progress, futures)
        skymodel = self.client.gather(futures)
        self.datacube._array = skymodel * U.Jy * U.pix**-2
        return self.datacube
