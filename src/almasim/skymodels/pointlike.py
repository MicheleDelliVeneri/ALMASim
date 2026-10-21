"""Pointlike sky model implementation."""

from typing import Any

import astropy.units as U
import numpy as np

from .base import SkyModel


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
        update_progress: Any | None = None,
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
        self.datacube._array[
            self.pos_x,
            self.pos_y,
            :,
        ] += self.spectrum() * U.Jy * U.pix**-2
        if self.update_progress is not None:
            self.update_progress.emit(100)
        return self.datacube
