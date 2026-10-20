"""Gaussian sky model implementation."""

import math
from typing import Any, Optional

import numpy as np
from dask.distributed import Client

from almasim.services.instrument.constants import FWHM_PER_SIGMA

from .base import SkyModel


def gaussian2d(n_px: int, cen_x: float, cen_y: float, fwhm_x: float, fwhm_y: float, angle: float):
    """Unnormalised 2D Gaussian on an ``n_px`` square grid, rotated by ``angle`` degrees."""
    x, y = np.meshgrid(np.arange(n_px), np.arange(n_px))
    angle_rad = math.radians(angle)
    # Rotate coordinates
    xp = (x - cen_x) * np.cos(angle_rad) - (y - cen_y) * np.sin(angle_rad) + cen_x
    yp = (x - cen_x) * np.sin(angle_rad) + (y - cen_y) * np.cos(angle_rad) + cen_y
    return np.exp(
        -(
            (xp - cen_x) ** 2 / (2 * (fwhm_x / FWHM_PER_SIGMA) ** 2)
            + (yp - cen_y) ** 2 / (2 * (fwhm_y / FWHM_PER_SIGMA) ** 2)
        )
    )


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
        shape = gaussian2d(self.n_px, self.pos_x, self.pos_y, self.fwhm_x, self.fwhm_y, self.angle)
        # Each channel holds its flux, spread as the normalised Gaussian.
        return self._add_cube((self.spectrum() / np.sum(shape))[:, None, None] * shape)
