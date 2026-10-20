"""Fakes shared by the unit tests (MADIV-28 T0.5).

Helpers used by several test modules live here instead of being copied into
each of them.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import astropy.units as U
import numpy as np
from martini import DataCube


def make_datacube(n_px: int = 16, n_chan: int = 8) -> DataCube:
    """A small martini DataCube centred at 100 GHz with 0.1 arcsec pixels."""
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


def mock_dask_client(n_px: int, n_chan: int) -> MagicMock:
    """A Dask client stand-in: compute passes tasks through, gather returns a random cube."""
    client = MagicMock()
    client.compute.side_effect = lambda tasks: tasks
    client.gather.side_effect = lambda futures: [
        np.random.rand(n_chan, n_px, n_px).astype(np.float32)
    ]
    return client
