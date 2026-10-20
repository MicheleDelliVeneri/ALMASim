"""Base class for sky model generation."""

from abc import ABC, abstractmethod
from typing import Any, Optional

import astropy.units as U
import numpy as np
from dask.distributed import Client

from .utils import gaussian


def channel_first(datacube: Any) -> np.ndarray:
    """The datacube's values as a ``(chan, y, x)`` array, in its own unit.

    martini's ``DataCube`` stores ``(x, y, chan)``; this and
    :func:`add_channel_first` are the only places that know it (MADIV-28
    T1.2), so no caller guesses the layout from the shape, which is
    ambiguous when the cube has as many pixels per side as channels.
    """
    array = datacube._array
    return np.asarray(array.to_value(array.unit)).T


def add_channel_first(datacube: Any, cube: np.ndarray) -> None:
    """Add a ``(chan, y, x)`` cube in Jy / pix^2 to the datacube's values.

    Sky models add to the cube rather than replace it, so sources inserted
    one after another (serendipitous ones, a background) all stay in it.
    """
    cube = np.asarray(cube)
    expected = datacube._array.shape[::-1]
    if cube.shape != expected:
        raise ValueError(f"expected a (chan, y, x) cube of shape {expected}, got {cube.shape}")
    datacube._array += cube.T * U.Jy * U.pix**-2


class _ImmediateFuture:
    """A resolved future wrapping an already-computed value."""

    def __init__(self, value: Any):
        self._value = value

    def done(self) -> bool:
        return True

    def result(self) -> Any:
        return self._value


class _LocalClient:
    """Minimal in-process stand-in for a ``dask.distributed.Client``.

    The sky-model builders were written against the Dask ``Client`` API
    (``compute`` / ``gather`` / ``scatter``), but the ``sync`` and ``local``
    compute backends do not expose a Dask client, so ``client`` arrives as
    ``None`` and the builders crash with ``'NoneType' object has no attribute
    'compute'``. This shim evaluates the delayed graphs eagerly in-process and
    returns resolved futures, so the builders work under any backend without a
    running Dask cluster.
    """

    def compute(self, tasks: Any, sync: bool = False, **kwargs: Any) -> Any:
        import dask

        if isinstance(tasks, (list, tuple)):
            results = dask.compute(*tasks)
            return [_ImmediateFuture(r) for r in results]
        return _ImmediateFuture(dask.compute(tasks)[0])

    def gather(self, futures: Any, **kwargs: Any) -> Any:
        if isinstance(futures, (list, tuple)):
            return [f.result() if hasattr(f, "result") else f for f in futures]
        return futures.result() if hasattr(futures, "result") else futures

    def scatter(self, data: Any, broadcast: bool = False, **kwargs: Any) -> Any:
        return data


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
        rng: Optional[np.random.Generator] = None,
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
        rng : Optional[np.random.Generator]
            Draws the model's random shape (MADIV-28 T5.1); fresh entropy when None
        """
        self.datacube = datacube
        self.continuum = continuum
        self.line_fluxes = line_fluxes
        self.pos_z = pos_z
        self.fwhm_z = fwhm_z
        self.n_px = n_px
        self.n_chan = n_chan
        # Fall back to an in-process client when no Dask cluster is provided
        # (sync / local backends pass client=None).
        self.client = client if client is not None else _LocalClient()
        self.update_progress = update_progress
        self.rng = rng if rng is not None else np.random.default_rng()

    def spectrum(self) -> np.ndarray:
        """Continuum plus emission lines per channel (Jy): the source's total flux."""
        z_idxs = np.arange(0, self.n_chan)
        lines = np.zeros(self.n_chan)
        for i in range(len(self.line_fluxes)):
            lines += gaussian(z_idxs, self.line_fluxes[i], self.pos_z[i], self.fwhm_z[i])
        return lines + np.asarray(self.continuum)

    def _add_cube(self, cube: np.ndarray) -> Any:
        """Add a ``(chan, y, x)`` cube to the datacube and report completion."""
        add_channel_first(self.datacube, cube)
        if self.update_progress is not None:
            self.update_progress.emit(100)
        return self.datacube

    def _add_template(self, template: np.ndarray) -> Any:
        """Add the spectrum spread over a fixed 2D ``template`` (``spectrum * template``)."""
        return self._add_cube(self.spectrum()[:, None, None] * template)

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
