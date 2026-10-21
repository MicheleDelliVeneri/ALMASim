"""Base computation backend interface."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, TypeVar

try:
    from dask import delayed as dask_delayed
except ImportError:  # pragma: no cover - dask is a core dependency
    dask_delayed = None

T = TypeVar("T")


class ComputationBackend(ABC):
    """Abstract base class for computation backends.

    This interface allows services to be backend-agnostic, supporting
    different execution environments (local, Dask, Slurm, Kubernetes).
    """

    @abstractmethod
    def scatter(self, data: Any, broadcast: bool = False) -> Any:
        """Scatter data to workers.

        Parameters
        ----------
        data : Any
            Data to scatter to workers
        broadcast : bool, optional
            Whether to broadcast data to all workers (default: False)

        Returns
        -------
        Any
            Scattered data reference (backend-specific)
        """
        pass

    @abstractmethod
    def compute(self, tasks: Any, sync: bool = True) -> Any:
        """Compute tasks.

        Parameters
        ----------
        tasks : Any
            Tasks to compute (can be a single task or list of tasks)
        sync : bool, optional
            Whether to wait for completion (default: True)

        Returns
        -------
        Any
            Computation result or future (backend-specific)
        """
        pass

    @abstractmethod
    def gather(self, futures: Any) -> list[Any]:
        """Gather results from futures.

        Parameters
        ----------
        futures : Any
            Futures or task references to gather results from

        Returns
        -------
        List[Any]
            List of results
        """
        pass

    @abstractmethod
    def delayed(self, func: Callable) -> Callable:
        """Create a delayed version of a function.

        Parameters
        ----------
        func : Callable
            Function to make delayed

        Returns
        -------
        Callable
            Delayed function decorator
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the backend and clean up resources."""
        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: :meth:`close`."""
        self.close()


class DaskClientBackend(ComputationBackend):
    """A backend that runs everything on a ``dask.distributed`` client (``self.client``).

    The Dask, Slurm and Kubernetes backends differ only in how they start and
    stop the cluster behind the client (MADIV-28 T6.5).
    """

    client: Any = None

    def _require_client(self) -> Any:
        if self.client is None:
            raise RuntimeError("Dask client not initialized")
        return self.client

    def scatter(self, data: Any, broadcast: bool = False) -> Any:
        """Scatter data to the workers."""
        return self._require_client().scatter(data, broadcast=broadcast)

    def compute(self, tasks: Any, sync: bool = True) -> Any:
        """Compute tasks on the workers."""
        return self._require_client().compute(tasks, sync=sync)

    def gather(self, futures: Any) -> list[Any]:
        """Gather results from futures (a single future gives a one-item list)."""
        client = self._require_client()
        if isinstance(futures, list):
            return client.gather(futures)
        return [client.gather([futures])[0]]

    def delayed(self, func: Callable) -> Callable:
        """A Dask delayed version of ``func``."""
        if dask_delayed is None:
            raise ImportError("Dask delayed is not available")
        return dask_delayed(func)
