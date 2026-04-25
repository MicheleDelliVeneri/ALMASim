"""Base computation backend interface."""

from abc import ABC, abstractmethod
from typing import Any, Callable, List, TypeVar

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
    def gather(self, futures: Any) -> List[Any]:
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

    @abstractmethod
    def __enter__(self):
        """Context manager entry."""
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass
