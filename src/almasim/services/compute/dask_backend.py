"""Dask computation backend."""
from typing import Any, Callable, List, Optional

try:
    from dask.distributed import Client, LocalCluster
    from dask import delayed as dask_delayed
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    Client = None
    LocalCluster = None
    dask_delayed = None

from .base import ComputationBackend


class DaskBackend(ComputationBackend):
    """Dask computation backend for distributed computing."""

    def __init__(
        self,
        scheduler: Optional[str] = None,
        n_workers: Optional[int] = None,
    ):
        """Initialize Dask backend.
        
        Parameters
        ----------
        scheduler : str, optional
            Scheduler address (e.g., "tcp://localhost:8786") or None for local
        n_workers : int, optional
            Number of workers (only used for local cluster)
        """
        if not DASK_AVAILABLE:
            raise ImportError(
                "Dask is not installed. Install it with: pip install dask distributed"
            )

        self.scheduler = scheduler
        self.n_workers = n_workers
        self.client: Optional[Client] = None
        self.cluster: Optional[LocalCluster] = None
        self._start_client()

    def _start_client(self) -> None:
        """Start Dask client."""
        if self.scheduler and self.scheduler != "threads":
            # Connect to external scheduler
            self.client = Client(self.scheduler)
        else:
            # Create local cluster with processes for true parallelism
            # Explicitly use LocalCluster to avoid threads parameter issues
            cluster_kwargs = {"processes": True}
            if self.n_workers is not None:
                cluster_kwargs["n_workers"] = self.n_workers
            self.cluster = LocalCluster(**cluster_kwargs)
            self.client = Client(self.cluster)

    def scatter(self, data: Any, broadcast: bool = False) -> Any:
        """Scatter data to Dask workers."""
        if self.client is None:
            raise RuntimeError("Dask client not initialized")
        return self.client.scatter(data, broadcast=broadcast)

    def compute(self, tasks: Any, sync: bool = True) -> Any:
        """Compute tasks using Dask."""
        if self.client is None:
            raise RuntimeError("Dask client not initialized")
        return self.client.compute(tasks, sync=sync)

    def gather(self, futures: Any) -> List[Any]:
        """Gather results from Dask futures."""
        if self.client is None:
            raise RuntimeError("Dask client not initialized")
        if isinstance(futures, list):
            return self.client.gather(futures)
        else:
            return [self.client.gather([futures])[0]]

    def delayed(self, func: Callable) -> Callable:
        """Create a Dask delayed version of a function.
        
        Returns a decorator that can be used to wrap function calls.
        """
        if dask_delayed is None:
            raise ImportError("Dask delayed is not available")
        # dask.delayed can be used as a decorator or function
        # When used as decorator: @delayed, when used as function: delayed(func)(*args)
        # We return it directly as it supports both patterns
        return dask_delayed(func)

    def close(self) -> None:
        """Close Dask client and cluster."""
        if self.client:
            self.client.close()
            self.client = None
        if self.cluster:
            self.cluster.close()
            self.cluster = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

