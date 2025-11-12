"""Slurm computation backend using dask-jobqueue."""
from typing import Any, Callable, List, Optional, Dict

try:
    from dask_jobqueue import SLURMCluster
    from dask.distributed import Client
    from dask import delayed as dask_delayed
    SLURM_AVAILABLE = True
except ImportError:
    SLURM_AVAILABLE = False
    SLURMCluster = None
    Client = None
    dask_delayed = None

from .base import ComputationBackend


class SlurmBackend(ComputationBackend):
    """Slurm computation backend using dask-jobqueue."""

    def __init__(
        self,
        queue: str = "normal",
        project: Optional[str] = None,
        walltime: str = "02:00:00",
        cores: int = 1,
        memory: str = "4GB",
        n_workers: int = 4,
        **kwargs: Dict[str, Any],
    ):
        """Initialize Slurm backend.
        
        Parameters
        ----------
        queue : str
            Slurm queue name (default: "normal")
        project : str, optional
            Slurm project/account name
        walltime : str
            Job walltime in HH:MM:SS format (default: "02:00:00")
        cores : int
            Number of cores per worker (default: 1)
        memory : str
            Memory per worker (default: "4GB")
        n_workers : int
            Number of workers to start (default: 4)
        **kwargs
            Additional arguments passed to SLURMCluster
        """
        if not SLURM_AVAILABLE:
            raise ImportError(
                "dask-jobqueue is not installed. Install it with: pip install dask-jobqueue"
            )

        self.queue = queue
        self.project = project
        self.walltime = walltime
        self.cores = cores
        self.memory = memory
        self.n_workers = n_workers
        self.kwargs = kwargs

        self.cluster: Optional[SLURMCluster] = None
        self.client: Optional[Client] = None
        self._start_cluster()

    def _start_cluster(self) -> None:
        """Start Slurm cluster and client."""
        cluster_kwargs = {
            "queue": self.queue,
            "walltime": self.walltime,
            "cores": self.cores,
            "memory": self.memory,
            **self.kwargs,
        }
        if self.project:
            cluster_kwargs["project"] = self.project

        self.cluster = SLURMCluster(**cluster_kwargs)
        self.cluster.scale(self.n_workers)
        self.client = Client(self.cluster)

    def scatter(self, data: Any, broadcast: bool = False) -> Any:
        """Scatter data to Slurm workers."""
        if self.client is None:
            raise RuntimeError("Dask client not initialized")
        return self.client.scatter(data, broadcast=broadcast)

    def compute(self, tasks: Any, sync: bool = True) -> Any:
        """Compute tasks using Slurm workers."""
        if self.client is None:
            raise RuntimeError("Dask client not initialized")
        return self.client.compute(tasks, sync=sync)

    def gather(self, futures: Any) -> List[Any]:
        """Gather results from Slurm workers."""
        if self.client is None:
            raise RuntimeError("Dask client not initialized")
        if isinstance(futures, list):
            return self.client.gather(futures)
        else:
            return [self.client.gather([futures])[0]]

    def delayed(self, func: Callable) -> Callable:
        """Create a Dask delayed version of a function."""
        if dask_delayed is None:
            raise ImportError("Dask delayed is not available")
        return dask_delayed(func)

    def close(self) -> None:
        """Close Slurm cluster and client."""
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

